from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
from PIL import Image

from .data_types import DetectionInstance, RGBDFrame
from .processing import mask_iou
from .slots import class_capacities

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


class InstanceDetector(Protocol):
    def detect(self, frame: RGBDFrame) -> list[DetectionInstance]:
        ...

    def detect_batch(
        self,
        frames: list[RGBDFrame],
    ) -> list[list[DetectionInstance]]:
        ...


class Sam3Detector:
    """Text-prompted SAM3 detector with true multi-image image batching.

    All synchronized camera images share one SAM3 image-backbone forward. Each
    text prompt is then grounded against the whole image batch in one model
    forward. The public single-image ``detect`` method is retained as a thin
    wrapper for debugging/tests.
    """

    _cache: ClassVar[dict[tuple[str, str], tuple[Any, Any]]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for SAM3")
        self.checkpoint = str(config.detector.checkpoint)
        self.device = str(config.runtime.device)
        self.use_bf16 = bool(config.runtime.get("use_bf16", True))
        self.capacities = class_capacities(config)
        self.prompts = list(self.capacities)
        self.score_threshold = float(config.detector.score_threshold)
        self.duplicate_iou = float(config.detector.duplicate_iou_threshold)
        self.min_mask_pixels = int(config.detector.get("min_mask_pixels", 30))
        self.last_filter_ms = 0.0
        self.last_counts_per_view: list[dict[str, int]] = []
        self.model, self.processor = self._get_or_build()

        # ``set_text_prompt`` used by the previous implementation already applied
        # the processor's own confidence/mask thresholds before our config-level
        # filtering. The batched path calls the model directly, so preserve that
        # effective behavior instead of accidentally admitting many extra queries.
        processor_score_threshold = float(
            getattr(self.processor, "confidence_threshold", 0.0)
        )
        self.effective_score_threshold = max(
            self.score_threshold, processor_score_threshold
        )
        self.effective_mask_threshold = float(
            getattr(self.processor, "mask_threshold", 0.5)
        )

    def _get_or_build(self) -> tuple[Any, Any]:
        key = (self.checkpoint, self.device)
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]

            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            checkpoint = Path(self.checkpoint)
            kwargs: dict[str, Any] = {
                "device": self.device,
                "eval_mode": True,
                "enable_segmentation": True,
                "compile": False,
            }
            if checkpoint.is_file():
                kwargs["checkpoint_path"] = str(checkpoint)
                kwargs["load_from_HF"] = False
            else:
                kwargs["load_from_HF"] = True

            model = build_sam3_image_model(**kwargs)
            processor = Sam3Processor(model)
            self._cache[key] = (model, processor)
            return model, processor

    @staticmethod
    def _pil(rgb: np.ndarray) -> Image.Image:
        return Image.fromarray(
            np.ascontiguousarray(rgb, dtype=np.uint8),
            mode="RGB",
        )

    def _append_candidate(
        self,
        detections: list[DetectionInstance],
        *,
        prompt: str,
        score: float,
        mask: np.ndarray,
        bbox: tuple[float, float, float, float] | None,
    ) -> None:
        # Keep this path intentionally tiny. Config-level thresholding, tiny-mask
        # rejection, same-class deduplication and per-class top-K happen together
        # after all prompt forwards, so one semantic class can never consume the
        # capacity reserved for another class.
        detections.append(
            DetectionInstance(
                detection_id=0,
                label=prompt,
                score=float(score),
                mask=np.asarray(mask, dtype=bool),
                bbox_xyxy=bbox,
                embedding=None,
            )
        )

    def _filter_view_candidates(
        self,
        candidates: list[DetectionInstance],
    ) -> tuple[list[DetectionInstance], dict[str, int]]:
        filtered: list[DetectionInstance] = []
        counts: dict[str, int] = {}
        for label in self.prompts:
            capacity = int(self.capacities[label])
            class_candidates = [
                item
                for item in candidates
                if item.label == label
                and float(item.score) >= self.effective_score_threshold
                and int(np.count_nonzero(item.mask)) >= self.min_mask_pixels
            ]
            class_candidates.sort(key=lambda item: float(item.score), reverse=True)

            kept: list[DetectionInstance] = []
            for candidate in class_candidates:
                if any(
                    mask_iou(candidate.mask, existing.mask) >= self.duplicate_iou
                    for existing in kept
                ):
                    continue
                kept.append(candidate)
                if len(kept) >= capacity:
                    break

            counts[label] = len(kept)
            filtered.extend(kept)

        for detection_id, item in enumerate(filtered, start=1):
            item.detection_id = detection_id
        return filtered, counts

    @torch.inference_mode()
    def detect_rgb_batch(
        self,
        rgbs: list[np.ndarray],
    ) -> list[list[DetectionInstance]]:
        """Run SAM3 on all synchronized RGB views in one image batch."""
        if not rgbs:
            return []
        if not hasattr(self.processor, "set_image_batch"):
            raise RuntimeError(
                "This SAM3 checkout has no Sam3Processor.set_image_batch(). "
                "Update SAM3 before using the EfficientTAM-only async pipeline."
            )

        from sam3.model import box_ops
        from sam3.model.data_misc import FindStage

        images = [self._pil(rgb) for rgb in rgbs]
        batch_size = len(images)
        autocast_enabled = self.use_bf16 and self.device.startswith("cuda")
        detections_per_view: list[list[DetectionInstance]] = [
            [] for _ in range(batch_size)
        ]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            state = self.processor.set_image_batch(images)
            device = torch.device(self.device)
            find_stage = FindStage(
                img_ids=torch.arange(batch_size, device=device, dtype=torch.long),
                text_ids=torch.zeros(batch_size, device=device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )

            for prompt in self.prompts:
                text_outputs = self.model.backbone.forward_text(
                    [prompt],
                    device=self.device,
                )
                state["backbone_out"].update(text_outputs)
                geometric_prompt = self.model._get_dummy_prompt(
                    num_prompts=batch_size
                )
                output = self.model.forward_grounding(
                    backbone_out=state["backbone_out"],
                    find_input=find_stage,
                    geometric_prompt=geometric_prompt,
                    find_target=None,
                )

                logits = output["pred_logits"]
                if logits.ndim == 2:
                    logits = logits.unsqueeze(-1)
                probabilities = logits.sigmoid().squeeze(-1)

                presence = output.get("presence_logit_dec")
                if presence is not None:
                    presence = presence.sigmoid().reshape(batch_size, -1)[:, :1]
                    probabilities = probabilities * presence

                boxes_xyxy = box_ops.box_cxcywh_to_xyxy(output["pred_boxes"])
                masks_logits = output["pred_masks"]

                for view_idx, rgb in enumerate(rgbs):
                    keep = probabilities[view_idx] > self.effective_score_threshold
                    if not bool(keep.any()):
                        continue

                    scores = probabilities[view_idx][keep]
                    boxes = boxes_xyxy[view_idx][keep]
                    masks = masks_logits[view_idx][keep]
                    height, width = rgb.shape[:2]
                    masks = F.interpolate(
                        masks.unsqueeze(1),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    ).sigmoid()[:, 0]

                    scale = torch.tensor(
                        [width, height, width, height],
                        dtype=boxes.dtype,
                        device=boxes.device,
                    )
                    boxes = boxes * scale

                    masks_np = masks.detach().float().cpu().numpy()
                    scores_np = scores.detach().float().cpu().numpy()
                    boxes_np = boxes.detach().float().cpu().numpy()

                    for index in range(len(scores_np)):
                        mask = masks_np[index] > self.effective_mask_threshold
                        if not mask.any():
                            continue
                        bbox_values = boxes_np[index].reshape(-1)
                        bbox = (
                            tuple(float(v) for v in bbox_values[:4])
                            if bbox_values.size >= 4
                            else None
                        )
                        self._append_candidate(
                            detections_per_view[view_idx],
                            prompt=prompt,
                            score=float(scores_np[index]),
                            mask=mask,
                            bbox=bbox,
                        )

        filter_started = time.perf_counter()
        filtered_per_view: list[list[DetectionInstance]] = []
        counts_per_view: list[dict[str, int]] = []
        for candidates in detections_per_view:
            filtered, counts = self._filter_view_candidates(candidates)
            filtered_per_view.append(filtered)
            counts_per_view.append(counts)
        self.last_filter_ms = 1000.0 * (time.perf_counter() - filter_started)
        self.last_counts_per_view = counts_per_view
        return filtered_per_view

    def detect_batch(
        self,
        frames: list[RGBDFrame],
    ) -> list[list[DetectionInstance]]:
        return self.detect_rgb_batch([frame.rgb for frame in frames])

    def detect(self, frame: RGBDFrame) -> list[DetectionInstance]:
        return self.detect_rgb_batch([frame.rgb])[0]


def build_detector(config) -> Sam3Detector:
    return Sam3Detector(config)
