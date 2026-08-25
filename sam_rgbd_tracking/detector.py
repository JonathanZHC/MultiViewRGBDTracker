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


# Run one SAM3 grounding pass over all (view, prompt) pairs instead of one pass
# per prompt. SAM3 pairs image and text features per row (FindStage.img_ids /
# text_ids), so the math is identical up to bf16 batch-shape rounding, which was
# measured at the same magnitude as the old code's own B=1 vs B=2 variation
# (~0.2 logit). Measured 160 -> 121 ms for 2 views x 3 prompts. Set False for the
# legacy per-prompt loop (text embeddings are cached in both modes).
GROUND_ALL_PROMPTS_IN_ONE_PASS = True


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

    def _text_features(self, prompts: list[str]) -> dict[str, Any]:
        """SAM3 text-encoder outputs for ``prompts``, computed once per process.

        The language backbone sees only the caption strings, so its output is a
        constant for a fixed prompt list. Tensors are cloned per call so a
        grounding forward can never mutate the cached copy. Must be called under
        the same autocast context as the grounding forward (dtype consistency).
        """
        key = tuple(prompts)
        cached = getattr(self, "_text_feature_cache", None)
        if cached is None or cached[0] != key:
            outputs = self.model.backbone.forward_text(list(prompts), device=self.device)
            cached = (key, outputs)
            self._text_feature_cache = cached
        return {
            name: (value.clone() if torch.is_tensor(value) else value)
            for name, value in cached[1].items()
        }

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

            # One grounding pass over all (view, prompt) pairs. SAM3 pairs image
            # features and text features per row via FindStage.img_ids/text_ids
            # (``language_features[:, text_ids]``, ``_get_img_feats(img_ids)``), so
            # this is the same math as one call per prompt, with the fusion
            # encoder/decoder launched once instead of len(prompts) times. Text
            # embeddings depend only on the constant prompt strings and are
            # encoded once per process instead of once per refresh.
            prompts = list(self.prompts)
            prompt_count = len(prompts)
            pair_count = batch_size * prompt_count

            def _ground(text_prompts: list[str], text_ids: torch.Tensor):
                state["backbone_out"].update(self._text_features(text_prompts))
                rows = int(text_ids.shape[0])
                stage = FindStage(
                    img_ids=torch.arange(batch_size, device=device, dtype=torch.long)
                    .repeat_interleave(rows // batch_size),
                    text_ids=text_ids,
                    input_boxes=None,
                    input_boxes_mask=None,
                    input_boxes_label=None,
                    input_points=None,
                    input_points_mask=None,
                )
                return self.model.forward_grounding(
                    backbone_out=state["backbone_out"],
                    find_input=stage,
                    geometric_prompt=self.model._get_dummy_prompt(num_prompts=rows),
                    find_target=None,
                )

            if GROUND_ALL_PROMPTS_IN_ONE_PASS:
                # Row order: (view 0, prompt 0), (view 0, prompt 1), ..., (view 1, prompt 0), ...
                output = _ground(
                    prompts,
                    torch.arange(prompt_count, device=device, dtype=torch.long).repeat(batch_size),
                )
                logits_all = output["pred_logits"]
                boxes_all = output["pred_boxes"]
                masks_all = output["pred_masks"]
                presence_all = output.get("presence_logit_dec")
            else:
                # Legacy per-prompt grounding (text still cached); rows regrouped
                # into the same (view, prompt) order as above.
                per_prompt = [
                    _ground([prompt], torch.zeros(batch_size, device=device, dtype=torch.long))
                    for prompt in prompts
                ]

                def _interleave(key):
                    values = [item[key] for item in per_prompt]
                    if any(v is None for v in values):
                        return None
                    stacked = torch.stack(values, dim=1)          # [B, P, ...]
                    return stacked.reshape(pair_count, *stacked.shape[2:])

                logits_all = _interleave("pred_logits")
                boxes_all = _interleave("pred_boxes")
                masks_all = _interleave("pred_masks")
                presence_all = _interleave("presence_logit_dec") if all(
                    item.get("presence_logit_dec") is not None for item in per_prompt
                ) else None

            logits = logits_all
            if logits.ndim == 2:
                logits = logits.unsqueeze(-1)
            probabilities = logits.sigmoid().squeeze(-1)

            if presence_all is not None:
                presence = presence_all.sigmoid().reshape(pair_count, -1)[:, :1]
                probabilities = probabilities * presence

            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_all)
            masks_logits = masks_all
            # One D2H for all pair/query scores; decide what to keep on the host
            # instead of one ``bool(keep.any())`` stream sync per pair.
            keep_all = probabilities > self.effective_score_threshold
            keep_all_cpu = keep_all.cpu().numpy()

            for pair_index in range(pair_count):
                view_idx = pair_index // prompt_count
                prompt = prompts[pair_index % prompt_count]
                rgb = rgbs[view_idx]
                if not keep_all_cpu[pair_index].any():
                    continue
                keep = keep_all[pair_index]

                if True:
                    scores = probabilities[pair_index][keep]
                    boxes = boxes_xyxy[pair_index][keep]
                    masks = masks_logits[pair_index][keep]
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
