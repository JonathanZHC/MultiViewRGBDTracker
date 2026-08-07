from __future__ import annotations

import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
from PIL import Image

from .data_types import DetectionInstance, RGBDFrame
from .processing import color_embedding, mask_iou
from .trackers.base import GLOBAL_CUDA_LOCK

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class InstanceDetector(Protocol):
    """Minimal detector interface used by :class:`SAMTrackingComponent`."""

    def detect(self, frame: RGBDFrame) -> list[DetectionInstance]:
        ...


class Sam3Detector:
    """Text-prompted SAM3 image detector with one shared model per GPU."""

    _cache: ClassVar[dict[tuple[str, str], tuple[Any, Any]]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for SAM3")
        self.checkpoint = str(config.detector.checkpoint)
        self.device = str(config.runtime.device)
        self.use_bf16 = bool(config.runtime.get("use_bf16", True))
        self.serialize_gpu = bool(config.runtime.get("serialize_gpu", True))
        self.prompts = [str(v) for v in config.detector.prompts]
        self.score_threshold = float(config.detector.score_threshold)
        self.mask_threshold = float(config.detector.mask_threshold)
        self.duplicate_iou = float(config.detector.duplicate_iou_threshold)
        self.model, self.processor = self._get_or_build()

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

    def detect(self, frame: RGBDFrame) -> list[DetectionInstance]:
        image = Image.fromarray(
            np.ascontiguousarray(frame.rgb).astype(np.uint8),
            mode="RGB",
        )
        detections: list[DetectionInstance] = []
        lock = (
            GLOBAL_CUDA_LOCK
            if self.serialize_gpu and self.device.startswith("cuda")
            else nullcontext()
        )
        autocast_enabled = self.use_bf16 and self.device.startswith("cuda")

        with lock, torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                state = self.processor.set_image(image)
                for prompt in self.prompts:
                    output = self.processor.set_text_prompt(
                        state=state,
                        prompt=prompt,
                    )
                    masks = output.get("masks")
                    scores = output.get("scores")
                    boxes = output.get("boxes")
                    if masks is None:
                        continue

                    masks_np = (
                        masks.detach().float().cpu().numpy()
                        if hasattr(masks, "detach")
                        else np.asarray(masks)
                    )
                    scores_np = (
                        scores.detach().float().cpu().numpy()
                        if scores is not None and hasattr(scores, "detach")
                        else np.asarray(
                            scores
                            if scores is not None
                            else np.ones(len(masks_np))
                        )
                    )
                    boxes_np = (
                        boxes.detach().float().cpu().numpy()
                        if boxes is not None and hasattr(boxes, "detach")
                        else (np.asarray(boxes) if boxes is not None else None)
                    )

                    while masks_np.ndim > 3 and masks_np.shape[0] == 1:
                        masks_np = masks_np[0]
                    if masks_np.ndim == 2:
                        masks_np = masks_np[None]
                    scores_np = scores_np.reshape(-1)

                    for index in range(min(len(masks_np), len(scores_np))):
                        score = float(scores_np[index])
                        if score < self.score_threshold:
                            continue
                        mask = np.asarray(masks_np[index]) > self.mask_threshold
                        if mask.shape != frame.depth_m.shape or not mask.any():
                            continue

                        bbox = None
                        if boxes_np is not None and index < len(boxes_np):
                            values = np.asarray(boxes_np[index]).reshape(-1)
                            if values.size >= 4:
                                bbox = tuple(float(v) for v in values[:4])

                        candidate = DetectionInstance(
                            detection_id=0,
                            label=prompt,
                            score=score,
                            mask=mask,
                            bbox_xyxy=bbox,
                            embedding=color_embedding(frame.rgb, mask),
                        )
                        duplicate = next(
                            (
                                item_index
                                for item_index, item in enumerate(detections)
                                if mask_iou(mask, item.mask) >= self.duplicate_iou
                            ),
                            None,
                        )
                        if duplicate is None:
                            detections.append(candidate)
                        elif score > detections[duplicate].score:
                            detections[duplicate] = candidate

        for detection_id, item in enumerate(detections, start=1):
            item.detection_id = detection_id
        return detections


def build_detector(config) -> Sam3Detector:
    return Sam3Detector(config)
