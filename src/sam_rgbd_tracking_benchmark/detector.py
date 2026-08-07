from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from .appearance import masked_color_embedding
from .association import mask_iou
from .data_types import DetectionInstance, RGBDFrame
from .trackers.sam2_adapter import GLOBAL_CUDA_LOCK

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class InstanceDetector(ABC):
    @abstractmethod
    def detect(self, frame: RGBDFrame) -> list[DetectionInstance]:
        raise NotImplementedError


class GroundTruthDetector(InstanceDetector):
    """Uses simulator instance masks to isolate tracker and post-processing behavior."""

    def detect(self, frame: RGBDFrame) -> list[DetectionInstance]:
        if frame.gt_instance_map is None:
            raise RuntimeError("Ground-truth detector requested but no GT instance map was received")
        detections: list[DetectionInstance] = []
        for instance_id in np.unique(frame.gt_instance_map):
            instance_id = int(instance_id)
            if instance_id <= 0:
                continue
            mask = frame.gt_instance_map == instance_id
            metadata = frame.gt_metadata.get(instance_id, {})
            label = str(metadata.get("label", f"instance_{instance_id}"))
            ys, xs = np.nonzero(mask)
            bbox = None
            if xs.size:
                bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            detections.append(
                DetectionInstance(
                    detection_id=instance_id,
                    label=label,
                    score=1.0,
                    mask=mask,
                    bbox_xyxy=bbox,
                    embedding=masked_color_embedding(frame.rgb, mask),
                )
            )
        return detections


class Sam3Detector(InstanceDetector):
    _cache: ClassVar[dict[tuple[str, str], tuple[Any, Any]]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        checkpoint: str,
        prompts: list[str],
        score_threshold: float,
        mask_threshold: float,
        duplicate_iou_threshold: float,
        device: str,
        use_bf16: bool,
        serialize_gpu: bool,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for SAM3")
        self.checkpoint = checkpoint
        self.prompts = list(prompts)
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.device = device
        self.use_bf16 = use_bf16
        self.serialize_gpu = serialize_gpu
        self.model, self.processor = self._get_or_build()

    def _get_or_build(self) -> tuple[Any, Any]:
        key = (str(Path(self.checkpoint).resolve()), self.device)
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            checkpoint = Path(self.checkpoint)
            model = build_sam3_image_model(
                checkpoint_path=str(checkpoint) if checkpoint.exists() else None,
                load_from_HF=not checkpoint.exists(),
                device=self.device,
                eval_mode=True,
                enable_segmentation=True,
                compile=False,
            )
            processor = Sam3Processor(model)
            self._cache[key] = (model, processor)
            return model, processor

    def detect(self, frame: RGBDFrame) -> list[DetectionInstance]:
        lock = GLOBAL_CUDA_LOCK if self.serialize_gpu else _NullLock()
        with lock, torch.inference_mode():
            autocast_enabled = self.device.startswith("cuda") and self.use_bf16
            context = torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled)
            with context:
                image = Image.fromarray(frame.rgb.astype(np.uint8), mode="RGB")
                state = self.processor.set_image(image)
                detections: list[DetectionInstance] = []
                detection_id = 0
                for prompt in self.prompts:
                    output = self.processor.set_text_prompt(state=state, prompt=prompt)
                    masks = self._to_numpy(output.get("masks"))
                    scores = self._to_numpy(output.get("scores")).reshape(-1)
                    boxes = self._to_numpy(output.get("boxes"))
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks = masks[:, 0]
                    if masks.ndim == 2:
                        masks = masks[None]
                    for index in range(masks.shape[0]):
                        score = float(scores[index]) if index < scores.size else 1.0
                        if score < self.score_threshold:
                            continue
                        mask_value = masks[index]
                        mask = mask_value > self.mask_threshold if mask_value.dtype != bool else mask_value
                        if not mask.any():
                            continue
                        bbox = None
                        if boxes.ndim >= 2 and index < boxes.shape[0]:
                            bbox = tuple(float(value) for value in boxes[index].reshape(-1)[:4])
                        candidate = DetectionInstance(
                            detection_id=detection_id,
                            label=prompt,
                            score=score,
                            mask=np.asarray(mask, dtype=bool),
                            bbox_xyxy=bbox,
                            embedding=masked_color_embedding(frame.rgb, mask),
                        )
                        detection_id += 1
                        if self._is_duplicate(candidate, detections):
                            continue
                        detections.append(candidate)
                return detections

    def _is_duplicate(self, candidate: DetectionInstance, accepted: list[DetectionInstance]) -> bool:
        for existing in accepted:
            if mask_iou(candidate.mask, existing.mask) >= self.duplicate_iou_threshold:
                if candidate.score > existing.score:
                    existing.mask = candidate.mask
                    existing.score = candidate.score
                    existing.label = candidate.label
                    existing.bbox_xyxy = candidate.bbox_xyxy
                return True
        return False

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0,), dtype=np.float32)
        if hasattr(value, "detach"):
            return value.detach().float().cpu().numpy()
        return np.asarray(value)


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def build_detector(config: Any) -> InstanceDetector:
    if config.detector.backend == "ground_truth":
        return GroundTruthDetector()
    if config.detector.backend == "sam3":
        return Sam3Detector(
            checkpoint=config.detector.checkpoint,
            prompts=list(config.detector.prompts),
            score_threshold=float(config.detector.score_threshold),
            mask_threshold=float(config.detector.mask_threshold),
            duplicate_iou_threshold=float(config.detector.duplicate_iou_threshold),
            device=config.runtime.device,
            use_bf16=bool(config.runtime.use_bf16),
            serialize_gpu=bool(config.runtime.serialize_gpu),
        )
    raise ValueError(f"Unknown detector backend: {config.detector.backend}")
