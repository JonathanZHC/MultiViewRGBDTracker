from __future__ import annotations

from abc import abstractmethod
from contextlib import nullcontext
from typing import Any, ClassVar
import threading

import numpy as np

from ..data_types import RGBDFrame, TrackerPrediction, TrackerSeed
from .base import GLOBAL_CUDA_LOCK, MultiObjectTracker
from .streaming_state import StreamingVideoState

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class Sam2StyleStreamingTracker(MultiObjectTracker):
    """Shared streaming adapter for EfficientTAM and SAM2-like predictors."""

    _predictor_cache: ClassVar[dict[tuple[str, ...], Any]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        *,
        config_path: str,
        checkpoint_path: str,
        device: str,
        non_overlap_masks: bool,
        offload_video_to_cpu: bool,
        offload_state_to_cpu: bool,
        vos_optimized: bool,
        serialize_gpu: bool,
        use_bf16: bool,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for tracker backends")
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.non_overlap_masks = bool(non_overlap_masks)
        self.offload_video_to_cpu = bool(offload_video_to_cpu)
        self.offload_state_to_cpu = bool(offload_state_to_cpu)
        self.vos_optimized = bool(vos_optimized)
        self.serialize_gpu = bool(serialize_gpu)
        self.use_bf16 = bool(use_bf16)
        self.predictor = self._get_or_build_predictor()
        self.stream: StreamingVideoState | None = None
        self.track_ids: list[int] = []

    @property
    @abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _build_predictor(self) -> Any:
        raise NotImplementedError

    def _cache_key(self) -> tuple[str, ...]:
        return (
            self.backend_name,
            self.config_path,
            self.checkpoint_path,
            self.device,
            str(self.non_overlap_masks),
            str(self.vos_optimized),
        )

    def _get_or_build_predictor(self) -> Any:
        key = self._cache_key()
        with self._cache_lock:
            if key not in self._predictor_cache:
                predictor = self._build_predictor()
                predictor.non_overlap_masks = self.non_overlap_masks
                self._predictor_cache[key] = predictor
            return self._predictor_cache[key]

    def _gpu_context(self):
        lock = GLOBAL_CUDA_LOCK if self.serialize_gpu else nullcontext()
        device_type = str(self.device).split(":", 1)[0]
        autocast = torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16 and device_type == "cuda",
        )
        return lock, autocast

    @staticmethod
    def _to_numpy_logits(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0, 0, 0), dtype=np.float32)
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 4 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3:
            raise ValueError(f"Unexpected tracker mask shape: {array.shape}")
        return array.astype(np.float32, copy=False)

    @staticmethod
    def _presence_from_logits(masks: np.ndarray) -> np.ndarray:
        if masks.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)
        positive_fraction = (masks > 0).reshape(masks.shape[0], -1).mean(axis=1)
        peak = masks.reshape(masks.shape[0], -1).max(axis=1)
        return (
            1.0 / (1.0 + np.exp(-np.clip(peak, -20.0, 20.0)))
            * np.clip(positive_fraction * 20.0, 0.0, 1.0)
        ).astype(np.float32)

    def _prediction(self, obj_ids: Any, logits: Any) -> TrackerPrediction:
        masks = self._to_numpy_logits(logits)
        ids = [int(v) for v in (obj_ids.tolist() if hasattr(obj_ids, "tolist") else obj_ids)]
        if len(ids) != masks.shape[0] and len(self.track_ids) == masks.shape[0]:
            ids = list(self.track_ids)
        return TrackerPrediction(
            ids, masks, self._presence_from_logits(masks), {"backend": self.backend_name}
        )

    def initialize(self, frame: RGBDFrame, seeds: list[TrackerSeed]) -> TrackerPrediction:
        if self.stream is not None:
            self.stream.close()
        self.stream = StreamingVideoState(
            self.predictor,
            frame.rgb,
            self.offload_video_to_cpu,
            self.offload_state_to_cpu,
        )
        self.track_ids = [seed.track_id for seed in seeds]
        if not seeds:
            return TrackerPrediction([], np.empty((0, *frame.depth_m.shape), np.float32), np.empty(0, np.float32), {"backend": self.backend_name})
        state = self.stream.state
        last_obj_ids: Any = self.track_ids
        last_logits: Any = np.stack([seed.mask.astype(np.float32) for seed in seeds])
        lock, autocast = self._gpu_context()
        with lock, torch.inference_mode(), autocast:
            for seed in seeds:
                output = self.predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=int(seed.track_id),
                    mask=seed.mask.astype(bool),
                )
                if isinstance(output, tuple) and len(output) >= 3:
                    _, last_obj_ids, last_logits = output[-3:]
        prediction = self._prediction(last_obj_ids, last_logits)
        # Some predictors return only the just-added object. In that case SAM3 masks
        # are the correct keyframe result and propagation starts on the next frame.
        if prediction.mask_logits.shape[0] != len(seeds):
            prediction = TrackerPrediction(
                list(self.track_ids),
                np.stack([seed.mask.astype(np.float32) for seed in seeds]),
                np.asarray([seed.confidence for seed in seeds], dtype=np.float32),
                {"backend": self.backend_name, "keyframe_masks": True},
            )
        return prediction

    def track(self, frame: RGBDFrame) -> TrackerPrediction:
        if self.stream is None:
            raise RuntimeError("Tracker has not been initialized. Run a SAM3 keyframe first.")
        frame_idx = self.stream.append(frame.rgb)
        lock, autocast = self._gpu_context()
        with lock, torch.inference_mode(), autocast:
            outputs = list(
                self.predictor.propagate_in_video(
                    self.stream.state,
                    start_frame_idx=frame_idx,
                    max_frame_num_to_track=1,
                    reverse=False,
                )
            )
        if not outputs:
            raise RuntimeError(f"{self.backend_name} returned no output for frame {frame_idx}")
        _, obj_ids, logits = outputs[-1]
        return self._prediction(obj_ids, logits)

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None
