from __future__ import annotations

import threading
from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np

from ..data_types import RGBDFrame, TrackerPrediction, TrackerSeed
from .base import MultiObjectTracker
from .streaming_state import StreamingVideoState

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


GLOBAL_CUDA_LOCK = threading.RLock()


class Sam2StyleStreamingTracker(MultiObjectTracker):
    """Shared streaming logic for SAM-MT and EfficientTAM predictors."""

    _predictor_cache: ClassVar[dict[tuple[str, ...], Any]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
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
            raise RuntimeError("PyTorch is required for real tracker backends")
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.non_overlap_masks = non_overlap_masks
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self.vos_optimized = vos_optimized
        self.serialize_gpu = serialize_gpu
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

    def _get_or_build_predictor(self) -> Any:
        key = (
            self.backend_name,
            self.config_path,
            self.checkpoint_path,
            self.device,
            str(self.non_overlap_masks),
            str(self.vos_optimized),
            str(self.use_bf16),
        )
        with self._cache_lock:
            if key not in self._predictor_cache:
                predictor = self._build_predictor()
                predictor.non_overlap_masks = self.non_overlap_masks
                self._predictor_cache[key] = predictor
            return self._predictor_cache[key]

    def _lock(self):
        return GLOBAL_CUDA_LOCK if self.serialize_gpu else _NullLock()

    def _autocast(self):
        device_type = str(self.device).split(":", 1)[0]
        enabled = self.use_bf16 and device_type == "cuda"
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=enabled)

    def initialize(self, frame: RGBDFrame, seeds: list[TrackerSeed]) -> TrackerPrediction:
        with self._lock(), torch.inference_mode(), self._autocast():
            if self.stream is not None:
                self.stream.close()
            self.stream = StreamingVideoState(
                self.predictor,
                frame.rgb,
                self.offload_video_to_cpu,
                self.offload_state_to_cpu,
            )
            self.track_ids = [int(seed.track_id) for seed in seeds]
            if not seeds:
                h, w = frame.rgb.shape[:2]
                return TrackerPrediction([], np.empty((0, h, w), np.float32), np.empty((0,), np.float32))
            latest_masks = None
            latest_ids = None
            for seed in seeds:
                _, latest_ids, latest_masks = self.predictor.add_new_mask(
                    inference_state=self.stream.state,
                    frame_idx=0,
                    obj_id=int(seed.track_id),
                    mask=np.asarray(seed.mask, dtype=bool),
                )
            logits = self._to_numpy_logits(latest_masks, frame.rgb.shape[:2])
            ids = [int(value) for value in latest_ids]
            return TrackerPrediction(
                track_ids=ids,
                mask_logits=logits,
                presence_scores=self._presence_from_logits(logits),
                backend_metadata={"backend": self.backend_name, "frame_index": 0},
            )

    def track(self, frame: RGBDFrame) -> TrackerPrediction:
        if self.stream is None:
            raise RuntimeError("Tracker must be initialized before track()")
        if not self.track_ids:
            h, w = frame.rgb.shape[:2]
            return TrackerPrediction([], np.empty((0, h, w), np.float32), np.empty((0,), np.float32))
        with self._lock(), torch.inference_mode(), self._autocast():
            frame_idx = self.stream.append(frame.rgb)
            output = None
            for output in self.predictor.propagate_in_video(
                self.stream.state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1,
                reverse=False,
            ):
                pass
            if output is None:
                raise RuntimeError("Upstream predictor yielded no output for the appended frame")
            out_frame_idx, obj_ids, mask_logits = output[:3]
            logits = self._to_numpy_logits(mask_logits, frame.rgb.shape[:2])
            return TrackerPrediction(
                track_ids=[int(value) for value in obj_ids],
                mask_logits=logits,
                presence_scores=self._presence_from_logits(logits),
                backend_metadata={
                    "backend": self.backend_name,
                    "frame_index": int(out_frame_idx),
                    "stream_length": int(self.stream.state["num_frames"]),
                },
            )

    @staticmethod
    def _presence_from_logits(logits: np.ndarray) -> np.ndarray:
        if logits.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)
        positive_fraction = (logits > 0).reshape(logits.shape[0], -1).mean(axis=1)
        peak = logits.reshape(logits.shape[0], -1).max(axis=1)
        return (1.0 / (1.0 + np.exp(-peak)) * np.clip(positive_fraction * 20.0, 0.0, 1.0)).astype(np.float32)

    @staticmethod
    def _to_numpy_logits(mask_logits: Any, original_hw: tuple[int, int]) -> np.ndarray:
        if mask_logits is None:
            return np.empty((0, *original_hw), dtype=np.float32)
        if hasattr(mask_logits, "detach"):
            array = mask_logits.detach().float().cpu().numpy()
        else:
            array = np.asarray(mask_logits, dtype=np.float32)
        if array.ndim == 4 and array.shape[1] == 1:
            array = array[:, 0]
        return array.astype(np.float32, copy=False)

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False
