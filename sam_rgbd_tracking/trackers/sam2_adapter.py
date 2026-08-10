from __future__ import annotations

import threading
import time
import traceback
from abc import abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Any, ClassVar, Iterator

import numpy as np

from ..data_types import RGBDFrame, TrackerPrediction, TrackerSeed
from .base import (
    GLOBAL_CUDA_LOCK,
    MultiObjectTracker,
    tracker_profile_context,
)
from .streaming_state import StreamingVideoState

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class Sam2StyleStreamingTracker(MultiObjectTracker):
    """Fast live-stream adapter used by the EfficientTAM SAM2-style predictor.

    CUDAGraph ordering is preserved:
      lock -> append/preprocess -> propagate -> D2H -> unlock

    The adapter also keeps the most recent successful masks. If upstream throws
    a rare bare ``AssertionError`` during propagation, the current streaming
    state is considered unsafe to continue using: the state is reset on the
    current RGB frame and reseeded once from those most recent masks. This
    prevents a single failed propagation from leaving the streaming history in
    a permanently inconsistent state.
    """

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
        stream_buffer_frames: int = 40,
        reuse_state_on_keyframe: bool = True,
        gpu_preprocess: bool = True,
        pin_input_memory: bool = True,
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
        self.stream_buffer_frames = max(2, int(stream_buffer_frames))
        self.reuse_state_on_keyframe = bool(reuse_state_on_keyframe)
        self.gpu_preprocess = bool(gpu_preprocess)
        self.pin_input_memory = bool(pin_input_memory)

        self.predictor = self._get_or_build_predictor()
        self.stream: StreamingVideoState | None = None
        self.track_ids: list[int] = []
        self._last_masks_by_track: dict[int, np.ndarray] = {}

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
            str(self.use_bf16),
        )

    def _get_or_build_predictor(self) -> Any:
        key = self._cache_key()
        with self._cache_lock:
            if key not in self._predictor_cache:
                predictor = self._build_predictor()
                predictor.non_overlap_masks = self.non_overlap_masks
                self._predictor_cache[key] = predictor
            return self._predictor_cache[key]

    @contextmanager
    def _gpu_guard(self) -> Iterator[None]:
        """Acquire the shared GPU lock and profile only time spent waiting."""
        if not self.serialize_gpu:
            yield
            return

        started = time.perf_counter()
        GLOBAL_CUDA_LOCK.acquire()
        self.record_profile(
            "tracker_lock_wait_cpu",
            1000.0 * (time.perf_counter() - started),
        )
        try:
            yield
        finally:
            GLOBAL_CUDA_LOCK.release()

    def _lock(self):
        # Backward-compatible alias for the EfficientTAM adapter.
        return GLOBAL_CUDA_LOCK if self.serialize_gpu else nullcontext()

    def _autocast(self):
        device_type = str(self.device).split(":", 1)[0]
        return torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16 and device_type == "cuda",
        )

    def _gpu_context(self):
        return self._gpu_guard(), self._autocast()

    def _new_stream(self, frame: RGBDFrame) -> StreamingVideoState:
        return StreamingVideoState(
            self.predictor,
            frame.rgb,
            self.offload_video_to_cpu,
            self.offload_state_to_cpu,
            buffer_frames=self.stream_buffer_frames,
            profiler=self.profiler,
            use_gpu_preprocess=self.gpu_preprocess,
            pin_input_memory=self.pin_input_memory,
        )

    def _reset_or_create_stream(self, frame: RGBDFrame) -> None:
        if self.stream is None:
            with self.profile_stage("tracker_first_init_cpu", cuda=False):
                self.stream = self._new_stream(frame)
            return

        if self.reuse_state_on_keyframe:
            with self.profile_stage("tracker_stream_reset_cpu", cuda=False):
                self.stream.reset(frame.rgb)
            return

        # Compatibility fallback: old behavior with repeated JPEG/init_state.
        self.stream.close()
        with self.profile_stage("tracker_first_init_cpu", cuda=False):
            self.stream = self._new_stream(frame)

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
        flat = masks.reshape(masks.shape[0], -1)
        positive_fraction = (flat > 0).mean(axis=1)
        peak = flat.max(axis=1)
        return (
            1.0 / (1.0 + np.exp(-np.clip(peak, -20.0, 20.0)))
            * np.clip(positive_fraction * 20.0, 0.0, 1.0)
        ).astype(np.float32)

    def _prediction(self, obj_ids: Any, logits: Any) -> TrackerPrediction:
        masks = self._to_numpy_logits(logits)
        raw_ids = obj_ids.tolist() if hasattr(obj_ids, "tolist") else obj_ids
        ids = [int(value) for value in raw_ids]
        if len(ids) != masks.shape[0] and len(self.track_ids) == masks.shape[0]:
            ids = list(self.track_ids)
        return TrackerPrediction(
            ids,
            masks,
            self._presence_from_logits(masks),
            {"backend": self.backend_name},
        )

    def _remember_prediction_masks(self, prediction: TrackerPrediction) -> None:
        logits = np.asarray(prediction.mask_logits)
        if logits.ndim == 2:
            logits = logits[None]
        for channel, track_id in enumerate(prediction.track_ids):
            if channel >= logits.shape[0]:
                continue
            mask = np.asarray(logits[channel] > 0.0, dtype=bool)
            if mask.ndim != 2 or not mask.any():
                continue
            self._last_masks_by_track[int(track_id)] = mask.copy()

        active = {int(value) for value in self.track_ids}
        self._last_masks_by_track = {
            track_id: mask
            for track_id, mask in self._last_masks_by_track.items()
            if track_id in active
        }

    def _recover_after_propagation_assertion(
        self,
        frame: RGBDFrame,
    ) -> TrackerPrediction:
        """Reset the unsafe state and reseed the current frame once.

        The current frame has already been appended when propagation failed, so
        simply continuing would leave a hole in EfficientTAM's temporal output
        dictionaries. Instead, discard that temporal state, make the current RGB
        frame frame-0, and use the most recent successful masks as prompts.
        """
        if self.stream is None:
            raise RuntimeError("Cannot recover tracker without a streaming state")

        recoverable_ids = [
            int(track_id)
            for track_id in self.track_ids
            if int(track_id) in self._last_masks_by_track
        ]
        if not recoverable_ids:
            raise RuntimeError(
                "EfficientTAM propagation asserted and no previous masks are "
                "available for automatic state recovery"
            )

        with self.profile_stage("tracker_assertion_recovery_gpu", cuda=True):
            self.stream.reset(frame.rgb)
            self.track_ids = recoverable_ids

            latest_obj_ids: Any = recoverable_ids
            latest_logits: Any = np.stack(
                [
                    self._last_masks_by_track[track_id].astype(np.float32)
                    for track_id in recoverable_ids
                ],
                axis=0,
            )

            for track_id in recoverable_ids:
                output = self.predictor.add_new_mask(
                    inference_state=self.stream.state,
                    frame_idx=0,
                    obj_id=int(track_id),
                    mask=self._last_masks_by_track[track_id],
                )
                if isinstance(output, tuple) and len(output) >= 3:
                    _, latest_obj_ids, latest_logits = output[-3:]

        with self.profile_stage("tracker_output_d2h_cpu", cuda=False):
            prediction = self._prediction(latest_obj_ids, latest_logits)

        prediction.metadata.update(
            {
                "frame_index": 0,
                "stream_length": 1,
                "recovered_from_assertion": True,
            }
        )
        self._remember_prediction_masks(prediction)
        return prediction

    def initialize(
        self,
        frame: RGBDFrame,
        seeds: list[TrackerSeed],
    ) -> TrackerPrediction:
        call_started = time.perf_counter()
        prediction: TrackerPrediction

        with self._gpu_guard():
            reinit_started = time.perf_counter()
            with (
                torch.inference_mode(),
                self._autocast(),
                tracker_profile_context(self.profiler),
            ):
                with self.profile_stage("tracker_reinit_gpu", cuda=True):
                    self._reset_or_create_stream(frame)
                    if self.stream is None:
                        raise RuntimeError("Tracker stream initialization failed")
                    self.track_ids = [int(seed.track_id) for seed in seeds]

                    if not seeds:
                        h, w = frame.rgb.shape[:2]
                        prediction = TrackerPrediction(
                            [],
                            np.empty((0, h, w), np.float32),
                            np.empty((0,), np.float32),
                            {"backend": self.backend_name, "frame_index": 0},
                        )
                    else:
                        latest_obj_ids: Any = self.track_ids
                        latest_logits: Any = np.stack(
                            [seed.mask.astype(np.float32) for seed in seeds],
                            axis=0,
                        )
                        with self.profile_stage("tracker_seed_gpu", cuda=True):
                            for seed in seeds:
                                output = self.predictor.add_new_mask(
                                    inference_state=self.stream.state,
                                    frame_idx=0,
                                    obj_id=int(seed.track_id),
                                    mask=np.asarray(seed.mask, dtype=bool),
                                )
                                if isinstance(output, tuple) and len(output) >= 3:
                                    _, latest_obj_ids, latest_logits = output[-3:]

                        with self.profile_stage("tracker_output_d2h_cpu", cuda=False):
                            prediction = self._prediction(
                                latest_obj_ids,
                                latest_logits,
                            )

            self.record_profile(
                "tracker_reinit_wall_cpu",
                1000.0 * (time.perf_counter() - reinit_started),
            )

        self.record_profile(
            "tracker_total_wall_cpu",
            1000.0 * (time.perf_counter() - call_started),
        )

        if seeds and prediction.mask_logits.shape[0] != len(seeds):
            prediction = TrackerPrediction(
                list(self.track_ids),
                np.stack(
                    [seed.mask.astype(np.float32) for seed in seeds],
                    axis=0,
                ),
                np.asarray(
                    [seed.confidence for seed in seeds],
                    dtype=np.float32,
                ),
                {"backend": self.backend_name, "keyframe_masks": True},
            )

        self._last_masks_by_track = {
            int(seed.track_id): np.asarray(seed.mask, dtype=bool).copy()
            for seed in seeds
            if np.asarray(seed.mask).ndim == 2 and np.asarray(seed.mask).any()
        }
        self._remember_prediction_masks(prediction)
        return prediction

    def track(self, frame: RGBDFrame) -> TrackerPrediction:
        if self.stream is None:
            raise RuntimeError(
                "Tracker has not been initialized. Run a SAM3 keyframe first."
            )
        if not self.track_ids:
            h, w = frame.rgb.shape[:2]
            return TrackerPrediction(
                [],
                np.empty((0, h, w), np.float32),
                np.empty((0,), np.float32),
                {"backend": self.backend_name},
            )

        call_started = time.perf_counter()
        with self._gpu_guard():
            with (
                torch.inference_mode(),
                self._autocast(),
                tracker_profile_context(self.profiler),
            ):
                try:
                    with self.profile_stage("tracker_total_gpu", cuda=True):
                        with self.profile_stage("tracker_append_cpu", cuda=False):
                            frame_idx = self.stream.append(frame.rgb)

                        with self.profile_stage("tracker_propagate_gpu", cuda=True):
                            output = None
                            for output in self.predictor.propagate_in_video(
                                self.stream.state,
                                start_frame_idx=frame_idx,
                                max_frame_num_to_track=1,
                                reverse=False,
                            ):
                                pass

                    if output is None:
                        raise RuntimeError(
                            f"{self.backend_name} returned no output for frame {frame_idx}"
                        )

                    out_frame_idx, obj_ids, logits = output[:3]
                    with self.profile_stage("tracker_output_d2h_cpu", cuda=False):
                        prediction = self._prediction(obj_ids, logits)
                    prediction.metadata.update(
                        {
                            "frame_index": int(out_frame_idx),
                            "stream_length": int(self.stream.state["num_frames"]),
                        }
                    )
                    self._remember_prediction_masks(prediction)

                except AssertionError:
                    # Do not continue using a state in which append succeeded but
                    # propagation aborted midway. Print the original upstream
                    # traceback once, then rebuild the current frame from the most
                    # recent successful masks.
                    print(
                        "[EfficientTAM] AssertionError during propagation; "
                        "resetting the current streaming state and reseeding from "
                        "the last successful masks. Original traceback:\n"
                        + traceback.format_exc(),
                        flush=True,
                    )
                    prediction = self._recover_after_propagation_assertion(frame)

        self.record_profile(
            "tracker_total_wall_cpu",
            1000.0 * (time.perf_counter() - call_started),
        )
        return prediction

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self._last_masks_by_track.clear()
