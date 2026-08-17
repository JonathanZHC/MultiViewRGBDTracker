from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, ClassVar

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..data_types import TrackerPrediction, TrackerSeed
from .base import GLOBAL_CUDA_LOCK, current_tracker_profiler
from .sam2_adapter import Sam2StyleStreamingTracker
from .streaming_state import BatchedStreamingPreprocessor, StreamingVideoState


def _move_rope_frequency_caches_to_device(
    predictor: Any,
    device: str,
) -> int:
    """Move RoPE frequency tensors to CUDA before the first compiled forward."""
    if torch is None:
        return 0

    target = torch.device(device)
    moved = 0
    attribute_names = (
        "freqs_cis",
        "freqs_cis_q",
        "freqs_cis_k",
    )

    for module in predictor.modules():
        for attribute_name in attribute_names:
            value = getattr(module, attribute_name, None)
            if not torch.is_tensor(value):
                continue
            if value.device == target:
                continue
            setattr(
                module,
                attribute_name,
                value.to(target, non_blocking=False),
            )
            moved += 1

    if moved:
        print(
            "[EfficientTAM] preloaded "
            f"{moved} RoPE frequency tensor(s) on {target} "
            "before first compiled forward",
            flush=True,
        )
    return moved


def _disable_unavailable_hole_fill_extension(predictor: Any) -> None:
    """Avoid repeatedly calling the unavailable optional ``_C`` extension."""
    fill_hole_area = int(getattr(predictor, "fill_hole_area", 0) or 0)
    if fill_hole_area <= 0:
        return

    predictor.fill_hole_area = 0
    print(
        "[EfficientTAM] optional _C hole-filling extension is unavailable in "
        "this container; fill_hole_area forced to 0",
        flush=True,
    )


def _install_memory_attention_clone_boundary(predictor: Any) -> None:
    """Keep CUDAGraph enabled and clone only memory-attention output."""
    if torch is None:
        raise RuntimeError("PyTorch is required for EfficientTAM")

    module = getattr(predictor, "memory_attention", None)
    if module is None:
        raise RuntimeError("EfficientTAM predictor has no memory_attention module")
    if getattr(module, "_sam_rgbd_memory_attention_clone_boundary", False):
        return

    compiled_forward = module.forward

    def clone_safe_forward(*args: Any, **kwargs: Any):
        output = compiled_forward(*args, **kwargs)
        if not torch.is_tensor(output):
            raise TypeError(
                "EfficientTAM memory_attention unexpectedly returned "
                f"{type(output)!r}; expected torch.Tensor"
            )

        profiler = current_tracker_profiler()
        context = (
            profiler.stage("tracker_state_clone_gpu", cuda=True)
            if profiler is not None
            else nullcontext()
        )
        with context:
            return output.clone()

    module.forward = clone_safe_forward
    module._sam_rgbd_memory_attention_clone_boundary = True

    print(
        "[EfficientTAM] selective CUDAGraph safety enabled: "
        "memory_attention output clone only",
        flush=True,
    )


def _synthetic_masks(height: int, width: int, count: int) -> list[np.ndarray]:
    """Create deterministic, non-overlapping prompts for compile warm-up."""
    count = max(1, int(count))
    masks: list[np.ndarray] = []
    margin_y = max(4, height // 12)
    margin_x = max(4, width // 12)
    usable_w = max(count, width - 2 * margin_x)
    cell_w = max(1, usable_w // count)

    for index in range(count):
        mask = np.zeros((height, width), dtype=bool)
        x0 = margin_x + index * cell_w + max(1, cell_w // 6)
        x1 = margin_x + (index + 1) * cell_w - max(1, cell_w // 6)
        y0 = margin_y + (index % 2) * max(1, height // 20)
        y1 = height - margin_y - ((index + 1) % 2) * max(1, height // 20)
        x0 = min(max(0, x0), width - 1)
        x1 = min(max(x0 + 1, x1), width)
        y0 = min(max(0, y0), height - 1)
        y1 = min(max(y0 + 1, y1), height)
        mask[y0:y1, x0:x1] = True
        masks.append(mask)
    return masks


class EfficientTAMTracker(Sam2StyleStreamingTracker):
    """EfficientTAM adapter retaining the native VOS/CUDAGraph fast path.

    The optional pre-warm path deliberately exercises the state shapes that
    otherwise tend to trigger lazy ``torch.compile``/CUDAGraph specialization
    during live operation: seed, several temporal-memory lengths, saturated
    memory, reset/reseed, and propagation after reset.
    """

    _prewarm_lock: ClassVar[threading.Lock] = threading.Lock()
    _prewarm_done: ClassVar[set[tuple[Any, ...]]] = set()

    def __init__(
        self,
        *args: Any,
        prewarm_enabled: bool = True,
        prewarm_object_counts: list[int] | tuple[int, ...] = (1,),
        prewarm_temporal_frames: int = 0,
        prewarm_post_reset_frames: int = 2,
        prewarm_passes: int = 2,
        execution_mode: str = "sequential",
        fixed_num_views: int = 2,
        object_slots_per_view: int = 4,
        slot_layout_key: tuple[str, ...] | list[str] = (),
        feature_history_frames: int = 32,
        use_max_autotune: bool = False,
        **kwargs: Any,
    ) -> None:
        self.execution_mode = str(execution_mode).strip().lower()
        if self.execution_mode not in {"sequential", "fixed_batch"}:
            raise ValueError(
                f"Unsupported EfficientTAM execution_mode={self.execution_mode!r}; "
                "use 'sequential' or 'fixed_batch'."
            )
        self.fixed_num_views = max(1, int(fixed_num_views))
        self.object_slots_per_view = max(1, int(object_slots_per_view))
        self.slot_layout_key = tuple(str(value) for value in slot_layout_key)
        self.feature_history_frames = max(2, int(feature_history_frames))
        self.use_max_autotune = bool(use_max_autotune)
        self.prewarm_enabled = bool(prewarm_enabled)
        cleaned = sorted({max(1, int(value)) for value in prewarm_object_counts})
        self.prewarm_object_counts = tuple(cleaned or [1])
        self.prewarm_temporal_frames = max(0, int(prewarm_temporal_frames))
        self.prewarm_post_reset_frames = max(1, int(prewarm_post_reset_frames))
        self.prewarm_passes = max(1, int(prewarm_passes))
        super().__init__(*args, **kwargs)
        # Raw model-input images use a fixed physical ring. Keep it slightly
        # larger than the persistent feature ring so every cached reference
        # remains addressable by logical frame index, including sequential mode.
        self.stream_buffer_frames = max(
            self.stream_buffer_frames,
            self.feature_history_frames + 2,
        )

    @property
    def backend_name(self) -> str:
        return "efficient_tam"

    def _cache_key(self) -> tuple[str, ...]:
        return super()._cache_key() + (
            self.execution_mode,
            str(self.fixed_num_views),
            str(self.object_slots_per_view),
            *self.slot_layout_key,
            str(self.feature_history_frames),
            str(self.use_max_autotune),
        )

    def _build_predictor(self) -> Any:
        from efficient_track_anything.build_efficienttam import (
            build_efficienttam_video_predictor,
        )

        predictor = build_efficienttam_video_predictor(
            config_file=self.config_path,
            ckpt_path=self.checkpoint_path,
            device=self.device,
            mode="eval",
            apply_postprocessing=True,
            vos_optimized=self.vos_optimized,
            execution_mode=self.execution_mode,
            fixed_num_views=self.fixed_num_views,
            max_objects_per_view=self.object_slots_per_view,
            use_max_autotune=self.use_max_autotune,
        )

        _move_rope_frequency_caches_to_device(predictor, self.device)
        _disable_unavailable_hole_fill_extension(predictor)

        if self.vos_optimized:
            _install_memory_attention_clone_boundary(predictor)
        return predictor

    def _resolved_prewarm_temporal_frames(self) -> int:
        if self.prewarm_temporal_frames > 0:
            return self.prewarm_temporal_frames

        # ``num_maskmem`` is the important shape transition: before saturation,
        # the number of memory tokens grows every frame. Run slightly beyond it
        # so both the growth phase and saturated phase are captured.
        num_maskmem = int(getattr(self.predictor, "num_maskmem", 7) or 7)
        return max(4, min(16, num_maskmem + 2))

    @staticmethod
    def _variant_rgb(rgb: np.ndarray, step: int) -> np.ndarray:
        # Shape-preserving deterministic variation prevents accidental reuse of
        # image-dependent caches while remaining very cheap to construct.
        value = np.ascontiguousarray(rgb, dtype=np.uint8).copy()
        if value.size:
            value[..., step % 3] = np.bitwise_xor(
                value[..., step % 3],
                np.uint8((17 * step) & 0xFF),
            )
        return value

    def _run_prewarm_sequence(
        self,
        rgb: np.ndarray,
        *,
        object_count: int,
        temporal_frames: int,
    ) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is required for EfficientTAM pre-warm")

        height, width = rgb.shape[:2]
        masks = _synthetic_masks(height, width, object_count)
        buffer_frames = max(self.stream_buffer_frames, temporal_frames + 4)
        stream = StreamingVideoState(
            self.predictor,
            rgb,
            self.offload_video_to_cpu,
            self.offload_state_to_cpu,
            buffer_frames=buffer_frames,
            profiler=None,
            use_gpu_preprocess=self.gpu_preprocess,
            pin_input_memory=self.pin_input_memory,
        )

        propagation_ms: list[float] = []
        reset_propagation_ms: list[float] = []
        try:
            # Seed path, including the object-count-dependent decoder shapes.
            for object_index, mask in enumerate(masks, start=1):
                self.predictor.add_new_mask(
                    inference_state=stream.state,
                    frame_idx=0,
                    obj_id=object_index,
                    mask=mask,
                )
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()

            # Exercise every growing memory length and at least one saturated
            # memory state. Synchronization is intentional: compile/capture cost
            # must finish here rather than leaking into the live benchmark.
            for frame_index in range(1, temporal_frames + 1):
                stream.append(self._variant_rgb(rgb, frame_index))
                started = time.perf_counter()
                output = None
                for output in self.predictor.propagate_in_video(
                    stream.state,
                    start_frame_idx=frame_index,
                    max_frame_num_to_track=1,
                    reverse=False,
                ):
                    pass
                if output is None:
                    raise RuntimeError(
                        "EfficientTAM pre-warm propagation returned no output"
                    )
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.synchronize()
                propagation_ms.append(1000.0 * (time.perf_counter() - started))

            # Warm the exact reset/reseed path used by SAM3 keyframes.
            stream.reset(self._variant_rgb(rgb, 101))
            for object_index, mask in enumerate(masks, start=1):
                self.predictor.add_new_mask(
                    inference_state=stream.state,
                    frame_idx=0,
                    obj_id=object_index,
                    mask=mask,
                )
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()

            for local_index in range(1, self.prewarm_post_reset_frames + 1):
                stream.append(self._variant_rgb(rgb, 101 + local_index))
                started = time.perf_counter()
                output = None
                for output in self.predictor.propagate_in_video(
                    stream.state,
                    start_frame_idx=local_index,
                    max_frame_num_to_track=1,
                    reverse=False,
                ):
                    pass
                if output is None:
                    raise RuntimeError(
                        "EfficientTAM post-reset pre-warm returned no output"
                    )
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.synchronize()
                reset_propagation_ms.append(
                    1000.0 * (time.perf_counter() - started)
                )
        finally:
            stream.close()

        return {
            "object_count": int(object_count),
            "temporal_frames": int(temporal_frames),
            "propagation_ms": propagation_ms,
            "reset_propagation_ms": reset_propagation_ms,
        }

    def prewarm(self, first_rgb: np.ndarray) -> dict[str, Any]:
        """Compile/capture common live EfficientTAM state shapes once.

        The predictor is shared between cameras, therefore the pre-warm itself
        is globally de-duplicated by predictor identity + resolution + settings.
        The second camera simply observes the completed warm-up and returns.
        """
        if not self.prewarm_enabled or not self.vos_optimized:
            return {"enabled": False, "performed": False}
        if torch is None:
            raise RuntimeError("PyTorch is required for EfficientTAM pre-warm")

        rgb = np.ascontiguousarray(first_rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB HxWx3 for pre-warm, got {rgb.shape}")

        temporal_frames = self._resolved_prewarm_temporal_frames()
        key = (
            id(self.predictor),
            int(rgb.shape[0]),
            int(rgb.shape[1]),
            self.prewarm_object_counts,
            temporal_frames,
            self.prewarm_post_reset_frames,
            self.prewarm_passes,
            str(self.device),
            bool(self.use_bf16),
        )

        with self._prewarm_lock:
            if key in self._prewarm_done:
                return {
                    "enabled": True,
                    "performed": False,
                    "already_warm": True,
                    "object_counts": list(self.prewarm_object_counts),
                    "temporal_frames": temporal_frames,
                }

            print(
                "[stage] EfficientTAM single-view pre-warm: "
                f"resolution={rgb.shape[1]}x{rgb.shape[0]}, "
                f"objects={list(self.prewarm_object_counts)}, "
                f"temporal_frames={temporal_frames}, "
                f"post_reset_frames={self.prewarm_post_reset_frames}, "
                f"passes={self.prewarm_passes}",
                flush=True,
            )

            started_total = time.perf_counter()
            results: list[dict[str, Any]] = []
            gpu_context = GLOBAL_CUDA_LOCK if self.serialize_gpu else nullcontext()
            with gpu_context:
                with torch.inference_mode(), self._autocast():
                    for pass_index in range(self.prewarm_passes):
                        for object_count in self.prewarm_object_counts:
                            started = time.perf_counter()
                            result = self._run_prewarm_sequence(
                                rgb,
                                object_count=object_count,
                                temporal_frames=temporal_frames,
                            )
                            result["pass"] = pass_index + 1
                            result["wall_ms"] = 1000.0 * (
                                time.perf_counter() - started
                            )
                            results.append(result)


            total_ms = 1000.0 * (time.perf_counter() - started_total)
            self._prewarm_done.add(key)

            # The last pass is a useful verification pass. If it is still slow,
            # say so explicitly rather than pretending all specialization is done.
            last_pass = [
                item for item in results if item["pass"] == self.prewarm_passes
            ]
            verify_max = 0.0
            for item in last_pass:
                verify_max = max(
                    verify_max,
                    *(item["propagation_ms"] or [0.0]),
                    *(item["reset_propagation_ms"] or [0.0]),
                )

            print(
                "[stage] EfficientTAM single-view pre-warm complete: "
                f"total={total_ms / 1000.0:.2f} s, "
                f"verification_max_propagation={verify_max:.2f} ms",
                flush=True,
            )
            if verify_max > 100.0:
                print(
                    "[WARN] EfficientTAM pre-warm verification still "
                    f"contained a {verify_max:.1f} ms propagation. This suggests "
                    "GPU contention or an un-covered dynamic specialization remains.",
                    flush=True,
                )

            return {
                "enabled": True,
                "performed": True,
                "already_warm": False,
                "total_ms": total_ms,
                "verification_max_ms": verify_max,
                "object_counts": list(self.prewarm_object_counts),
                "temporal_frames": temporal_frames,
                "results": results,
            }


class EfficientTAMMultiViewTracker(EfficientTAMTracker):
    """One EfficientTAM predictor shared by all synchronized camera views."""

    def __init__(
        self,
        *args: Any,
        num_views: int,
        **kwargs: Any,
    ) -> None:
        self.num_views = max(1, int(num_views))
        kwargs["fixed_num_views"] = self.num_views
        super().__init__(*args, **kwargs)
        self.streams: list[StreamingVideoState | None] = [None] * self.num_views
        self.track_ids_per_view: list[list[int]] = [
            [] for _ in range(self.num_views)
        ]
        self._live_prepared = False
        self._feature_snapshots: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._batched_preprocessor: BatchedStreamingPreprocessor | None = None

        required_api = (
            "snapshot_multiview_image_features",
            "correct_multiview_from_reference",
        )
        missing = [name for name in required_api if not hasattr(self.predictor, name)]
        if missing:
            raise RuntimeError(
                "EfficientTAM checkout is missing the asynchronous direct-reference "
                f"API: {missing}. Rebuild the container from the updated EfficientTAM repo."
            )

    def _live_states(self) -> list[dict[str, Any]]:
        if any(stream is None for stream in self.streams):
            raise RuntimeError("EfficientTAM multi-view streams are not initialized")
        return [stream.state for stream in self.streams if stream is not None]

    def _cache_feature_snapshot(self, snapshot: dict[str, Any]) -> None:
        frame_idx = int(snapshot["frame_idx"])
        self._feature_snapshots[frame_idx] = snapshot
        self._feature_snapshots.move_to_end(frame_idx)
        while len(self._feature_snapshots) > self.feature_history_frames:
            self._feature_snapshots.popitem(last=False)

    def has_feature_snapshot(self, frame_idx: int) -> bool:
        return int(frame_idx) in self._feature_snapshots

    @property
    def cached_feature_frames(self) -> tuple[int, ...]:
        return tuple(self._feature_snapshots.keys())

    def _state_prepared(self, state: dict[str, Any]) -> bool:
        if not bool(state.get("multiview_prepared", False)):
            return False
        return self.execution_mode != "fixed_batch" or bool(
            state.get("fixed_batch_prepared", False)
        )

    def _state_real_ids(self, state: dict[str, Any]) -> list[int]:
        key = (
            "fixed_batch_real_obj_ids"
            if self.execution_mode == "fixed_batch"
            else "obj_ids"
        )
        return [
            int(value)
            for value in state.get(key, [])
            if isinstance(value, (int, np.integer))
        ]

    @property
    def live_ready(self) -> bool:
        if not self._live_prepared or any(stream is None for stream in self.streams):
            return False
        states = [stream.state for stream in self.streams if stream is not None]
        return all(self._state_prepared(state) for state in states) and [
            self._state_real_ids(state) for state in states
        ] == self.track_ids_per_view

    def _assert_prepared_states(self, states: list[dict[str, Any]]) -> None:
        if len(states) != self.num_views:
            raise RuntimeError(
                f"Expected {self.num_views} live EfficientTAM states, got {len(states)}"
            )
        prepared = [self._state_prepared(state) for state in states]
        actual_ids = [self._state_real_ids(state) for state in states]
        if not all(prepared) or actual_ids != self.track_ids_per_view:
            raise RuntimeError(
                "EfficientTAM coordinated prepare failed: "
                f"execution_mode={self.execution_mode}, prepared={prepared}, "
                f"expected_ids={self.track_ids_per_view}, actual_ids={actual_ids}"
            )

    @staticmethod
    def _clear_multiview_tags(state: dict[str, Any]) -> None:
        for key in (
            "fixed_batch_real_obj_ids",
            "fixed_batch_real_obj_count",
            "fixed_batch_dummy_obj_ids",
            "fixed_batch_prepared",
            "multiview_prepared",
            "multiview_execution_mode",
        ):
            state.pop(key, None)

    def _new_stream_from_rgb(
        self,
        rgb: np.ndarray,
        *,
        profiler: Any | None = None,
        buffer_frames: int | None = None,
    ) -> StreamingVideoState:
        return StreamingVideoState(
            self.predictor,
            np.ascontiguousarray(rgb, dtype=np.uint8),
            self.offload_video_to_cpu,
            self.offload_state_to_cpu,
            buffer_frames=(
                self.stream_buffer_frames
                if buffer_frames is None
                else max(2, int(buffer_frames))
            ),
            profiler=profiler,
            use_gpu_preprocess=self.gpu_preprocess,
            pin_input_memory=self.pin_input_memory,
        )

    def _reset_or_create_streams(self, rgbs: list[np.ndarray]) -> None:
        if len(rgbs) != self.num_views:
            raise ValueError(f"Expected {self.num_views} RGB views, got {len(rgbs)}")

        self._live_prepared = False
        self._feature_snapshots.clear()
        for view_idx, rgb in enumerate(rgbs):
            stream = self.streams[view_idx]
            if stream is None:
                stream = self._new_stream_from_rgb(rgb)
            elif self.reuse_state_on_keyframe:
                stream.reset(rgb)
            else:
                stream.close()
                stream = self._new_stream_from_rgb(rgb)
            self.streams[view_idx] = stream
            self._clear_multiview_tags(stream.state)

        # Allocate/reuse the shared hot-path stager during the sparse keyframe
        # reset instead of paying its one-time allocation on the next live frame.
        self._ensure_batched_preprocessor()

    def _ensure_batched_preprocessor(
        self,
    ) -> BatchedStreamingPreprocessor | None:
        streams = [stream for stream in self.streams if stream is not None]
        if len(streams) != self.num_views:
            return None
        use_batch = (
            self.gpu_preprocess
            and self.num_views > 1
            and all(stream._gpu_preprocess for stream in streams)
            and all(stream.storage_device.type == "cuda" for stream in streams)
        )
        if not use_batch:
            return None
        if (
            self._batched_preprocessor is None
            or not self._batched_preprocessor.compatible(streams)
        ):
            self._batched_preprocessor = BatchedStreamingPreprocessor(streams)
        return self._batched_preprocessor

    def _append_synchronized_rgbs(self, rgbs: list[np.ndarray]) -> int:
        streams = [stream for stream in self.streams if stream is not None]
        if len(streams) != self.num_views:
            raise RuntimeError("EfficientTAM multi-view streams are not initialized")

        batch_preprocessor = self._ensure_batched_preprocessor()
        if batch_preprocessor is not None:
            return batch_preprocessor.append(streams, rgbs)

        frame_indices = [
            stream.append(rgb) for stream, rgb in zip(streams, rgbs)
        ]
        if len(set(frame_indices)) != 1:
            raise RuntimeError(
                "Multi-view streams became misaligned: "
                f"frame_indices={frame_indices}"
            )
        return int(frame_indices[0])

    def _prediction_from_view_result(
        self,
        result: dict[str, Any],
    ) -> TrackerPrediction:
        """Keep live mask logits on CUDA for the batched postprocessor.

        EfficientTAM already produced these logits on the GPU.  The old adapter
        copied the full O×H×W tensor to CPU only for postprocess to immediately
        copy it back to CUDA for resize/threshold/erosion.  Keep the tensor on
        device and transfer only the tiny presence vector plus compact masks later.
        """
        value = result["video_res_masks"]
        if torch is not None and torch.is_tensor(value):
            masks = value.detach()
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.ndim == 4 and masks.shape[0] == 1:
                masks = masks[0]
            if masks.ndim == 2:
                masks = masks[None]
            if masks.ndim != 3:
                raise ValueError(
                    f"Unexpected EfficientTAM mask shape: {tuple(masks.shape)}"
                )
            # Presence is a tiny O-vector; computing it on device avoids a full
            # logits D2H while preserving the existing heuristic exactly.
            flat = masks.float().reshape(masks.shape[0], -1)
            positive_fraction = (flat > 0).float().mean(dim=1)
            peak = flat.amax(dim=1)
            presence = (
                torch.sigmoid(peak.clamp(-20.0, 20.0))
                * (positive_fraction * 20.0).clamp(0.0, 1.0)
            ).cpu().numpy().astype(np.float32, copy=False)
        else:
            masks = self._to_numpy_logits(value)
            presence = self._presence_from_logits(masks)

        track_ids = [int(value) for value in result.get("obj_ids", [])]
        if len(track_ids) != int(masks.shape[0]):
            raise RuntimeError(
                "EfficientTAM output ID/mask count mismatch: "
                f"ids={len(track_ids)} masks={int(masks.shape[0])}"
            )
        return TrackerPrediction(
            track_ids,
            masks,
            presence,
            {
                "backend": self.backend_name,
                "execution_mode": self.execution_mode,
                "frame_index": int(result.get("frame_idx", -1)),
                "num_real_objects": int(
                    result.get("num_real_objects", len(track_ids))
                ),
                "num_dummy_objects": int(result.get("num_dummy_objects", 0)),
            },
        )

    def _seed_prediction(
        self,
        rgb: np.ndarray,
        seeds: list[TrackerSeed],
    ) -> TrackerPrediction:
        height, width = rgb.shape[:2]
        if seeds:
            masks = np.stack(
                [np.asarray(seed.mask, dtype=np.float32) for seed in seeds],
                axis=0,
            )
            presence = np.asarray(
                [float(seed.confidence) for seed in seeds],
                dtype=np.float32,
            )
        else:
            masks = np.empty((0, height, width), dtype=np.float32)
            presence = np.empty((0,), dtype=np.float32)
        return TrackerPrediction(
            [int(seed.track_id) for seed in seeds],
            masks,
            presence,
            {
                "backend": self.backend_name,
                "execution_mode": self.execution_mode,
                "frame_index": 0,
            },
        )

    def _prepare_live_states(self, states: list[dict[str, Any]]) -> None:
        if self.execution_mode == "fixed_batch":
            self.predictor.prepare_multiview_states(
                states,
                conditioning_frame_idx=0,
            )
            return

        # Sequential propagation accepts an empty view, but upstream prepare
        # requires at least one object. Mark empty views as prepared locally.
        for state in states:
            if state.get("obj_ids"):
                self.predictor.prepare_multiview_states(
                    [state],
                    conditioning_frame_idx=0,
                )
            else:
                state["multiview_prepared"] = True
                state["multiview_execution_mode"] = "sequential"

    def correct_views(
        self,
        rgbs: list[np.ndarray],
        seeds_per_view: list[list[TrackerSeed]],
    ) -> list[TrackerPrediction]:
        """Reset, reseed and prepare all views on one coordinated keyframe."""
        if len(seeds_per_view) != self.num_views:
            raise ValueError(
                f"Expected seeds for {self.num_views} views, got {len(seeds_per_view)}"
            )
        if self.execution_mode == "fixed_batch":
            for view_idx, seeds in enumerate(seeds_per_view):
                if len(seeds) != self.object_slots_per_view:
                    raise RuntimeError(
                        f"View {view_idx} must seed every configured fixed slot: "
                        f"got {len(seeds)}, expected {self.object_slots_per_view}"
                    )

        call_started = time.perf_counter()
        with self._gpu_guard():
            with torch.inference_mode(), self._autocast():
                self._reset_or_create_streams(rgbs)
                states: list[dict[str, Any]] = []
                predictions: list[TrackerPrediction] = []

                for view_idx, (rgb, seeds) in enumerate(zip(rgbs, seeds_per_view)):
                    stream = self.streams[view_idx]
                    assert stream is not None
                    state = stream.state
                    self.track_ids_per_view[view_idx] = [
                        int(seed.track_id) for seed in seeds
                    ]
                    for seed in seeds:
                        self.predictor.add_new_mask(
                            inference_state=state,
                            frame_idx=0,
                            obj_id=int(seed.track_id),
                            mask=np.asarray(seed.mask, dtype=bool),
                        )
                    states.append(state)
                    predictions.append(self._seed_prediction(rgb, seeds))

                self._prepare_live_states(states)
                self._assert_prepared_states(states)
                self._live_prepared = True
                snapshot = self.predictor.snapshot_multiview_image_features(
                    states,
                    frame_idx=0,
                )
                self._cache_feature_snapshot(snapshot)

        self.record_profile(
            "tracker_total_wall_cpu",
            1000.0 * (time.perf_counter() - call_started),
        )
        return predictions

    def track_views(
        self,
        rgbs: list[np.ndarray],
        *,
        correction_reference_frame_idx: int | None = None,
        correction_masks_per_view: list[list[np.ndarray]] | None = None,
    ) -> list[TrackerPrediction]:
        """Append one synchronized frame and run normal or direct correction.

        Every frame is encoded exactly once through
        ``snapshot_multiview_image_features``. The persistent snapshot is cached
        in a bounded GPU ring and is reused immediately for ordinary propagation.
        If a SAM3 correction for historical frame ``x`` is supplied, the same
        current snapshot is instead used for direct ``x -> current`` correction.
        """
        if len(rgbs) != self.num_views:
            raise ValueError(f"Expected {self.num_views} RGB views, got {len(rgbs)}")
        if not self.live_ready:
            raise RuntimeError("EfficientTAM is not initialized/prepared")

        call_started = time.perf_counter()
        with self._gpu_guard():
            with torch.inference_mode(), self._autocast():
                states = self._live_states()
                frame_idx = self._append_synchronized_rgbs(rgbs)
                source = "propagation"
                reference_idx: int | None = None
                reference_snapshot: dict[str, Any] | None = None
                if correction_reference_frame_idx is not None:
                    reference_idx = int(correction_reference_frame_idx)
                    if correction_masks_per_view is None:
                        raise ValueError(
                            "correction_masks_per_view is required with a reference frame"
                        )
                    # Hold the reference locally before inserting the current
                    # snapshot, so a just-at-the-ring-limit reference is still usable.
                    reference_snapshot = self._feature_snapshots.get(reference_idx)
                    if reference_snapshot is None:
                        raise KeyError(
                            f"No cached EfficientTAM feature snapshot for frame {reference_idx}; "
                            f"cached={list(self._feature_snapshots)}"
                        )

                current_snapshot = self.predictor.snapshot_multiview_image_features(
                    states,
                    frame_idx=frame_idx,
                )
                self._cache_feature_snapshot(current_snapshot)

                if correction_reference_frame_idx is not None:
                    assert reference_idx is not None and reference_snapshot is not None
                    if reference_idx >= frame_idx:
                        raise ValueError(
                            f"Direct correction requires reference < current, got "
                            f"{reference_idx} -> {frame_idx}"
                        )
                    results = self.predictor.correct_multiview_from_reference(
                        states,
                        reference_feature_snapshot=reference_snapshot,
                        reference_masks=correction_masks_per_view,
                        current_frame_idx=frame_idx,
                        current_feature_snapshot=current_snapshot,
                        reverse=False,
                    )
                    source = "direct_correction"
                else:
                    results = self.predictor.propagate_multiview_step(
                        states,
                        frame_idx=frame_idx,
                        reverse=False,
                        image_feature_snapshot=current_snapshot,
                    )

                predictions = [
                    self._prediction_from_view_result(result) for result in results
                ]
                for prediction in predictions:
                    prediction.metadata["tracking_source"] = source
                    prediction.metadata["feature_cache_frames"] = len(
                        self._feature_snapshots
                    )
                    if reference_idx is not None:
                        prediction.metadata["reference_frame_idx"] = reference_idx

        self.record_profile(
            "tracker_total_wall_cpu",
            1000.0 * (time.perf_counter() - call_started),
        )
        return predictions

    def prewarm_views(self, first_rgbs: list[np.ndarray]) -> dict[str, Any]:
        """Compile the selected multi-view execution shape before live tracking."""
        if not self.prewarm_enabled or not self.vos_optimized:
            return {"enabled": False, "performed": False}
        if len(first_rgbs) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} prewarm views, got {len(first_rgbs)}"
            )

        rgbs = [np.ascontiguousarray(rgb, dtype=np.uint8) for rgb in first_rgbs]
        shapes = {rgb.shape for rgb in rgbs}
        if len(shapes) != 1:
            raise RuntimeError(
                f"All prewarm views must share one resolution, got {sorted(shapes)}"
            )

        temporal_frames = self._resolved_prewarm_temporal_frames()
        key = (
            id(self.predictor),
            "multiview",
            self.execution_mode,
            self.num_views,
            self.object_slots_per_view,
            tuple(rgbs[0].shape),
            temporal_frames,
            self.prewarm_post_reset_frames,
            self.prewarm_passes,
            str(self.device),
            bool(self.use_bf16),
        )

        with self._prewarm_lock:
            if key in self._prewarm_done:
                return {"enabled": True, "performed": False, "already_warm": True}

            print(
                "[stage] EfficientTAM pre-warm: "
                f"mode={self.execution_mode}, views={self.num_views}, "
                f"object_slots/view={self.object_slots_per_view}, "
                f"temporal_frames={temporal_frames}, passes={self.prewarm_passes}",
                flush=True,
            )
            started_total = time.perf_counter()
            propagation_ms: list[float] = []
            buffer_frames = max(self.stream_buffer_frames, temporal_frames + 4)

            with self._gpu_guard():
                with torch.inference_mode(), self._autocast():
                    for pass_index in range(self.prewarm_passes):
                        # Normal path: encode+snapshot once, then propagate using
                        # that persistent snapshot exactly like the live tracker.
                        temp_streams = [
                            self._new_stream_from_rgb(rgb, buffer_frames=buffer_frames)
                            for rgb in rgbs
                        ]
                        try:
                            states = [stream.state for stream in temp_streams]
                            synthetic_per_view: list[list[np.ndarray]] = []
                            for view_idx, state in enumerate(states):
                                masks = _synthetic_masks(
                                    rgbs[view_idx].shape[0],
                                    rgbs[view_idx].shape[1],
                                    self.object_slots_per_view,
                                )
                                synthetic_per_view.append(masks)
                                for obj_idx, mask in enumerate(masks, start=1):
                                    self.predictor.add_new_mask(
                                        inference_state=state,
                                        frame_idx=0,
                                        obj_id=obj_idx,
                                        mask=mask,
                                    )
                            self.predictor.prepare_multiview_states(
                                states,
                                conditioning_frame_idx=0,
                            )

                            for frame_idx in range(1, temporal_frames + 1):
                                for view_idx, stream in enumerate(temp_streams):
                                    stream.append(
                                        self._variant_rgb(
                                            rgbs[view_idx],
                                            frame_idx + 31 * view_idx,
                                        )
                                    )
                                started = time.perf_counter()
                                snapshot = self.predictor.snapshot_multiview_image_features(
                                    states,
                                    frame_idx=frame_idx,
                                )
                                self.predictor.propagate_multiview_step(
                                    states,
                                    frame_idx=frame_idx,
                                    reverse=False,
                                    image_feature_snapshot=snapshot,
                                )
                                if (
                                    torch.cuda.is_available()
                                    and str(self.device).startswith("cuda")
                                ):
                                    torch.cuda.synchronize(torch.device(self.device))
                                propagation_ms.append(
                                    1000.0 * (time.perf_counter() - started)
                                )
                        finally:
                            for stream in temp_streams:
                                stream.close()

                        # Direct-correction path: corrected reference x -> current t,
                        # followed by one ordinary propagation. This keeps the first
                        # live asynchronous SAM3 result from triggering compilation.
                        direct_streams = [
                            self._new_stream_from_rgb(rgb, buffer_frames=buffer_frames)
                            for rgb in rgbs
                        ]
                        try:
                            states = [stream.state for stream in direct_streams]
                            masks_per_view: list[list[np.ndarray]] = []
                            for view_idx, state in enumerate(states):
                                masks = _synthetic_masks(
                                    rgbs[view_idx].shape[0],
                                    rgbs[view_idx].shape[1],
                                    self.object_slots_per_view,
                                )
                                masks_per_view.append(masks)
                                for obj_idx, mask in enumerate(masks, start=1):
                                    self.predictor.add_new_mask(
                                        inference_state=state,
                                        frame_idx=0,
                                        obj_id=obj_idx,
                                        mask=mask,
                                    )
                            self.predictor.prepare_multiview_states(
                                states,
                                conditioning_frame_idx=0,
                            )

                            for view_idx, stream in enumerate(direct_streams):
                                stream.append(self._variant_rgb(rgbs[view_idx], 301 + view_idx))
                            reference_snapshot = self.predictor.snapshot_multiview_image_features(
                                states,
                                frame_idx=1,
                            )
                            self.predictor.propagate_multiview_step(
                                states,
                                frame_idx=1,
                                reverse=False,
                                image_feature_snapshot=reference_snapshot,
                            )

                            for view_idx, stream in enumerate(direct_streams):
                                stream.append(self._variant_rgb(rgbs[view_idx], 401 + view_idx))
                            current_snapshot = self.predictor.snapshot_multiview_image_features(
                                states,
                                frame_idx=2,
                            )
                            self.predictor.correct_multiview_from_reference(
                                states,
                                reference_feature_snapshot=reference_snapshot,
                                reference_masks=masks_per_view,
                                current_frame_idx=2,
                                current_feature_snapshot=current_snapshot,
                                reverse=False,
                            )

                            for view_idx, stream in enumerate(direct_streams):
                                stream.append(self._variant_rgb(rgbs[view_idx], 501 + view_idx))
                            next_snapshot = self.predictor.snapshot_multiview_image_features(
                                states,
                                frame_idx=3,
                            )
                            self.predictor.propagate_multiview_step(
                                states,
                                frame_idx=3,
                                reverse=False,
                                image_feature_snapshot=next_snapshot,
                            )
                            if (
                                torch.cuda.is_available()
                                and str(self.device).startswith("cuda")
                            ):
                                torch.cuda.synchronize(torch.device(self.device))
                        finally:
                            for stream in direct_streams:
                                stream.close()

            total_ms = 1000.0 * (time.perf_counter() - started_total)
            self._prewarm_done.add(key)
            verify_max = max(propagation_ms[-temporal_frames:] or [0.0])
            print(
                "[stage] EfficientTAM pre-warm complete: "
                f"total={total_ms / 1000.0:.2f} s, "
                f"verification_max={verify_max:.2f} ms",
                flush=True,
            )
            return {
                "enabled": True,
                "performed": True,
                "total_ms": total_ms,
                "verification_max_ms": verify_max,
                "execution_mode": self.execution_mode,
                "views": self.num_views,
                "object_slots_per_view": self.object_slots_per_view,
            }

    def close(self) -> None:
        self._live_prepared = False
        self._feature_snapshots.clear()
        self._batched_preprocessor = None
        for stream in self.streams:
            if stream is not None:
                stream.close()
        self.streams = [None] * self.num_views
        self.track_ids_per_view = [[] for _ in range(self.num_views)]
