from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from typing import Any, ClassVar

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .base import GLOBAL_CUDA_LOCK, current_tracker_profiler
from .sam2_adapter import Sam2StyleStreamingTracker
from .streaming_state import StreamingVideoState


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
        **kwargs: Any,
    ) -> None:
        self.prewarm_enabled = bool(prewarm_enabled)
        cleaned = sorted({max(1, int(value)) for value in prewarm_object_counts})
        self.prewarm_object_counts = tuple(cleaned or [1])
        self.prewarm_temporal_frames = max(0, int(prewarm_temporal_frames))
        self.prewarm_post_reset_frames = max(1, int(prewarm_post_reset_frames))
        self.prewarm_passes = max(1, int(prewarm_passes))
        super().__init__(*args, **kwargs)

    @property
    def backend_name(self) -> str:
        return "efficient_tam"

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
                "[EfficientTAM warmup] starting full VOS pre-warm: "
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

                            prop = result["propagation_ms"]
                            reset_prop = result["reset_propagation_ms"]
                            prop_max = max(prop) if prop else 0.0
                            reset_max = max(reset_prop) if reset_prop else 0.0
                            print(
                                "[EfficientTAM warmup] "
                                f"pass={pass_index + 1}/{self.prewarm_passes} "
                                f"objects={object_count}: wall={result['wall_ms']:.1f} ms, "
                                f"prop_max={prop_max:.1f} ms, "
                                f"post_reset_max={reset_max:.1f} ms",
                                flush=True,
                            )

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
                "[EfficientTAM warmup] complete: "
                f"total={total_ms / 1000.0:.2f} s, "
                f"verification_max_propagation={verify_max:.2f} ms",
                flush=True,
            )
            if verify_max > 100.0:
                print(
                    "[EfficientTAM warmup] WARNING: the verification pass still "
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
