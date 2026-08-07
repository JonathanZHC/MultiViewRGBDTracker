from __future__ import annotations

from contextlib import nullcontext
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .base import current_tracker_profiler
from .sam2_adapter import Sam2StyleStreamingTracker


def _install_memory_attention_clone_boundary(predictor: Any) -> None:
    """Keep CUDAGraph enabled and clone only memory-attention output.

    EfficientTAMVideoPredictorVOS already clones the compiled image-encoder,
    prompt-encoder, mask-decoder and memory-encoder outputs at their required
    graph lifetime boundaries. The missing boundary observed in this benchmark
    is memory_attention. Its output is cloned *outside* torch.compile so the
    persistent video state never retains a CUDAGraph-owned static output buffer.
    """
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

        # The predictor is shared by camera workers, so select the profiler of
        # the worker currently executing via a thread-local context.
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


class EfficientTAMTracker(Sam2StyleStreamingTracker):
    """EfficientTAM adapter retaining the native VOS/CUDAGraph fast path."""

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

        if self.vos_optimized:
            _install_memory_attention_clone_boundary(predictor)
        return predictor
