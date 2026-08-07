from __future__ import annotations

from contextlib import nullcontext
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .base import current_tracker_profiler
from .sam2_adapter import Sam2StyleStreamingTracker


def _move_rope_frequency_caches_to_device(
    predictor: Any,
    device: str,
) -> int:
    """Move RoPE frequency tensors to CUDA before the first compiled forward.

    EfficientTAM's RoPE attention stores ``freqs_cis`` (and, depending on the
    implementation, ``freqs_cis_q`` / ``freqs_cis_k``) as ordinary tensor
    attributes rather than registered buffers. They are therefore often still
    on CPU after the rest of the model has been moved to CUDA.

    With full VOS compilation, the first memory-attention forward can then be
    traced with a CPU tensor argument and TorchInductor prints messages such as

        skipping cudagraphs due to cpu device (...)

    ``torch.compile`` is lazy, so moving these tensor attributes immediately
    after predictor construction happens before their first compiled execution
    and keeps the corresponding specialization CUDA-only.
    """
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
    """Disable the optional EfficientTAM connected-components extension.

    The current container does not provide ``efficient_track_anything._C``.
    Upstream catches that import failure and continues, but repeats a warning
    every time hole filling is requested. Since the fallback already skips the
    operation, setting ``fill_hole_area`` to zero is behaviorally equivalent to
    what the current runtime actually does, while avoiding repeated exception
    handling and log spam.
    """
    fill_hole_area = int(getattr(predictor, "fill_hole_area", 0) or 0)
    if fill_hole_area <= 0:
        return

    predictor.fill_hole_area = 0
    print(
        "[EfficientTAM] optional _C hole-filling extension is unavailable in "
        "this container; fill_hole_area forced to 0 (same behavior as the "
        "upstream warning fallback)",
        flush=True,
    )


def _install_memory_attention_clone_boundary(predictor: Any) -> None:
    """Keep CUDAGraph enabled and clone only memory-attention output.

    EfficientTAMVideoPredictorVOS already clones the compiled image-encoder,
    prompt-encoder, mask-decoder and memory-encoder outputs at their required
    graph-lifetime boundaries. The missing boundary observed in this benchmark
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

        # Do these before the first lazy torch.compile execution.
        _move_rope_frequency_caches_to_device(
            predictor,
            self.device,
        )
        _disable_unavailable_hole_fill_extension(predictor)

        if self.vos_optimized:
            _install_memory_attention_clone_boundary(predictor)
        return predictor
