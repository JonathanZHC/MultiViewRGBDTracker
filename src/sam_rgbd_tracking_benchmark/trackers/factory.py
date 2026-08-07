from __future__ import annotations

from typing import Any

from .base import MultiObjectTracker
from .efficient_tam import EfficientTAMTracker
from .mock import MockOpticalFlowTracker
from .sam_mt import SamMTTracker


def build_tracker(config: Any, backend_override: str | None = None) -> MultiObjectTracker:
    backend = backend_override or config.tracker.backend
    if backend == "mock":
        return MockOpticalFlowTracker()
    common = dict(
        device=config.runtime.device,
        offload_video_to_cpu=bool(config.tracker.offload_video_to_cpu),
        offload_state_to_cpu=bool(config.tracker.offload_state_to_cpu),
        vos_optimized=bool(config.tracker.vos_optimized),
        serialize_gpu=bool(config.runtime.serialize_gpu),
        use_bf16=bool(config.runtime.use_bf16),
    )
    if backend == "sam_mt":
        return SamMTTracker(
            config_path=config.tracker.sam_mt.config,
            checkpoint_path=config.tracker.sam_mt.checkpoint,
            non_overlap_masks=bool(config.tracker.sam_mt.non_overlap_masks),
            points_per_object=int(config.tracker.sam_mt.points_per_object),
            **common,
        )
    if backend == "efficient_tam":
        return EfficientTAMTracker(
            config_path=config.tracker.efficient_tam.config,
            checkpoint_path=config.tracker.efficient_tam.checkpoint,
            non_overlap_masks=bool(config.tracker.efficient_tam.non_overlap_masks),
            **common,
        )
    raise ValueError(f"Unknown tracker backend: {backend}")
