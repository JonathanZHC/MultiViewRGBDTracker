from __future__ import annotations

from .efficient_tam import EfficientTAMTracker
from .sam_mt import SamMTTracker


def build_tracker(config):
    backend = str(config.tracker.backend)
    common = dict(
        device=str(config.runtime.device),
        offload_video_to_cpu=bool(config.tracker.offload_video_to_cpu),
        offload_state_to_cpu=bool(config.tracker.offload_state_to_cpu),
        vos_optimized=bool(config.tracker.vos_optimized),
        serialize_gpu=bool(config.runtime.serialize_gpu),
        use_bf16=bool(config.runtime.use_bf16),
    )
    if backend == "sam_mt":
        cfg = config.tracker.sam_mt
        return SamMTTracker(
            config_path=str(cfg.config),
            checkpoint_path=str(cfg.checkpoint),
            non_overlap_masks=bool(cfg.non_overlap_masks),
            points_per_object=int(cfg.points_per_object),
            **common,
        )
    if backend == "efficient_tam":
        cfg = config.tracker.efficient_tam
        return EfficientTAMTracker(
            config_path=str(cfg.config),
            checkpoint_path=str(cfg.checkpoint),
            non_overlap_masks=bool(cfg.non_overlap_masks),
            **common,
        )
    raise ValueError(f"Unknown tracker backend: {backend!r}; use sam_mt or efficient_tam")
