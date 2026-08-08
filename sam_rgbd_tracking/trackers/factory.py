from __future__ import annotations

from .efficient_tam import EfficientTAMTracker
from .sam_mt import SamMTTracker


def build_tracker(config):
    backend = str(config.tracker.backend)

    # The frame buffer only needs to cover one detector-refresh interval plus a
    # small margin. It grows automatically if a session becomes longer.
    hz = float(config.runtime.target_hz)
    refresh_seconds = float(config.detector.refresh_seconds)
    default_buffer_frames = max(8, int(round(hz * refresh_seconds)) + 4)

    common = dict(
        device=str(config.runtime.device),
        offload_video_to_cpu=bool(config.tracker.offload_video_to_cpu),
        offload_state_to_cpu=bool(config.tracker.offload_state_to_cpu),
        vos_optimized=bool(config.tracker.vos_optimized),
        serialize_gpu=bool(config.runtime.serialize_gpu),
        use_bf16=bool(config.runtime.use_bf16),
        stream_buffer_frames=int(
            config.tracker.get("stream_buffer_frames", default_buffer_frames)
        ),
        reuse_state_on_keyframe=bool(
            config.tracker.get("reuse_state_on_keyframe", True)
        ),
        gpu_preprocess=bool(config.tracker.get("gpu_preprocess", True)),
        pin_input_memory=bool(config.tracker.get("pin_input_memory", True)),
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

        # By default warm every object-count specialization up to the number of
        # configured text prompts. This keeps the one-prompt benchmark cheap
        # while automatically covering 1..N for the usual multi-prompt setup.
        prompts = list(config.detector.get("prompts", []))
        default_max_objects = max(1, len(prompts))
        default_object_counts = list(range(1, default_max_objects + 1))
        configured_counts = cfg.get("prewarm_object_counts", default_object_counts)
        if isinstance(configured_counts, (int, float)):
            configured_counts = [int(configured_counts)]

        return EfficientTAMTracker(
            config_path=str(cfg.config),
            checkpoint_path=str(cfg.checkpoint),
            non_overlap_masks=bool(cfg.non_overlap_masks),
            prewarm_enabled=bool(cfg.get("prewarm_enabled", True)),
            prewarm_object_counts=[int(v) for v in configured_counts],
            prewarm_temporal_frames=int(cfg.get("prewarm_temporal_frames", 0)),
            prewarm_post_reset_frames=int(cfg.get("prewarm_post_reset_frames", 2)),
            prewarm_passes=int(cfg.get("prewarm_passes", 2)),
            **common,
        )

    raise ValueError(
        f"Unknown tracker backend: {backend!r}; use sam_mt or efficient_tam"
    )
