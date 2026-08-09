from __future__ import annotations

from .efficient_tam import EfficientTAMMultiViewTracker, EfficientTAMTracker
from .sam_mt import SamMTTracker


def _common_tracker_kwargs(config) -> dict:
    hz = float(config.runtime.target_hz)
    refresh_seconds = float(config.detector.refresh_seconds)
    default_buffer_frames = max(8, int(round(hz * refresh_seconds)) + 4)
    return dict(
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


def _efficient_tam_kwargs(config, *, num_views: int) -> dict:
    cfg = config.tracker.efficient_tam
    prompts = list(config.detector.get("prompts", []))
    default_max_objects = max(1, len(prompts))
    default_object_counts = list(range(1, default_max_objects + 1))
    configured_counts = cfg.get("prewarm_object_counts", default_object_counts)
    if isinstance(configured_counts, (int, float)):
        configured_counts = [int(configured_counts)]

    return dict(
        config_path=str(cfg.config),
        checkpoint_path=str(cfg.checkpoint),
        non_overlap_masks=bool(cfg.non_overlap_masks),
        prewarm_enabled=bool(cfg.get("prewarm_enabled", True)),
        prewarm_object_counts=[int(v) for v in configured_counts],
        prewarm_temporal_frames=int(cfg.get("prewarm_temporal_frames", 0)),
        prewarm_post_reset_frames=int(cfg.get("prewarm_post_reset_frames", 2)),
        prewarm_passes=int(cfg.get("prewarm_passes", 2)),
        execution_mode=str(cfg.get("execution_mode", "sequential")),
        fixed_num_views=int(num_views),
        max_objects_per_view=int(cfg.get("max_objects_per_view", default_max_objects)),
        **_common_tracker_kwargs(config),
    )


def build_tracker(config):
    """Build the original per-camera tracker path.

    SAM-MT continues to use this function unchanged. EfficientTAM sequential
    compatibility is kept for direct/component use, but the ROS node uses
    ``build_multiview_efficient_tam_tracker`` so both EfficientTAM execution
    modes share one predictor across all camera views.
    """
    backend = str(config.tracker.backend)
    common = _common_tracker_kwargs(config)

    if backend == "sam_mt":
        cfg = config.tracker.sam_mt
        return SamMTTracker(
            config_path=str(cfg.config),
            checkpoint_path=str(cfg.checkpoint),
            non_overlap_masks=bool(cfg.non_overlap_masks),
            points_per_object=int(cfg.points_per_object),
            compile_image_encoder=bool(cfg.get("compile_image_encoder", True)),
            prewarm_enabled=bool(cfg.get("prewarm_enabled", True)),
            prewarm_passes=int(cfg.get("prewarm_passes", 2)),
            **common,
        )

    if backend == "efficient_tam":
        # Direct single-camera component compatibility. The ROS integration uses
        # build_multiview_efficient_tam_tracker() below; a standalone component
        # keeps the original native per-object propagation semantics.
        kwargs = _efficient_tam_kwargs(config, num_views=1)
        kwargs["execution_mode"] = "sequential"
        return EfficientTAMTracker(**kwargs)

    raise ValueError(
        f"Unknown tracker backend: {backend!r}; use sam_mt or efficient_tam"
    )


def build_multiview_efficient_tam_tracker(config, *, num_views: int):
    if str(config.tracker.backend) != "efficient_tam":
        raise ValueError(
            "build_multiview_efficient_tam_tracker is only valid for "
            "tracker.backend=efficient_tam"
        )
    return EfficientTAMMultiViewTracker(
        num_views=int(num_views),
        **_efficient_tam_kwargs(config, num_views=int(num_views)),
    )
