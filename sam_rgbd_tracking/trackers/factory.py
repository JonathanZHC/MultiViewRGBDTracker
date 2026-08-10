from __future__ import annotations

from .efficient_tam import EfficientTAMMultiViewTracker
from ..slots import object_slots_per_view, slot_layout_key


def _common_tracker_kwargs(config) -> dict:
    hz = float(config.runtime.target_hz)
    refresh_seconds = float(config.detector.refresh_seconds)
    default_buffer_frames = max(8, int(round(hz * refresh_seconds)) + 8)
    return dict(
        device=str(config.runtime.device),
        offload_video_to_cpu=bool(config.tracker.offload_video_to_cpu),
        offload_state_to_cpu=bool(config.tracker.offload_state_to_cpu),
        vos_optimized=bool(config.tracker.vos_optimized),
        serialize_gpu=bool(config.runtime.get("serialize_gpu", False)),
        use_bf16=bool(config.runtime.use_bf16),
        stream_buffer_frames=int(
            config.tracker.get("stream_buffer_frames", default_buffer_frames)
        ),
        reuse_state_on_keyframe=True,
        gpu_preprocess=bool(config.tracker.get("gpu_preprocess", True)),
        pin_input_memory=bool(config.tracker.get("pin_input_memory", True)),
    )


def build_multiview_efficient_tam_tracker(config, *, num_views: int):
    cfg = config.tracker.efficient_tam
    slots_per_view = object_slots_per_view(config)
    # Fixed-capacity mode always seeds every configured slot (inactive slots use
    # zero masks), so only the production B=views*slots shape needs prewarming.
    default_object_counts = [slots_per_view]
    configured_counts = cfg.get("prewarm_object_counts", default_object_counts)
    if isinstance(configured_counts, (int, float)):
        configured_counts = [int(configured_counts)]

    return EfficientTAMMultiViewTracker(
        num_views=int(num_views),
        config_path=str(cfg.config),
        checkpoint_path=str(cfg.checkpoint),
        non_overlap_masks=bool(cfg.non_overlap_masks),
        prewarm_enabled=bool(cfg.get("prewarm_enabled", True)),
        prewarm_object_counts=[int(v) for v in configured_counts],
        prewarm_temporal_frames=int(cfg.get("prewarm_temporal_frames", 0)),
        prewarm_post_reset_frames=int(cfg.get("prewarm_post_reset_frames", 2)),
        prewarm_passes=int(cfg.get("prewarm_passes", 2)),
        execution_mode=str(cfg.get("execution_mode", "fixed_batch")),
        fixed_num_views=int(num_views),
        object_slots_per_view=slots_per_view,
        slot_layout_key=slot_layout_key(config),
        feature_history_frames=int(cfg.get("feature_history_frames", 32)),
        use_max_autotune=bool(cfg.get("use_max_autotune", False)),
        **_common_tracker_kwargs(config),
    )
