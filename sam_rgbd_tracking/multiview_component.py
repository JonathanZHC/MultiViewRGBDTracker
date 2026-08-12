from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .alignment import CrossFrameAligner, CrossViewAligner
from .batched_postprocess import BatchedPostprocessor
from .async_sam3 import SAM3BatchResult
from .component import SAMTrackingComponent
from .config import Config, load_config
from .data_types import (
    DetectionInstance,
    FrameResult,
    MultiViewInstance,
    RGBDFrame,
    TrackerPrediction,
)
from .profiler import FrameProfiler
from .trackers import build_multiview_efficient_tam_tracker


class MultiViewEfficientTAMComponent:
    """Fixed-capacity multi-view tracking + spatial/temporal instance alignment."""

    def __init__(
        self,
        config: str | Path | Config = "configs/tracking.yaml",
        *,
        camera_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            config = load_config(config)

        self.config = config
        self.camera_names = [
            str(name)
            for name in (
                camera_names if camera_names is not None else config.runtime.camera_names
            )
        ]
        if not self.camera_names:
            raise ValueError("At least one camera is required")

        self.views = [
            SAMTrackingComponent(config, camera_name=camera_name)
            for camera_name in self.camera_names
        ]
        # Multi-view execution has one shared profiler. Disable the legacy
        # per-view profilers so the same work is never timed twice.
        for view in self.views:
            view.profiler.enabled = False
        self.tracker = build_multiview_efficient_tam_tracker(
            config,
            num_views=len(self.camera_names),
        )
        self.profiler = FrameProfiler(
            config, name="batched_pipeline", auto_print=False
        )
        self.cross_view = CrossViewAligner(config)
        self.postprocessor = BatchedPostprocessor(
            config,
            len(self.camera_names),
            voxelizer=self.cross_view.voxelizer,
        )
        self.cross_frame = CrossFrameAligner(config, num_views=len(self.camera_names))
        self.last_multiview_instances: list[MultiViewInstance] = []

        hz = float(config.runtime.target_hz)
        self.refresh_frames = max(
            1,
            int(round(float(config.detector.refresh_seconds) * hz)),
        )
        self.min_trigger_gap = max(
            1,
            int(config.detector.min_frames_between_triggers),
        )
        self.last_sam3_submit_frame = -10**9

    @property
    def live_ready(self) -> bool:
        return bool(self.tracker.live_ready)

    @property
    def execution_mode(self) -> str:
        return str(self.tracker.execution_mode)

    @property
    def object_slots_per_view(self) -> int:
        return int(self.tracker.object_slots_per_view)

    @property
    def feature_history_frames(self) -> int:
        return int(self.tracker.feature_history_frames)

    @property
    def fixed_tracking_batch_size(self) -> int:
        return len(self.camera_names) * self.object_slots_per_view

    @property
    def execution_description(self) -> str:
        if self.execution_mode == "fixed_batch":
            return (
                f"E(B={len(self.camera_names)}) + "
                f"tracking(B={self.fixed_tracking_batch_size})"
            )
        return f"E(B={len(self.camera_names)}) + per-object B1"

    def get_last_multiview_instances(self) -> list[MultiViewInstance]:
        return self.last_multiview_instances

    def prewarm_tracker(self, first_rgbs: list[np.ndarray]) -> dict[str, Any]:
        return dict(self.tracker.prewarm_views(first_rgbs))

    def make_frames_batch(self, view_inputs: list[dict[str, Any]]) -> list[RGBDFrame]:
        if len(view_inputs) != len(self.views):
            raise ValueError(
                f"Expected {len(self.views)} view inputs, got {len(view_inputs)}"
            )
        return [
            view.make_frame(
                item["rgb"],
                item["depth_m"],
                fx=float(item["fx"]),
                fy=float(item["fy"]),
                cx=float(item["cx"]),
                cy=float(item["cy"]),
                timestamp_ns=int(item.get("timestamp_ns", 0)),
                world_from_camera=item.get("world_from_camera"),
            )
            for view, item in zip(self.views, view_inputs)
        ]

    def _run_shared_tracker_stage(
        self,
        stage_name: str,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # EfficientTAM already executes all views/slots as one fixed batch. Profile
        # that batch once instead of duplicating the same CUDA interval per camera.
        with self.profiler.stage(stage_name, cuda=True):
            return function(*args, **kwargs)

    def _align_results(self, results: list[FrameResult]) -> list[FrameResult]:
        with self.profiler.stage("alignment_total", cuda=False):
            groups, cross_view_counts = self.cross_view.align(
                results, profiler=self.profiler
            )
            cross_frame_counts = self.cross_frame.align(
                groups, profiler=self.profiler
            )
        self.last_multiview_instances = groups
        counters = {**cross_view_counts, **cross_frame_counts}
        for result in results:
            result.metadata.update(counters)
            result.metadata["num_multiview_instances"] = len(groups)
        return results

    def _finish_frame(self, results: list[FrameResult]) -> list[FrameResult]:
        timings = self.profiler.end_frame()
        profiler_warmup_excluded = bool(self.profiler.last_frame_excluded)
        active_per_view = [
            int(result.metadata.get("num_active_instances_per_view", 0))
            for result in results
        ]
        dummy_per_view = [
            int(result.metadata.get("num_dummy_slots_per_view", 0))
            for result in results
        ]
        for result in results:
            # Every camera belongs to the same synchronized batched execution, so
            # every result carries the same one-per-bundle timing dictionary.
            result.timings_ms = dict(timings)
            result.metadata["profiling_warmup_excluded"] = profiler_warmup_excluded
            result.metadata["profiling_warmup_seen"] = int(self.profiler.seen_frames)
            result.metadata["active_instances_per_view"] = active_per_view
            result.metadata["dummy_slots_per_view"] = dummy_per_view
        return results

    def initialize_arrays_batch(
        self,
        view_inputs: list[dict[str, Any]],
        detections_per_view: list[list[DetectionInstance]],
        *,
        sam3_wall_ms: float | None = None,
        sam3_filter_ms: float | None = None,
        sam3_counts_per_view: list[dict[str, int]] | None = None,
    ) -> list[FrameResult]:
        return self.initialize_frames_batch(
            self.make_frames_batch(view_inputs),
            detections_per_view,
            sam3_wall_ms=sam3_wall_ms,
            sam3_filter_ms=sam3_filter_ms,
            sam3_counts_per_view=sam3_counts_per_view,
        )

    def initialize_frames_batch(
        self,
        frames: list[RGBDFrame],
        detections_per_view: list[list[DetectionInstance]],
        *,
        sam3_wall_ms: float | None = None,
        sam3_filter_ms: float | None = None,
        sam3_counts_per_view: list[dict[str, int]] | None = None,
    ) -> list[FrameResult]:
        """One-time B=views SAM3 seed into every configured fixed local slot."""
        if len(frames) != len(self.views):
            raise ValueError("Initial frame view count mismatch")
        if len(detections_per_view) != len(self.views):
            raise ValueError("Initial SAM3 result view count mismatch")
        self.profiler.begin_frame()
        if sam3_wall_ms is not None:
            self.profiler.record("sam3_async", float(sam3_wall_ms))
        if sam3_filter_ms is not None:
            self.profiler.record("sam3_filter", float(sam3_filter_ms))

        with self.profiler.stage("sam3_slot_assoc", cuda=False):
            seeds_per_view = [
                view.initialize_tracks(frame, detections)
                for view, frame, detections in zip(
                    self.views, frames, detections_per_view
                )
            ]
        predictions = self._run_shared_tracker_stage(
            "tracker_reinit",
            self.tracker.correct_views,
            [frame.rgb for frame in frames],
            seeds_per_view,
        )
        self.last_sam3_submit_frame = int(frames[0].frame_index)

        trigger_reasons = [["initial_sam3"] for _ in self.views]
        extra_metadata = [
            {
                "tracking_source": "initial_sam3",
                "sam3_refresh_due": False,
                "num_sam3_detections_per_class": (
                    sam3_counts_per_view[index]
                    if sam3_counts_per_view and index < len(sam3_counts_per_view)
                    else {}
                ),
            }
            for index in range(len(self.views))
        ]
        with self.profiler.stage("postprocess_alignment_total", cuda=False):
            results = self.postprocessor.process(
                self.views,
                frames,
                predictions,
                keyframe=True,
                trigger_reasons=trigger_reasons,
                extra_metadata_per_view=extra_metadata,
                profiler=self.profiler,
            )
            results = self._align_results(results)
        return self._finish_frame(results)

    def _build_correction_masks(
        self,
        correction: SAM3BatchResult,
    ) -> tuple[list[list[np.ndarray]], int]:
        if len(correction.reference_frames) != len(self.views):
            raise ValueError("SAM3 correction reference view count mismatch")
        if len(correction.detections_per_view) != len(self.views):
            raise ValueError("SAM3 correction detection view count mismatch")
        if len(correction.fallback_masks_per_view) != len(self.views):
            raise ValueError("SAM3 correction fallback view count mismatch")

        # Views are independent here. Reuse the persistent postprocess worker pool
        # instead of serializing two mask-IoU associations.
        with self.profiler.stage("sam3_slot_assoc", cuda=False):
            futures = []
            for view, reference_frame, detections, expected_ids, fallback_masks in zip(
                self.views,
                correction.reference_frames,
                correction.detections_per_view,
                self.tracker.track_ids_per_view,
                correction.fallback_masks_per_view,
            ):
                futures.append(
                    self.postprocessor.submit(
                        view.build_direct_correction_masks,
                        reference_frame,
                        detections,
                        expected_ids,
                        fallback_masks,
                    )
                )
            values = [future.result() for future in futures]
        return [value[0] for value in values], sum(int(value[1]) for value in values)

    def _refresh_due(
        self,
        frame_idx: int,
        predictions: list[TrackerPrediction],
    ) -> tuple[bool, str | None]:
        if frame_idx - self.last_sam3_submit_frame < self.min_trigger_gap:
            return False, None

        if bool(self.config.detector.trigger_on_anomaly):
            threshold = float(
                self.config.detector.get("anomaly_presence_threshold", 0.05)
            )
            for prediction in predictions:
                if (
                    prediction.presence_scores.size == 0
                    or float(np.min(prediction.presence_scores)) < threshold
                ):
                    return True, "tracking_anomaly"

        if frame_idx - self.last_sam3_submit_frame >= self.refresh_frames:
            return True, "periodic"
        return False, None

    def mark_sam3_submitted(self, frame_idx: int) -> None:
        self.last_sam3_submit_frame = int(frame_idx)

    def process_arrays_batch(
        self,
        view_inputs: list[dict[str, Any]],
        *,
        correction: SAM3BatchResult | None = None,
    ) -> list[FrameResult]:
        """Track one synchronized bundle; optionally consume one SAM3 result."""
        if not self.tracker.live_ready:
            raise RuntimeError("Call initialize_arrays_batch before propagation")

        self.profiler.begin_frame()
        frames = self.make_frames_batch(view_inputs)
        current_frame_idx = int(frames[0].frame_index)
        correction_applied = False
        correction_drop_reason: str | None = None
        correction_masks: list[list[np.ndarray]] | None = None
        reference_idx: int | None = None
        activated_slots = 0

        if correction is not None:
            reference_idx = int(correction.frame_index)
            self.profiler.record("sam3_async", float(correction.wall_ms))
            self.profiler.record("sam3_filter", float(correction.filter_cpu_ms))
            if reference_idx >= current_frame_idx:
                correction_drop_reason = "reference_not_older_than_current"
            elif not self.tracker.has_feature_snapshot(reference_idx):
                correction_drop_reason = "reference_feature_expired"
            else:
                correction_masks, activated_slots = self._build_correction_masks(correction)
                correction_applied = True

        if correction_applied:
            predictions = self._run_shared_tracker_stage(
                "tracker_direct_correction",
                self.tracker.track_views,
                [frame.rgb for frame in frames],
                correction_reference_frame_idx=reference_idx,
                correction_masks_per_view=correction_masks,
            )
            trigger_reasons = [
                [f"async_sam3_correction:{reference_idx}"] for _ in self.views
            ]
        else:
            predictions = self._run_shared_tracker_stage(
                "tracker_propagate",
                self.tracker.track_views,
                [frame.rgb for frame in frames],
            )
            if correction is not None:
                trigger_reasons = [
                    [f"sam3_result_dropped:{correction_drop_reason}"]
                    for _ in self.views
                ]
            else:
                trigger_reasons = [[] for _ in self.views]

        refresh_due, refresh_reason = self._refresh_due(
            current_frame_idx,
            predictions,
        )
        extra_metadata: list[dict[str, Any]] = []
        for view_index in range(len(self.views)):
            counts = {}
            if (
                correction is not None
                and correction.detections_per_class is not None
                and view_index < len(correction.detections_per_class)
            ):
                counts = correction.detections_per_class[view_index]
            extra_metadata.append(
                {
                    "tracking_source": (
                        "direct_correction" if correction_applied else "propagation"
                    ),
                    "sam3_refresh_due": bool(refresh_due),
                    "sam3_refresh_reason": refresh_reason,
                    "sam3_reference_frame": reference_idx,
                    "sam3_correction_applied": bool(correction_applied),
                    "sam3_correction_drop_reason": correction_drop_reason,
                    "sam3_activated_slots": int(activated_slots),
                    "num_sam3_detections_per_class": counts,
                    "feature_cache_frames": len(self.tracker.cached_feature_frames),
                }
            )
        with self.profiler.stage("postprocess_alignment_total", cuda=False):
            results = self.postprocessor.process(
                self.views,
                frames,
                predictions,
                keyframe=correction_applied,
                trigger_reasons=trigger_reasons,
                extra_metadata_per_view=extra_metadata,
                profiler=self.profiler,
            )
            results = self._align_results(results)
        return self._finish_frame(results)

    def fallback_masks_from_results(
        self,
        results: list[FrameResult],
    ) -> list[dict[int, np.ndarray]]:
        return [
            view.raw_masks_by_track(result)
            for view, result in zip(self.views, results)
        ]

    def print_stats(self) -> None:
        self.profiler.print_summary()

    def close(self) -> None:
        self.postprocessor.close()
        self.tracker.close()
        for view in self.views:
            view.close()
