from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .component import SAMTrackingComponent
from .config import Config, load_config
from .data_types import FrameResult, RGBDFrame, TrackerPrediction
from .trackers.factory import build_multiview_efficient_tam_tracker


class MultiViewEfficientTAMComponent:
    """Run per-view SAM3/RGB-D processing around one shared EfficientTAM tracker."""

    def __init__(
        self,
        config: str | Path | Config = "configs/tracking.yaml",
        *,
        camera_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            config = load_config(config)
        if str(config.tracker.backend) != "efficient_tam":
            raise ValueError(
                "MultiViewEfficientTAMComponent requires tracker.backend=efficient_tam"
            )

        self.config = config
        self.camera_names = [
            str(name)
            for name in (
                camera_names if camera_names is not None else config.runtime.camera_names
            )
        ]
        if not self.camera_names:
            raise ValueError("At least one camera is required")

        # Each view owns detector/association/post-processing state. Only the
        # EfficientTAM predictor and propagation state are shared.
        self.views = [
            SAMTrackingComponent(
                config,
                camera_name=camera_name,
                build_tracker_backend=False,
            )
            for camera_name in self.camera_names
        ]
        self.tracker = build_multiview_efficient_tam_tracker(
            config,
            num_views=len(self.camera_names),
        )

    @property
    def execution_mode(self) -> str:
        return str(self.tracker.execution_mode)

    @property
    def max_objects_per_view(self) -> int:
        return int(self.tracker.max_objects_per_view)

    @property
    def fixed_tracking_batch_size(self) -> int:
        return len(self.camera_names) * self.max_objects_per_view

    @property
    def execution_description(self) -> str:
        if self.execution_mode == "fixed_batch":
            return (
                f"E(B={len(self.camera_names)}) + "
                f"tracking(B={self.fixed_tracking_batch_size})"
            )
        return f"{len(self.camera_names)}E + per-object B1"

    def prewarm_tracker(self, first_rgbs: list[np.ndarray]) -> dict[str, Any]:
        return dict(self.tracker.prewarm_views(first_rgbs))

    def _make_frames(self, view_inputs: list[dict[str, Any]]) -> list[RGBDFrame]:
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
        stage_names: tuple[tuple[str, bool], ...],
        function: Callable[..., Any],
        *args: Any,
    ) -> Any:
        """Profile one shared tracker call in every per-view profiler."""
        started = time.perf_counter()
        with ExitStack() as stack:
            for view in self.views:
                for name, cuda in stage_names:
                    stack.enter_context(view.profiler.stage(name, cuda=cuda))
            result = function(*args)

        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        for view in self.views:
            view.profiler.record("tracker_total_wall_cpu", elapsed_ms)
        return result

    def _run_coordinated_keyframe(
        self,
        frames: list[RGBDFrame],
    ) -> list[TrackerPrediction]:
        seeds_per_view = [
            view.detect_and_seed(frame)
            for view, frame in zip(self.views, frames)
        ]
        predictions = self._run_shared_tracker_stage(
            (
                ("tracker_reinit_gpu", True),
                ("tracker_total_gpu", True),
            ),
            self.tracker.correct_views,
            [frame.rgb for frame in frames],
            seeds_per_view,
        )
        for view, frame in zip(self.views, frames):
            view.mark_keyframe(frame.frame_index)
        return predictions

    def process_arrays_batch(
        self,
        view_inputs: list[dict[str, Any]],
    ) -> list[FrameResult]:
        """Process one synchronized multi-camera RGB-D bundle."""
        frames = self._make_frames(view_inputs)
        for view in self.views:
            view.begin_external_frame()

        tracker_unprepared = not self.tracker.live_ready
        periodic_requested = any(
            view.needs_periodic_keyframe(frame.frame_index)
            for view, frame in zip(self.views, frames)
        )

        keyframe = tracker_unprepared or periodic_requested
        update_raw_observations = False

        if keyframe:
            if tracker_unprepared:
                reason = (
                    "initial"
                    if all(not view.tracks for view in self.views)
                    else "tracker_unprepared"
                )
            else:
                reason = "coordinated_periodic"
            trigger_reasons = [[reason] for _ in self.views]
            predictions = self._run_coordinated_keyframe(frames)
        else:
            predictions = self._run_shared_tracker_stage(
                (
                    ("tracker_propagate_gpu", True),
                    ("tracker_total_gpu", True),
                ),
                self.tracker.track_views,
                [frame.rgb for frame in frames],
            )

            anomaly_requested = any(
                view.needs_anomaly_keyframe(frame.frame_index, prediction)
                for view, frame, prediction in zip(self.views, frames, predictions)
            )
            if anomaly_requested:
                keyframe = True
                trigger_reasons = [
                    ["coordinated_tracking_anomaly"] for _ in self.views
                ]
                predictions = self._run_coordinated_keyframe(frames)
            else:
                trigger_reasons = [[] for _ in self.views]
                update_raw_observations = True

        return [
            view.finalize_external_prediction(
                frame,
                prediction,
                keyframe=keyframe,
                trigger_reasons=reasons,
                update_raw_observations=update_raw_observations,
            )
            for view, frame, prediction, reasons in zip(
                self.views,
                frames,
                predictions,
                trigger_reasons,
            )
        ]

    def print_stats(self) -> None:
        for view in self.views:
            view.print_stats()

    def close(self) -> None:
        self.tracker.close()
        for view in self.views:
            view.close()
