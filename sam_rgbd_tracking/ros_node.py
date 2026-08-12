from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

from .async_sam3 import AsyncSAM3Worker
from .config import load_config
from .multiview_component import MultiViewEfficientTAMComponent
from .visualization import FusedRvizPublisher, RvizPublisher
from .slots import object_slots_per_view


# EfficientTAM TorchInductor/CUDAGraph state stays on one persistent OS thread.
# SAM3 has its own persistent worker/thread and CUDA stream (AsyncSAM3Worker).
_TRACKER_GPU_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="efficienttam-gpu-owner",
)


def _run_on_tracker_owner(function, /, *args, **kwargs):
    return _TRACKER_GPU_EXECUTOR.submit(function, *args, **kwargs).result()


@dataclass
class _Packet:
    color: Any
    depth: Any
    info: Any
    enqueue_wall_s: float


class _RateDiagnostics:
    """Windowed camera, synchronization, and worker diagnostics.

    This helper measures three different layers independently:

    1. Raw ROS topic delivery for color, depth, and CameraInfo.
    2. ApproximateTimeSynchronizer output and timestamp skew.
    3. Bounded worker queue, component runtime, and RViz publishing.

    Keeping these layers separate is important: a fast tracking pipeline cannot
    produce 30 Hz if one raw camera topic is slow or if the three header stamps
    cannot be paired by the synchronizer.
    """

    RAW_STREAMS = ("color", "depth", "info")

    WORKER_STAGE_ORDER = (
        "queue_wait",
        "ros_decode",
        "tf_lookup",
        "component",
        "rviz_pointcloud_build_cpu",
        "rviz_markers_build_cpu",
        "rviz_overlay_build_cpu",
        "rviz_raw_mask_build_cpu",
        "rviz_filtered_mask_build_cpu",
        "rviz_fused_build_cpu",
        "rviz_build",
        "publish_points_cpu",
        "publish_markers_cpu",
        "publish_overlay_cpu",
        "publish_raw_mask_cpu",
        "publish_filtered_mask_cpu",
        "publish_fused_cpu",
        "ros_publish",
        "worker_total",
    )

    PIPELINE_STAGE_ORDER = (
        "pipeline_total",
        "sam3_filter",
        "sam3_slot_assoc",
        "sam3_async",
        "tracker_reinit",
        "tracker_propagate",
        "tracker_direct_correction",
        "postprocess_alignment_total",
        "postprocess_total",
        "postprocess_masks",
        "postprocess_components",
        "postprocess_geometry",
        "postprocess_finalize",
        "alignment_total",
        "cross_view_voxelize",
        "cross_view_bbox_gate",
        "cross_view_voxel_match",
        "cross_view_hungarian",
        "cross_view_fusion",
        "cross_view_total",
        "cross_frame_gate",
        "cross_frame_chamfer",
        "cross_frame_hungarian",
        "cross_frame_total",
    )

    SYNC_SKEW_ORDER = (
        "color_depth_ms",
        "color_info_ms",
        "depth_info_ms",
        "tuple_span_ms",
    )

    def __init__(
        self,
        node: Any,
        camera_name: str,
        config,
        *,
        camera_only: bool = False,
    ) -> None:
        self.node = node
        self.camera_name = camera_name
        self.camera_only = bool(camera_only)
        self.interval_s = max(
            0.5,
            float(config.profiling.get("rate_summary_interval_seconds", 5.0)),
        )
        self.enabled = bool(config.profiling.get("rate_diagnostics", True))
        self.camera_enabled = bool(
            config.profiling.get("camera_diagnostics", True)
        )
        # Normal tracking stays quiet during model warm-up/initialization.
        # Camera-only mode is diagnostic by definition, so it reports immediately.
        self._reporting_enabled = bool(camera_only)
        # Normal tracking can suppress all rate/worker/pipeline statistics during
        # the same warm-up epoch used by FrameProfiler. Camera-only diagnostics
        # are interactive diagnostics, so they continue to collect immediately.
        self._statistics_enabled = bool(camera_only)
        self.sync_slop_ms = 1000.0 * float(config.ros.sync_slop_seconds)

        self._lock = threading.Lock()
        now = time.perf_counter()
        self._window_start_s = now
        self._last_report_s = now

        self._window_input = 0
        self._window_processed = 0
        self._window_published = 0
        self._window_dropped = 0
        self._window_errors = 0
        self._window_keyframes = 0

        self._total_input = 0
        self._total_processed = 0
        self._total_published = 0
        self._total_dropped = 0
        self._total_errors = 0
        self._total_keyframes = 0

        self._worker_samples: dict[str, list[float]] = {}
        self._pipeline_samples: dict[str, list[float]] = {}
        self._latest_instance_counters: dict[str, Any] = {}

        self._topics: dict[str, str] = {}

        self._raw_window: dict[str, dict[str, Any]] = {
            name: self._new_raw_window_state() for name in self.RAW_STREAMS
        }
        self._raw_total_count: dict[str, int] = {
            name: 0 for name in self.RAW_STREAMS
        }
        self._raw_last_seen_stamp_ns: dict[str, int | None] = {
            name: None for name in self.RAW_STREAMS
        }
        self._raw_total_nonmonotonic: dict[str, int] = {
            name: 0 for name in self.RAW_STREAMS
        }

        self._sync_skew_samples: dict[str, list[float]] = {}

    @staticmethod
    def _new_raw_window_state() -> dict[str, Any]:
        return {
            "count": 0,
            "wall_first_s": None,
            "wall_last_s": None,
            "stamp_first_ns": None,
            "stamp_last_ns": None,
            "last_wall_s": None,
            "last_stamp_ns": None,
            "wall_gap_ms": [],
            "stamp_gap_ms": [],
            "nonmonotonic": 0,
        }

    def set_topics(self, *, color: str, depth: str, info: str) -> None:
        with self._lock:
            self._topics = {
                "color": str(color),
                "depth": str(depth),
                "info": str(info),
            }

    def start_reporting(self, *, collect_immediately: bool = True) -> None:
        """Start a fresh live diagnostics epoch.

        When ``collect_immediately`` is false the report clock starts, but all
        samples/counters remain disabled until ``enable_statistics`` is called.
        This lets the worker execute real warm-up bundles without contaminating
        worker/rate statistics.
        """
        self.reset_all()
        self._reporting_enabled = True
        self._statistics_enabled = bool(collect_immediately)

    def enable_statistics(self) -> None:
        """Begin a fresh measured epoch after runtime warm-up."""
        self.reset_all()
        self._statistics_enabled = True
        self._reporting_enabled = True

    def reset_all(self) -> None:
        """Start a fresh diagnostics epoch."""
        now = time.perf_counter()
        with self._lock:
            self._window_start_s = now
            self._last_report_s = now
            self._window_input = 0
            self._window_processed = 0
            self._window_published = 0
            self._window_dropped = 0
            self._window_errors = 0
            self._window_keyframes = 0
            self._total_input = 0
            self._total_processed = 0
            self._total_published = 0
            self._total_dropped = 0
            self._total_errors = 0
            self._total_keyframes = 0
            self._worker_samples = {}
            self._pipeline_samples = {}
            self._latest_instance_counters = {}
            self._raw_window = {
                name: self._new_raw_window_state() for name in self.RAW_STREAMS
            }
            self._raw_total_count = {name: 0 for name in self.RAW_STREAMS}
            self._raw_last_seen_stamp_ns = {name: None for name in self.RAW_STREAMS}
            self._raw_total_nonmonotonic = {name: 0 for name in self.RAW_STREAMS}
            self._sync_skew_samples = {}

    @staticmethod
    def _stamp_ns(stamp: Any) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _summary(values: list[float]) -> tuple[float, float, float, float]:
        array = np.asarray(values, dtype=np.float64)
        return (
            float(array.mean()),
            float(np.median(array)),
            float(np.percentile(array, 95)),
            float(array.max()),
        )

    @staticmethod
    def _rate_from_span(count: int, first: float | int | None, last: float | int | None, scale: float = 1.0) -> float | None:
        if count < 2 or first is None or last is None or last <= first:
            return None
        return (count - 1) * scale / (last - first)

    def on_raw_message(self, stream: str, message: Any) -> None:
        """Record every message delivered by one raw message_filters subscriber."""
        if (
            not self.enabled
            or not self.camera_enabled
            or not self._statistics_enabled
        ):
            return
        if stream not in self.RAW_STREAMS:
            raise ValueError(f"unknown raw stream: {stream}")

        now_s = time.perf_counter()
        stamp_ns = self._stamp_ns(message.header.stamp)

        with self._lock:
            state = self._raw_window[stream]
            state["count"] += 1
            self._raw_total_count[stream] += 1

            if state["wall_first_s"] is None:
                state["wall_first_s"] = now_s
            state["wall_last_s"] = now_s
            if state["stamp_first_ns"] is None:
                state["stamp_first_ns"] = stamp_ns
            state["stamp_last_ns"] = stamp_ns

            last_wall_s = state["last_wall_s"]
            if last_wall_s is not None and now_s > last_wall_s:
                state["wall_gap_ms"].append(1000.0 * (now_s - last_wall_s))
            state["last_wall_s"] = now_s

            last_window_stamp_ns = state["last_stamp_ns"]
            if last_window_stamp_ns is not None and stamp_ns > last_window_stamp_ns:
                state["stamp_gap_ms"].append(
                    (stamp_ns - last_window_stamp_ns) * 1e-6
                )
            state["last_stamp_ns"] = stamp_ns

            last_seen_stamp_ns = self._raw_last_seen_stamp_ns[stream]
            if last_seen_stamp_ns is not None and stamp_ns <= last_seen_stamp_ns:
                state["nonmonotonic"] += 1
                self._raw_total_nonmonotonic[stream] += 1
            self._raw_last_seen_stamp_ns[stream] = stamp_ns

    def on_sync_input(self, color: Any, depth: Any, info: Any) -> None:
        if not self.enabled or not self._statistics_enabled:
            return

        color_ns = self._stamp_ns(color.header.stamp)
        depth_ns = self._stamp_ns(depth.header.stamp)
        info_ns = self._stamp_ns(info.header.stamp)

        with self._lock:
            self._window_input += 1
            self._total_input += 1

            if self.camera_enabled:
                skew_values = {
                    "color_depth_ms": abs(color_ns - depth_ns) * 1e-6,
                    "color_info_ms": abs(color_ns - info_ns) * 1e-6,
                    "depth_info_ms": abs(depth_ns - info_ns) * 1e-6,
                    "tuple_span_ms": (
                        max(color_ns, depth_ns, info_ns)
                        - min(color_ns, depth_ns, info_ns)
                    ) * 1e-6,
                }
                for name, value in skew_values.items():
                    self._sync_skew_samples.setdefault(name, []).append(value)

    def on_drop(self, count: int = 1) -> None:
        if not self.enabled or not self._statistics_enabled:
            return
        with self._lock:
            self._window_dropped += int(count)
            self._total_dropped += int(count)

    def on_error(self) -> None:
        if not self.enabled or not self._statistics_enabled:
            return
        with self._lock:
            self._window_errors += 1
            self._total_errors += 1

    def on_processed(self, *, keyframe: bool) -> None:
        if not self.enabled or not self._statistics_enabled:
            return
        with self._lock:
            self._window_processed += 1
            self._total_processed += 1
            if keyframe:
                self._window_keyframes += 1
                self._total_keyframes += 1

    def on_published(self) -> None:
        if not self.enabled or not self._statistics_enabled:
            return
        with self._lock:
            self._window_published += 1
            self._total_published += 1

    def record_worker_stage(self, name: str, value_ms: float) -> None:
        if not self.enabled or not self._statistics_enabled:
            return
        with self._lock:
            self._worker_samples.setdefault(name, []).append(float(value_ms))

    def record_pipeline_timings(self, timings_ms: dict[str, float]) -> None:
        if not self.enabled or not self._statistics_enabled:
            return
        with self._lock:
            for name, value in timings_ms.items():
                self._pipeline_samples.setdefault(name, []).append(float(value))

    def record_instance_counters(self, metadata: dict[str, Any]) -> None:
        if not self.enabled or not self._statistics_enabled:
            return
        names = (
            "num_active_instances_per_view",
            "num_dummy_slots_per_view",
            "num_cross_view_candidate_pairs",
            "num_cross_view_matches",
            "num_cross_frame_candidate_pairs",
            "num_cross_frame_matches",
            "num_multiview_instances",
            "num_fused_points_before_downsample",
            "num_fused_points_after_downsample",
            "active_instances_per_view",
            "dummy_slots_per_view",
        )
        with self._lock:
            for name in names:
                if name in metadata:
                    self._latest_instance_counters[name] = metadata[name]
            sam3_counts = metadata.get("num_sam3_detections_per_class")
            if sam3_counts:
                self._latest_instance_counters[
                    "num_sam3_detections_per_class"
                ] = dict(sam3_counts)

    def maybe_report(self, *, force: bool = False, queue_depth: int | None = None) -> None:
        if (
            not self.enabled
            or not self._reporting_enabled
            or not self._statistics_enabled
        ):
            return
        now = time.perf_counter()
        with self._lock:
            elapsed = now - self._window_start_s
            if not force and (now - self._last_report_s) < self.interval_s:
                return
            if elapsed <= 0.0:
                return

            snapshot = {
                "elapsed": elapsed,
                "input": self._window_input,
                "processed": self._window_processed,
                "published": self._window_published,
                "dropped": self._window_dropped,
                "errors": self._window_errors,
                "keyframes": self._window_keyframes,
                "worker": self._worker_samples,
                "pipeline": self._pipeline_samples,
                "instance_counters": dict(self._latest_instance_counters),
                "total_input": self._total_input,
                "total_processed": self._total_processed,
                "total_published": self._total_published,
                "total_dropped": self._total_dropped,
                "total_errors": self._total_errors,
                "total_keyframes": self._total_keyframes,
                "topics": dict(self._topics),
                "raw": {
                    name: {
                        "count": int(state["count"]),
                        "wall_first_s": state["wall_first_s"],
                        "wall_last_s": state["wall_last_s"],
                        "stamp_first_ns": state["stamp_first_ns"],
                        "stamp_last_ns": state["stamp_last_ns"],
                        "wall_gap_ms": list(state["wall_gap_ms"]),
                        "stamp_gap_ms": list(state["stamp_gap_ms"]),
                        "nonmonotonic": int(state["nonmonotonic"]),
                        "total_count": int(self._raw_total_count[name]),
                        "total_nonmonotonic": int(
                            self._raw_total_nonmonotonic[name]
                        ),
                    }
                    for name, state in self._raw_window.items()
                },
                "sync_skew": {
                    name: list(values)
                    for name, values in self._sync_skew_samples.items()
                },
            }

            self._window_start_s = now
            self._last_report_s = now
            self._window_input = 0
            self._window_processed = 0
            self._window_published = 0
            self._window_dropped = 0
            self._window_errors = 0
            self._window_keyframes = 0
            self._worker_samples = {}
            self._pipeline_samples = {}
            self._latest_instance_counters = {}
            self._raw_window = {
                name: self._new_raw_window_state() for name in self.RAW_STREAMS
            }
            self._sync_skew_samples = {}

        self._print_snapshot(snapshot, queue_depth=queue_depth)

    @classmethod
    def _append_stats_table(
        cls,
        lines: list[str],
        title: str,
        samples: dict[str, list[float]],
        names: tuple[str, ...] | list[str],
    ) -> None:
        rows: list[tuple[str, int, float, float, float, float]] = []
        for name in names:
            values = samples.get(name, [])
            if not values:
                continue
            mean, median, p95, maximum = cls._summary(values)
            rows.append((name, len(values), mean, median, p95, maximum))
        if not rows:
            return

        lines.append(f"  {title} (ms):")
        lines.append(
            "    "
            f"{'stage':<34} {'n':>5} {'mean':>8} {'median':>8} "
            f"{'p95':>8} {'max':>8}"
        )
        for name, count, mean, median, p95, maximum in rows:
            lines.append(
                "    "
                f"{name:<34} {count:>5d} {mean:>8.2f} {median:>8.2f} "
                f"{p95:>8.2f} {maximum:>8.2f}"
            )

    def _print_camera_section(
        self,
        lines: list[str],
        snapshot: dict[str, Any],
    ) -> None:
        if not self.camera_enabled:
            return

        elapsed = float(snapshot["elapsed"])
        sync_count = int(snapshot["input"])
        raw = snapshot["raw"]
        topics = snapshot["topics"]

        lines.append("  camera input:")
        lines.append(
            "    "
            f"{'stream':<7} {'n':>5} {'wall Hz':>9} {'stamp Hz':>9} "
            f"{'nonmono':>8}"
        )
        for stream in self.RAW_STREAMS:
            state = raw[stream]
            count = int(state["count"])
            wall_hz = count / elapsed
            stamp_hz = self._rate_from_span(
                count,
                state["stamp_first_ns"],
                state["stamp_last_ns"],
                1e9,
            )
            stamp_text = "n/a" if stamp_hz is None else f"{stamp_hz:.2f}"
            lines.append(
                "    "
                f"{stream:<7} {count:>5d} {wall_hz:>9.2f} {stamp_text:>9} "
                f"{int(state['nonmonotonic']):>8d}"
            )
        if topics:
            lines.append(
                "    topics: "
                + " | ".join(
                    f"{stream}={topics.get(stream, '?')}"
                    for stream in self.RAW_STREAMS
                )
            )

        gap_rows: list[tuple[str, str, int, float, float, float, float]] = []
        for stream in self.RAW_STREAMS:
            state = raw[stream]
            for clock, key in (("wall", "wall_gap_ms"), ("stamp", "stamp_gap_ms")):
                values = state[key]
                if not values:
                    continue
                mean, median, p95, maximum = self._summary(values)
                gap_rows.append(
                    (stream, clock, len(values), mean, median, p95, maximum)
                )
        if gap_rows:
            lines.append("  camera gaps (ms):")
            lines.append(
                "    "
                f"{'stream':<7} {'clock':<6} {'n':>5} {'mean':>8} "
                f"{'median':>8} {'p95':>8} {'max':>8}"
            )
            for stream, clock, count, mean, median, p95, maximum in gap_rows:
                lines.append(
                    "    "
                    f"{stream:<7} {clock:<6} {count:>5d} {mean:>8.2f} "
                    f"{median:>8.2f} {p95:>8.2f} {maximum:>8.2f}"
                )

        lines.append(f"  synchronization: slop={self.sync_slop_ms:.2f} ms")
        ratios = []
        for stream in self.RAW_STREAMS:
            raw_count = int(raw[stream]["count"])
            ratio = 100.0 * sync_count / max(1, raw_count)
            ratios.append(f"{stream}={ratio:.1f}% ({sync_count}/{raw_count})")
        lines.append("    sync/raw: " + " | ".join(ratios))

        sync_skew = snapshot["sync_skew"]
        skew_rows = []
        for name in self.SYNC_SKEW_ORDER:
            values = sync_skew.get(name, [])
            if not values:
                continue
            mean, median, p95, maximum = self._summary(values)
            skew_rows.append((name, len(values), mean, median, p95, maximum))
        if skew_rows:
            lines.append(
                "    "
                f"{'skew':<20} {'n':>5} {'mean':>8} {'median':>8} "
                f"{'p95':>8} {'max':>8}"
            )
            for name, count, mean, median, p95, maximum in skew_rows:
                lines.append(
                    "    "
                    f"{name:<20} {count:>5d} {mean:>8.3f} {median:>8.3f} "
                    f"{p95:>8.3f} {maximum:>8.3f}"
                )

    def _print_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        queue_depth: int | None,
    ) -> None:
        elapsed = float(snapshot["elapsed"])
        input_count = int(snapshot["input"])
        processed_count = int(snapshot["processed"])
        published_count = int(snapshot["published"])
        dropped_count = int(snapshot["dropped"])

        input_hz = input_count / elapsed
        processed_hz = processed_count / elapsed
        published_hz = published_count / elapsed
        drop_pct = 100.0 * dropped_count / max(1, input_count)

        if self.camera_only:
            lines = [
                f"[CameraOnly:{self.camera_name}] {elapsed:.2f}s | "
                f"input={input_hz:.2f} Hz ({input_count})"
            ]
        else:
            queue_text = "n/a" if queue_depth is None else str(int(queue_depth))
            lines = [
                f"[Rate:{self.camera_name}] {elapsed:.2f}s | "
                f"input={input_hz:.2f} Hz | processed={processed_hz:.2f} Hz | "
                f"published={published_hz:.2f} Hz | drop={drop_pct:.1f}% "
                f"({dropped_count}/{input_count})",
                f"  state: queue={queue_text} | errors={int(snapshot['errors'])} | "
                f"keyframes={int(snapshot['keyframes'])}",
            ]

        self._print_camera_section(lines, snapshot)

        raw = snapshot["raw"]
        if self.camera_only:
            lines.append(
                f"  cumulative: sync_input={int(snapshot['total_input'])} | "
                + " | ".join(
                    f"{stream}={int(raw[stream]['total_count'])}"
                    f"(nonmono={int(raw[stream]['total_nonmonotonic'])})"
                    for stream in self.RAW_STREAMS
                )
            )
            print("\n".join(lines), flush=True)
            return

        worker_samples = snapshot["worker"]
        self._append_stats_table(
            lines,
            "worker core",
            worker_samples,
            (
                "queue_wait",
                "ros_decode",
                "tf_lookup",
                "component",
                "worker_total",
            ),
        )
        self._append_stats_table(
            lines,
            "visualization build",
            worker_samples,
            (
                "rviz_pointcloud_build_cpu",
                "rviz_markers_build_cpu",
                "rviz_overlay_build_cpu",
                "rviz_raw_mask_build_cpu",
                "rviz_filtered_mask_build_cpu",
                "rviz_fused_build_cpu",
                "rviz_build",
            ),
        )
        self._append_stats_table(
            lines,
            "ROS publish",
            worker_samples,
            (
                "publish_points_cpu",
                "publish_markers_cpu",
                "publish_overlay_cpu",
                "publish_raw_mask_cpu",
                "publish_filtered_mask_cpu",
                "publish_fused_cpu",
                "ros_publish",
            ),
        )

        counters = snapshot.get("instance_counters", {})
        if counters:
            lines.append("  instances / alignment (latest):")
            active = counters.get(
                "active_instances_per_view",
                counters.get("num_active_instances_per_view", "-"),
            )
            dummy = counters.get(
                "dummy_slots_per_view",
                counters.get("num_dummy_slots_per_view", "-"),
            )
            lines.append(
                "    "
                f"active/view={active} | dummy/view={dummy} | "
                f"multiview={counters.get('num_multiview_instances', '-')}"
            )
            lines.append(
                "    "
                f"cross-view: candidates={counters.get('num_cross_view_candidate_pairs', '-')} "
                f"matches={counters.get('num_cross_view_matches', '-')} | "
                f"cross-frame: candidates={counters.get('num_cross_frame_candidate_pairs', '-')} "
                f"matches={counters.get('num_cross_frame_matches', '-')}"
            )
            before = counters.get("num_fused_points_before_downsample")
            after = counters.get("num_fused_points_after_downsample")
            if before is not None and after is not None:
                reduction = 100.0 * (1.0 - float(after) / max(1.0, float(before)))
                lines.append(
                    f"    fused PCD: {int(before)} -> {int(after)} points "
                    f"({reduction:.1f}% removed at shared voxel resolution)"
                )
            sam3_counts = counters.get("num_sam3_detections_per_class")
            if sam3_counts:
                lines.append(
                    "    SAM3/class: "
                    + " | ".join(
                        f"{label}={count}" for label, count in sam3_counts.items()
                    )
                )

        pipeline_samples = snapshot["pipeline"]
        self._append_stats_table(
            lines,
            "batched pipeline",
            pipeline_samples,
            ("pipeline_total",),
        )
        self._append_stats_table(
            lines,
            "postprocess + alignment",
            pipeline_samples,
            ("postprocess_alignment_total",),
        )
        self._append_stats_table(
            lines,
            "batched postprocess",
            pipeline_samples,
            (
                "postprocess_total",
                "postprocess_masks",
                "postprocess_components",
                "postprocess_geometry",
                "postprocess_finalize",
            ),
        )
        self._append_stats_table(
            lines,
            "alignment",
            pipeline_samples,
            ("alignment_total",),
        )
        self._append_stats_table(
            lines,
            "cross-view",
            pipeline_samples,
            (
                "cross_view_voxelize",
                "cross_view_bbox_gate",
                "cross_view_voxel_match",
                "cross_view_hungarian",
                "cross_view_fusion",
                "cross_view_total",
            ),
        )
        self._append_stats_table(
            lines,
            "cross-frame",
            pipeline_samples,
            (
                "cross_frame_gate",
                "cross_frame_chamfer",
                "cross_frame_hungarian",
                "cross_frame_total",
            ),
        )
        self._append_stats_table(
            lines,
            "SAM3 refresh",
            pipeline_samples,
            (
                "sam3_filter",
                "sam3_slot_assoc",
                "sam3_async",
            ),
        )
        self._append_stats_table(
            lines,
            "tracker",
            pipeline_samples,
            (
                "tracker_reinit",
                "tracker_propagate",
                "tracker_direct_correction",
            ),
        )

        lines.append(
            f"  cumulative: input={int(snapshot['total_input'])} | "
            f"processed={int(snapshot['total_processed'])} | "
            f"published={int(snapshot['total_published'])} | "
            f"dropped={int(snapshot['total_dropped'])} | "
            f"errors={int(snapshot['total_errors'])} | "
            f"keyframes={int(snapshot['total_keyframes'])}"
        )
        print("\n".join(lines), flush=True)


class _MultiViewEfficientTAMWorker:
    """Synchronize camera bundles and run one shared EfficientTAM predictor.

    Cameras keep separate ROS subscriptions/RViz publishers, while tracker,
    post-processing, alignment, profiling, and rate reporting operate once per
    synchronized multi-view bundle.
    """

    def __init__(self, node: Any, config) -> None:
        import message_filters
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image
        from rclpy.qos import qos_profile_sensor_data

        self.node = node
        self.config = config
        self.camera_names = [str(name) for name in config.runtime.camera_names]
        self.bridge = CvBridge()
        self.component = _run_on_tracker_owner(
            MultiViewEfficientTAMComponent,
            config,
            camera_names=self.camera_names,
        )
        # SAM3 owns a separate persistent thread + CUDA stream so sparse B=views
        # detection can overlap continuous EfficientTAM tracking.
        self.sam3_worker = AsyncSAM3Worker(config)
        self.visualization_enabled = bool(
            config.runtime.get("enable_visualization", True)
        )
        self.visualizers = (
            {
                name: RvizPublisher(node, name, config)
                for name in self.camera_names
            }
            if self.visualization_enabled
            else {}
        )
        self.fused_visualizer = (
            FusedRvizPublisher(node, config) if self.visualization_enabled else None
        )
        # Normal tracking reports one synchronized-bundle stream. Per-camera raw
        # diagnostics remain available in CameraOnlyNode, but are intentionally not
        # attached here so the batched real-time path pays no diagnostic callback cost.
        self.batch_diagnostics = _RateDiagnostics(node, "batch", config)
        self.batch_diagnostics.camera_enabled = False

        self.queue: queue.Queue[list[_Packet]] = queue.Queue(
            maxsize=max(1, int(config.runtime.queue_size))
        )
        self.stop_event = threading.Event()
        self._packet_lock = threading.Lock()
        self._latest_packets: dict[str, _Packet] = {}
        self._cross_view_slop_ns = int(
            1e9
            * float(
                config.ros.get(
                    "multiview_sync_slop_seconds",
                    config.ros.sync_slop_seconds,
                )
            )
        )
        self._prewarm_pending = bool(
            config.tracker.efficient_tam.get("prewarm_enabled", True)
        )
        self._live = False
        self._initialized = False
        self._statistics_warmup_frames = max(
            0, int(config.profiling.get("warmup_frames", 0))
        )
        self._statistics_warmup_seen = 0
        self._statistics_started = False

        # Reuse the small Python orchestration containers every cycle.  Frame
        # arrays themselves are not recycled because asynchronous SAM3 keeps
        # reference RGB-D frames alive until its correction result returns.
        self._view_inputs: list[dict[str, Any]] = [
            {} for _ in self.camera_names
        ]
        self._rgbs: list[np.ndarray] = [
            np.empty((0, 0, 3), dtype=np.uint8)
            for _ in self.camera_names
        ]

        self.color_subs: dict[str, Any] = {}
        self.depth_subs: dict[str, Any] = {}
        self.info_subs: dict[str, Any] = {}
        self.syncs: dict[str, Any] = {}

        for camera_name in self.camera_names:
            color_topic = str(config.ros.color_topic).format(camera=camera_name)
            depth_topic = str(config.ros.depth_topic).format(camera=camera_name)
            info_topic = str(config.ros.camera_info_topic).format(camera=camera_name)
            color_sub = message_filters.Subscriber(
                node,
                Image,
                color_topic,
                qos_profile=qos_profile_sensor_data,
            )
            depth_sub = message_filters.Subscriber(
                node,
                Image,
                depth_topic,
                qos_profile=qos_profile_sensor_data,
            )
            info_sub = message_filters.Subscriber(
                node,
                CameraInfo,
                info_topic,
                qos_profile=qos_profile_sensor_data,
            )
            sync = message_filters.ApproximateTimeSynchronizer(
                [color_sub, depth_sub, info_sub],
                queue_size=max(4, int(config.runtime.queue_size) * 2),
                slop=float(config.ros.sync_slop_seconds),
            )
            sync.registerCallback(
                lambda color, depth, info, name=camera_name: self._sync_callback(
                    name, color, depth, info
                )
            )

            self.color_subs[camera_name] = color_sub
            self.depth_subs[camera_name] = depth_sub
            self.info_subs[camera_name] = info_sub
            self.syncs[camera_name] = sync

        self.thread = threading.Thread(
            target=self._run,
            name="tracking-efficient-tam-multiview",
            daemon=True,
        )
        self.thread.start()

        self.node.get_logger().info(
            "EfficientTAM multi-view initialized: "
            f"cameras={self.camera_names}, "
            f"execution_mode={self.component.execution_mode}, "
            f"{self.component.execution_description}; "
            "waiting for pre-warm/initial SAM3 seed"
        )

    @staticmethod
    def _stamp_ns(message: Any) -> int:
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _sync_callback(
        self,
        camera_name: str,
        color: Any,
        depth: Any,
        info: Any,
    ) -> None:
        packet = _Packet(
            color=color,
            depth=depth,
            info=info,
            enqueue_wall_s=time.perf_counter(),
        )

        ready_bundle: list[_Packet] | None = None
        with self._packet_lock:
            # Keep only the newest synchronized packet from each view until a
            # complete cross-view bundle is available.
            self._latest_packets[camera_name] = packet

            while all(name in self._latest_packets for name in self.camera_names):
                stamps = {
                    name: self._stamp_ns(self._latest_packets[name].color)
                    for name in self.camera_names
                }
                oldest_stamp = min(stamps.values())
                newest_stamp = max(stamps.values())
                if newest_stamp - oldest_stamp <= self._cross_view_slop_ns:
                    ready_bundle = [
                        self._latest_packets.pop(name)
                        for name in self.camera_names
                    ]
                    break

                # The oldest view cannot match the newer bundle anymore.  Drop
                # only that stale synchronized packet and wait for its next one.
                for name, stamp_ns in stamps.items():
                    if stamp_ns == oldest_stamp:
                        self._latest_packets.pop(name, None)

        if ready_bundle is not None:
            first = ready_bundle[0]
            self.batch_diagnostics.on_sync_input(first.color, first.depth, first.info)
            self._enqueue_bundle(ready_bundle)
        self.batch_diagnostics.maybe_report(queue_depth=self.queue.qsize())

    def _enqueue_bundle(self, bundle: list[_Packet]) -> None:
        try:
            self.queue.put_nowait(bundle)
            return
        except queue.Full:
            pass

        if not bool(self.config.runtime.drop_when_busy):
            self.batch_diagnostics.on_drop()
            return

        try:
            self.queue.get_nowait()
            self.batch_diagnostics.on_drop()
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait(bundle)
        except queue.Full:
            self.batch_diagnostics.on_drop()

    def _world_from_camera(self, frame_id: str, stamp: Any) -> np.ndarray | None:
        try:
            from rclpy.duration import Duration
            from rclpy.time import Time
            from tf2_ros import Buffer, TransformListener

            if not hasattr(self, "tf_buffer"):
                self.tf_buffer = Buffer()
                self.tf_listener = TransformListener(
                    self.tf_buffer,
                    self.node,
                    spin_thread=False,
                )
            transform = self.tf_buffer.lookup_transform(
                str(self.config.ros.world_frame),
                frame_id,
                Time.from_msg(stamp),
                timeout=Duration(seconds=0.01),
            )
        except Exception:
            return None

        translation = transform.transform.translation
        quaternion = transform.transform.rotation
        x, y, z, w = (
            float(quaternion.x),
            float(quaternion.y),
            float(quaternion.z),
            float(quaternion.w),
        )
        rotation = np.array(
            [
                [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
            ],
            dtype=np.float32,
        )
        result = np.eye(4, dtype=np.float32)
        result[:3, :3] = rotation
        result[:3, 3] = [
            float(translation.x),
            float(translation.y),
            float(translation.z),
        ]
        return result

    @staticmethod
    def _elapsed_ms(start_s: float) -> float:
        return 1000.0 * (time.perf_counter() - start_s)

    def _decode_bundle(
        self,
        bundle: list[_Packet],
    ) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
        # ``self._view_inputs`` and ``self._rgbs`` are persistent containers.
        # Only their references/values are refreshed here, eliminating per-frame
        # list/dict construction without recycling image arrays that SAM3 may
        # still own asynchronously.
        decode_ms = 0.0
        tf_ms = 0.0
        for view_index, (camera_name, packet) in enumerate(
            zip(self.camera_names, bundle)
        ):
            stage_started = time.perf_counter()
            rgb_decoded = self.bridge.imgmsg_to_cv2(
                packet.color,
                desired_encoding="rgb8",
            )
            depth = np.asarray(
                self.bridge.imgmsg_to_cv2(
                    packet.depth,
                    desired_encoding="passthrough",
                )
            )
            if depth.dtype == np.uint16:
                # One owning float32 depth array is intentional: an async SAM3
                # reference can outlive this worker iteration and correction uses
                # its historical depth for association.
                depth_m = depth.astype(np.float32)
                depth_m *= np.float32(0.001)
            else:
                depth_m = np.asarray(depth, dtype=np.float32)
                if not depth_m.flags.c_contiguous:
                    depth_m = np.ascontiguousarray(depth_m)

            rgb = np.asarray(rgb_decoded, dtype=np.uint8)
            if not rgb.flags.c_contiguous:
                rgb = np.ascontiguousarray(rgb)
            decode_ms += self._elapsed_ms(stage_started)

            intrinsics = packet.info.k
            frame_id = (
                packet.color.header.frame_id
                or f"{camera_name}_optical_frame"
            )
            stamp = packet.color.header.stamp

            stage_started = time.perf_counter()
            world_from_camera = self._world_from_camera(frame_id, stamp)
            tf_ms += self._elapsed_ms(stage_started)
            timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

            view_input = self._view_inputs[view_index]
            view_input["rgb"] = rgb
            view_input["depth_m"] = depth_m
            view_input["fx"] = float(intrinsics[0])
            view_input["fy"] = float(intrinsics[4])
            view_input["cx"] = float(intrinsics[2])
            view_input["cy"] = float(intrinsics[5])
            view_input["timestamp_ns"] = timestamp_ns
            view_input["world_from_camera"] = world_from_camera
            self._rgbs[view_index] = rgb

        self.batch_diagnostics.record_worker_stage("ros_decode", decode_ms)
        self.batch_diagnostics.record_worker_stage("tf_lookup", tf_ms)
        return self._view_inputs, self._rgbs

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                bundle = self.queue.get(timeout=0.1)
            except queue.Empty:
                self.batch_diagnostics.maybe_report(queue_depth=self.queue.qsize())
                continue

            worker_started = time.perf_counter()
            became_live = False
            processed_bundle = False
            self.batch_diagnostics.record_worker_stage(
                "queue_wait",
                max(1000.0 * (worker_started - packet.enqueue_wall_s) for packet in bundle),
            )

            try:
                view_inputs, rgbs = self._decode_bundle(bundle)

                if self._prewarm_pending:
                    self.node.get_logger().info(
                        "EfficientTAM multi-view pre-warm started; "
                        f"execution_mode={self.component.execution_mode}, "
                        f"views={len(self.camera_names)}, "
                        f"object_slots_per_view={self.component.object_slots_per_view}"
                    )
                    warmup_result = _run_on_tracker_owner(
                        self.component.prewarm_tracker,
                        rgbs,
                    )
                    self._prewarm_pending = False
                    while True:
                        try:
                            self.queue.get_nowait()
                        except queue.Empty:
                            break
                    with self._packet_lock:
                        self._latest_packets.clear()
                    self.batch_diagnostics.reset_all()
                    self.node.get_logger().info(
                        "EfficientTAM multi-view pre-warm complete "
                        f"performed={warmup_result.get('performed', False)}; "
                        "waiting for initial batched SAM3 seed"
                    )
                    continue

                stage_started = time.perf_counter()
                if not self._initialized:
                    # The very first SAM3 inference has no old tracker state to run
                    # against, so it is the only detector call that the live path
                    # waits for. It still executes on the dedicated SAM3 thread.
                    initial_frames = _run_on_tracker_owner(
                        self.component.make_frames_batch,
                        view_inputs,
                    )
                    initial_sam3 = self.sam3_worker.run_blocking(
                        frame_index=int(initial_frames[0].frame_index),
                        reference_frames=initial_frames,
                    )
                    # _make_frames already consumed frame index 0; initialize from
                    # those exact frames without constructing them a second time.
                    results = _run_on_tracker_owner(
                        self.component.initialize_frames_batch,
                        initial_frames,
                        initial_sam3.detections_per_view,
                        sam3_wall_ms=initial_sam3.wall_ms,
                        sam3_filter_ms=initial_sam3.filter_cpu_ms,
                        sam3_counts_per_view=initial_sam3.detections_per_class,
                    )
                    self._initialized = True
                else:
                    pending_correction = self.sam3_worker.poll()
                    results = _run_on_tracker_owner(
                        self.component.process_arrays_batch,
                        view_inputs,
                        correction=pending_correction,
                    )
                    if pending_correction is not None and results:
                        if bool(results[0].metadata.get("sam3_correction_applied", False)):
                            self.node.get_logger().info(
                                "SAM3 direct correction applied: "
                                f"reference={pending_correction.frame_index} -> "
                                f"current={results[0].frame.frame_index}, "
                                f"sam3_wall={pending_correction.wall_ms:.1f} ms"
                            )
                        else:
                            self.node.get_logger().warning(
                                "SAM3 result dropped: "
                                f"reference={pending_correction.frame_index}, "
                                f"reason={results[0].metadata.get('sam3_correction_drop_reason')}"
                            )
                component_ms = self._elapsed_ms(stage_started)

                if results:
                    processed_bundle = True
                    self.batch_diagnostics.record_worker_stage("component", component_ms)
                    self.batch_diagnostics.record_pipeline_timings(results[0].timings_ms)
                    self.batch_diagnostics.record_instance_counters(results[0].metadata)
                    self.batch_diagnostics.on_processed(keyframe=bool(results[0].keyframe))

                if self.visualization_enabled:
                    build_started = time.perf_counter()
                    built_messages = []
                    for camera_name, packet, result in zip(self.camera_names, bundle, results):
                        built_messages.append(
                            (
                                self.visualizers[camera_name],
                                self.visualizers[camera_name].build_messages(
                                    result, packet.color.header.stamp
                                ),
                            )
                        )
                    groups = (
                        _run_on_tracker_owner(self.component.get_last_multiview_instances)
                        if self.fused_visualizer is not None and results
                        else []
                    )
                    fused_messages = (
                        self.fused_visualizer.build_messages(
                            groups, bundle[0].color.header.stamp
                        )
                        if self.fused_visualizer is not None and results
                        else None
                    )
                    self.batch_diagnostics.record_worker_stage(
                        "rviz_build", self._elapsed_ms(build_started)
                    )

                    publish_started = time.perf_counter()
                    for visualizer, messages in built_messages:
                        visualizer.publish_messages(messages)
                    if self.fused_visualizer is not None and fused_messages is not None:
                        self.fused_visualizer.publish_messages(fused_messages)
                    self.batch_diagnostics.record_worker_stage(
                        "ros_publish", self._elapsed_ms(publish_started)
                    )
                    self.batch_diagnostics.on_published()

                # After frame x has been encoded/cached and published, trigger
                # sparse SAM3 on that exact synchronized bundle. At most one job
                # can be outstanding, so detector backlog is impossible.
                if results and bool(results[0].metadata.get("sam3_refresh_due", False)):
                    reference_frames = [result.frame for result in results]
                    fallback_masks = _run_on_tracker_owner(
                        self.component.fallback_masks_from_results,
                        results,
                    )
                    submitted = self.sam3_worker.submit(
                        frame_index=int(reference_frames[0].frame_index),
                        reference_frames=reference_frames,
                        fallback_masks_per_view=fallback_masks,
                    )
                    if submitted:
                        _run_on_tracker_owner(
                            self.component.mark_sam3_submitted,
                            int(reference_frames[0].frame_index),
                        )
                        self.node.get_logger().info(
                            "SAM3 async refresh submitted: "
                            f"frame={reference_frames[0].frame_index}, "
                            f"reason={results[0].metadata.get('sam3_refresh_reason')}, "
                            f"batch={len(reference_frames)}"
                        )

                became_live = not self._live
            except Exception as error:
                self.batch_diagnostics.on_error()
                self.node.get_logger().error(
                    "EfficientTAM multi-view worker: "
                    f"{type(error).__name__}: {error}\n"
                    f"{traceback.format_exc()}"
                )
            finally:
                total_ms = self._elapsed_ms(worker_started)
                self.batch_diagnostics.record_worker_stage("worker_total", total_ms)
                if became_live:
                    self._live = True
                    self.batch_diagnostics.start_reporting(
                        collect_immediately=self._statistics_warmup_frames == 0
                    )
                    self._statistics_started = self._statistics_warmup_frames == 0
                    self.node.get_logger().info(
                        "EfficientTAM tracking LIVE: "
                        f"execution_mode={self.component.execution_mode}, "
                        f"{self.component.execution_description}; "
                        f"profiling_warmup_frames={self._statistics_warmup_frames}"
                    )

                # Count only successfully completed tracking bundles. Enable rate/
                # worker statistics *after* the final warm-up bundle has completely
                # finished, so no decode/component/RViz/publish fragment from that
                # bundle leaks into the measured epoch. FrameProfiler independently
                # applies the same warmup_frames value to cumulative stage history.
                if (
                    processed_bundle
                    and self._live
                    and not self._statistics_started
                ):
                    self._statistics_warmup_seen += 1
                    if (
                        self._statistics_warmup_seen
                        >= self._statistics_warmup_frames
                    ):
                        self.batch_diagnostics.enable_statistics()
                        self._statistics_started = True
                        self.node.get_logger().info(
                            "Profiling warm-up complete; statistics start now: "
                            f"excluded_frames={self._statistics_warmup_seen}"
                        )

                self.batch_diagnostics.maybe_report(queue_depth=self.queue.qsize())

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.batch_diagnostics.maybe_report(
            force=True, queue_depth=self.queue.qsize()
        )
        self.sam3_worker.close()
        _run_on_tracker_owner(self.component.print_stats)
        _run_on_tracker_owner(self.component.close)


class _CameraOnlyWorker:
    """Measure raw/synchronized camera transport without constructing any model."""

    def __init__(self, node: Any, camera_name: str, config) -> None:
        import message_filters
        from sensor_msgs.msg import CameraInfo, Image
        from rclpy.qos import qos_profile_sensor_data

        self.node = node
        self.camera_name = camera_name
        self.config = config
        self.diagnostics = _RateDiagnostics(
            node,
            camera_name,
            config,
            camera_only=True,
        )

        color_topic = str(config.ros.color_topic).format(camera=camera_name)
        depth_topic = str(config.ros.depth_topic).format(camera=camera_name)
        info_topic = str(config.ros.camera_info_topic).format(camera=camera_name)
        self.diagnostics.set_topics(
            color=color_topic,
            depth=depth_topic,
            info=info_topic,
        )

        self.color_sub = message_filters.Subscriber(
            node,
            Image,
            color_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_sub = message_filters.Subscriber(
            node,
            Image,
            depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.info_sub = message_filters.Subscriber(
            node,
            CameraInfo,
            info_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.color_sub.registerCallback(self._raw_color_callback)
        self.depth_sub.registerCallback(self._raw_depth_callback)
        self.info_sub.registerCallback(self._raw_info_callback)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=max(4, int(config.runtime.queue_size) * 2),
            slop=float(config.ros.sync_slop_seconds),
        )
        self.sync.registerCallback(self._sync_callback)

        self.node.get_logger().info(
            f"{camera_name} CAMERA-ONLY baseline: "
            f"color={color_topic}, depth={depth_topic}, info={info_topic}, "
            f"sync_slop={1000.0 * float(config.ros.sync_slop_seconds):.1f} ms"
        )

    def _raw_color_callback(self, message: Any) -> None:
        self.diagnostics.on_raw_message("color", message)
        self.diagnostics.maybe_report()

    def _raw_depth_callback(self, message: Any) -> None:
        self.diagnostics.on_raw_message("depth", message)
        self.diagnostics.maybe_report()

    def _raw_info_callback(self, message: Any) -> None:
        self.diagnostics.on_raw_message("info", message)
        self.diagnostics.maybe_report()

    def _sync_callback(self, color: Any, depth: Any, info: Any) -> None:
        self.diagnostics.on_sync_input(color, depth, info)
        self.diagnostics.maybe_report()

    def close(self) -> None:
        self.diagnostics.maybe_report(force=True)


class CameraOnlyNode:
    """ROS camera-rate baseline with no SAM3/EfficientTAM/GPU tracker load."""

    def __init__(self, config) -> None:
        from rclpy.node import Node

        class _Node(Node):
            pass

        self.node = _Node("sam_rgbd_camera_baseline")
        self.workers = [
            _CameraOnlyWorker(self.node, str(name), config)
            for name in config.runtime.camera_names
        ]
        self.node.get_logger().info(
            "CAMERA-ONLY diagnostics running; no detector or tracker was constructed. "
            f"cameras={list(config.runtime.camera_names)}"
        )

    def close(self) -> None:
        for worker in self.workers:
            worker.close()
        self.node.destroy_node()


class TrackingNode:
    def __init__(self, config) -> None:
        from rclpy.node import Node

        class _Node(Node):
            pass

        self.node = _Node("sam_rgbd_tracking")
        self.workers = [_MultiViewEfficientTAMWorker(self.node, config)]
        execution_mode = str(
            config.tracker.efficient_tam.get("execution_mode", "fixed_batch")
        )
        object_slots = object_slots_per_view(config)
        history = int(config.tracker.efficient_tam.get("feature_history_frames", 32))
        use_max_autotune = bool(
            config.tracker.efficient_tam.get("use_max_autotune", False)
        )
        visualization_enabled = bool(
            config.runtime.get("enable_visualization", True)
        )
        self.node.get_logger().info(
            "Tracking "
            f"cameras={list(config.runtime.camera_names)} "
            f"backend=efficient_tam execution_mode={execution_mode} "
            f"object_slots_per_view={object_slots} "
            f"feature_history_frames={history} "
            f"compile_mode={'max-autotune' if use_max_autotune else 'default'} "
            f"visualization={'on' if visualization_enabled else 'off'}"
        )

    def close(self) -> None:
        for worker in self.workers:
            worker.close()
        self.node.destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tracking.yaml")
    parser.add_argument(
        "--efficient-tam-execution-mode",
        choices=("sequential", "fixed_batch"),
        default=None,
        help=(
            "Override tracker.efficient_tam.execution_mode from tracking.yaml "
            "for this run."
        ),
    )
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Measure raw RGB/depth/CameraInfo and synchronization rates without constructing SAM3 or a tracker.",
    )
    return parser.parse_args()


def main() -> None:
    import rclpy

    args = parse_args()
    config = load_config(
        args.config,
        efficient_tam_execution_mode=args.efficient_tam_execution_mode,
    )
    rclpy.init()
    wrapper = CameraOnlyNode(config) if args.camera_only else TrackingNode(config)
    try:
        rclpy.spin(wrapper.node)
    except KeyboardInterrupt:
        pass
    finally:
        wrapper.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
