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

from .component import SAMTrackingComponent
from .config import load_config
from .visualization import RvizPublisher


# torch.compile + Inductor CUDAGraph Trees keep capture-manager state in
# thread-local storage. EfficientTAM uses a predictor shared by both cameras, so
# all model construction, pre-warm, keyframe correction, and propagation must
# execute on one persistent OS thread. A mutex alone is insufficient: it
# serializes CUDA work but still alternates between the two camera worker
# threads, which can make deferred CUDAGraph capture fail in PyTorch TLS.
_GPU_OWNER_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="sam-gpu-owner",
)


def _run_on_gpu_owner(function, /, *args, **kwargs):
    """Run model-owning work on the single persistent CUDAGraph thread."""
    return _GPU_OWNER_EXECUTOR.submit(function, *args, **kwargs).result()


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
        "queue_wait_cpu",
        "ros_decode_cpu",
        "tf_lookup_cpu",
        "component_cpu",
        "rviz_build_cpu",
        "ros_publish_cpu",
        "worker_total_cpu",
    )

    PIPELINE_STAGE_ORDER = (
        "pipeline_total",
        "postprocess_cpu",
        "sam3_total_gpu",
        "tracker_lock_wait_cpu",
        "tracker_propagate_gpu",
        "tracker_total_gpu",
        "tracker_total_wall_cpu",
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

    def reset_all(self) -> None:
        """Start a fresh measurement epoch (used after expensive model pre-warm)."""
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
        if not self.enabled or not self.camera_enabled:
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
        if not self.enabled:
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
        if not self.enabled:
            return
        with self._lock:
            self._window_dropped += int(count)
            self._total_dropped += int(count)

    def on_error(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._window_errors += 1
            self._total_errors += 1

    def on_processed(self, *, keyframe: bool) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._window_processed += 1
            self._total_processed += 1
            if keyframe:
                self._window_keyframes += 1
                self._total_keyframes += 1

    def on_published(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._window_published += 1
            self._total_published += 1

    def record_worker_stage(self, name: str, value_ms: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._worker_samples.setdefault(name, []).append(float(value_ms))

    def record_pipeline_timings(self, timings_ms: dict[str, float]) -> None:
        if not self.enabled:
            return
        with self._lock:
            for name, value in timings_ms.items():
                self._pipeline_samples.setdefault(name, []).append(float(value))

    def maybe_report(self, *, force: bool = False, queue_depth: int | None = None) -> None:
        if not self.enabled:
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
            self._raw_window = {
                name: self._new_raw_window_state() for name in self.RAW_STREAMS
            }
            self._sync_skew_samples = {}

        self._print_snapshot(snapshot, queue_depth=queue_depth)

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

        lines.append("  raw camera topics:")
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
            topic = topics.get(stream, "?")
            stamp_text = "n/a" if stamp_hz is None else f"{stamp_hz:.2f} Hz"
            lines.append(
                f"    {stream:<5} wall={wall_hz:6.2f} Hz ({count:4d} msgs), "
                f"stamp={stamp_text}, nonmono={int(state['nonmonotonic'])}, "
                f"topic={topic}"
            )

            wall_gaps = state["wall_gap_ms"]
            if wall_gaps:
                mean, median, p95, maximum = self._summary(wall_gaps)
                lines.append(
                    f"      wall gap: n={len(wall_gaps)}, mean={mean:.2f} ms, "
                    f"median={median:.2f} ms, p95={p95:.2f} ms, max={maximum:.2f} ms"
                )
            stamp_gaps = state["stamp_gap_ms"]
            if stamp_gaps:
                mean, median, p95, maximum = self._summary(stamp_gaps)
                lines.append(
                    f"      stamp gap: n={len(stamp_gaps)}, mean={mean:.2f} ms, "
                    f"median={median:.2f} ms, p95={p95:.2f} ms, max={maximum:.2f} ms"
                )

        lines.append("  synchronization:")
        lines.append(
            f"    ApproximateTimeSynchronizer slop = {self.sync_slop_ms:.2f} ms"
        )
        for stream in self.RAW_STREAMS:
            raw_count = int(raw[stream]["count"])
            ratio = 100.0 * sync_count / max(1, raw_count)
            lines.append(
                f"    sync/raw {stream:<5} count ratio = {ratio:6.1f}% "
                f"({sync_count}/{raw_count})"
            )

        sync_skew = snapshot["sync_skew"]
        for name in self.SYNC_SKEW_ORDER:
            values = sync_skew.get(name, [])
            if not values:
                continue
            mean, median, p95, maximum = self._summary(values)
            lines.append(
                f"    {name}: n={len(values)}, mean={mean:.3f} ms, "
                f"median={median:.3f} ms, p95={p95:.3f} ms, max={maximum:.3f} ms"
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
                f"[CameraOnly:{self.camera_name}] window={elapsed:.2f} s",
                f"  RGB-D synchronized input = {input_hz:.2f} Hz ({input_count} packets)",
            ]
        else:
            lines = [
                f"[Rate:{self.camera_name}] window={elapsed:.2f} s",
                f"  RGB-D synchronized input = {input_hz:.2f} Hz ({input_count} packets)",
                f"  processed                = {processed_hz:.2f} Hz ({processed_count} frames)",
                f"  published                = {published_hz:.2f} Hz ({published_count} frames)",
                f"  dropped                  = {dropped_count} ({drop_pct:.1f}% of synchronized input)",
                f"  worker errors             = {int(snapshot['errors'])}",
                f"  keyframes                 = {int(snapshot['keyframes'])}",
            ]
            if queue_depth is not None:
                lines.append(f"  queue depth now           = {int(queue_depth)}")

        self._print_camera_section(lines, snapshot)

        if self.camera_only:
            raw = snapshot["raw"]
            lines.extend(
                [
                    "  cumulative:",
                    f"    sync_input={int(snapshot['total_input'])}",
                    "    raw=" + ", ".join(
                        f"{stream}:{int(raw[stream]['total_count'])}"
                        f"(nonmono={int(raw[stream]['total_nonmonotonic'])})"
                        for stream in self.RAW_STREAMS
                    ),
                ]
            )
            print("\n".join(lines), flush=True)
            return

        lines.append("  worker timing:")
        worker_samples = snapshot["worker"]
        for name in self.WORKER_STAGE_ORDER:
            values = worker_samples.get(name, [])
            if not values:
                continue
            mean, median, p95, maximum = self._summary(values)
            lines.append(
                f"    {name}: n={len(values)}, mean={mean:.2f} ms, "
                f"median={median:.2f} ms, p95={p95:.2f} ms, max={maximum:.2f} ms"
            )

        lines.append("  pipeline / periodic stalls:")
        pipeline_samples = snapshot["pipeline"]
        for name in self.PIPELINE_STAGE_ORDER:
            values = pipeline_samples.get(name, [])
            if not values:
                continue
            mean, median, p95, maximum = self._summary(values)
            lines.append(
                f"    {name}: n={len(values)}, mean={mean:.2f} ms, "
                f"median={median:.2f} ms, p95={p95:.2f} ms, max={maximum:.2f} ms"
            )

        raw = snapshot["raw"]
        lines.extend(
            [
                "  cumulative:",
                f"    sync_input={int(snapshot['total_input'])}, "
                f"processed={int(snapshot['total_processed'])}, "
                f"published={int(snapshot['total_published'])}, "
                f"dropped={int(snapshot['total_dropped'])}, "
                f"errors={int(snapshot['total_errors'])}, "
                f"keyframes={int(snapshot['total_keyframes'])}",
                "    raw=" + ", ".join(
                    f"{stream}:{int(raw[stream]['total_count'])}"
                    f"(nonmono={int(raw[stream]['total_nonmonotonic'])})"
                    for stream in self.RAW_STREAMS
                ),
            ]
        )
        print("\n".join(lines), flush=True)


class _CameraWorker:
    """Keep only the newest synchronized RGB-D frame for one camera."""

    def __init__(self, node: Any, camera_name: str, config) -> None:
        import message_filters
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image
        from rclpy.qos import qos_profile_sensor_data

        self.node = node
        self.camera_name = camera_name
        self.config = config
        self.bridge = CvBridge()
        self.component = _run_on_gpu_owner(
            SAMTrackingComponent,
            config,
            camera_name=camera_name,
        )
        self.visualizer = RvizPublisher(node, camera_name, config)
        self.diagnostics = _RateDiagnostics(node, camera_name, config)
        self._prewarm_pending = bool(
            str(config.tracker.backend) == "efficient_tam"
            and bool(config.tracker.efficient_tam.get("prewarm_enabled", True))
        )
        self.queue: queue.Queue[_Packet] = queue.Queue(
            maxsize=int(config.runtime.queue_size)
        )
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name=f"tracking-{camera_name}",
            daemon=True,
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

        # Tap each raw message_filters subscriber before synchronization. These
        # callbacks do not decode images and do not alter the messages; they only
        # count arrivals and inspect header timestamps.
        self.color_sub.registerCallback(self._raw_color_callback)
        self.depth_sub.registerCallback(self._raw_depth_callback)
        self.info_sub.registerCallback(self._raw_info_callback)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=max(4, int(config.runtime.queue_size) * 2),
            slop=float(config.ros.sync_slop_seconds),
        )
        self.sync.registerCallback(self._sync_callback)
        self.thread.start()

        self.node.get_logger().info(
            f"{camera_name} camera diagnostics topics: "
            f"color={color_topic}, depth={depth_topic}, info={info_topic}, "
            f"sync_slop={1000.0 * float(config.ros.sync_slop_seconds):.1f} ms"
        )

    def _raw_color_callback(self, message: Any) -> None:
        self.diagnostics.on_raw_message("color", message)

    def _raw_depth_callback(self, message: Any) -> None:
        self.diagnostics.on_raw_message("depth", message)

    def _raw_info_callback(self, message: Any) -> None:
        self.diagnostics.on_raw_message("info", message)

    def _sync_callback(self, color: Any, depth: Any, info: Any) -> None:
        self.diagnostics.on_sync_input(color, depth, info)
        packet = _Packet(
            color=color,
            depth=depth,
            info=info,
            enqueue_wall_s=time.perf_counter(),
        )
        try:
            self.queue.put_nowait(packet)
            self.diagnostics.maybe_report(queue_depth=self.queue.qsize())
            return
        except queue.Full:
            pass

        if not bool(self.config.runtime.drop_when_busy):
            # The newest synchronized packet itself is rejected.
            self.diagnostics.on_drop()
            self.diagnostics.maybe_report(queue_depth=self.queue.qsize())
            return

        # Keep only the newest packet. The pending packet removed from the
        # bounded queue is a real dropped synchronized input frame.
        try:
            self.queue.get_nowait()
            self.diagnostics.on_drop()
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            # Very unlikely race; if it happens, the new packet is also lost.
            self.diagnostics.on_drop()
        self.diagnostics.maybe_report(queue_depth=self.queue.qsize())

    def _world_from_camera(
        self,
        frame_id: str,
        stamp: Any,
    ) -> np.ndarray | None:
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

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                packet = self.queue.get(timeout=0.1)
            except queue.Empty:
                self.diagnostics.maybe_report(queue_depth=self.queue.qsize())
                continue

            worker_start = time.perf_counter()
            self.diagnostics.record_worker_stage(
                "queue_wait_cpu",
                1000.0 * (worker_start - packet.enqueue_wall_s),
            )

            try:
                stage_start = time.perf_counter()
                rgb = self.bridge.imgmsg_to_cv2(
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
                    depth_m = depth.astype(np.float32) * 0.001
                else:
                    depth_m = depth.astype(np.float32, copy=False)
                self.diagnostics.record_worker_stage(
                    "ros_decode_cpu",
                    self._elapsed_ms(stage_start),
                )

                if self._prewarm_pending:
                    self.node.get_logger().info(
                        f"{self.camera_name}: starting EfficientTAM full pre-warm; "
                        "camera/rate diagnostics will reset when it completes"
                    )
                    warmup_result = _run_on_gpu_owner(
                        self.component.prewarm_tracker,
                        rgb,
                    )
                    self._prewarm_pending = False

                    # Frames accumulated while compilation/capture was running are
                    # intentionally stale. Drop them without counting them in the
                    # live benchmark, then start a fresh diagnostics epoch.
                    while True:
                        try:
                            self.queue.get_nowait()
                        except queue.Empty:
                            break
                    self.diagnostics.reset_all()
                    self.node.get_logger().info(
                        f"{self.camera_name}: EfficientTAM pre-warm complete "
                        f"performed={warmup_result.get('performed', False)}; "
                        "live diagnostics restarted from zero"
                    )
                    worker_start = None
                    continue

                intrinsics = packet.info.k
                frame_id = (
                    packet.color.header.frame_id
                    or f"{self.camera_name}_optical_frame"
                )
                stamp = packet.color.header.stamp

                stage_start = time.perf_counter()
                world_from_camera = self._world_from_camera(frame_id, stamp)
                self.diagnostics.record_worker_stage(
                    "tf_lookup_cpu",
                    self._elapsed_ms(stage_start),
                )

                timestamp_ns = (
                    int(stamp.sec) * 1_000_000_000
                    + int(stamp.nanosec)
                )

                stage_start = time.perf_counter()
                result = _run_on_gpu_owner(
                    self.component.process_arrays,
                    rgb,
                    depth_m,
                    fx=float(intrinsics[0]),
                    fy=float(intrinsics[4]),
                    cx=float(intrinsics[2]),
                    cy=float(intrinsics[5]),
                    timestamp_ns=timestamp_ns,
                    world_from_camera=world_from_camera,
                )
                self.diagnostics.record_worker_stage(
                    "component_cpu",
                    self._elapsed_ms(stage_start),
                )
                self.diagnostics.record_pipeline_timings(result.timings_ms)
                self.diagnostics.on_processed(keyframe=bool(result.keyframe))

                stage_start = time.perf_counter()
                messages = self.visualizer.build_messages(result, stamp)
                self.diagnostics.record_worker_stage(
                    "rviz_build_cpu",
                    self._elapsed_ms(stage_start),
                )

                stage_start = time.perf_counter()
                self.visualizer.publish_messages(messages)
                self.diagnostics.record_worker_stage(
                    "ros_publish_cpu",
                    self._elapsed_ms(stage_start),
                )
                self.diagnostics.on_published()
            except Exception as error:
                self.diagnostics.on_error()
                self.node.get_logger().error(
                    f"{self.camera_name}: "
                    f"{type(error).__name__}: {error}\n"
                    f"{traceback.format_exc()}"
                )
            finally:
                if worker_start is not None:
                    self.diagnostics.record_worker_stage(
                        "worker_total_cpu",
                        self._elapsed_ms(worker_start),
                    )
                self.diagnostics.maybe_report(queue_depth=self.queue.qsize())

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.diagnostics.maybe_report(
            force=True,
            queue_depth=self.queue.qsize(),
        )
        _run_on_gpu_owner(self.component.print_stats)
        _run_on_gpu_owner(self.component.close)


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
        self.workers = [
            _CameraWorker(self.node, str(name), config)
            for name in config.runtime.camera_names
        ]
        self.node.get_logger().info(
            "Tracking "
            f"cameras={list(config.runtime.camera_names)} "
            f"backend={config.tracker.backend}"
        )

    def close(self) -> None:
        for worker in self.workers:
            worker.close()
        self.node.destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tracking.yaml")
    parser.add_argument(
        "--tracker",
        choices=("sam_mt", "efficient_tam"),
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
    config = load_config(args.config, tracker=args.tracker)
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
