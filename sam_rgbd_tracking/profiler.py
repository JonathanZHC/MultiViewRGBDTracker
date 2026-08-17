from __future__ import annotations

import csv
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class FrameProfiler:
    """Per-frame CPU/CUDA profiler with one synchronization at frame end.

    The tracker now exposes its internal stages instead of treating the whole
    tracker call as one opaque CUDA interval. This is important with two camera
    workers because lock waiting and CPU preprocessing must not be mistaken for
    EfficientTAM inference time.
    """

    PREFERRED_ORDER = [
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
        "postprocess_geometry_prepare",
        "postprocess_geometry_gpu",
        "postprocess_geometry_d2h",
        "postprocess_direct_fallback",
        "postprocess_finalize",
        "alignment_total",
        "cross_view_voxelize",
        "cross_view_bbox_gate",
        "cross_view_voxel_match",
        "cross_view_hungarian",
        "cross_view_fusion",
        "cross_view_gpu_fusion",
        "cross_view_total",
        "cross_frame_gate",
        "cross_frame_cloud_upload",
        "cross_frame_cloud_d2d",
        "cross_frame_chamfer",
        "cross_frame_hungarian",
        "cross_frame_total",
    ]

    def __init__(
        self,
        config,
        name: str = "tracking",
        *,
        auto_print: bool = True,
    ) -> None:
        self.name = name
        self.auto_print = bool(auto_print)
        self.enabled = bool(config.profiling.get("enabled", True))
        self.interval = int(config.profiling.get("summary_interval_frames", 100))
        # The ROS rate report already prints the same batched stage statistics in
        # a compact rolling window. Avoid a second periodic profiler dump; a final
        # cumulative profiler summary is still printed on shutdown.
        if name == "batched_pipeline" and bool(
            config.profiling.get("rate_diagnostics", True)
        ):
            self.interval = 0
        self.use_cuda_events = bool(config.profiling.get("cuda_events", True))
        # Runtime frames can still contain one-time CUDA/BLAS/workspace lazy-init
        # costs after the explicit model pre-warm. Execute those frames normally,
        # but keep them out of benchmark statistics.
        self.warmup_frames = max(
            0, int(config.profiling.get("warmup_frames", 0))
        )
        raw_csv = str(config.profiling.get("csv_path", ""))
        self.csv_path = Path(raw_csv) if raw_csv else None
        if self.csv_path is not None and name != "tracking":
            safe = name.replace("/", "_").replace(" ", "_")
            self.csv_path = self.csv_path.with_name(
                f"{self.csv_path.stem}_{safe}{self.csv_path.suffix}"
            )
        # Buffer CSV rows and write them in batches. Per-frame open/write/close
        # creates avoidable filesystem and scheduler traffic on the real-time path.
        self.csv_flush_interval = max(
            1, int(config.profiling.get("csv_flush_interval_frames", 100))
        )
        self._csv_buffer: list[tuple[int, dict[str, float]]] = []
        self._csv_fields: list[str] | None = None

        self._history: dict[str, list[float]] = {}
        self._history_frames: dict[str, list[int]] = {}
        self._current_cpu: dict[str, float] = {}
        self._current_cuda: dict[str, list[tuple[object, object]]] = {}
        self._frame_start = 0.0
        # _seen_frames counts every completed runtime frame. _frames counts only
        # frames admitted to statistics after warm-up.
        self._seen_frames = 0
        self._frames = 0
        self._excluded_warmup_frames = 0
        self._last_frame_excluded = False
        self._lock = threading.Lock()
        self.last_frame: dict[str, float] = {}

    def begin_frame(self) -> None:
        if not self.enabled:
            return
        self._frame_start = time.perf_counter()
        self._current_cpu = {}
        self._current_cuda = {}

    def record(self, name: str, value_ms: float) -> None:
        """Add an already measured CPU/wall-clock duration to this frame."""
        if not self.enabled:
            return
        self._current_cpu[name] = self._current_cpu.get(name, 0.0) + float(value_ms)

    @contextmanager
    def stage(self, name: str, *, cuda: bool = False) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        can_cuda = (
            cuda
            and self.use_cuda_events
            and torch is not None
            and torch.cuda.is_available()
        )
        if can_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._current_cuda.setdefault(name, []).append((start, end))
        else:
            start_t = time.perf_counter()
            try:
                yield
            finally:
                elapsed = 1000.0 * (time.perf_counter() - start_t)
                self.record(name, elapsed)

    def end_frame(self) -> dict[str, float]:
        if not self.enabled:
            return {}

        timings = dict(self._current_cpu)
        if self._current_cuda:
            # One current-stream synchronization point per frame. All stage end
            # events have already been enqueued before this fence; once it has
            # completed, elapsed_time() is valid for every recorded pair. This
            # avoids one CUDA runtime synchronization call per profiled stage.
            fence = torch.cuda.Event()
            fence.record()
            fence.synchronize()
            for name, pairs in self._current_cuda.items():
                timings[name] = sum(
                    float(start.elapsed_time(end)) for start, end in pairs
                )

        timings["pipeline_total"] = 1000.0 * (
            time.perf_counter() - self._frame_start
        )

        flush_csv = False
        print_now = False
        with self._lock:
            self._seen_frames += 1
            if self._seen_frames <= self.warmup_frames:
                # The frame was fully executed (including CUDA synchronization) so
                # caches/workspaces are genuinely warmed, but it contributes no
                # samples, CSV row, max/worst frame, or rolling pipeline timing.
                self._excluded_warmup_frames += 1
                self._last_frame_excluded = True
                self.last_frame = {}
                return {}

            self._last_frame_excluded = False
            self._frames += 1
            for key, value in timings.items():
                self._history.setdefault(key, []).append(float(value))
                # Keep the original runtime-frame number for "worst" so it stays
                # easy to correlate with logs even though warm-up was excluded.
                self._history_frames.setdefault(key, []).append(self._seen_frames)
            self.last_frame = timings
            if self.csv_path is not None:
                self._csv_buffer.append((self._seen_frames, dict(timings)))
                flush_csv = len(self._csv_buffer) >= self.csv_flush_interval
            print_now = (
                self.auto_print
                and self.interval > 0
                and self._frames % self.interval == 0
            )

        # File I/O and summary formatting are deliberately outside the state lock.
        if flush_csv:
            self._flush_csv()
        if print_now:
            self.print_summary()
        return timings

    def _flush_csv(self) -> None:
        path = self.csv_path
        if path is None:
            return
        with self._lock:
            if not self._csv_buffer:
                return
            rows = self._csv_buffer
            self._csv_buffer = []

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            exists = path.exists()
            if self._csv_fields is None:
                if exists and path.stat().st_size > 0:
                    with path.open("r", newline="", encoding="utf-8") as handle:
                        self._csv_fields = next(csv.reader(handle), None)
                if not self._csv_fields:
                    extras = sorted(
                        {
                            key
                            for _, timings in rows
                            for key in timings
                            if key not in self.PREFERRED_ORDER
                        }
                    )
                    self._csv_fields = ["frame", *self.PREFERRED_ORDER, *extras]

            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=self._csv_fields,
                    extrasaction="ignore",
                )
                if not exists or path.stat().st_size == 0:
                    writer.writeheader()
                writer.writerows(
                    {"frame": frame_number, **timings}
                    for frame_number, timings in rows
                )
        except (OSError, ValueError, StopIteration):
            # A mounted log folder can be read-only, and an existing CSV may
            # have an incompatible header. Profiling must never stop real-time.
            self.csv_path = None
            with self._lock:
                self._csv_buffer.clear()

    def close(self) -> None:
        """Flush any buffered profiler rows without changing timing statistics."""
        self._flush_csv()

    @staticmethod
    def _summary(values: list[float]) -> tuple[float, float, float, float, int]:
        array = np.asarray(values, dtype=np.float64)
        max_index = int(np.argmax(array))
        return (
            float(array.mean()),
            float(np.median(array)),
            float(np.percentile(array, 95)),
            float(array.max()),
            max_index,
        )

    def print_summary(self) -> None:
        if not self.enabled or self._frames == 0:
            return
        lines = [
            (
                f"[Profiler:{self.name}] frames={self._frames}"
                f" | warmup_excluded={self._excluded_warmup_frames}"
            ),
            "  "
            f"{'stage':<36} {'n':>6} {'mean':>8} {'median':>8} "
            f"{'p95':>8} {'max':>8} {'worst':>7}",
        ]
        keys = self.PREFERRED_ORDER + sorted(
            key for key in self._history if key not in self.PREFERRED_ORDER
        )
        for key in keys:
            values = self._history.get(key)
            if not values:
                continue
            mean, median, p95, maximum, max_index = self._summary(values)
            frames = self._history_frames.get(key, [])
            worst_frame = frames[max_index] if max_index < len(frames) else -1
            lines.append(
                "  "
                f"{key:<36} {len(values):>6d} {mean:>8.2f} {median:>8.2f} "
                f"{p95:>8.2f} {maximum:>8.2f} {worst_frame:>7d}"
            )
        print("\n".join(lines), flush=True)

    @property
    def frames(self) -> int:
        """Number of frames included in statistics."""
        return self._frames

    @property
    def seen_frames(self) -> int:
        """Number of completed runtime frames, including excluded warm-up."""
        return self._seen_frames

    @property
    def warmup_complete(self) -> bool:
        return self._seen_frames >= self.warmup_frames

    @property
    def last_frame_excluded(self) -> bool:
        return self._last_frame_excluded
