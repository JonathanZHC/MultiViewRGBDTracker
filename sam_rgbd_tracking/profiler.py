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
        raw_csv = str(config.profiling.get("csv_path", ""))
        self.csv_path = Path(raw_csv) if raw_csv else None
        if self.csv_path is not None and name != "tracking":
            safe = name.replace("/", "_").replace(" ", "_")
            self.csv_path = self.csv_path.with_name(
                f"{self.csv_path.stem}_{safe}{self.csv_path.suffix}"
            )

        self._history: dict[str, list[float]] = {}
        self._history_frames: dict[str, list[int]] = {}
        self._current_cpu: dict[str, float] = {}
        self._current_cuda: dict[str, list[tuple[object, object]]] = {}
        self._frame_start = 0.0
        self._frames = 0
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
            # One synchronization point per frame. Repeated stage names are
            # accumulated, e.g. memory_attention can run more than once.
            for pairs in self._current_cuda.values():
                for _, end in pairs:
                    end.synchronize()
            for name, pairs in self._current_cuda.items():
                timings[name] = sum(
                    float(start.elapsed_time(end)) for start, end in pairs
                )

        timings["pipeline_total"] = 1000.0 * (
            time.perf_counter() - self._frame_start
        )

        with self._lock:
            self._frames += 1
            for key, value in timings.items():
                self._history.setdefault(key, []).append(float(value))
                self._history_frames.setdefault(key, []).append(self._frames)
            self.last_frame = timings
            if self.csv_path is not None:
                self._append_csv(self._frames, timings)
            if (
                self.auto_print
                and self.interval > 0
                and self._frames % self.interval == 0
            ):
                self.print_summary()
        return timings

    def _append_csv(self, frame_number: int, timings: dict[str, float]) -> None:
        path = self.csv_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        fields = ["frame", *self.PREFERRED_ORDER]
        # Keep any future/new stages instead of silently dropping them.
        fields.extend(sorted(k for k in timings if k not in fields))
        try:
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fields,
                    extrasaction="ignore",
                )
                if not exists:
                    writer.writeheader()
                writer.writerow({"frame": frame_number, **timings})
        except (OSError, ValueError):
            # A mounted log folder can be read-only, and an existing CSV may
            # have an older header. Profiling must never stop the real-time path.
            self.csv_path = None

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
            f"[Profiler:{self.name}] frames={self._frames}",
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
        return self._frames
