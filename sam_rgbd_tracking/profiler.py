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
    """Low-overhead cycle profiler.

    GPU stages use CUDA events and are synchronized once at frame end. The
    printed names intentionally stay stable because they are convenient for
    comparing SAM3, tracker, CPU postprocess and end-to-end cycle time.
    """

    def __init__(self, config, name: str = "tracking") -> None:
        self.name = name
        self.enabled = bool(config.profiling.get("enabled", True))
        self.interval = int(config.profiling.get("summary_interval_frames", 100))
        self.use_cuda_events = bool(config.profiling.get("cuda_events", True))
        raw_csv = str(config.profiling.get("csv_path", ""))
        self.csv_path = Path(raw_csv) if raw_csv else None
        if self.csv_path is not None and name != "tracking":
            safe = name.replace("/", "_").replace(" ", "_")
            self.csv_path = self.csv_path.with_name(f"{self.csv_path.stem}_{safe}{self.csv_path.suffix}")
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
                self._current_cpu[name] = self._current_cpu.get(name, 0.0) + elapsed

    def end_frame(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        timings = dict(self._current_cpu)
        if self._current_cuda:
            # Synchronize only at frame end; repeated stages in the same frame are
            # accumulated (e.g. normal tracking followed by anomaly correction).
            for pairs in self._current_cuda.values():
                for _, end in pairs:
                    end.synchronize()
            for name, pairs in self._current_cuda.items():
                timings[name] = sum(float(start.elapsed_time(end)) for start, end in pairs)
        timings["pipeline_total"] = 1000.0 * (time.perf_counter() - self._frame_start)
        with self._lock:
            self._frames += 1
            for key, value in timings.items():
                self._history.setdefault(key, []).append(float(value))
                self._history_frames.setdefault(key, []).append(self._frames)
            self.last_frame = timings
            if self.csv_path is not None:
                self._append_csv(self._frames, timings)
            if self.interval > 0 and self._frames % self.interval == 0:
                self.print_summary()
        return timings

    def _append_csv(self, frame_number: int, timings: dict[str, float]) -> None:
        path = self.csv_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        fields = [
            "frame",
            "pipeline_total",
            "postprocess_cpu",
            "sam3_total_gpu",
            "tracker_total_gpu",
        ]
        try:
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                row = {"frame": frame_number, **timings}
                writer.writerow(row)
        except OSError:
            # Profiling must never stop the real-time path because a mounted log
            # directory is read-only. Console statistics remain available.
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
        if not self.enabled:
            return
        print(f"[Profiler:{self.name}] frames={self._frames}", flush=True)
        preferred = ["pipeline_total", "postprocess_cpu", "sam3_total_gpu", "tracker_total_gpu"]
        keys = preferred + sorted(k for k in self._history if k not in preferred)
        for key in keys:
            values = self._history.get(key)
            if not values:
                continue
            mean, median, p95, maximum, max_index = self._summary(values)
            frames = self._history_frames.get(key, [])
            worst_frame = frames[max_index] if max_index < len(frames) else -1
            print(
                f"  {key}: n={len(values)}, mean={mean:.2f} ms, "
                f"median={median:.2f} ms, p95={p95:.2f} ms, "
                f"max={maximum:.2f} ms (frame={worst_frame})",
                flush=True,
            )

    @property
    def frames(self) -> int:
        return self._frames
