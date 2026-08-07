from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class FrameProfiler:
    """CPU/CUDA profiler that synchronizes once per frame, not per stage."""

    def __init__(
        self,
        csv_path: str,
        jsonl_path: str,
        use_cuda_events: bool = True,
        summary_interval: int = 100,
        history_size: int = 10000,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        # One profiler instance represents one benchmark run. Avoid silently
        # mixing frames from an older process that used the same log directory.
        for path in (self.csv_path, self.jsonl_path):
            if path.exists():
                path.unlink()
        self.use_cuda = bool(
            use_cuda_events and torch is not None and torch.cuda.is_available()
        )
        self.summary_interval = summary_interval
        self.frame_data: dict[str, float] = {}
        self._cpu_start: dict[str, float] = {}
        self._cuda_events: dict[str, tuple[object, object]] = {}
        self.history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.frame_count = 0
        self._csv_header: list[str] = []
        self._csv_rows: list[dict[str, object]] = []
        self.frame_context: dict[str, object] = {}

    def begin_frame(self, **context: object) -> None:
        self.frame_data = {}
        self._cpu_start = {}
        self._cuda_events = {}
        self.frame_context = context
        self._frame_wall_start = time.perf_counter()

    @contextmanager
    def stage(self, name: str, cuda: bool = False) -> Iterator[None]:
        start = time.perf_counter()
        start_event = end_event = None
        if cuda and self.use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        try:
            yield
        finally:
            if end_event is not None:
                end_event.record()
                self._cuda_events[name] = (start_event, end_event)
            self.frame_data[f"{name}_cpu"] = (time.perf_counter() - start) * 1000.0

    def end_frame(self, **extra: object) -> dict[str, float]:
        if self.use_cuda and self._cuda_events:
            torch.cuda.synchronize()
            for name, (start_event, end_event) in self._cuda_events.items():
                self.frame_data[f"{name}_gpu"] = float(start_event.elapsed_time(end_event))
        self.frame_data["pipeline_total"] = (
            time.perf_counter() - self._frame_wall_start
        ) * 1000.0
        for key, value in self.frame_data.items():
            self.history[key].append(float(value))
        self.frame_count += 1
        record: dict[str, object] = {
            **self.frame_context,
            **extra,
            **self.frame_data,
        }
        if torch is not None and torch.cuda.is_available():
            record.update(
                gpu_allocated_mb=torch.cuda.memory_allocated() / 2**20,
                gpu_reserved_mb=torch.cuda.memory_reserved() / 2**20,
                gpu_peak_mb=torch.cuda.max_memory_allocated() / 2**20,
            )
        self._write_jsonl(record)
        self._write_csv(record)
        if self.summary_interval > 0 and self.frame_count % self.summary_interval == 0:
            print(self.format_summary())
        return dict(self.frame_data)

    def _write_jsonl(self, record: dict[str, object]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _write_csv(self, record: dict[str, object]) -> None:
        flattened = {
            key: value
            for key, value in record.items()
            if not isinstance(value, (dict, list))
        }
        self._csv_rows.append(flattened)
        new_fields = [key for key in flattened if key not in self._csv_header]
        if new_fields:
            self._csv_header.extend(new_fields)
            # A keyframe and a tracking-only frame expose different stage names.
            # Rewrite when the schema expands so the CSV never silently drops a
            # timing column that appears after the first frame.
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._csv_header)
                writer.writeheader()
                for row in self._csv_rows:
                    writer.writerow({key: row.get(key, "") for key in self._csv_header})
            return
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_header)
            writer.writerow({key: flattened.get(key, "") for key in self._csv_header})

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = (len(ordered) - 1) * quantile
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        alpha = index - lower
        return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha

    def summary(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for name, values_deque in self.history.items():
            values = list(values_deque)
            if not values:
                continue
            result[name] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p95": self._percentile(values, 0.95),
                "p99": self._percentile(values, 0.99),
                "max": max(values),
            }
        return result

    def format_summary(self) -> str:
        lines = [f"[Profiler] frames={self.frame_count}"]
        for name, stats in sorted(self.summary().items()):
            if name in {"pipeline_total", "tracker_total_gpu", "sam3_total_gpu", "postprocess_cpu"}:
                lines.append(
                    f"  {name}: mean={stats['mean']:.2f} ms, "
                    f"p95={stats['p95']:.2f} ms, max={stats['max']:.2f} ms"
                )
        return "\n".join(lines)
