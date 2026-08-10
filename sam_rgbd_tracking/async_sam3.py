from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data_types import DetectionInstance, RGBDFrame
from .detector import Sam3Detector

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class SAM3BatchResult:
    frame_index: int
    reference_frames: list[RGBDFrame]
    detections_per_view: list[list[DetectionInstance]]
    fallback_masks_per_view: list[dict[int, np.ndarray]]
    wall_ms: float
    filter_cpu_ms: float = 0.0
    detections_per_class: list[dict[str, int]] | None = None


class AsyncSAM3Worker:
    """Single outstanding SAM3 B=views job on a dedicated CUDA stream/thread."""

    def __init__(self, config) -> None:
        self.config = config
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="sam3-async",
        )
        self._detector, self._stream = self._executor.submit(
            self._build_runtime
        ).result()
        self._future: Future[SAM3BatchResult] | None = None

    def _build_runtime(self) -> tuple[Sam3Detector, Any | None]:
        detector = Sam3Detector(self.config)
        stream = None
        if (
            torch is not None
            and torch.cuda.is_available()
            and str(self.config.runtime.device).startswith("cuda")
        ):
            stream = torch.cuda.Stream(device=torch.device(self.config.runtime.device))
        return detector, stream

    @property
    def busy(self) -> bool:
        return self._future is not None

    def _infer(
        self,
        frame_index: int,
        reference_frames: list[RGBDFrame],
        fallback_masks_per_view: list[dict[int, np.ndarray]],
    ) -> SAM3BatchResult:
        started = time.perf_counter()
        stream_context = (
            torch.cuda.stream(self._stream)
            if torch is not None and self._stream is not None
            else nullcontext()
        )
        with stream_context:
            detections = self._detector.detect_batch(reference_frames)
        if self._stream is not None:
            self._stream.synchronize()
        return SAM3BatchResult(
            frame_index=int(frame_index),
            reference_frames=reference_frames,
            detections_per_view=detections,
            fallback_masks_per_view=fallback_masks_per_view,
            wall_ms=1000.0 * (time.perf_counter() - started),
            filter_cpu_ms=float(self._detector.last_filter_ms),
            detections_per_class=list(self._detector.last_counts_per_view),
        )

    def run_blocking(
        self,
        frame_index: int,
        reference_frames: list[RGBDFrame],
        fallback_masks_per_view: list[dict[int, np.ndarray]] | None = None,
    ) -> SAM3BatchResult:
        if self._future is not None:
            raise RuntimeError("Cannot run blocking SAM3 while an async job exists")
        fallback = fallback_masks_per_view or [
            {} for _ in reference_frames
        ]
        return self._executor.submit(
            self._infer,
            int(frame_index),
            reference_frames,
            fallback,
        ).result()

    def submit(
        self,
        frame_index: int,
        reference_frames: list[RGBDFrame],
        fallback_masks_per_view: list[dict[int, np.ndarray]],
    ) -> bool:
        """Submit one job; return False rather than building a backlog."""
        if self._future is not None:
            return False
        self._future = self._executor.submit(
            self._infer,
            int(frame_index),
            reference_frames,
            fallback_masks_per_view,
        )
        return True

    def poll(self) -> SAM3BatchResult | None:
        future = self._future
        if future is None or not future.done():
            return None
        self._future = None
        return future.result()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
