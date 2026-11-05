from __future__ import annotations

import threading
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
    yield_wait_ms: float = 0.0          # cooperative: time spent waiting for the latency-critical loop to go idle
    yield_timeouts: int = 0             # cooperative: waits that hit cooperative_max_wait_s
    detections_per_class: list[dict[str, int]] | None = None


class AsyncSAM3Worker:
    """Single outstanding SAM3 B=views job on a dedicated CUDA stream/thread.

    detector.cooperative_scheduling (default false): an asynchronous job yields the GPU to the latency-critical
    loop. Before every vision-trunk block, fusion encoder/decoder layer, neck conv and the segmentation head it
    waits until its previously queued kernels finished and the loop is idle (set_idle()/set_busy(), called by the
    owner around each cycle). A job then overlaps a cycle by at most one module (<= ~3 ms; ~7 ms for the head)
    instead of its whole duration (~0.1 s for two views on an RTX 5090), at the price of a longer job (spread over the
    idle gaps of several frames). The blocking initial detection never yields. detector.cooperative_max_wait_s bounds
    each wait. The owner MUST call set_busy()/set_idle() (ScenePredictor's pipeline does; a caller that never does
    leaves the job unthrottled apart from one stream sync per module).
    """

    # Module lists whose elements get a yield point (names in the SAM3 image model); the text encoder runs once.
    _YIELD_MODULE_LISTS = (
        "backbone.vision_backbone.trunk.blocks",
        "backbone.vision_backbone.convs",
        "geometry_encoder.encode",
        "transformer.encoder.layers",
        "transformer.decoder.layers",
        "segmentation_head.pixel_decoder.conv_layers",
    )
    # Single modules with a yield point. Their work grows with the (view, prompt) rows of one grounding pass, so
    # with several prompts one unhooked stretch (geometry encoder, head internals) reached ~15 ms of GPU work.
    _YIELD_MODULES = (
        "geometry_encoder",
        "segmentation_head",
        "segmentation_head.pixel_decoder",
        "segmentation_head.mask_predictor",
        "dot_prod_scoring",
    )

    def __init__(self, config) -> None:
        self.config = config
        self._cooperative = bool(config.detector.get("cooperative_scheduling", False))
        self._max_wait_s = float(config.detector.get("cooperative_max_wait_s", 0.25))
        self._idle = threading.Event()
        self._idle.set()
        self._yielding = False             # True only while an asynchronous job runs (executor thread)
        self._wait_s = 0.0
        self._timeouts = 0
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
        if self._cooperative:
            self._install_yield_hooks(detector, stream)
        return detector, stream

    def _install_yield_hooks(self, detector, stream) -> None:
        """Forward pre-hooks at the SAM3 module boundaries (vision trunk blocks, neck convs, fusion encoder and
        decoder layers, segmentation head); the text encoder runs once per process and is not hooked."""

        def wait_for_idle(_module, _args) -> None:
            if not self._yielding:
                return
            if stream is not None:
                stream.synchronize()       # bound the queued SAM3 work to the module about to be issued
            started = time.perf_counter()
            if not self._idle.wait(timeout=self._max_wait_s):
                self._timeouts += 1
            self._wait_s += time.perf_counter() - started

        modules = dict(detector.model.named_modules())
        hooked = 0
        for name in self._YIELD_MODULE_LISTS:
            module_list = modules.get(name)
            if module_list is None:
                print(f"[SAM3] cooperative scheduling: module list {name!r} not found, no yield points there", flush=True)
                continue
            for block in module_list:
                block.register_forward_pre_hook(wait_for_idle)
                hooked += 1
        for name in self._YIELD_MODULES:
            module = modules.get(name)
            if module is None:
                print(f"[SAM3] cooperative scheduling: module {name!r} not found, no yield point", flush=True)
                continue
            module.register_forward_pre_hook(wait_for_idle)
            hooked += 1
        # The detector's per-(view, prompt) mask post-processing (upsampling to full resolution) yields as well.
        detector.yield_point = lambda: wait_for_idle(None, None)
        print(f"[SAM3] cooperative scheduling: {hooked} module yield points + post-processing", flush=True)

    def set_busy(self) -> None:
        """The latency-critical loop starts a cycle: cooperative jobs pause at their next module boundary."""
        self._idle.clear()

    def set_idle(self) -> None:
        """The latency-critical loop finished its cycle: cooperative jobs may issue GPU work."""
        self._idle.set()

    @property
    def busy(self) -> bool:
        return self._future is not None

    def _infer(
        self,
        frame_index: int,
        reference_frames: list[RGBDFrame],
        fallback_masks_per_view: list[dict[int, np.ndarray]],
        cooperative: bool = False,
    ) -> SAM3BatchResult:
        started = time.perf_counter()
        self._yielding = bool(cooperative and self._cooperative)
        self._wait_s = 0.0
        self._timeouts = 0
        stream_context = (
            torch.cuda.stream(self._stream)
            if torch is not None and self._stream is not None
            else nullcontext()
        )
        try:
            with stream_context:
                detections = self._detector.detect_batch(reference_frames)
            if self._stream is not None:
                self._stream.synchronize()
        finally:
            self._yielding = False
        return SAM3BatchResult(
            frame_index=int(frame_index),
            reference_frames=reference_frames,
            detections_per_view=detections,
            fallback_masks_per_view=fallback_masks_per_view,
            wall_ms=1000.0 * (time.perf_counter() - started),
            filter_cpu_ms=float(self._detector.last_filter_ms),
            detections_per_class=list(self._detector.last_counts_per_view),
            yield_wait_ms=1000.0 * self._wait_s,
            yield_timeouts=int(self._timeouts),
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
            cooperative=True,
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
