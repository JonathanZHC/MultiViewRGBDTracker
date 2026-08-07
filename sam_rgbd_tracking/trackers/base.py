from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

from ..data_types import RGBDFrame, TrackerPrediction, TrackerSeed

# SAM3 and all tracker backends share this lock when runtime.serialize_gpu=True.
# This keeps the two camera workers from interleaving stateful CUDA-graph calls.
GLOBAL_CUDA_LOCK = threading.RLock()

# The EfficientTAM memory-attention clone wrapper lives on the shared predictor,
# while each camera owns a different profiler. A thread-local pointer lets that
# wrapper attribute clone time to the camera that is currently executing.
_TRACKER_PROFILE_CONTEXT = threading.local()


def current_tracker_profiler() -> Any | None:
    return getattr(_TRACKER_PROFILE_CONTEXT, "profiler", None)


@contextmanager
def tracker_profile_context(profiler: Any | None) -> Iterator[None]:
    previous = getattr(_TRACKER_PROFILE_CONTEXT, "profiler", None)
    _TRACKER_PROFILE_CONTEXT.profiler = profiler
    try:
        yield
    finally:
        _TRACKER_PROFILE_CONTEXT.profiler = previous


class MultiObjectTracker(ABC):
    """Common tracker interface plus optional profiler attachment."""

    def set_profiler(self, profiler: Any | None) -> None:
        self._profiler = profiler

    @property
    def profiler(self) -> Any | None:
        return getattr(self, "_profiler", None)

    def profile_stage(self, name: str, *, cuda: bool = False):
        profiler = self.profiler
        if profiler is None:
            return nullcontext()
        return profiler.stage(name, cuda=cuda)

    def record_profile(self, name: str, value_ms: float) -> None:
        profiler = self.profiler
        if profiler is not None:
            profiler.record(name, value_ms)

    @abstractmethod
    def initialize(
        self,
        frame: RGBDFrame,
        seeds: list[TrackerSeed],
    ) -> TrackerPrediction:
        raise NotImplementedError

    @abstractmethod
    def track(self, frame: RGBDFrame) -> TrackerPrediction:
        raise NotImplementedError

    def correct(
        self,
        frame: RGBDFrame,
        seeds: list[TrackerSeed],
    ) -> TrackerPrediction:
        return self.initialize(frame, seeds)

    def close(self) -> None:
        return None
