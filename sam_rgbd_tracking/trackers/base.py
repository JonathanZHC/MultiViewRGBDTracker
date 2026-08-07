from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from ..data_types import RGBDFrame, TrackerPrediction, TrackerSeed

GLOBAL_CUDA_LOCK = threading.RLock()


class MultiObjectTracker(ABC):
    @abstractmethod
    def initialize(self, frame: RGBDFrame, seeds: list[TrackerSeed]) -> TrackerPrediction:
        raise NotImplementedError

    @abstractmethod
    def track(self, frame: RGBDFrame) -> TrackerPrediction:
        raise NotImplementedError

    def correct(self, frame: RGBDFrame, seeds: list[TrackerSeed]) -> TrackerPrediction:
        return self.initialize(frame, seeds)

    def close(self) -> None:
        return None
