from __future__ import annotations

from abc import ABC, abstractmethod

from ..data_types import RGBDFrame, TrackerPrediction, TrackerSeed


class MultiObjectTracker(ABC):
    """Common interface used by both real backends and the mock backend."""

    @abstractmethod
    def initialize(self, frame: RGBDFrame, seeds: list[TrackerSeed]) -> TrackerPrediction:
        raise NotImplementedError

    @abstractmethod
    def track(self, frame: RGBDFrame) -> TrackerPrediction:
        raise NotImplementedError

    def correct(self, frame: RGBDFrame, seeds: list[TrackerSeed]) -> TrackerPrediction:
        """Reset the short streaming window from a fresh keyframe.

        Reinitializing every SAM3 keyframe bounds video-frame memory and makes
        the A/B comparison deterministic. Track IDs remain unchanged.
        """
        return self.initialize(frame, seeds)

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
