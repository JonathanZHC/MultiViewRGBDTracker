from __future__ import annotations

import numpy as np

from ..data_types import RGBDFrame, TrackerPrediction, TrackerSeed
from .base import MultiObjectTracker

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class MockOpticalFlowTracker(MultiObjectTracker):
    """Small deterministic backend used for CI and full-pipeline smoke tests."""

    def __init__(self) -> None:
        self.track_ids: list[int] = []
        self.masks = np.empty((0, 1, 1), dtype=bool)
        self.previous_gray: np.ndarray | None = None

    def initialize(self, frame: RGBDFrame, seeds: list[TrackerSeed]) -> TrackerPrediction:
        self.track_ids = [seed.track_id for seed in seeds]
        self.masks = (
            np.stack([np.asarray(seed.mask, dtype=bool) for seed in seeds])
            if seeds
            else np.empty((0, *frame.rgb.shape[:2]), dtype=bool)
        )
        self.previous_gray = self._gray(frame.rgb)
        return self._prediction()

    def track(self, frame: RGBDFrame) -> TrackerPrediction:
        current_gray = self._gray(frame.rgb)
        if self.previous_gray is not None and self.masks.shape[0] and cv2 is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.previous_gray,
                current_gray,
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=15,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            warped: list[np.ndarray] = []
            height, width = current_gray.shape
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            map_x = (grid_x - flow[..., 0]).astype(np.float32)
            map_y = (grid_y - flow[..., 1]).astype(np.float32)
            for mask in self.masks:
                warped_mask = cv2.remap(
                    mask.astype(np.uint8),
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                warped.append(warped_mask.astype(bool))
            self.masks = np.stack(warped)
        self.previous_gray = current_gray
        return self._prediction()

    @staticmethod
    def _gray(rgb: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return rgb.mean(axis=2).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def _prediction(self) -> TrackerPrediction:
        logits = np.where(self.masks, 8.0, -8.0).astype(np.float32)
        presence = np.array([float(mask.any()) for mask in self.masks], dtype=np.float32)
        return TrackerPrediction(self.track_ids.copy(), logits, presence, {"backend": "mock"})

    def close(self) -> None:
        self.previous_gray = None
