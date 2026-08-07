from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_types import TrackState
from .geometry import mask_median_depth


@dataclass
class TriggerDecision:
    trigger: bool
    reasons: list[str]


class KeyframeTrigger:
    def __init__(
        self,
        refresh_seconds: float,
        target_hz: float,
        min_frames_between_triggers: int,
        area_change_threshold: float = 0.35,
        depth_jump_threshold_m: float = 0.12,
        confidence_threshold: float = 0.25,
    ) -> None:
        self.refresh_frames = max(1, round(refresh_seconds * target_hz))
        self.min_frames_between = min_frames_between_triggers
        self.area_change_threshold = area_change_threshold
        self.depth_jump_threshold_m = depth_jump_threshold_m
        self.confidence_threshold = confidence_threshold
        self.last_trigger_frame = -10**9

    def evaluate(
        self,
        frame_index: int,
        depth_m: np.ndarray,
        tracks: dict[int, TrackState],
        force: bool = False,
        phase_offset_frames: int = 0,
    ) -> TriggerDecision:
        reasons: list[str] = []
        if force or not tracks:
            reasons.append("initialization")
        if (frame_index - phase_offset_frames) % self.refresh_frames == 0:
            reasons.append("periodic")
        for track in tracks.values():
            if track.tracking_confidence < self.confidence_threshold:
                reasons.append(f"low_confidence:{track.track_id}")
            previous_area = max(int(track.last_raw_mask.sum()), 1)
            current_area = int(track.last_mask.sum())
            if abs(current_area - previous_area) / previous_area > self.area_change_threshold:
                reasons.append(f"area_jump:{track.track_id}")
            current_depth = mask_median_depth(depth_m, track.last_mask)
            if (
                current_depth is not None
                and track.depth_model.initialized
                and abs(current_depth - track.depth_model.median_m) > self.depth_jump_threshold_m
            ):
                reasons.append(f"depth_jump:{track.track_id}")
        enough_gap = frame_index - self.last_trigger_frame >= self.min_frames_between
        trigger = bool(reasons) and enough_gap
        if trigger:
            self.last_trigger_frame = frame_index
        return TriggerDecision(trigger=trigger, reasons=reasons if trigger else [])
