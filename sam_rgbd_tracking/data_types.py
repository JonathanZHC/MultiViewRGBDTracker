from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class RGBDFrame:
    camera_name: str
    frame_index: int
    timestamp_ns: int
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    world_from_camera: np.ndarray | None = None


@dataclass
class DetectionInstance:
    detection_id: int
    label: str
    score: float
    mask: np.ndarray
    bbox_xyxy: tuple[float, float, float, float] | None = None
    embedding: np.ndarray | None = None


@dataclass
class TrackerSeed:
    track_id: int
    mask: np.ndarray
    label: str = "object"
    confidence: float = 1.0


@dataclass
class TrackerPrediction:
    track_ids: list[int]
    mask_logits: np.ndarray
    presence_scores: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthModel:
    median_m: float
    mad_m: float


class VisibilityState(str, Enum):
    VISIBLE = "visible"
    PARTIAL = "partial"
    OCCLUDED = "occluded"
    LOST = "lost"


@dataclass
class TrackState:
    track_id: int
    label: str
    semantic_confidence: float
    embedding: np.ndarray | None = None
    last_mask: np.ndarray | None = None
    last_raw_mask: np.ndarray | None = None
    depth_model: DepthModel | None = None
    centroid_camera: np.ndarray | None = None
    centroid_world: np.ndarray | None = None
    tracking_confidence: float = 1.0
    visible_ratio: float = 1.0
    missing_frames: int = 0
    last_seen_frame: int = 0


@dataclass
class ProcessedInstance:
    track_id: int
    label: str
    semantic_confidence: float
    tracking_confidence: float
    motion_prediction_confidence: float
    raw_mask: np.ndarray
    mask: np.ndarray
    points_camera: np.ndarray
    points_world: np.ndarray | None
    colors_rgb: np.ndarray
    centroid_camera: np.ndarray | None
    centroid_world: np.ndarray | None
    bbox_min: np.ndarray | None
    bbox_max: np.ndarray | None
    visible_ratio: float
    depth_consistency: float
    status: VisibilityState


@dataclass
class FrameResult:
    frame: RGBDFrame
    instances: list[ProcessedInstance]
    owner_track_map: np.ndarray
    keyframe: bool
    timings_ms: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
