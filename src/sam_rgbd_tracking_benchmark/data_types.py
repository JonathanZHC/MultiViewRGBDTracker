from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


@dataclass
class RGBDFrame:
    camera_name: str
    frame_index: int
    stamp_ns: int
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    world_from_camera: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    gt_instance_map: np.ndarray | None = None
    gt_metadata: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class DetectionInstance:
    detection_id: int
    label: str
    score: float
    mask: np.ndarray
    bbox_xyxy: tuple[float, float, float, float] | None = None
    embedding: np.ndarray | None = None
    centroid_camera: np.ndarray | None = None
    median_depth_m: float | None = None


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
    backend_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthModel:
    initialized: bool = False
    median_m: float = 0.0
    mad_m: float = 0.0
    q05_m: float = 0.0
    q95_m: float = 0.0
    valid_pixels: int = 0


class VisibilityState(StrEnum):
    VISIBLE = "visible"
    PARTIAL = "partial"
    OCCLUDED = "occluded"
    LOST = "lost"


@dataclass
class TrackState:
    track_id: int
    label: str
    semantic_confidence: float
    last_mask: np.ndarray
    last_raw_mask: np.ndarray
    depth_model: DepthModel = field(default_factory=DepthModel)
    centroid_camera: np.ndarray | None = None
    centroid_world: np.ndarray | None = None
    bbox_3d_min: np.ndarray | None = None
    bbox_3d_max: np.ndarray | None = None
    visible_ratio: float = 1.0
    tracking_confidence: float = 1.0
    motion_prediction_confidence: float = 1.0
    status: VisibilityState = VisibilityState.VISIBLE
    last_seen_frame: int = 0
    missing_frames: int = 0
    appearance_embedding: np.ndarray | None = None


@dataclass
class ProcessedInstance:
    track_id: int
    label: str
    semantic_confidence: float
    tracking_confidence: float
    motion_prediction_confidence: float
    raw_mask: np.ndarray
    exclusive_mask: np.ndarray
    depth_filtered_mask: np.ndarray
    depth_rejected_mask: np.ndarray
    points_camera: np.ndarray
    points_world: np.ndarray
    colors_rgb: np.ndarray
    centroid_camera: np.ndarray | None
    centroid_world: np.ndarray | None
    bbox_3d_min: np.ndarray | None
    bbox_3d_max: np.ndarray | None
    visible_ratio: float
    depth_consistency: float
    status: VisibilityState


@dataclass
class FrameResult:
    frame: RGBDFrame
    instances: list[ProcessedInstance]
    owner_track_map: np.ndarray
    keyframe: bool
    anomaly_triggered: bool
    timings_ms: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
