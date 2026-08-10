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
    mask_logits: Any
    presence_scores: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class VisibilityState(str, Enum):
    """Tracking-level state only.

    Depth-based partial/occluded classification was intentionally removed.
    """

    VISIBLE = "visible"
    LOST = "lost"


@dataclass
class TrackState:
    track_id: int
    label: str
    semantic_confidence: float
    tracker_slot: int = -1
    class_slot: int = -1
    active: bool = True
    embedding: np.ndarray | None = None
    last_mask: np.ndarray | None = None
    last_raw_mask: np.ndarray | None = None
    centroid_camera: np.ndarray | None = None
    centroid_world: np.ndarray | None = None
    tracking_confidence: float = 1.0
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
    bbox_2d_xyxy: tuple[int, int, int, int] | None
    status: VisibilityState
    tracker_slot: int = -1
    class_slot: int = -1
    multiview_group_id: int | None = None
    global_track_id: int | None = None
    # Optional shared-world voxel cache.  It is produced once during batched
    # postprocess and reused by cross-view alignment/fusion, avoiding a second
    # quantize+unique pass over the same point cloud.
    voxel_coords: np.ndarray | None = None
    voxel_keys: np.ndarray | None = None
    voxel_points: np.ndarray | None = None
    voxel_colors: np.ndarray | None = None
    voxel_bbox_min: np.ndarray | None = None
    voxel_bbox_max: np.ndarray | None = None


@dataclass
class MultiViewInstance:
    group_id: int
    semantic_label: str
    members: list[tuple[str, ProcessedInstance]]
    points_world: np.ndarray
    colors_rgb: np.ndarray
    centroid_world: np.ndarray | None
    bbox_min: np.ndarray | None
    bbox_max: np.ndarray | None
    global_track_id: int | None = None


@dataclass
class FrameResult:
    frame: RGBDFrame
    instances: list[ProcessedInstance]
    owner_track_map: np.ndarray
    keyframe: bool
    timings_ms: dict[str, float]
    # Visualization-only compact uint8 instance-code rasters (0=background,
    # 1..N=current processed-instance order).  True track-ID ownership remains in
    # owner_track_map.
    raw_instance_map: np.ndarray | None = None
    filtered_instance_map: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
