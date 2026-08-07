from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .appearance import cosine_distance
from .data_types import DetectionInstance, TrackState
from .geometry import mask_median_depth


@dataclass
class AssociationResult:
    matches: list[tuple[int, int]]
    unmatched_track_ids: list[int]
    unmatched_detection_indices: list[int]
    cost_matrix: np.ndarray


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def mask_centroid_3d(
    depth_m: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    z = depth_m[ys, xs]
    valid = np.isfinite(z) & (z > 0)
    if valid.sum() < 5:
        return None
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    points = np.column_stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z))
    return np.median(points, axis=0).astype(np.float32)


def associate_detections(
    tracks: dict[int, TrackState],
    detections: list[DetectionInstance],
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_centroid_distance_m: float,
    min_mask_iou: float,
    label_mismatch_cost: float,
    weight_centroid: float,
    weight_iou: float,
    weight_depth: float,
    weight_embedding: float,
) -> AssociationResult:
    track_ids = list(tracks)
    if not track_ids or not detections:
        return AssociationResult(
            matches=[],
            unmatched_track_ids=track_ids,
            unmatched_detection_indices=list(range(len(detections))),
            cost_matrix=np.empty((len(track_ids), len(detections)), dtype=np.float32),
        )

    detection_centroids = [
        mask_centroid_3d(depth_m, det.mask, fx, fy, cx, cy) for det in detections
    ]
    detection_depths = [mask_median_depth(depth_m, det.mask) for det in detections]
    cost = np.full((len(track_ids), len(detections)), 1e6, dtype=np.float32)

    for row, track_id in enumerate(track_ids):
        track = tracks[track_id]
        for col, detection in enumerate(detections):
            iou = mask_iou(track.last_mask, detection.mask)
            centroid_distance = 0.0
            if track.centroid_camera is not None and detection_centroids[col] is not None:
                centroid_distance = float(
                    np.linalg.norm(track.centroid_camera - detection_centroids[col])
                )
                if centroid_distance > max_centroid_distance_m and iou < min_mask_iou:
                    continue
            elif iou < min_mask_iou:
                continue
            depth_cost = 0.0
            if track.depth_model.initialized and detection_depths[col] is not None:
                depth_cost = abs(track.depth_model.median_m - float(detection_depths[col]))
            label_cost = 0.0 if track.label == detection.label else label_mismatch_cost
            embedding_cost = cosine_distance(track.appearance_embedding, detection.embedding)
            cost[row, col] = (
                weight_centroid * centroid_distance
                + weight_iou * (1.0 - iou)
                + weight_depth * depth_cost
                + weight_embedding * embedding_cost
                + label_cost
            )

    rows, cols = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        if cost[row, col] >= 1e5:
            continue
        matches.append((track_ids[row], col))
        matched_rows.add(row)
        matched_cols.add(col)

    return AssociationResult(
        matches=matches,
        unmatched_track_ids=[track_ids[i] for i in range(len(track_ids)) if i not in matched_rows],
        unmatched_detection_indices=[i for i in range(len(detections)) if i not in matched_cols],
        cost_matrix=cost,
    )
