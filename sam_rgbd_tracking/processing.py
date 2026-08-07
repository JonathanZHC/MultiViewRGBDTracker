from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .data_types import CameraIntrinsics, DepthModel, DetectionInstance, TrackState


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    a = a.astype(bool, copy=False)
    b = b.astype(bool, copy=False)
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def color_embedding(rgb: np.ndarray, mask: np.ndarray, bins: int = 8) -> np.ndarray | None:
    """Compact joint RGB histogram for keyframe association."""
    pixels = np.asarray(rgb, dtype=np.uint8)[np.asarray(mask, dtype=bool)]
    if pixels.shape[0] < 8:
        return None
    bins = max(2, int(bins))
    quantized = np.minimum((pixels.astype(np.int32) * bins) // 256, bins - 1)
    indices = quantized[:, 0] * bins * bins + quantized[:, 1] * bins + quantized[:, 2]
    histogram = np.bincount(indices, minlength=bins**3).astype(np.float32)
    norm = float(np.linalg.norm(histogram))
    return histogram / norm if norm > 1e-12 else None

def cosine_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape or a.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def blend_embedding(old: np.ndarray | None, new: np.ndarray | None, alpha: float = 0.25) -> np.ndarray | None:
    if new is None:
        return old
    if old is None or old.shape != new.shape:
        return new.copy()
    out = (1.0 - alpha) * old + alpha * new
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 0 else out


def valid_depth_mask(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    return np.isfinite(depth_m) & (depth_m >= min_m) & (depth_m <= max_m)


def depth_edge_validity(depth_m: np.ndarray, threshold_m: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if threshold_m <= 0:
        return valid
    padded_x = np.pad(depth, ((0, 0), (1, 1)), mode="edge")
    padded_y = np.pad(depth, ((1, 1), (0, 0)), mode="edge")
    dx = np.abs(padded_x[:, 2:] - padded_x[:, :-2]) * 0.5
    dy = np.abs(padded_y[2:, :] - padded_y[:-2, :]) * 0.5
    return valid & ~((dx > threshold_m) | (dy > threshold_m))

def estimate_depth_model(depth_m: np.ndarray, mask: np.ndarray, min_pixels: int) -> DepthModel | None:
    values = depth_m[mask & np.isfinite(depth_m)]
    if values.size < min_pixels:
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return DepthModel(median_m=median, mad_m=max(mad, 1e-4))


def blend_depth_model(old: DepthModel | None, new: DepthModel | None, alpha: float = 0.2) -> DepthModel | None:
    if new is None:
        return old
    if old is None:
        return new
    return DepthModel(
        median_m=(1.0 - alpha) * old.median_m + alpha * new.median_m,
        mad_m=(1.0 - alpha) * old.mad_m + alpha * new.mad_m,
    )


def robust_centroid(depth_m: np.ndarray, mask: np.ndarray, K: CameraIntrinsics) -> np.ndarray | None:
    ys, xs = np.nonzero(mask & np.isfinite(depth_m))
    if xs.size == 0:
        return None
    z = depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - K.cx) * z / K.fx
    y = (ys.astype(np.float32) - K.cy) * z / K.fy
    return np.median(np.stack((x, y, z), axis=1), axis=0).astype(np.float32)


def backproject_mask(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    K: CameraIntrinsics,
    stride: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(mask, dtype=bool).copy()
    if stride > 1:
        sampling = np.zeros_like(selected)
        sampling[::stride, ::stride] = True
        selected &= sampling
    ys, xs = np.nonzero(selected)
    if max_points > 0 and xs.size > max_points:
        take = np.linspace(0, xs.size - 1, max_points, dtype=np.int64)
        ys, xs = ys[take], xs[take]
    if xs.size == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    z = depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - K.cx) * z / K.fx
    y = (ys.astype(np.float32) - K.cy) * z / K.fy
    return np.column_stack((x, y, z)).astype(np.float32), np.asarray(rgb[ys, xs, :3], dtype=np.uint8)

def transform_points(points: np.ndarray, transform: np.ndarray | None) -> np.ndarray | None:
    if transform is None:
        return None
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    hom = np.concatenate((points, np.ones((points.shape[0], 1), dtype=np.float32)), axis=1)
    return (hom @ transform.T)[:, :3].astype(np.float32)


def bbox_3d(points: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if points is None or points.size == 0:
        return None, None
    return (
        np.quantile(points, 0.01, axis=0).astype(np.float32),
        np.quantile(points, 0.99, axis=0).astype(np.float32),
    )


def erode_and_filter(mask: np.ndarray, erosion_pixels: int, min_component_pixels: int) -> np.ndarray:
    out = mask.astype(np.uint8)
    if erosion_pixels > 0 and out.any():
        size = 2 * erosion_pixels + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        out = cv2.erode(out, kernel, iterations=1)
    if min_component_pixels > 1 and out.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        keep = np.zeros_like(out, dtype=np.uint8)
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) >= min_component_pixels:
                keep[labels == index] = 1
        out = keep
    return out.astype(bool)


def exclusive_depth_masks(
    depth_m: np.ndarray,
    logits: np.ndarray,
    track_ids: list[int],
    tracks: dict[int, TrackState],
    valid_depth: np.ndarray,
    threshold: float,
    mad_scale: float,
    min_gate_m: float,
    max_gate_m: float,
    *,
    overlap_depth_only: bool = True,
    logit_weight: float = 0.15,
) -> tuple[list[np.ndarray], list[float]]:
    """Depth-aware ownership: same runtime rule as the full benchmark, consolidated here."""
    if logits.size == 0:
        return [], []
    n, h, w = logits.shape
    candidates = logits > threshold
    candidate_count = candidates.sum(axis=0)
    any_candidate = candidate_count > 0
    costs = np.full((n, h, w), np.inf, dtype=np.float32)
    depth_valid = np.zeros((n, h, w), dtype=bool)
    for idx, track_id in enumerate(track_ids):
        model = tracks[track_id].depth_model
        candidate = candidates[idx] & valid_depth
        if model is None:
            depth_cost = np.zeros((h, w), dtype=np.float32)
            depth_ok = valid_depth.copy()
        else:
            gate = float(np.clip(mad_scale * 1.4826 * max(model.mad_m, 1e-5), min_gate_m, max_gate_m))
            error = np.abs(depth_m - model.median_m)
            depth_cost = error / max(gate, 1e-6)
            depth_ok = error <= gate
        depth_valid[idx] = candidate & depth_ok
        costs[idx] = np.where(candidate, depth_cost - logit_weight * logits[idx], np.inf)
    owner = np.argmin(costs, axis=0).astype(np.int32)
    owner[(~np.isfinite(np.min(costs, axis=0))) | (~any_candidate)] = -1
    if overlap_depth_only:
        sole = candidate_count == 1
        sole_owner = np.argmax(candidates, axis=0).astype(np.int32)
        owner[sole] = sole_owner[sole]
    exclusive = np.stack([owner == idx for idx in range(n)])
    filtered = exclusive & depth_valid
    consistency = []
    for idx in range(n):
        denom = int(exclusive[idx].sum())
        consistency.append(float(filtered[idx].sum() / denom) if denom else 0.0)
    return [filtered[idx] for idx in range(n)], consistency

def associate(
    detections: list[DetectionInstance],
    tracks: dict[int, TrackState],
    detection_centroids: list[np.ndarray | None],
    config,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Hungarian keyframe association with 3-D, IoU, depth, label and appearance evidence."""
    track_ids = list(tracks)
    if not detections or not track_ids:
        return [], set(range(len(detections))), set(track_ids)
    detection_depths = []
    for centroid in detection_centroids:
        detection_depths.append(None if centroid is None else float(centroid[2]))
    cost = np.full((len(track_ids), len(detections)), 1e6, dtype=np.float32)
    min_iou = float(config.min_mask_iou)
    max_distance = float(config.max_centroid_distance_m)
    for row, track_id in enumerate(track_ids):
        track = tracks[track_id]
        for col, det in enumerate(detections):
            iou = mask_iou(track.last_mask, det.mask)
            centroid_distance = 0.0
            if track.centroid_camera is not None and detection_centroids[col] is not None:
                centroid_distance = float(np.linalg.norm(track.centroid_camera - detection_centroids[col]))
                if centroid_distance > max_distance and iou < min_iou:
                    continue
            elif iou < min_iou:
                continue
            depth_cost = 0.0
            if track.depth_model is not None and detection_depths[col] is not None:
                depth_cost = abs(track.depth_model.median_m - detection_depths[col])
            label_cost = 0.0 if track.label == det.label else float(config.label_mismatch_cost)
            cost[row, col] = (
                float(config.weight_centroid) * centroid_distance
                + float(config.weight_iou) * (1.0 - iou)
                + float(config.weight_depth) * depth_cost
                + float(config.weight_embedding) * cosine_distance(track.embedding, det.embedding)
                + label_cost
            )
    rows, cols = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    matched_d: set[int] = set()
    matched_t: set[int] = set()
    for row, col in zip(rows.tolist(), cols.tolist()):
        if cost[row, col] >= 1e5:
            continue
        track_id = track_ids[row]
        matches.append((col, track_id))
        matched_d.add(col)
        matched_t.add(track_id)
    return matches, set(range(len(detections))) - matched_d, set(track_ids) - matched_t

