from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .data_types import CameraIntrinsics, DetectionInstance, TrackState


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


def blend_embedding(
    old: np.ndarray | None,
    new: np.ndarray | None,
    alpha: float = 0.25,
) -> np.ndarray | None:
    if new is None:
        return old
    if old is None or old.shape != new.shape:
        return new.copy()
    out = (1.0 - alpha) * old + alpha * new
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 0 else out


def valid_depth_mask(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """Depth validity for 3-D geometry only.

    This mask is never used to modify, arbitrate, or exclude 2-D instance masks.
    """
    return np.isfinite(depth_m) & (depth_m >= min_m) & (depth_m <= max_m)


def robust_centroid(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: CameraIntrinsics,
) -> np.ndarray | None:
    """3-D centroid used for keyframe association, not mask occlusion filtering."""
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth_m) & (depth_m > 0.0)
    ys, xs = np.nonzero(valid)
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
    """Back-project valid depth samples inside a 2-D mask.

    Invalid depth is discarded from the point cloud only. The caller's 2-D mask
    is never modified.
    """
    selected = np.asarray(mask, dtype=bool).copy()
    selected &= np.isfinite(depth_m) & (depth_m > 0.0)
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
    return (
        np.column_stack((x, y, z)).astype(np.float32),
        np.asarray(rgb[ys, xs, :3], dtype=np.uint8),
    )


def transform_points(points: np.ndarray, transform: np.ndarray | None) -> np.ndarray | None:
    if transform is None:
        return None
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    # Rigid xyz transform without constructing an Nx4 homogeneous temporary.
    # This is on the per-instance postprocess hot path.
    transform = np.asarray(transform, dtype=np.float32)
    return (
        points @ transform[:3, :3].T + transform[:3, 3][None, :]
    ).astype(np.float32, copy=False)


def bbox_3d(points: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if points is None or points.size == 0:
        return None, None
    return (
        np.quantile(points, 0.01, axis=0).astype(np.float32),
        np.quantile(points, 0.99, axis=0).astype(np.float32),
    )


@lru_cache(maxsize=16)
def _erosion_kernel(erosion_pixels: int) -> np.ndarray:
    """Cache the tiny morphology kernels instead of allocating per mask/frame."""
    radius = max(0, int(erosion_pixels))
    size = 2 * radius + 1
    return np.ones((size, size), dtype=np.uint8)


def erode_filter_and_bbox(
    mask: np.ndarray,
    erosion_pixels: int,
    min_component_pixels: int,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Pure 2-D morphology plus an already-computed 2-D bbox.

    The connected-component pass already computes component extents, so reuse
    those statistics for visualization rather than rescanning each final mask
    later in the RViz stage.  Component selection is LUT/vectorized instead of
    repeatedly evaluating ``labels == index`` in Python.
    """
    out = np.asarray(mask, dtype=np.uint8)
    if erosion_pixels > 0 and np.any(out):
        out = cv2.erode(
            out,
            _erosion_kernel(int(erosion_pixels)),
            iterations=1,
        )

    bbox: tuple[int, int, int, int] | None = None
    if min_component_pixels > 1 and np.any(out):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            out,
            connectivity=8,
        )
        if count > 1:
            component_ids = np.arange(1, count, dtype=np.int32)
            valid = component_ids[
                stats[1:, cv2.CC_STAT_AREA] >= int(min_component_pixels)
            ]
            if valid.size:
                keep_lut = np.zeros(count, dtype=bool)
                keep_lut[valid] = True
                filtered = keep_lut[labels]

                valid_stats = stats[valid]
                x0 = int(valid_stats[:, cv2.CC_STAT_LEFT].min())
                y0 = int(valid_stats[:, cv2.CC_STAT_TOP].min())
                x1 = int(
                    (
                        valid_stats[:, cv2.CC_STAT_LEFT]
                        + valid_stats[:, cv2.CC_STAT_WIDTH]
                        - 1
                    ).max()
                )
                y1 = int(
                    (
                        valid_stats[:, cv2.CC_STAT_TOP]
                        + valid_stats[:, cv2.CC_STAT_HEIGHT]
                        - 1
                    ).max()
                )
                bbox = (x0, y0, x1, y1)
                return filtered, bbox
        return np.zeros_like(out, dtype=bool), None

    out_bool = out.astype(bool, copy=False)
    if np.any(out_bool):
        x, y, width, height = cv2.boundingRect(out)
        bbox = (int(x), int(y), int(x + width - 1), int(y + height - 1))
    return out_bool, bbox


def erode_and_filter(
    mask: np.ndarray,
    erosion_pixels: int,
    min_component_pixels: int,
) -> np.ndarray:
    """Compatibility wrapper returning only the filtered mask."""
    filtered, _ = erode_filter_and_bbox(
        mask,
        erosion_pixels,
        min_component_pixels,
    )
    return filtered


def nonoverlap_owner_map(
    masks: list[np.ndarray] | np.ndarray,
    track_ids: list[int],
    height: int,
    width: int,
) -> np.ndarray:
    """Diagnostic owner map without resolving overlaps.

    Pixels covered by exactly one instance receive that track ID. Pixels covered
    by zero or multiple instance masks are left as 0. Therefore this helper does
    not perform mask exclusion or occlusion reasoning and can be replaced by the
    user's own overlap/occlusion policy later.
    """
    owner = np.zeros((height, width), dtype=np.int32)
    if not track_ids:
        return owner
    stack = np.asarray(masks, dtype=bool)
    if stack.size == 0:
        return owner
    coverage = stack.sum(axis=0)
    unique = coverage == 1
    for index, track_id in enumerate(track_ids):
        owner[stack[index] & unique] = int(track_id)
    return owner


def associate(
    detections: list[DetectionInstance],
    tracks: dict[int, TrackState],
    detection_centroids: list[np.ndarray | None],
    config,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Hungarian keyframe association with 3-D centroid, IoU, label and appearance.

    The former per-track depth-model term has been removed with the depth-based
    mask-occlusion subsystem. 3-D centroid distance is retained because it is an
    association cue, not a mask-exclusion rule.
    """
    track_ids = list(tracks)
    if not detections or not track_ids:
        return [], set(range(len(detections))), set(track_ids)

    cost = np.full((len(track_ids), len(detections)), 1e6, dtype=np.float32)
    min_iou = float(config.min_mask_iou)
    max_distance = float(config.max_centroid_distance_m)

    for row, track_id in enumerate(track_ids):
        track = tracks[track_id]
        for col, det in enumerate(detections):
            iou = mask_iou(track.last_mask, det.mask)
            centroid_distance = 0.0
            if track.centroid_camera is not None and detection_centroids[col] is not None:
                centroid_distance = float(
                    np.linalg.norm(track.centroid_camera - detection_centroids[col])
                )
                if centroid_distance > max_distance and iou < min_iou:
                    continue
            elif iou < min_iou:
                continue

            label_cost = 0.0 if track.label == det.label else float(config.label_mismatch_cost)
            cost[row, col] = (
                float(config.weight_centroid) * centroid_distance
                + float(config.weight_iou) * (1.0 - iou)
                + float(config.weight_embedding)
                * cosine_distance(track.embedding, det.embedding)
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
