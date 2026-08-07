from __future__ import annotations

import numpy as np

from .data_types import CameraIntrinsics


def valid_depth_mask(depth_m: np.ndarray, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    depth = np.asarray(depth_m)
    return np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)


def backproject_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    rgb: np.ndarray | None = None,
    stride: int = 1,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(mask, dtype=bool).copy()
    if stride > 1:
        sampling = np.zeros_like(selected)
        sampling[::stride, ::stride] = True
        selected &= sampling
    ys, xs = np.nonzero(selected)
    if ys.size == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    if max_points and ys.size > max_points:
        indices = np.linspace(0, ys.size - 1, max_points, dtype=np.int64)
        ys, xs = ys[indices], xs[indices]
    z = depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - intrinsics.cx) * z / intrinsics.fx
    y = (ys.astype(np.float32) - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x, y, z)).astype(np.float32)
    if rgb is None:
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)
    else:
        colors = np.asarray(rgb[ys, xs, :3], dtype=np.uint8)
    return points, colors


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3).astype(np.float32)
    rotation = transform[:3, :3].astype(np.float32)
    translation = transform[:3, 3].astype(np.float32)
    return points @ rotation.T + translation


def robust_centroid(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] == 0:
        return None
    return np.median(points, axis=0).astype(np.float32)


def bbox_3d(points: np.ndarray, lower_q: float = 0.01, upper_q: float = 0.99) -> tuple[np.ndarray | None, np.ndarray | None]:
    if points.shape[0] == 0:
        return None, None
    return (
        np.quantile(points, lower_q, axis=0).astype(np.float32),
        np.quantile(points, upper_q, axis=0).astype(np.float32),
    )


def mask_median_depth(depth_m: np.ndarray, mask: np.ndarray) -> float | None:
    values = depth_m[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if values.size else None
