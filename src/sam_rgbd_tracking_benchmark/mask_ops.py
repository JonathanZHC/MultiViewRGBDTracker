from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def ensure_nhw(logits: np.ndarray, height: int, width: int) -> np.ndarray:
    array = np.asarray(logits, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, height, width), dtype=np.float32)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"Expected [N,H,W] mask logits, got {array.shape}")
    if array.shape[-2:] != (height, width):
        if cv2 is None:
            raise RuntimeError("OpenCV is required to resize mask logits")
        array = np.stack(
            [cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR) for mask in array]
        )
    return array


def exclusive_masks_from_logits(
    logits: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidates, mutually exclusive masks and channel owner map.

    The owner map is -1 for background and 0..N-1 for an object channel.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [N,H,W]")
    n, h, w = logits.shape
    if n == 0:
        return (
            np.zeros((0, h, w), dtype=bool),
            np.zeros((0, h, w), dtype=bool),
            np.full((h, w), -1, dtype=np.int32),
        )
    candidates = logits > threshold
    any_candidate = candidates.any(axis=0)
    masked_logits = np.where(candidates, logits, -np.inf)
    owner = np.argmax(masked_logits, axis=0).astype(np.int32)
    owner[~any_candidate] = -1
    exclusive = np.stack([(owner == i) for i in range(n)])
    return candidates, exclusive, owner


def channel_owner_to_track_owner(owner: np.ndarray, track_ids: list[int]) -> np.ndarray:
    result = np.full(owner.shape, -1, dtype=np.int32)
    for channel, track_id in enumerate(track_ids):
        result[owner == channel] = int(track_id)
    return result


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    if pixels <= 0 or not binary.any():
        return binary.astype(bool)
    if cv2 is None:
        from scipy.ndimage import binary_erosion

        return binary_erosion(binary, iterations=pixels)
    kernel_size = 2 * pixels + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(binary, kernel, iterations=1).astype(bool)


def keep_large_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    if min_pixels <= 1 or not binary.any():
        return binary.astype(bool)
    if cv2 is None:
        from scipy.ndimage import label

        labels, count = label(binary)
        result = np.zeros_like(binary, dtype=bool)
        for idx in range(1, count + 1):
            component = labels == idx
            if int(component.sum()) >= min_pixels:
                result |= component
        return result
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = np.zeros_like(binary, dtype=bool)
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] >= min_pixels:
            result[labels == idx] = True
    return result


def depth_edge_validity(depth_m: np.ndarray, threshold_m: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if threshold_m <= 0:
        return valid
    padded_x = np.pad(depth, ((0, 0), (1, 1)), mode="edge")
    padded_y = np.pad(depth, ((1, 1), (0, 0)), mode="edge")
    dx = np.abs(padded_x[:, 2:] - padded_x[:, :-2]) * 0.5
    dy = np.abs(padded_y[2:, :] - padded_y[:-2, :]) * 0.5
    edge = (dx > threshold_m) | (dy > threshold_m)
    return valid & ~edge
