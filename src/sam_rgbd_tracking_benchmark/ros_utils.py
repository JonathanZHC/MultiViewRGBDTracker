from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np


def image_to_numpy(message: Any, bridge: Any, desired_encoding: str | None = None) -> np.ndarray:
    encoding = desired_encoding or "passthrough"
    return np.asarray(bridge.imgmsg_to_cv2(message, desired_encoding=encoding))


def depth_message_to_meters(message: Any, bridge: Any) -> np.ndarray:
    depth = image_to_numpy(message, bridge, "passthrough")
    if message.encoding in {"16UC1", "mono16"}:
        return depth.astype(np.float32) * 0.001
    return depth.astype(np.float32)


def matrix_from_transform(transform: Any) -> np.ndarray:
    tx = transform.transform.translation.x
    ty = transform.transform.translation.y
    tz = transform.transform.translation.z
    qx = transform.transform.rotation.x
    qy = transform.transform.rotation.y
    qz = transform.transform.rotation.z
    qw = transform.transform.rotation.w
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    rotation = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (tx, ty, tz)
    return matrix


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def metadata_from_json_message(message: Any | None) -> dict[int, dict[str, Any]]:
    if message is None:
        return {}
    try:
        raw = json.loads(message.data)
    except (json.JSONDecodeError, AttributeError):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for key, value in raw.items():
        try:
            result[int(key)] = dict(value)
        except (TypeError, ValueError):
            continue
    return result


def pack_rgb_float(rgb: np.ndarray) -> np.ndarray:
    colors = np.asarray(rgb, dtype=np.uint8)
    packed = (
        colors[:, 0].astype(np.uint32) << 16
        | colors[:, 1].astype(np.uint32) << 8
        | colors[:, 2].astype(np.uint32)
    )
    return packed.view(np.float32)


def color_for_track(track_id: int) -> tuple[int, int, int]:
    # Deterministic high-contrast color without a global palette dependency.
    hue = (track_id * 0.61803398875) % 1.0
    import colorsys

    red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)
