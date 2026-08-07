from __future__ import annotations

import math

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return value / norm


def rotation_matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized xyzw quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got {m.shape}.")

    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    return normalize(quaternion)


def camera_pose(
    position_world: np.ndarray,
    look_at_world: np.ndarray,
    up_world: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ROS-optical world transform and USD-camera xyzw quaternion.

    ROS optical axes are +X right, +Y down, +Z forward. USD camera axes are
    +X right, +Y up, -Z forward. The returned transform therefore maps optical
    points into world coordinates, while the quaternion is suitable for the
    USD camera prim.
    """
    eye = np.asarray(position_world, dtype=np.float64)
    target = np.asarray(look_at_world, dtype=np.float64)
    up_hint = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if up_world is None
        else normalize(np.asarray(up_world, dtype=np.float64))
    )

    forward = normalize(target - eye)
    right = np.cross(forward, up_hint)
    if float(np.linalg.norm(right)) <= 1.0e-8:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback)
    right = normalize(right)
    camera_up = normalize(np.cross(right, forward))

    rotation_world_from_optical = np.column_stack(
        (right, -camera_up, forward)
    )
    rotation_world_from_usd = np.column_stack(
        (right, camera_up, -forward)
    )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_world_from_optical
    transform[:3, 3] = eye

    quaternion_usd_xyzw = rotation_matrix_to_quaternion_xyzw(
        rotation_world_from_usd
    )
    return transform, quaternion_usd_xyzw
