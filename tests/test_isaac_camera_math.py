from __future__ import annotations

import numpy as np

from isaac_sim.camera_math import camera_pose, rotation_matrix_to_quaternion_xyzw


def test_camera_pose_uses_ros_optical_axes() -> None:
    transform, quaternion = camera_pose(
        np.array([1.0, -1.0, 1.0]),
        np.array([0.0, 0.0, 0.5]),
    )
    assert transform.shape == (4, 4)
    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6)
    expected_forward = np.array([-1.0, 1.0, -0.5])
    expected_forward /= np.linalg.norm(expected_forward)
    assert np.allclose(transform[:3, 2], expected_forward, atol=1e-6)


def test_identity_quaternion_is_xyzw() -> None:
    quaternion = rotation_matrix_to_quaternion_xyzw(np.eye(3))
    assert np.allclose(quaternion, [0.0, 0.0, 0.0, 1.0], atol=1e-7)
