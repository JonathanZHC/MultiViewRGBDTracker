import numpy as np

from sam_rgbd_tracking_benchmark.data_types import CameraIntrinsics
from sam_rgbd_tracking_benchmark.geometry import backproject_mask, transform_points


def test_backprojection_center_pixel():
    depth = np.ones((3, 3), np.float32) * 2.0
    mask = np.zeros((3, 3), bool)
    mask[1, 1] = True
    intrinsics = CameraIntrinsics(3, 3, 2.0, 2.0, 1.0, 1.0)
    points, _ = backproject_mask(depth, mask, intrinsics)
    np.testing.assert_allclose(points, [[0.0, 0.0, 2.0]])
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(transform_points(points, transform), [[1.0, 2.0, 5.0]])
