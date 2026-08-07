#!/usr/bin/env python3
"""Sanity-check the CUDA merged point-cloud builder."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import warp as wp

from pointcloud_debug_gpu import GpuMergedPointCloudBuilder


def make_frame(
    name: str,
    depth_value: float,
    rgb_value: tuple[int, int, int],
    translation_x: float,
):
    height, width = 48, 64

    depth = np.full(
        (height, width),
        depth_value,
        dtype=np.float32,
    )
    depth[0, 0] = 0.0

    rgb = np.zeros(
        (height, width, 4),
        dtype=np.uint8,
    )
    rgb[..., 0] = rgb_value[0]
    rgb[..., 1] = rgb_value[1]
    rgb[..., 2] = rgb_value[2]
    rgb[..., 3] = 255

    K = np.asarray(
        [
            [60.0, 0.0, (width - 1) * 0.5],
            [0.0, 60.0, (height - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = translation_x

    runtime = SimpleNamespace(
        spec=SimpleNamespace(name=name),
        K=K,
        T_world_from_camera_optical=transform,
    )

    return SimpleNamespace(
        runtime=runtime,
        depth_gpu=wp.array(
            depth,
            dtype=wp.float32,
            device="cuda:0",
        ),
        rgb_gpu=wp.array(
            rgb,
            dtype=wp.uint8,
            device="cuda:0",
        ),
    )


def main() -> None:
    wp.init()

    builder = GpuMergedPointCloudBuilder(
        voxel_size_m=0.05,
        max_points=5000,
        max_depth_m=10.0,
        device="cuda:0",
    )

    points, packed_rgb = builder.build(
        [
            make_frame(
                "camera_0",
                depth_value=1.0,
                rgb_value=(255, 0, 0),
                translation_x=0.0,
            ),
            make_frame(
                "camera_1",
                depth_value=1.2,
                rgb_value=(0, 255, 0),
                translation_x=0.5,
            ),
        ]
    )

    assert points.ndim == 2
    assert points.shape[1] == 3
    assert points.shape[0] > 0
    assert points.shape[0] <= 5000
    assert packed_rgb.shape == (points.shape[0],)
    assert np.isfinite(points).all()

    print("GPU point-cloud sanity check passed.")
    print("Output points:", points.shape[0])
    print("Output bytes:", points.nbytes + packed_rgb.nbytes)


if __name__ == "__main__":
    main()
