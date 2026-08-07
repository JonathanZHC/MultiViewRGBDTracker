#!/usr/bin/env python3
"""GPU construction and voxel downsampling for the merged RViz point cloud.

The complete point-cloud preparation path stays on CUDA:

1. Read the existing Warp RGB and depth annotator arrays without copying.
2. Filter invalid depth pixels.
3. Back-project depth into each optical camera frame.
4. Transform both point sets into the world frame.
5. Merge the camera point sets.
6. Quantize to voxels and retain one point per voxel on the GPU.
7. Limit the visualization cloud to a fixed maximum point count.
8. Pack RGB values on the GPU.

Only the final compact point array and packed RGB array are copied to the CPU
for sensor_msgs/PointCloud2 serialization.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import warp as wp


class GpuMergedPointCloudBuilder:
    """Build one merged, voxelized world-frame cloud on CUDA."""

    def __init__(
        self,
        voxel_size_m: float,
        max_points: int,
        max_depth_m: float,
        device: str = "cuda:0",
    ) -> None:
        if voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be positive.")
        if max_points <= 0:
            raise ValueError("max_points must be positive.")
        if max_depth_m <= 0.0:
            raise ValueError("max_depth_m must be positive.")
        if not wp.is_cuda_available():
            raise RuntimeError("CUDA is unavailable to NVIDIA Warp.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to PyTorch.")

        self.voxel_size_m = float(voxel_size_m)
        self.max_points = int(max_points)
        self.max_depth_m = float(max_depth_m)
        self.device = device
        self.torch_device = torch.device(
            wp.device_to_torch(device)
        )

        self._pixel_grid_cache: dict[
            tuple[int, int],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._transform_cache: dict[
            str,
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

    def _pixel_grid(
        self,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached flattened pixel coordinates on CUDA."""

        key = (height, width)
        cached = self._pixel_grid_cache.get(key)
        if cached is not None:
            return cached

        rows = torch.arange(
            height,
            device=self.torch_device,
            dtype=torch.float32,
        )
        columns = torch.arange(
            width,
            device=self.torch_device,
            dtype=torch.float32,
        )
        v, u = torch.meshgrid(
            rows,
            columns,
            indexing="ij",
        )
        cached = (u.reshape(-1), v.reshape(-1))
        self._pixel_grid_cache[key] = cached
        return cached

    def _world_transform(
        self,
        runtime: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached world-from-optical rotation and translation."""

        name = runtime.spec.name
        cached = self._transform_cache.get(name)
        if cached is not None:
            return cached

        transform = np.asarray(
            runtime.T_world_from_camera_optical,
            dtype=np.float32,
        )
        if transform.shape != (4, 4):
            raise ValueError(
                f"{name} transform shape is {transform.shape}."
            )

        rotation = torch.as_tensor(
            transform[:3, :3],
            device=self.torch_device,
            dtype=torch.float32,
        )
        translation = torch.as_tensor(
            transform[:3, 3],
            device=self.torch_device,
            dtype=torch.float32,
        )
        cached = (rotation, translation)
        self._transform_cache[name] = cached
        return cached

    @staticmethod
    def _voxel_keys(
        voxel_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Encode signed XYZ voxel coordinates into collision-free int64 keys."""

        minimum = voxel_coordinates.amin(dim=0)
        shifted = voxel_coordinates - minimum
        spans = shifted.amax(dim=0) + 1

        return (
            shifted[:, 0]
            + spans[0]
            * (
                shifted[:, 1]
                + spans[1] * shifted[:, 2]
            )
        )

    def build(
        self,
        frames: Iterable[Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return compact world XYZ and packed RGB arrays on the CPU."""

        frame_list = list(frames)
        if not frame_list:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.uint32),
            )

        world_points: list[torch.Tensor] = []
        point_colors: list[torch.Tensor] = []

        with torch.inference_mode():
            for frame in frame_list:
                depth = wp.to_torch(frame.depth_gpu).squeeze()
                rgb = wp.to_torch(frame.rgb_gpu)

                if depth.ndim != 2:
                    raise RuntimeError(
                        f"{frame.runtime.spec.name} depth shape is "
                        f"{tuple(depth.shape)}."
                    )
                if rgb.ndim != 3 or rgb.shape[2] < 3:
                    raise RuntimeError(
                        f"{frame.runtime.spec.name} RGB shape is "
                        f"{tuple(rgb.shape)}."
                    )
                if tuple(rgb.shape[:2]) != tuple(depth.shape):
                    raise RuntimeError(
                        f"{frame.runtime.spec.name} RGB/depth mismatch: "
                        f"{tuple(rgb.shape)} vs {tuple(depth.shape)}."
                    )

                height, width = int(depth.shape[0]), int(depth.shape[1])
                u, v = self._pixel_grid(height, width)

                z = depth.reshape(-1)
                valid = (
                    torch.isfinite(z)
                    & (z > 0.0)
                    & (z < self.max_depth_m)
                )

                z_valid = z[valid]
                if z_valid.numel() == 0:
                    continue

                K = frame.runtime.K
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                cx = float(K[0, 2])
                cy = float(K[1, 2])

                x = (u[valid] - cx) * z_valid / fx
                y = (v[valid] - cy) * z_valid / fy
                points_camera = torch.stack(
                    (x, y, z_valid),
                    dim=1,
                )

                rotation, translation = self._world_transform(
                    frame.runtime
                )
                points_world = (
                    points_camera @ rotation.transpose(0, 1)
                    + translation
                )

                colors = (
                    rgb[..., :3]
                    .reshape(-1, 3)[valid]
                    .to(dtype=torch.int32)
                )

                world_points.append(points_world)
                point_colors.append(colors)

            if not world_points:
                return (
                    np.empty((0, 3), dtype=np.float32),
                    np.empty((0,), dtype=np.uint32),
                )

            merged_points = torch.cat(world_points, dim=0)
            merged_colors = torch.cat(point_colors, dim=0)

            voxel_coordinates = torch.floor(
                merged_points / self.voxel_size_m
            ).to(dtype=torch.int64)

            keys = self._voxel_keys(voxel_coordinates)
            sorted_keys, sorted_indices = torch.sort(keys)

            keep = torch.ones(
                sorted_keys.shape,
                dtype=torch.bool,
                device=self.torch_device,
            )
            if sorted_keys.numel() > 1:
                keep[1:] = (
                    sorted_keys[1:] != sorted_keys[:-1]
                )

            selected_indices = sorted_indices[keep]
            selected_count = int(selected_indices.shape[0])

            if selected_count > self.max_points:
                step = (
                    selected_count + self.max_points - 1
                ) // self.max_points
                selected_indices = selected_indices[
                    ::step
                ][: self.max_points]

            selected_points = merged_points[
                selected_indices
            ].contiguous()
            selected_colors = merged_colors[
                selected_indices
            ]

            packed_rgb = (
                (selected_colors[:, 0] << 16)
                | (selected_colors[:, 1] << 8)
                | selected_colors[:, 2]
            ).contiguous()

            # These are the only point-cloud device-to-host transfers.
            points_cpu = (
                selected_points
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            packed_rgb_cpu = (
                packed_rgb
                .cpu()
                .numpy()
                .astype(np.uint32, copy=False)
            )

        return (
            np.ascontiguousarray(points_cpu),
            np.ascontiguousarray(packed_rgb_cpu),
        )
