from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from .data_types import FrameResult, MultiViewInstance, ProcessedInstance
from .slots import max_cross_frame_candidate_pairs

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(slots=True)
class _VoxelObservation:
    view_index: int
    camera_name: str
    instance: ProcessedInstance
    coords: np.ndarray
    keys: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    coverage_origin: np.ndarray | None = None
    coverage_grid: np.ndarray | None = None


class SharedWorldVoxelizer:
    """Persistent sparse world voxelizer shared by all camera observations."""

    _BITS = 21
    _BIAS = 1 << 20
    _MASK = (1 << _BITS) - 1

    def __init__(self, config) -> None:
        cfg = config.get("shared_voxel_grid", {})
        self.voxel_size_m = float(cfg.get("voxel_size_m", 0.01))
        if self.voxel_size_m <= 0.0:
            raise ValueError("shared_voxel_grid.voxel_size_m must be > 0")
        self.inv_voxel_size = 1.0 / self.voxel_size_m
        self.origin_world = np.asarray(
            cfg.get("origin_world", [0.0, 0.0, 0.0]), dtype=np.float32
        ).reshape(3)
        self.match_radius = max(0, int(cfg.get("match_radius_voxels", 1)))
        self.min_alignment_score = float(cfg.get("min_alignment_score", 0.45))
        self.min_bidirectional_coverage = float(
            cfg.get("min_bidirectional_coverage", 0.20)
        )
        # Sparse voxel keys remain canonical. For compact object extents, derive
        # a tiny *local* dilated occupancy cache to make neighborhood coverage O(N)
        # rather than O(27*N*logN). This is never a global dense world volume.
        self.max_local_dense_voxels = max(0, int(cfg.get("max_local_dense_voxels", 1_000_000)))
        self._neighbor_offsets = np.asarray(
            list(
                product(
                    range(-self.match_radius, self.match_radius + 1),
                    repeat=3,
                )
            ),
            dtype=np.int64,
        )

    @classmethod
    def _encode_keys(cls, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.int64)
        shifted = coords + cls._BIAS
        if shifted.size and (
            np.any(shifted < 0) or np.any(shifted > cls._MASK)
        ):
            raise ValueError(
                "World voxel coordinate exceeded the compact 21-bit/key range. "
                "Move shared_voxel_grid.origin_world closer to the workspace or "
                "increase voxel_size_m."
            )
        x = shifted[:, 0].astype(np.uint64, copy=False)
        y = shifted[:, 1].astype(np.uint64, copy=False)
        z = shifted[:, 2].astype(np.uint64, copy=False)
        return (x << np.uint64(42)) | (y << np.uint64(21)) | z

    def downsample_points(
        self,
        points_world: np.ndarray,
        colors_rgb: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Voxel-deduplicate a fused cloud on the exact shared world lattice.

        Cross-view matching already uses ``voxel_size_m`` and ``origin_world``.
        Reusing the same lattice here removes duplicate/overlapping samples from
        different views before the cloud is visualized or sent to cross-frame
        Chamfer. One representative point/color is kept per occupied voxel.
        """
        points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        if points.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )

        colors = (
            np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)
            if colors_rgb is not None
            else np.zeros((len(points), 3), dtype=np.uint8)
        )
        if len(colors) != len(points):
            raise ValueError(
                "Fused point/color count mismatch during shared-voxel downsample"
            )

        coords = np.floor(
            (points - self.origin_world[None, :]) * self.inv_voxel_size
        ).astype(np.int64)
        keys = self._encode_keys(coords)
        _, keep = np.unique(keys, return_index=True)
        # np.unique sorts by key; sort representative indices back into original
        # point order so visualization does not get an arbitrary spatial ordering.
        keep.sort()
        return (
            np.ascontiguousarray(points[keep], dtype=np.float32),
            np.ascontiguousarray(colors[keep], dtype=np.uint8),
        )

    def prepare_points(
        self,
        points_world: np.ndarray,
        colors_rgb: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Quantize+deduplicate one local cloud once for all later alignment work."""
        points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        if points.size == 0:
            return None
        coords = np.floor(
            (points - self.origin_world[None, :]) * self.inv_voxel_size
        ).astype(np.int64)
        keys = self._encode_keys(coords)
        unique_keys, unique_indices = np.unique(keys, return_index=True)
        unique_coords = np.ascontiguousarray(coords[unique_indices], dtype=np.int64)
        unique_points = np.ascontiguousarray(points[unique_indices], dtype=np.float32)
        if colors_rgb is None:
            unique_colors = np.empty((0, 3), dtype=np.uint8)
        else:
            source_colors = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)
            if source_colors.size == 0:
                unique_colors = np.empty((0, 3), dtype=np.uint8)
            else:
                if len(source_colors) != len(points):
                    raise ValueError(
                        "Point/color count mismatch while voxelizing local instance"
                    )
                unique_colors = np.ascontiguousarray(
                    source_colors[unique_indices], dtype=np.uint8
                )
        bbox_min = unique_coords.min(axis=0)
        bbox_max = unique_coords.max(axis=0)
        return (
            unique_coords,
            unique_keys,
            unique_points,
            unique_colors,
            bbox_min,
            bbox_max,
        )

    def voxelize(
        self,
        view_index: int,
        camera_name: str,
        instance: ProcessedInstance,
    ) -> _VoxelObservation | None:
        # Fast path: batched postprocess already quantized this cloud while the
        # points were hot in cache.  Cross-view only builds the tiny local
        # occupancy acceleration structure here.
        if (
            instance.voxel_coords is not None
            and instance.voxel_keys is not None
            and instance.voxel_points is not None
            and instance.voxel_bbox_min is not None
            and instance.voxel_bbox_max is not None
        ):
            unique_coords = instance.voxel_coords
            unique_keys = instance.voxel_keys
            unique_points = instance.voxel_points
            unique_colors = (
                instance.voxel_colors
                if instance.voxel_colors is not None
                else np.empty((0, 3), dtype=np.uint8)
            )
            bbox_min = instance.voxel_bbox_min
            bbox_max = instance.voxel_bbox_max
        else:
            points = instance.points_world
            if points is None or points.size == 0:
                return None
            prepared = self.prepare_points(points, instance.colors_rgb)
            if prepared is None:
                return None
            (
                unique_coords,
                unique_keys,
                unique_points,
                unique_colors,
                bbox_min,
                bbox_max,
            ) = prepared
        coverage_origin = None
        coverage_grid = None
        if self.max_local_dense_voxels > 0:
            radius = self.match_radius
            local_min = bbox_min - radius
            local_max = bbox_max + radius
            shape = (local_max - local_min + 1).astype(np.int64)
            volume = int(np.prod(shape, dtype=np.int64))
            if 0 < volume <= self.max_local_dense_voxels:
                occupancy = np.zeros(tuple(int(v) for v in shape), dtype=np.uint8)
                local = unique_coords - local_min[None, :]
                occupancy[local[:, 0], local[:, 1], local[:, 2]] = 1
                if radius > 0:
                    occupancy = maximum_filter(
                        occupancy,
                        size=2 * radius + 1,
                        mode="constant",
                        cval=0,
                    )
                coverage_origin = local_min
                coverage_grid = occupancy

        return _VoxelObservation(
            view_index=int(view_index),
            camera_name=str(camera_name),
            instance=instance,
            coords=unique_coords,
            keys=unique_keys,
            points=unique_points,
            colors=unique_colors,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            coverage_origin=coverage_origin,
            coverage_grid=coverage_grid,
        )

    def bbox_compatible(self, a: _VoxelObservation, b: _VoxelObservation) -> bool:
        radius = self.match_radius
        return bool(
            np.all(a.bbox_max + radius >= b.bbox_min)
            and np.all(b.bbox_max + radius >= a.bbox_min)
        )

    @staticmethod
    def _membership(sorted_keys: np.ndarray, query_keys: np.ndarray) -> np.ndarray:
        if sorted_keys.size == 0 or query_keys.size == 0:
            return np.zeros(query_keys.shape[0], dtype=bool)
        positions = np.searchsorted(sorted_keys, query_keys)
        valid = positions < sorted_keys.size
        out = np.zeros(query_keys.shape[0], dtype=bool)
        if np.any(valid):
            out[valid] = sorted_keys[positions[valid]] == query_keys[valid]
        return out

    def directional_coverage(
        self,
        source: _VoxelObservation,
        target: _VoxelObservation,
    ) -> float:
        count = int(source.coords.shape[0])
        if count == 0 or target.keys.size == 0:
            return 0.0

        # Fast path for compact objects: target.coverage_grid is already dilated
        # by match_radius, so coverage is one bounds check + indexed gather.
        if target.coverage_grid is not None and target.coverage_origin is not None:
            local = source.coords - target.coverage_origin[None, :]
            shape = np.asarray(target.coverage_grid.shape, dtype=np.int64)
            valid = np.all((local >= 0) & (local < shape[None, :]), axis=1)
            if not np.any(valid):
                return 0.0
            hit = np.zeros(count, dtype=bool)
            indices = local[valid]
            hit[valid] = target.coverage_grid[
                indices[:, 0], indices[:, 1], indices[:, 2]
            ] != 0
            return float(hit.mean())

        if self.match_radius == 0:
            return float(self._membership(target.keys, source.keys).mean())

        # Sparse fallback for large object extents. Evaluate the complete
        # Chebyshev neighborhood with one vectorized searchsorted per chunk.
        # searchsorted per chunk instead of 27 Python/search calls for r=1.
        # Chunking bounds the temporary (N x neighbors x xyz) integer array.
        offsets = self._neighbor_offsets
        neighbor_count = max(1, int(offsets.shape[0]))
        chunk_size = max(256, min(count, 131072 // neighbor_count))
        matched_count = 0
        for start in range(0, count, chunk_size):
            end = min(count, start + chunk_size)
            shifted = (
                source.coords[start:end, None, :] + offsets[None, :, :]
            ).reshape(-1, 3)
            query = self._encode_keys(shifted)
            hit = self._membership(target.keys, query).reshape(
                end - start, neighbor_count
            )
            matched_count += int(np.any(hit, axis=1).sum())
        return float(matched_count / count)

    def symmetric_score(
        self,
        a: _VoxelObservation,
        b: _VoxelObservation,
    ) -> tuple[float, float, float]:
        a_to_b = self.directional_coverage(a, b)
        b_to_a = self.directional_coverage(b, a)
        return 0.5 * (a_to_b + b_to_a), a_to_b, b_to_a


class CrossViewAligner:
    """Class gate -> sparse world voxel geometry -> Hungarian -> fusion."""

    def __init__(self, config) -> None:
        self.voxelizer = SharedWorldVoxelizer(config)
        self.visualization_enabled = bool(
            config.runtime.get("enable_visualization", True)
        )

    def align(
        self,
        results: list[FrameResult],
        profiler: Any | None = None,
    ) -> tuple[list[MultiViewInstance], dict[str, int]]:
        observations: list[_VoxelObservation] = []
        started = time.perf_counter()
        context = (
            profiler.stage("cross_view_voxelize", cuda=False)
            if profiler is not None
            else _NullContext()
        )
        with context:
            for view_index, result in enumerate(results):
                for instance in result.instances:
                    voxelized = self.voxelizer.voxelize(
                        view_index,
                        result.frame.camera_name,
                        instance,
                    )
                    if voxelized is not None:
                        observations.append(voxelized)

        counters = {
            "num_cross_view_candidate_pairs": 0,
            "num_cross_view_matches": 0,
            "num_fused_points_before_downsample": 0,
            "num_fused_points_after_downsample": 0,
        }
        if not observations:
            if profiler is not None:
                profiler.record(
                    "cross_view_total",
                    1000.0 * (time.perf_counter() - started),
                )
            return [], counters

        by_view: dict[int, list[int]] = {}
        for index, obs in enumerate(observations):
            by_view.setdefault(obs.view_index, []).append(index)

        edges: list[tuple[float, int, int]] = []
        bbox_started = time.perf_counter()
        voxel_match_ms = 0.0
        hungarian_ms = 0.0

        for view_a, view_b in combinations(sorted(by_view), 2):
            indices_a = by_view[view_a]
            indices_b = by_view[view_b]
            labels = sorted(
                set(observations[index].instance.label for index in indices_a)
                & set(observations[index].instance.label for index in indices_b)
            )
            for label in labels:
                class_a = [
                    index
                    for index in indices_a
                    if observations[index].instance.label == label
                ]
                class_b = [
                    index
                    for index in indices_b
                    if observations[index].instance.label == label
                ]
                if not class_a or not class_b:
                    continue

                scores = np.full(
                    (len(class_a), len(class_b)), -1.0, dtype=np.float32
                )
                directional_min = np.zeros_like(scores)
                for row, index_a in enumerate(class_a):
                    obs_a = observations[index_a]
                    for col, index_b in enumerate(class_b):
                        obs_b = observations[index_b]
                        if not self.voxelizer.bbox_compatible(obs_a, obs_b):
                            continue
                        counters["num_cross_view_candidate_pairs"] += 1
                        match_started = time.perf_counter()
                        score, a_to_b, b_to_a = self.voxelizer.symmetric_score(
                            obs_a, obs_b
                        )
                        voxel_match_ms += 1000.0 * (
                            time.perf_counter() - match_started
                        )
                        scores[row, col] = score
                        directional_min[row, col] = min(a_to_b, b_to_a)

                valid = (
                    (scores >= self.voxelizer.min_alignment_score)
                    & (
                        directional_min
                        >= self.voxelizer.min_bidirectional_coverage
                    )
                )
                if not np.any(valid):
                    continue
                cost = np.where(valid, 1.0 - scores, 1e6).astype(np.float32)
                hungarian_started = time.perf_counter()
                rows, cols = linear_sum_assignment(cost)
                hungarian_ms += 1000.0 * (
                    time.perf_counter() - hungarian_started
                )
                for row, col in zip(rows, cols):
                    if not valid[row, col]:
                        continue
                    edges.append(
                        (
                            float(scores[row, col]),
                            class_a[int(row)],
                            class_b[int(col)],
                        )
                    )

        if profiler is not None:
            profiler.record(
                "cross_view_bbox_gate",
                max(
                    0.0,
                    1000.0 * (time.perf_counter() - bbox_started)
                    - voxel_match_ms
                    - hungarian_ms,
                ),
            )
            profiler.record("cross_view_voxel_match", voxel_match_ms)
            profiler.record("cross_view_hungarian", hungarian_ms)

        # Highest-confidence geometry edges win. The constrained DSU guarantees
        # at most one observation from each camera in a fused group.
        parent = list(range(len(observations)))
        cameras = [{observations[index].view_index} for index in range(len(observations))]

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> bool:
            root_a, root_b = find(a), find(b)
            if root_a == root_b:
                return False
            if cameras[root_a] & cameras[root_b]:
                return False
            if len(cameras[root_a]) < len(cameras[root_b]):
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a
            cameras[root_a] |= cameras[root_b]
            return True

        for score, index_a, index_b in sorted(edges, reverse=True):
            del score
            if union(index_a, index_b):
                counters["num_cross_view_matches"] += 1

        groups: dict[int, list[_VoxelObservation]] = {}
        for index, observation in enumerate(observations):
            groups.setdefault(find(index), []).append(observation)

        fusion_context = (
            profiler.stage("cross_view_fusion", cuda=False)
            if profiler is not None
            else _NullContext()
        )
        fused: list[MultiViewInstance] = []
        with fusion_context:
            for group_index, members in enumerate(groups.values(), start=1):
                local_members = [member.instance for member in members]
                raw_point_count = sum(
                    int(len(member.instance.points_world))
                    for member in members
                    if member.instance.points_world is not None
                )
                counters["num_fused_points_before_downsample"] += raw_point_count

                # Reuse the exact sparse voxel keys already computed for
                # cross-view matching. This avoids quantizing the same points a
                # second time: final fusion is just a union of occupied voxels.
                if len(members) == 1:
                    points_world = members[0].points
                    colors_rgb = members[0].colors
                else:
                    fused_keys = np.concatenate(
                        [member.keys for member in members], axis=0
                    )
                    fused_points = np.concatenate(
                        [member.points for member in members], axis=0
                    )
                    _, keep = np.unique(fused_keys, return_index=True)
                    keep.sort()
                    points_world = np.ascontiguousarray(
                        fused_points[keep], dtype=np.float32
                    )
                    if self.visualization_enabled:
                        fused_colors = np.concatenate(
                            [member.colors for member in members], axis=0
                        )
                        colors_rgb = np.ascontiguousarray(
                            fused_colors[keep], dtype=np.uint8
                        )
                    else:
                        colors_rgb = np.empty((0, 3), dtype=np.uint8)

                counters["num_fused_points_after_downsample"] += int(
                    len(points_world)
                )

                if points_world.size > 0:
                    centroid = np.median(points_world, axis=0).astype(np.float32)
                    if self.visualization_enabled:
                        # The group already lives on one shared voxel lattice.
                        # Reuse the union voxel bounds instead of two O(N)
                        # quantile passes over the fused cloud every frame.
                        voxel_min = np.min(
                            np.stack([member.bbox_min for member in members], axis=0),
                            axis=0,
                        )
                        voxel_max = np.max(
                            np.stack([member.bbox_max for member in members], axis=0),
                            axis=0,
                        )
                        bbox_min = (
                            self.voxelizer.origin_world
                            + voxel_min.astype(np.float32)
                            * np.float32(self.voxelizer.voxel_size_m)
                        )
                        bbox_max = (
                            self.voxelizer.origin_world
                            + (voxel_max.astype(np.float32) + 1.0)
                            * np.float32(self.voxelizer.voxel_size_m)
                        )
                    else:
                        bbox_min = bbox_max = None
                else:
                    centroid = None
                    bbox_min = None
                    bbox_max = None
                group = MultiViewInstance(
                    group_id=group_index,
                    semantic_label=members[0].instance.label,
                    members=[
                        (member.camera_name, member.instance) for member in members
                    ],
                    points_world=points_world,
                    colors_rgb=colors_rgb,
                    centroid_world=centroid,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                )
                for member in local_members:
                    member.multiview_group_id = group_index
                fused.append(group)

        if profiler is not None:
            profiler.record(
                "cross_view_total",
                1000.0 * (time.perf_counter() - started),
            )
        return fused, counters


class _ChamferWorkspace:
    """Reusable pinned-host + CUDA buffers for per-frame batched Chamfer.

    The pair dimension is fixed once from the configured per-view semantic
    capacities and camera count.
    Point dimensions may still grow geometrically when a larger fused cloud is
    observed.  This keeps the hot path allocation-free for the expected scene
    while making a violated instance-capacity assumption fail loudly.
    """

    def __init__(
        self,
        device: str,
        max_pairs: int,
        max_workspace_mb: float = 256.0,
    ) -> None:
        self.device = torch.device(device) if torch is not None else None
        self.max_pairs = max(1, int(max_pairs))
        self.max_workspace_bytes = max(16, int(max_workspace_mb)) * 1024 * 1024
        self.capacity = (self.max_pairs, 0, 0)
        self.host_a = None
        self.host_b = None
        self.host_count_a = None
        self.host_count_b = None
        self.gpu_a = None
        self.gpu_b = None
        self.gpu_count_a = None
        self.gpu_count_b = None
        self.index_a = None
        self.index_b = None
        self.origin = None
        self.best_b_sq = None
        self.total_a = None

    @staticmethod
    def _grow_points(required: int, current: int) -> int:
        """Power-of-two growth for point buffers; pair capacity stays fixed."""
        required = max(1, int(required))
        current = max(0, int(current))
        if required <= current:
            return current
        return 1 << (required - 1).bit_length()

    def _check_pair_count(self, pairs: int) -> None:
        if int(pairs) > self.max_pairs:
            raise RuntimeError(
                "Cross-frame Chamfer candidate count exceeded the strict "
                f"multi-view same-class upper bound: got {pairs}, "
                f"capacity={self.max_pairs}. This bound already accounts for "
                "all configured cameras and unmatched cross-view observations; "
                "check detector.prompts capacities or cross-view grouping for "
                "unexpected duplicate observations."
            )

    def reserve_points(self, points: int) -> None:
        """Optionally reserve the expected fused-cloud size at startup."""
        if torch is None or self.device is None:
            return
        if self.device.type == "cuda" and not torch.cuda.is_available():
            return
        if points > 0:
            self._ensure(self.max_pairs, int(points), int(points))

    def _ensure(self, pairs: int, max_a: int, max_b: int) -> None:
        if torch is None or self.device is None:
            return
        self._check_pair_count(pairs)
        _, old_a, old_b = self.capacity
        if max_a <= old_a and max_b <= old_b and self.host_a is not None:
            return

        new_a = self._grow_points(max_a, old_a)
        new_b = self._grow_points(max_b, old_b)
        pin = bool(torch.cuda.is_available())

        # Pair capacity is exact and fixed. Point capacity grows only when a larger
        # fused cloud appears. Padding is intentionally left uninitialized because
        # count masks make those values semantically irrelevant.
        self.host_a = torch.empty(
            (self.max_pairs, new_a, 3), dtype=torch.float32, device="cpu", pin_memory=pin
        )
        self.host_b = torch.empty(
            (self.max_pairs, new_b, 3), dtype=torch.float32, device="cpu", pin_memory=pin
        )
        self.host_count_a = torch.empty(
            (self.max_pairs,), dtype=torch.long, device="cpu", pin_memory=pin
        )
        self.host_count_b = torch.empty(
            (self.max_pairs,), dtype=torch.long, device="cpu", pin_memory=pin
        )

        self.gpu_a = torch.empty(
            (self.max_pairs, new_a, 3), dtype=torch.float32, device=self.device
        )
        self.gpu_b = torch.empty(
            (self.max_pairs, new_b, 3), dtype=torch.float32, device=self.device
        )
        self.gpu_count_a = torch.empty(
            (self.max_pairs,), dtype=torch.long, device=self.device
        )
        self.gpu_count_b = torch.empty(
            (self.max_pairs,), dtype=torch.long, device=self.device
        )

        self.origin = torch.empty(
            (self.max_pairs, 1, 3), dtype=torch.float32, device=self.device
        )
        self.best_b_sq = torch.empty(
            (self.max_pairs, new_b), dtype=torch.float32, device=self.device
        )
        self.total_a = torch.empty(
            (self.max_pairs,), dtype=torch.float32, device=self.device
        )
        self.index_a = torch.arange(new_a, device=self.device, dtype=torch.long)
        self.index_b = torch.arange(new_b, device=self.device, dtype=torch.long)
        self.capacity = (self.max_pairs, new_a, new_b)

    def _symmetric_gpu(
        self,
        a: Any,
        b: Any,
        count_a: Any,
        count_b: Any,
        *,
        max_a: int,
        max_b: int,
    ) -> Any:
        """One pairwise-distance pass computes both Chamfer directions.

        Squared Euclidean distances are formed by batched GEMM. A common
        per-pair origin improves numerical stability; workspace point buffers are
        overwritten on the next frame, so the in-place shift adds no allocation.
        """
        pairs = int(a.shape[0])
        assert self.origin is not None
        assert self.best_b_sq is not None and self.total_a is not None
        assert self.index_a is not None and self.index_b is not None

        origin = self.origin[:pairs]
        origin.copy_(a[:, :1, :])
        a.sub_(origin)
        b.sub_(origin)

        target = b[:, :max_b]
        target_sq = (target * target).sum(dim=2)
        valid_b = self.index_b[:max_b][None, :] < count_b[:, None]
        best_b_sq = self.best_b_sq[:pairs, :max_b]
        best_b_sq.fill_(float("inf"))
        total_a = self.total_a[:pairs]
        total_a.zero_()

        bytes_per_source_point = max(1, pairs * max_b * 4)
        chunk = max(
            1,
            min(max_a, self.max_workspace_bytes // bytes_per_source_point),
        )
        target_t = target.transpose(1, 2)
        for start in range(0, max_a, chunk):
            end = min(max_a, start + chunk)
            source = a[:, start:end]
            source_sq = (source * source).sum(dim=2)
            # D^2 = ||a||^2 + ||b||^2 - 2 a^T b. The same matrix supplies
            # row minima (A->B) and column minima (B->A).
            dist_sq = torch.bmm(source, target_t).mul_(-2.0)
            dist_sq.add_(source_sq[:, :, None])
            dist_sq.add_(target_sq[:, None, :])
            dist_sq.clamp_min_(0.0)

            valid_a = self.index_a[start:end][None, :] < count_a[:, None]
            dist_sq.masked_fill_(~valid_b[:, None, :], float("inf"))

            nearest_a_sq = dist_sq.amin(dim=2)
            nearest_a_sq.masked_fill_(~valid_a, 0.0)
            total_a.add_(torch.sqrt(nearest_a_sq).sum(dim=1))

            dist_sq.masked_fill_(~valid_a[:, :, None], float("inf"))
            torch.minimum(best_b_sq, dist_sq.amin(dim=1), out=best_b_sq)

        best_b_sq.masked_fill_(~valid_b, 0.0)
        a_to_b = total_a / count_a.clamp_min(1).to(torch.float32)
        b_to_a = (
            torch.sqrt(best_b_sq).sum(dim=1)
            / count_b.clamp_min(1).to(torch.float32)
        )
        return 0.5 * (a_to_b + b_to_a)

    def compute(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        if not pairs:
            return np.empty((0,), dtype=np.float32)
        self._check_pair_count(len(pairs))
        if (
            torch is None
            or self.device is None
            or self.device.type != "cuda"
            or not torch.cuda.is_available()
        ):
            return self._compute_cpu(pairs)

        counts_a = [len(a) for a, _ in pairs]
        counts_b = [len(b) for _, b in pairs]
        max_a = max(counts_a, default=0)
        max_b = max(counts_b, default=0)
        if max_a == 0 or max_b == 0:
            return np.full((len(pairs),), np.inf, dtype=np.float32)

        self._ensure(len(pairs), max_a, max_b)
        assert self.host_a is not None and self.host_b is not None
        assert self.host_count_a is not None and self.host_count_b is not None
        assert self.gpu_a is not None and self.gpu_b is not None
        assert self.gpu_count_a is not None and self.gpu_count_b is not None

        pair_count = len(pairs)
        host_a = self.host_a[:pair_count, :max_a]
        host_b = self.host_b[:pair_count, :max_b]
        host_a_np = host_a.numpy()
        host_b_np = host_b.numpy()
        for index, (a, b) in enumerate(pairs):
            host_a_np[index, : len(a)] = np.asarray(a, dtype=np.float32)
            host_b_np[index, : len(b)] = np.asarray(b, dtype=np.float32)

        host_count_a = self.host_count_a[:pair_count]
        host_count_b = self.host_count_b[:pair_count]
        host_count_a.numpy()[:] = counts_a
        host_count_b.numpy()[:] = counts_b

        gpu_a = self.gpu_a[:pair_count, :max_a]
        gpu_b = self.gpu_b[:pair_count, :max_b]
        gpu_count_a = self.gpu_count_a[:pair_count]
        gpu_count_b = self.gpu_count_b[:pair_count]
        gpu_a.copy_(host_a, non_blocking=host_a.is_pinned())
        gpu_b.copy_(host_b, non_blocking=host_b.is_pinned())
        gpu_count_a.copy_(host_count_a, non_blocking=host_count_a.is_pinned())
        gpu_count_b.copy_(host_count_b, non_blocking=host_count_b.is_pinned())

        result = self._symmetric_gpu(
            gpu_a,
            gpu_b,
            gpu_count_a,
            gpu_count_b,
            max_a=max_a,
            max_b=max_b,
        )
        return result.detach().cpu().numpy()

    @staticmethod
    def _compute_cpu(
        pairs: list[tuple[np.ndarray, np.ndarray]],
        chunk: int = 256,
    ) -> np.ndarray:
        values: list[float] = []
        for a, b in pairs:
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            if a.size == 0 or b.size == 0:
                values.append(float("inf"))
                continue
            total_a = 0.0
            best_b_sq = np.full((len(b),), np.inf, dtype=np.float32)
            for start in range(0, len(a), chunk):
                source = a[start : start + chunk]
                diff = source[:, None, :] - b[None, :, :]
                dist_sq = np.sum(diff * diff, axis=2)
                total_a += float(np.sqrt(np.min(dist_sq, axis=1)).sum())
                best_b_sq = np.minimum(best_b_sq, np.min(dist_sq, axis=0))
            a_to_b = total_a / len(a)
            b_to_a = float(np.sqrt(best_b_sq).sum()) / len(b)
            values.append(0.5 * (a_to_b + b_to_a))
        return np.asarray(values, dtype=np.float32)


class CrossFrameAligner:
    """Class+centroid hard gate -> batched Chamfer -> Hungarian."""

    def __init__(self, config, *, num_views: int | None = None) -> None:
        cfg = config.get("cross_frame_alignment", {})
        self.centroid_gate_m = float(cfg.get("centroid_gate_m", 0.20))
        threshold = cfg.get("max_chamfer_m", None)
        self.max_chamfer_m = None if threshold in (None, "", False) else float(threshold)
        device = str(config.runtime.get("device", "cuda"))
        workspace_mb = float(cfg.get("chamfer_max_workspace_mb", 256.0))
        pair_capacity = max_cross_frame_candidate_pairs(config, num_views=num_views)
        self.workspace = _ChamferWorkspace(
            device,
            max_pairs=pair_capacity,
            max_workspace_mb=workspace_mb,
        )
        self.workspace.reserve_points(int(cfg.get("chamfer_preallocate_points", 0)))
        self.previous: list[MultiViewInstance] = []
        self.next_global_track_id = 1

    def _assign_new_id(self, instance: MultiViewInstance) -> None:
        instance.global_track_id = self.next_global_track_id
        self.next_global_track_id += 1

    def align(
        self,
        current: list[MultiViewInstance],
        profiler: Any | None = None,
    ) -> dict[str, int]:
        started = time.perf_counter()
        counters = {
            "num_cross_frame_candidate_pairs": 0,
            "num_cross_frame_matches": 0,
        }
        if not current:
            self.previous = []
            if profiler is not None:
                profiler.record(
                    "cross_frame_total",
                    1000.0 * (time.perf_counter() - started),
                )
            return counters

        if not self.previous:
            for instance in current:
                self._assign_new_id(instance)
                self._propagate_id_to_members(instance)
            self.previous = current
            if profiler is not None:
                profiler.record(
                    "cross_frame_total",
                    1000.0 * (time.perf_counter() - started),
                )
            return counters

        gate_context = (
            profiler.stage("cross_frame_gate", cuda=False)
            if profiler is not None
            else _NullContext()
        )
        with gate_context:
            current_labels = np.asarray([item.semantic_label for item in current], dtype=object)
            previous_labels = np.asarray(
                [item.semantic_label for item in self.previous], dtype=object
            )
            current_centroids = np.stack(
                [
                    item.centroid_world
                    if item.centroid_world is not None
                    else np.full(3, np.nan, dtype=np.float32)
                    for item in current
                ]
            )
            previous_centroids = np.stack(
                [
                    item.centroid_world
                    if item.centroid_world is not None
                    else np.full(3, np.nan, dtype=np.float32)
                    for item in self.previous
                ]
            )
            same_class = current_labels[:, None] == previous_labels[None, :]
            delta = current_centroids[:, None, :] - previous_centroids[None, :, :]
            centroid_distance = np.linalg.norm(delta, axis=2)
            candidate_mask = (
                same_class
                & np.isfinite(centroid_distance)
                & (centroid_distance <= self.centroid_gate_m)
            )

        candidate_indices = np.argwhere(candidate_mask)
        counters["num_cross_frame_candidate_pairs"] = int(len(candidate_indices))
        cost = np.full(
            (len(current), len(self.previous)), 1e6, dtype=np.float32
        )

        if len(candidate_indices):
            candidate_clouds = [
                (
                    current[int(row)].points_world,
                    self.previous[int(col)].points_world,
                )
                for row, col in candidate_indices
            ]
            cuda_enabled = (
                torch is not None
                and torch.cuda.is_available()
                and self.workspace.device is not None
                and self.workspace.device.type == "cuda"
            )
            chamfer_context = (
                profiler.stage(
                    "cross_frame_chamfer"
                    if cuda_enabled
                    else "cross_frame_chamfer",
                    cuda=cuda_enabled,
                )
                if profiler is not None
                else _NullContext()
            )
            with chamfer_context:
                chamfer = self.workspace.compute(candidate_clouds)
            for (row, col), distance in zip(candidate_indices, chamfer):
                if not math.isfinite(float(distance)):
                    continue
                if self.max_chamfer_m is not None and distance > self.max_chamfer_m:
                    continue
                cost[int(row), int(col)] = float(distance)

        hungarian_context = (
            profiler.stage("cross_frame_hungarian", cuda=False)
            if profiler is not None
            else _NullContext()
        )
        matched_current: set[int] = set()
        with hungarian_context:
            if cost.size and np.any(cost < 1e6):
                rows, cols = linear_sum_assignment(cost)
                for row, col in zip(rows, cols):
                    if cost[row, col] >= 1e6:
                        continue
                    global_id = self.previous[int(col)].global_track_id
                    if global_id is None:
                        global_id = self.next_global_track_id
                        self.next_global_track_id += 1
                    current[int(row)].global_track_id = int(global_id)
                    matched_current.add(int(row))
                    counters["num_cross_frame_matches"] += 1

        for index, instance in enumerate(current):
            if index not in matched_current:
                self._assign_new_id(instance)
            self._propagate_id_to_members(instance)

        self.previous = current
        if profiler is not None:
            profiler.record(
                "cross_frame_total",
                1000.0 * (time.perf_counter() - started),
            )
        return counters

    @staticmethod
    def _propagate_id_to_members(instance: MultiViewInstance) -> None:
        for _, member in instance.members:
            member.global_track_id = instance.global_track_id


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
