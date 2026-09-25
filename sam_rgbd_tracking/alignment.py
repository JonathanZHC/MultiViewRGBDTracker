from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from .data_types import FrameResult, MultiViewInstance, ProcessedInstance
from .slots import max_cross_frame_candidate_pairs, max_cross_frame_instances

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# GPU-alignment mode: deduplicate voxels shared by several cameras before Chamfer
# staging (legacy semantics; costs an np.unique + gather kernels per group) or
# simply concatenate the member clouds (default; zero kernels — DifFlow's voxel-2
# collapses duplicates on its own grid, the Chamfer gate is only marginally
# reweighted).
GPU_FUSION_DEDUP = False

logger = logging.getLogger(__name__)


def gpu_alignment_enabled(config) -> bool:
    """GPU-resident cross-view fusion / Chamfer staging (production default).

    Requires the CUDA geometry backend and visualization off (visualization
    consumes NumPy clouds). ``postprocess.gpu_alignment: false`` restores the
    legacy path where voxel representatives are copied to the host, fused in
    NumPy and uploaded again for Chamfer.
    """
    if torch is None or not torch.cuda.is_available():
        return False
    if not str(config.runtime.get("device", "cuda")).startswith("cuda"):
        return False
    if bool(config.runtime.get("enable_visualization", True)):
        return False
    postprocess = config.get("postprocess", {})
    return bool(postprocess.get("gpu_geometry", True)) and bool(
        postprocess.get("gpu_alignment", True)
    )


@dataclass(slots=True)
class _VoxelObservation:
    view_index: int
    camera_name: str
    instance: ProcessedInstance
    # coords/keys/bbox are None only in GPU-alignment mode with a single camera
    # (no cross-view matching happens, so no CPU voxel data is materialized).
    coords: np.ndarray | None
    keys: np.ndarray | None
    # NumPy voxel representatives; None in GPU-alignment mode (see points_gpu).
    points: np.ndarray | None
    colors: np.ndarray
    bbox_min: np.ndarray | None
    bbox_max: np.ndarray | None
    coverage_origin: np.ndarray | None = None
    coverage_grid: np.ndarray | None = None
    # CUDA view of the same unique representatives (GPU-alignment mode).
    points_gpu: Any | None = None
    # The dense coverage grid is only needed when this observation is the target
    # of a cross-view score; it is built lazily on first use (see
    # ``SharedWorldVoxelizer.ensure_coverage``) instead of on every frame.
    coverage_built: bool = False


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
        # Cross-view association test. "containment" (default): intersection volume of the two
        # observations' world AABBs divided by the SMALLER box volume. It stays high when the
        # visible surfaces do not overlap at all (a human seen from the front by one camera and
        # from the back by the other: the boxes coincide, the surfaces are a body depth apart)
        # and when one camera sees only part of the object (its box lies inside the other).
        # "coverage": legacy bidirectional voxel-surface coverage on the shared 5 mm lattice.
        self.association = str(cfg.get("association", "containment"))
        if self.association not in ("containment", "coverage"):
            raise ValueError("shared_voxel_grid.association must be 'containment' or 'coverage'")
        self.min_containment_ratio = float(cfg.get("min_containment_ratio", 0.5))
        # Temporal hysteresis of a cross-view pairing, keyed by the per-camera slots (EfficientTAM
        # keeps a slot on the same object over time): merge after this many consecutive passing
        # frames, split after this many consecutive failing frames.
        self.merge_confirm_frames = max(1, int(cfg.get("merge_confirm_frames", 2)))
        self.split_confirm_frames = max(1, int(cfg.get("split_confirm_frames", 5)))
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
            and (instance.voxel_points is not None or instance.voxel_points_gpu is not None)
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
            points_gpu=instance.voxel_points_gpu,
        )

    @staticmethod
    def observe_gpu_only(
        view_index: int,
        camera_name: str,
        instance: ProcessedInstance,
    ) -> _VoxelObservation | None:
        """Single-camera GPU-alignment observation: no CPU voxel data at all."""
        points_gpu = instance.voxel_points_gpu
        if points_gpu is None or int(points_gpu.shape[0]) == 0:
            return None
        return _VoxelObservation(
            view_index=int(view_index),
            camera_name=str(camera_name),
            instance=instance,
            coords=None,
            keys=None,
            points=None,
            colors=np.empty((0, 3), dtype=np.uint8),
            bbox_min=None,
            bbox_max=None,
            points_gpu=points_gpu,
        )

    def ensure_coverage(self, observation: _VoxelObservation) -> None:
        """Build the dilated dense occupancy grid for ``observation`` on demand.

        Previously built eagerly for every observation on every frame: a ~1 MB
        ``np.zeros`` + scipy ``maximum_filter`` (3-5 ms) for objects that fit the
        ``max_local_dense_voxels`` cap, even with a single camera where no
        cross-view score is ever computed.
        """
        if observation.coverage_built:
            return
        observation.coverage_built = True
        if self.max_local_dense_voxels <= 0:
            return
        radius = self.match_radius
        local_min = observation.bbox_min - radius
        local_max = observation.bbox_max + radius
        shape = (local_max - local_min + 1).astype(np.int64)
        volume = int(np.prod(shape, dtype=np.int64))
        if not (0 < volume <= self.max_local_dense_voxels):
            return
        occupancy = np.zeros(tuple(int(v) for v in shape), dtype=np.uint8)
        local = observation.coords - local_min[None, :]
        occupancy[local[:, 0], local[:, 1], local[:, 2]] = 1
        if radius > 0:
            occupancy = maximum_filter(
                occupancy,
                size=2 * radius + 1,
                mode="constant",
                cval=0,
            )
        observation.coverage_origin = local_min
        observation.coverage_grid = occupancy

    def containment_ratio(self, a: _VoxelObservation, b: _VoxelObservation) -> float:
        """Intersection volume of the two voxel AABBs over the smaller AABB volume, in [0, 1].
        Boxes are inclusive voxel index ranges, so one voxel has extent 1."""
        if a.bbox_min is None or b.bbox_min is None:
            return 0.0
        lo = np.maximum(a.bbox_min, b.bbox_min)
        hi = np.minimum(a.bbox_max, b.bbox_max)
        inter = np.prod(np.maximum(hi - lo + 1, 0).astype(np.float64))
        if inter <= 0.0:
            return 0.0
        vol_a = np.prod((a.bbox_max - a.bbox_min + 1).astype(np.float64))
        vol_b = np.prod((b.bbox_max - b.bbox_min + 1).astype(np.float64))
        return float(inter / max(1.0, min(vol_a, vol_b)))

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
        self.ensure_coverage(target)
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


@dataclass
class _PairState:
    agree: int = 0
    disagree: int = 0
    merged: bool = False
    last_frame: int = 0


class CrossViewAligner:
    """Class gate -> AABB containment with per-slot hysteresis (or legacy voxel coverage)
    -> Hungarian -> constrained union-find -> fusion."""

    def __init__(self, config) -> None:
        self.voxelizer = SharedWorldVoxelizer(config)
        self.visualization_enabled = bool(
            config.runtime.get("enable_visualization", True)
        )
        self.gpu_fusion = gpu_alignment_enabled(config)
        self.device = (
            torch.device(str(config.runtime.get("device", "cuda")))
            if self.gpu_fusion
            else None
        )
        self._centroid_host = None  # pinned staging for batched centroid D2H
        # (view_a, slot_a, view_b, slot_b) -> _PairState; the hysteresis memory of the
        # containment association (see SharedWorldVoxelizer.association).
        self._pairs: dict[tuple[int, int, int, int], _PairState] = {}
        self._frame = 0

    def _pair_passes(self, key: tuple[int, int, int, int], passed: bool, label: str, score: float) -> bool:
        """Update the hysteresis state of one cross-view pair; return whether it counts as merged."""
        state = self._pairs.get(key)
        if state is None:
            state = self._pairs[key] = _PairState()
        state.last_frame = self._frame
        if passed:
            state.agree += 1
            state.disagree = 0
            if not state.merged and state.agree >= self.voxelizer.merge_confirm_frames:
                state.merged = True
                logger.info("cross-view merge: %s view%d[slot %d] + view%d[slot %d] (containment %.2f)",
                            label, key[0], key[1], key[2], key[3], score)
        else:
            state.disagree += 1
            state.agree = 0
            if state.merged and state.disagree >= self.voxelizer.split_confirm_frames:
                state.merged = False
                logger.info("cross-view split: %s view%d[slot %d] | view%d[slot %d] (containment %.2f)",
                            label, key[0], key[1], key[2], key[3], score)
        return state.merged

    def _prune_pairs(self) -> None:
        if self._frame % 300 == 0:
            stale = [k for k, st in self._pairs.items() if self._frame - st.last_frame > 300]
            for k in stale:
                del self._pairs[k]

    def align(
        self,
        results: list[FrameResult],
        profiler: Any | None = None,
    ) -> tuple[list[MultiViewInstance], dict[str, int]]:
        observations: list[_VoxelObservation] = []
        started = time.perf_counter()
        self._frame += 1
        self._prune_pairs()
        containment = self.voxelizer.association == "containment"
        context = (
            profiler.stage("cross_view_voxelize", cuda=False)
            if profiler is not None
            else _NullContext()
        )
        # With one camera no cross-view matching can happen, so GPU-alignment
        # mode skips CPU voxel data entirely (the geometry stage did not even
        # copy it back). Multi-camera GPU mode still matches on CPU keys/coords.
        gpu_single_view = self.gpu_fusion and len(results) == 1
        with context:
            for view_index, result in enumerate(results):
                for instance in result.instances:
                    if gpu_single_view:
                        voxelized = self.voxelizer.observe_gpu_only(
                            view_index, result.frame.camera_name, instance
                        )
                    else:
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
                if containment:
                    valid = np.zeros(scores.shape, dtype=bool)
                    for row, index_a in enumerate(class_a):
                        obs_a = observations[index_a]
                        for col, index_b in enumerate(class_b):
                            obs_b = observations[index_b]
                            counters["num_cross_view_candidate_pairs"] += 1
                            ratio = self.voxelizer.containment_ratio(obs_a, obs_b)
                            scores[row, col] = ratio
                            key = (view_a, int(obs_a.instance.track_id), view_b, int(obs_b.instance.track_id))
                            valid[row, col] = self._pair_passes(
                                key, ratio >= self.voxelizer.min_containment_ratio, label, ratio
                            )
                else:
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
            if self.gpu_fusion:
                fused = self._fuse_groups_gpu(list(groups.values()), counters)
            for group_index, members in enumerate(groups.values(), start=1):
                if self.gpu_fusion:
                    break
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
                    # Union of the members' voxel bounds (the cross-frame tracker keys on the
                    # AABB centre, which unlike the visible-surface centroid is view-invariant).
                    bbox_min, bbox_max = self._union_bbox_world(members)
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
                    point_count=int(len(points_world)),
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

    def _fuse_groups_gpu(
        self,
        groups: list[list[_VoxelObservation]],
        counters: dict[str, int],
    ) -> list[MultiViewInstance]:
        """Cross-view fusion without moving any point cloud through the host.

        Single-member groups (the common case) alias the geometry backend's
        unique-representative slice. Multi-member groups reuse the exact voxel
        key union computed on CPU for matching (``np.unique`` on int64 keys) and
        gather the survivors on device. Centroids are medians computed on the
        GPU and copied back in one batched transfer.
        """
        assert torch is not None and self.device is not None
        fused: list[MultiViewInstance] = []
        clouds: list[Any] = []
        # Groups whose centroid could not be derived from per-record means
        # (legacy geometry backend without record centroids) fall back to a GPU
        # median + one batched readback below.
        need_gpu_centroid: list[int] = []
        for group_index, members in enumerate(groups, start=1):
            raw_point_count = 0
            for member in members:
                if member.instance.points_world is not None:
                    raw_point_count += int(len(member.instance.points_world))
                elif member.points_gpu is not None:
                    raw_point_count += int(member.points_gpu.shape[0])
            counters["num_fused_points_before_downsample"] += raw_point_count

            member_clouds = [
                member.points_gpu for member in members if member.points_gpu is not None
            ]
            member_list = None
            if len(member_clouds) == 0:
                fused_gpu = torch.empty((0, 3), dtype=torch.float32, device=self.device)
                count = 0
            elif len(member_clouds) == 1:
                fused_gpu = member_clouds[0]
                count = int(fused_gpu.shape[0])
            elif GPU_FUSION_DEDUP:
                # Key union on CPU (keys are already resident for matching);
                # the survivor gather stays on device.
                fused_keys = np.concatenate([member.keys for member in members], axis=0)
                _, keep = np.unique(fused_keys, return_index=True)
                keep.sort()
                # Pinned staging + non-blocking copy: a pageable ``.to()`` would
                # block the host on the (possibly contended) stream.
                keep_gpu = self._stage_index(keep)
                fused_gpu = torch.cat(member_clouds, dim=0).index_select(0, keep_gpu)
                count = int(fused_gpu.shape[0])
            else:
                # No kernels here: the bank staging concatenates the member
                # slices with plain D2D copies (copy engine, not SMs), which is
                # what keeps this stage cheap while SAM3 occupies the GPU.
                fused_gpu = None
                member_list = member_clouds
                count = int(sum(int(cloud.shape[0]) for cloud in member_clouds))
            counters["num_fused_points_after_downsample"] += count
            clouds.append(fused_gpu if (count and fused_gpu is not None) else None)

            # Centroid: count-weighted combination of the per-record means that
            # the geometry stage already shipped with its metadata. No GPU work
            # and no host wait here; under SAM3 contention every extra small
            # kernel + sync in this stage cost milliseconds.
            centroid = None
            member_centroids = [
                (member.instance.centroid_world, int(member.points_gpu.shape[0]))
                for member in members
                if member.points_gpu is not None and member.instance.centroid_world is not None
            ]
            if count and member_centroids and len(member_centroids) == len(member_clouds):
                weights = np.asarray([w for _, w in member_centroids], dtype=np.float64)
                stacked = np.stack([c for c, _ in member_centroids]).astype(np.float64)
                centroid = (stacked * weights[:, None]).sum(0) / max(1.0, weights.sum())
                centroid = centroid.astype(np.float32)
            elif count and fused_gpu is not None:
                need_gpu_centroid.append(len(fused))
            elif count:
                # Multi-member without record means and without a fused tensor:
                # fall back to the (deduplicating) gather so a centroid exists.
                fused_gpu = torch.cat(member_list, dim=0)
                member_list = None
                clouds[-1] = fused_gpu
                need_gpu_centroid.append(len(fused))

            bbox_min, bbox_max = self._union_bbox_world(members)
            group = MultiViewInstance(
                group_id=group_index,
                semantic_label=members[0].instance.label,
                members=[(member.camera_name, member.instance) for member in members],
                points_world=None,
                colors_rgb=np.empty((0, 3), dtype=np.uint8),
                centroid_world=centroid,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                point_count=count,
                fused_points_gpu=fused_gpu,
                fused_member_clouds_gpu=member_list,
            )
            for member in members:
                member.instance.multiview_group_id = group_index
            fused.append(group)

        # Fallback centroid readback (only when record means were unavailable).
        valid = [index for index in need_gpu_centroid if clouds[index] is not None]
        if valid:
            medians = torch.stack(
                [torch.quantile(clouds[index], 0.5, dim=0) for index in valid], dim=0
            )
            host = self._centroid_host
            if host is None or host.shape[0] < len(valid):
                host = torch.empty((max(8, 2 * len(valid)), 3), dtype=torch.float32, pin_memory=True)
                self._centroid_host = host
            host[: len(valid)].copy_(medians, non_blocking=True)
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(self.device))
            done.synchronize()
            centroids = host[: len(valid)].numpy()
            for row, index in enumerate(valid):
                fused[index].centroid_world = np.array(centroids[row], dtype=np.float32, copy=True)
        return fused

    def _union_bbox_world(self, members: list[_VoxelObservation]):
        """World-frame AABB (metres) of the union of the members' voxel boxes, or (None, None)."""
        boxes = [(m.bbox_min, m.bbox_max) for m in members if m.bbox_min is not None and m.bbox_max is not None]
        if not boxes:
            return None, None
        vmin = np.min(np.stack([b[0] for b in boxes], axis=0), axis=0).astype(np.float32)
        vmax = np.max(np.stack([b[1] for b in boxes], axis=0), axis=0).astype(np.float32)
        size = np.float32(self.voxelizer.voxel_size_m)
        return (self.voxelizer.origin_world + vmin * size,
                self.voxelizer.origin_world + (vmax + 1.0) * size)

    def _stage_index(self, keep: np.ndarray):
        """Upload an int64 index array through a reusable pinned buffer (non-blocking)."""
        assert torch is not None and self.device is not None
        n = int(keep.shape[0])
        host = getattr(self, "_index_host", None)
        if host is None or host.shape[0] < n:
            host = torch.empty(max(4096, 2 * n), dtype=torch.int64, pin_memory=True)
            self._index_host = host
            self._index_dev = torch.empty(host.shape[0], dtype=torch.int64, device=self.device)
        host[:n].numpy()[:] = keep
        dev = self._index_dev[:n]
        dev.copy_(host[:n], non_blocking=True)
        # The staging buffer is reused next frame; the copy has completed long
        # before then (the frame ends with a stream fence).
        return dev


class _ChamferWorkspace:
    """Persistent unique-cloud GPU banks plus reusable batched Chamfer buffers.

    Current-frame fused clouds are packed/uploaded exactly once.  The previous
    frame stays resident on CUDA and becomes the temporal reference by swapping
    the two banks at frame end.  Candidate pairs carry only (current, previous)
    indices; pair-shaped tensors are populated by device-to-device gathers, so a
    cloud participating in several candidates is never uploaded several times.
    """

    def __init__(
        self,
        device: str,
        max_pairs: int,
        max_instances: int,
        max_workspace_mb: float = 256.0,
    ) -> None:
        self.device = torch.device(device) if torch is not None else None
        self.max_pairs = max(1, int(max_pairs))
        self.max_instances = max(1, int(max_instances))
        self.max_workspace_bytes = max(16, int(max_workspace_mb)) * 1024 * 1024
        self.point_capacity = 0

        self.host_current = None
        self.host_previous = None
        self.host_current_count = None
        self.host_previous_count = None
        self.gpu_current = None
        self.gpu_previous = None
        self.gpu_current_count = None
        self.gpu_previous_count = None

        self.pair_a = None
        self.pair_b = None
        self.pair_count_a = None
        self.pair_count_b = None
        self.host_pair_current = None
        self.host_pair_previous = None
        self.gpu_pair_current = None
        self.gpu_pair_previous = None

        self.index_a = None
        self.index_b = None
        self.origin = None
        self.best_b_sq = None
        self.total_a = None

        self.current_size = 0
        self.previous_size = 0
        self.current_counts_np = np.empty((0,), dtype=np.int64)
        self.previous_counts_np = np.empty((0,), dtype=np.int64)
        # Two reusable events follow the two ping-pong GPU banks. They provide a
        # stream dependency from the tracker-owner H2D upload to the
        # ScenePredictor adapter without a host synchronize.
        self.current_ready_event = None
        self.previous_ready_event = None

    @property
    def cuda_enabled(self) -> bool:
        return bool(
            torch is not None
            and self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )

    @staticmethod
    def _grow_points(required: int, current: int) -> int:
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
                f"capacity={self.max_pairs}."
            )

    def _check_instance_count(self, count: int) -> None:
        if int(count) > self.max_instances:
            raise RuntimeError(
                "Cross-frame instance count exceeded the strict configured "
                f"multi-view slot bound: got {count}, capacity={self.max_instances}."
            )

    def reserve_points(self, points: int) -> None:
        if self.cuda_enabled and points > 0:
            self._ensure_point_capacity(int(points))

    def _allocate_bank(self, points: int):
        assert torch is not None and self.device is not None
        pin = bool(torch.cuda.is_available())
        host = torch.empty(
            (self.max_instances, points, 3),
            dtype=torch.float32,
            device="cpu",
            pin_memory=pin,
        )
        host_count = torch.empty(
            (self.max_instances,), dtype=torch.long, device="cpu", pin_memory=pin
        )
        gpu = torch.empty(
            (self.max_instances, points, 3), dtype=torch.float32, device=self.device
        )
        gpu_count = torch.empty(
            (self.max_instances,), dtype=torch.long, device=self.device
        )
        return host, host_count, gpu, gpu_count

    def _ensure_point_capacity(self, required: int) -> None:
        if not self.cuda_enabled:
            return
        if required <= self.point_capacity and self.gpu_current is not None:
            return

        old_previous = self.gpu_previous
        old_previous_count = self.gpu_previous_count
        old_capacity = self.point_capacity
        previous_size = self.previous_size

        new_points = self._grow_points(required, self.point_capacity)
        (
            self.host_current,
            self.host_current_count,
            self.gpu_current,
            self.gpu_current_count,
        ) = self._allocate_bank(new_points)
        (
            self.host_previous,
            self.host_previous_count,
            self.gpu_previous,
            self.gpu_previous_count,
        ) = self._allocate_bank(new_points)

        assert torch is not None and self.device is not None
        self.pair_a = torch.empty(
            (self.max_pairs, new_points, 3), dtype=torch.float32, device=self.device
        )
        self.pair_b = torch.empty(
            (self.max_pairs, new_points, 3), dtype=torch.float32, device=self.device
        )
        self.pair_count_a = torch.empty(
            (self.max_pairs,), dtype=torch.long, device=self.device
        )
        self.pair_count_b = torch.empty(
            (self.max_pairs,), dtype=torch.long, device=self.device
        )

        pin = bool(torch.cuda.is_available())
        self.host_pair_current = torch.empty(
            (self.max_pairs,), dtype=torch.long, device="cpu", pin_memory=pin
        )
        self.host_pair_previous = torch.empty(
            (self.max_pairs,), dtype=torch.long, device="cpu", pin_memory=pin
        )
        self.gpu_pair_current = torch.empty(
            (self.max_pairs,), dtype=torch.long, device=self.device
        )
        self.gpu_pair_previous = torch.empty(
            (self.max_pairs,), dtype=torch.long, device=self.device
        )

        self.origin = torch.empty(
            (self.max_pairs, 1, 3), dtype=torch.float32, device=self.device
        )
        self.best_b_sq = torch.empty(
            (self.max_pairs, new_points), dtype=torch.float32, device=self.device
        )
        self.total_a = torch.empty(
            (self.max_pairs,), dtype=torch.float32, device=self.device
        )
        self.index_a = torch.arange(new_points, device=self.device, dtype=torch.long)
        self.index_b = torch.arange(new_points, device=self.device, dtype=torch.long)
        self.point_capacity = new_points

        # Point-capacity growth is rare (normally covered by startup reserve).
        # Preserve the resident previous frame with one D2D copy so growth does
        # not change temporal matching semantics.
        if old_previous is not None and previous_size > 0 and old_capacity > 0:
            keep_points = min(old_capacity, new_points)
            self.gpu_previous[:previous_size, :keep_points].copy_(
                old_previous[:previous_size, :keep_points]
            )
            if old_previous_count is not None:
                self.gpu_previous_count[:previous_size].copy_(
                    old_previous_count[:previous_size]
                )

    def _ensure_ready_events(self) -> None:
        if not self.cuda_enabled or self.current_ready_event is not None:
            return
        assert torch is not None and self.device is not None
        with torch.cuda.device(self.device):
            self.current_ready_event = torch.cuda.Event(
                enable_timing=False, blocking=False
            )
            self.previous_ready_event = torch.cuda.Event(
                enable_timing=False, blocking=False
            )

    def current_cloud_views(self):
        """Return exact current-cloud CUDA views plus their upload-ready event."""
        if not self.cuda_enabled or self.gpu_current is None:
            return [], None
        views = [
            self.gpu_current[index, : int(count)]
            for index, count in enumerate(self.current_counts_np)
        ]
        return views, self.current_ready_event

    def stage_current(self, clouds: list[np.ndarray]) -> None:
        """Pack each unique current fused cloud once and upload one bank."""
        if not self.cuda_enabled:
            return
        self._check_instance_count(len(clouds))
        self._ensure_ready_events()
        counts = np.asarray([len(cloud) for cloud in clouds], dtype=np.int64)
        max_points = int(counts.max(initial=0))
        self._ensure_point_capacity(max(1, max_points))

        assert self.host_current is not None and self.host_current_count is not None
        assert self.gpu_current is not None and self.gpu_current_count is not None
        current_size = len(clouds)
        host_np = self.host_current.numpy()
        for index, cloud in enumerate(clouds):
            count = int(counts[index])
            if count:
                host_np[index, :count] = np.asarray(cloud, dtype=np.float32)
        if current_size:
            self.host_current_count[:current_size].numpy()[:] = counts
            used_points = max(1, max_points)
            host = self.host_current[:current_size, :used_points]
            gpu = self.gpu_current[:current_size, :used_points]
            gpu.copy_(host, non_blocking=host.is_pinned())
            host_count = self.host_current_count[:current_size]
            self.gpu_current_count[:current_size].copy_(
                host_count, non_blocking=host_count.is_pinned()
            )

        self.current_size = current_size
        self.current_counts_np = counts
        if current_size and self.current_ready_event is not None:
            assert self.device is not None
            self.current_ready_event.record(torch.cuda.current_stream(self.device))

    def stage_current_gpu(self, clouds: list[Any], counts: list[int]) -> None:
        """Stage CUDA-resident fused clouds into the current bank by D2D copy.

        Same layout/semantics as ``stage_current`` but without the host pack
        and H2D upload; the counts are already known on the host (they came
        back with the geometry metadata), so only the tiny count vector is
        uploaded.
        """
        if not self.cuda_enabled:
            return
        self._check_instance_count(len(clouds))
        self._ensure_ready_events()
        counts_np = np.asarray(counts, dtype=np.int64)
        max_points = int(counts_np.max(initial=0))
        self._ensure_point_capacity(max(1, max_points))
        assert self.gpu_current is not None and self.gpu_current_count is not None
        assert self.host_current_count is not None
        current_size = len(clouds)
        for index, (cloud, count) in enumerate(zip(clouds, counts)):
            if not count:
                continue
            if isinstance(cloud, (list, tuple)):
                # Multi-camera group: concatenate member slices into the row.
                offset = 0
                for part in cloud:
                    n = int(part.shape[0])
                    if n:
                        self.gpu_current[index, offset : offset + n].copy_(part, non_blocking=True)
                        offset += n
            else:
                self.gpu_current[index, :count].copy_(cloud[:count], non_blocking=True)
        if current_size:
            host_count = self.host_current_count[:current_size]
            host_count.numpy()[:] = counts_np
            self.gpu_current_count[:current_size].copy_(
                host_count, non_blocking=host_count.is_pinned()
            )
        self.current_size = current_size
        self.current_counts_np = counts_np
        if current_size and self.current_ready_event is not None:
            assert self.device is not None
            self.current_ready_event.record(torch.cuda.current_stream(self.device))

    def promote_current(self) -> None:
        """Make the just-uploaded current bank the next frame's resident previous."""
        if not self.cuda_enabled:
            return
        self.host_current, self.host_previous = self.host_previous, self.host_current
        self.host_current_count, self.host_previous_count = (
            self.host_previous_count,
            self.host_current_count,
        )
        self.gpu_current, self.gpu_previous = self.gpu_previous, self.gpu_current
        self.gpu_current_count, self.gpu_previous_count = (
            self.gpu_previous_count,
            self.gpu_current_count,
        )
        self.current_ready_event, self.previous_ready_event = (
            self.previous_ready_event,
            self.current_ready_event,
        )
        self.previous_size = self.current_size
        self.previous_counts_np = self.current_counts_np
        self.current_size = 0
        self.current_counts_np = np.empty((0,), dtype=np.int64)

    def clear_previous(self) -> None:
        self.previous_size = 0
        self.current_size = 0
        self.previous_counts_np = np.empty((0,), dtype=np.int64)
        self.current_counts_np = np.empty((0,), dtype=np.int64)

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

    def compute_indices(self, candidate_indices: np.ndarray) -> np.ndarray:
        """Compute Chamfer for candidate bank indices with no repeated H2D clouds."""
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        if candidate_indices.size == 0:
            return np.empty((0,), dtype=np.float32)
        if candidate_indices.ndim != 2 or candidate_indices.shape[1] != 2:
            raise ValueError("candidate_indices must have shape [P, 2]")
        pair_count = int(candidate_indices.shape[0])
        self._check_pair_count(pair_count)
        if not self.cuda_enabled:
            raise RuntimeError("compute_indices requires CUDA")
        if self.current_size <= 0 or self.previous_size <= 0:
            return np.full((pair_count,), np.inf, dtype=np.float32)

        rows = candidate_indices[:, 0]
        cols = candidate_indices[:, 1]
        if np.any(rows < 0) or np.any(rows >= self.current_size):
            raise IndexError("Current Chamfer candidate index is out of range")
        if np.any(cols < 0) or np.any(cols >= self.previous_size):
            raise IndexError("Previous Chamfer candidate index is out of range")

        counts_a_np = self.current_counts_np[rows]
        counts_b_np = self.previous_counts_np[cols]
        max_a = int(counts_a_np.max(initial=0))
        max_b = int(counts_b_np.max(initial=0))
        if max_a == 0 or max_b == 0:
            return np.full((pair_count,), np.inf, dtype=np.float32)

        assert self.host_pair_current is not None and self.host_pair_previous is not None
        assert self.gpu_pair_current is not None and self.gpu_pair_previous is not None
        assert self.gpu_current is not None and self.gpu_previous is not None
        assert self.gpu_current_count is not None and self.gpu_previous_count is not None
        assert self.pair_a is not None and self.pair_b is not None
        assert self.pair_count_a is not None and self.pair_count_b is not None

        self.host_pair_current[:pair_count].numpy()[:] = rows
        self.host_pair_previous[:pair_count].numpy()[:] = cols
        gpu_rows = self.gpu_pair_current[:pair_count]
        gpu_cols = self.gpu_pair_previous[:pair_count]
        host_rows = self.host_pair_current[:pair_count]
        host_cols = self.host_pair_previous[:pair_count]
        gpu_rows.copy_(host_rows, non_blocking=host_rows.is_pinned())
        gpu_cols.copy_(host_cols, non_blocking=host_cols.is_pinned())

        # Gather the full preallocated point dimension so ``out`` remains a
        # contiguous tensor. Padding is ignored by the count masks below.
        torch.index_select(
            self.gpu_current[: self.current_size],
            0,
            gpu_rows,
            out=self.pair_a[:pair_count],
        )
        torch.index_select(
            self.gpu_previous[: self.previous_size],
            0,
            gpu_cols,
            out=self.pair_b[:pair_count],
        )
        pair_a = self.pair_a[:pair_count, :max_a]
        pair_b = self.pair_b[:pair_count, :max_b]
        torch.index_select(
            self.gpu_current_count[: self.current_size],
            0,
            gpu_rows,
            out=self.pair_count_a[:pair_count],
        )
        torch.index_select(
            self.gpu_previous_count[: self.previous_size],
            0,
            gpu_cols,
            out=self.pair_count_b[:pair_count],
        )

        result = self._symmetric_gpu(
            pair_a,
            pair_b,
            self.pair_count_a[:pair_count],
            self.pair_count_b[:pair_count],
            max_a=max_a,
            max_b=max_b,
        )
        return result.detach().cpu().numpy()

    @staticmethod
    def compute_cpu(
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


@dataclass
class _Track:
    track_id: int
    label: str
    bbox_min: np.ndarray        # last observed world AABB (m)
    bbox_max: np.ndarray
    created: int
    hits: int = 1
    misses: int = 0
    last_seen_s: float = 0.0    # frame time of the last observation
    coast_s: float = 0.0        # time unobserved


class CrossFrameAligner:
    """Persistent global ids by AABB containment between consecutive observations.

    No motion model: a track is its last observed world AABB. Oldest track first, every track
    takes the unmatched same-class observation with the highest affinity, where the affinity is
    the AABB containment ratio (intersection volume over the smaller box) and reaches 1.0 when
    one box contains the other's centre (expanded by ``centre_margin_m``). At 30 fps a walking
    person moves ~5 cm per frame, far less than a body box, so consecutive boxes overlap; a
    mask that suddenly grows or shrinks keeps the smaller box inside the larger one. Unmatched
    tracks keep their id for ``max_age_s`` (a missed detection or a brief occlusion), unmatched
    observations start a new track; when two groups merge the survivor keeps the oldest id.

    The persistent CUDA bank of fused clouds is still filled here (``_stage_current_bank``): the
    ScenePredictor adapter consumes those views; only the Chamfer matching was retired.
    """

    def __init__(self, config, *, num_views: int | None = None) -> None:
        cfg = config.get("cross_frame_alignment", {})
        self.min_containment_ratio = float(cfg.get("min_containment_ratio",
                                                   config.get("shared_voxel_grid", {}).get("min_containment_ratio", 0.5)))
        self.centre_margin_m = float(cfg.get("centre_margin_m", 0.15))
        self.max_age_s = float(cfg.get("max_age_s", 0.5))
        self.nominal_dt_s = float(cfg.get("nominal_dt_s", 1.0 / 30.0))
        device = str(config.runtime.get("device", "cuda"))
        workspace_mb = float(cfg.get("chamfer_max_workspace_mb", 256.0))
        pair_capacity = max_cross_frame_candidate_pairs(config, num_views=num_views)
        instance_capacity = max_cross_frame_instances(config, num_views=num_views)
        self.workspace = _ChamferWorkspace(
            device,
            max_pairs=pair_capacity,
            max_instances=instance_capacity,
            max_workspace_mb=workspace_mb,
        )
        self.workspace.reserve_points(int(cfg.get("chamfer_preallocate_points", 0)))
        self.previous: list[MultiViewInstance] = []
        self.next_global_track_id = 1
        self.tracks: dict[int, _Track] = {}
        self._frame = 0
        self._time_s = 0.0
        self._last_stamp_s: float | None = None
        self._warned_no_box = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _box(item: MultiViewInstance):
        """World AABB of a fused instance (metres) or None."""
        if item.bbox_min is not None and item.bbox_max is not None:
            return np.asarray(item.bbox_min, np.float64), np.asarray(item.bbox_max, np.float64)
        if item.centroid_world is not None:                         # degenerate box at the centroid
            c = np.asarray(item.centroid_world, np.float64)
            return c.copy(), c.copy()
        return None

    @staticmethod
    def _containment(lo_a, hi_a, lo_b, hi_b) -> float:
        """Intersection volume over the smaller box volume for two world AABBs (metres)."""
        inter = np.prod(np.maximum(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b), 0.0))
        if inter <= 0.0:
            return 0.0
        vol = min(np.prod(np.maximum(hi_a - lo_a, 1e-6)), np.prod(np.maximum(hi_b - lo_b, 1e-6)))
        return float(inter / vol)

    def _affinity(self, track: _Track, lo_o: np.ndarray, hi_o: np.ndarray) -> float:
        """Containment of the track's last box with the observation; 1.0 when one box contains
        the other's centre (expanded by centre_margin_m)."""
        ratio = self._containment(track.bbox_min, track.bbox_max, lo_o, hi_o)
        g = self.centre_margin_m
        centre_t = 0.5 * (track.bbox_min + track.bbox_max)
        centre_o = 0.5 * (lo_o + hi_o)
        if np.all(centre_t >= lo_o - g) and np.all(centre_t <= hi_o + g):
            return 1.0
        if np.all(centre_o >= track.bbox_min - g) and np.all(centre_o <= track.bbox_max + g):
            return 1.0
        return ratio

    def _assign_new_id(self, instance: MultiViewInstance, box=None) -> None:
        instance.global_track_id = self.next_global_track_id
        self.next_global_track_id += 1
        if box is not None:
            self.tracks[instance.global_track_id] = _Track(
                track_id=instance.global_track_id, label=instance.semantic_label,
                bbox_min=box[0].copy(), bbox_max=box[1].copy(), created=self._frame, last_seen_s=self._time_s,
            )

    def _update_track(self, track: _Track, item: MultiViewInstance, box) -> None:
        track.bbox_min, track.bbox_max = box[0].copy(), box[1].copy()
        track.misses = 0
        track.coast_s = 0.0
        track.last_seen_s = self._time_s
        track.hits += 1
        item.global_track_id = track.track_id

    def _stage_current_bank(
        self,
        current: list[MultiViewInstance],
        profiler: Any | None,
    ) -> None:
        if not self.workspace.cuda_enabled:
            return
        context = (
            profiler.stage("cross_frame_cloud_upload", cuda=True)
            if profiler is not None
            else _NullContext()
        )
        with context:
            gpu_mode = bool(current) and all(
                item.fused_points_gpu is not None or item.fused_member_clouds_gpu is not None
                for item in current
            )
            if gpu_mode:
                # GPU-alignment mode: fused clouds never left the device.
                self.workspace.stage_current_gpu(
                    [
                        item.fused_member_clouds_gpu
                        if item.fused_member_clouds_gpu is not None
                        else item.fused_points_gpu
                        for item in current
                    ],
                    [int(item.point_count) for item in current],
                )
            else:
                self.workspace.stage_current([item.points_world for item in current])

        # Export zero-copy views of the bank. ScenePredictor reuses these views instead of
        # uploading the same fused cloud a second time.
        views, ready_event = self.workspace.current_cloud_views()
        if len(views) != len(current):
            raise RuntimeError(
                "Cross-frame GPU bank view count does not match current fused "
                f"instances: {len(views)} != {len(current)}"
            )
        for item, view in zip(current, views):
            item.points_world_gpu = view
            item.points_world_gpu_ready_event = ready_event
            if item.fused_member_clouds_gpu is not None:
                item.fused_points_gpu = view

    # ------------------------------------------------------------------ main
    def align(
        self,
        current: list[MultiViewInstance],
        profiler: Any | None = None,
        stamp_ns: int | None = None,
    ) -> dict[str, int]:
        started = time.perf_counter()
        counters = {
            "num_cross_frame_candidate_pairs": 0,
            "num_cross_frame_matches": 0,
            "num_cross_frame_new_ids": 0,
            "num_cross_frame_coasting": 0,
            "num_cross_frame_dropped": 0,
        }
        self._frame += 1
        stamp_s = None if not stamp_ns else stamp_ns * 1e-9
        if stamp_s is not None and self._last_stamp_s is not None and 0.0 < stamp_s - self._last_stamp_s < 5.0:
            dt = stamp_s - self._last_stamp_s
        else:
            dt = self.nominal_dt_s
        if stamp_s is not None:
            self._last_stamp_s = stamp_s
        self._time_s += dt
        if current:
            self._stage_current_bank(current, profiler)

        gate_context = (
            profiler.stage("cross_frame_gate", cuda=False)
            if profiler is not None
            else _NullContext()
        )
        with gate_context:
            boxes = [self._box(item) for item in current]
            if any(b is None for b in boxes) and not self._warned_no_box:
                self._warned_no_box = True
                logger.warning("cross-frame: fused instance without AABB/centroid; it gets a fresh id every frame")
            free = set(range(len(current)))
            matched_tracks: set[int] = set()
            # Oldest track first ("oldest id wins" when two groups merge into one observation).
            for track in sorted(self.tracks.values(), key=lambda t: (t.created, t.track_id)):
                best, best_a = None, self.min_containment_ratio
                for i in free:
                    if boxes[i] is None or current[i].semantic_label != track.label:
                        continue
                    counters["num_cross_frame_candidate_pairs"] += 1
                    a = self._affinity(track, *boxes[i])
                    if a >= best_a:
                        best, best_a = i, a
                if best is None:
                    continue
                free.discard(best)
                matched_tracks.add(track.track_id)
                self._update_track(track, current[best], boxes[best])
                counters["num_cross_frame_matches"] += 1
            for i in sorted(free):
                why = ""
                if boxes[i] is not None and logger.isEnabledFor(logging.INFO):
                    same = [t for t in self.tracks.values() if t.label == current[i].semantic_label and t.track_id not in matched_tracks]
                    if same:
                        near = max(same, key=lambda t: self._affinity(t, *boxes[i]))
                        size = np.round(boxes[i][1] - boxes[i][0], 2).tolist()
                        why = (f"; best unmatched id {near.track_id}: affinity {self._affinity(near, *boxes[i]):.2f} < {self.min_containment_ratio:g}, "
                               f"unseen {self._time_s - near.last_seen_s:.2f} s, obs box size {size}, dt {dt * 1e3:.0f} ms")
                self._assign_new_id(current[i], boxes[i])
                counters["num_cross_frame_new_ids"] += 1
                centre = None if boxes[i] is None else np.round(0.5 * (boxes[i][0] + boxes[i][1]), 3).tolist()
                logger.info("cross-frame: new id %d for %s at %s%s", current[i].global_track_id, current[i].semantic_label, centre, why)
            for track_id in list(self.tracks):
                track = self.tracks[track_id]
                if track_id in matched_tracks or track.created == self._frame:
                    continue
                track.misses += 1
                track.coast_s = self._time_s - track.last_seen_s
                if track.coast_s > self.max_age_s:
                    del self.tracks[track_id]
                    counters["num_cross_frame_dropped"] += 1
                    logger.info("cross-frame: id %d (%s) dropped after %.2f s / %d frames unobserved", track_id, track.label,
                                track.coast_s, track.misses)
                    continue
                counters["num_cross_frame_coasting"] += 1
            for instance in current:
                self._propagate_id_to_members(instance)

        self.previous = current
        if current and self.workspace.cuda_enabled:
            self.workspace.promote_current()
        elif not current:
            self.workspace.clear_previous()
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
