"""Which instances of the cameras are one object, and which object it was in the frame before.

CrossViewAligner   same time, different cameras: surface gap under the per-class instance count
                   (cross_view_gap.py), then the fusion of the grouped clouds.
CrossFrameAligner  different times: persistent global ids by the nearest world box, and the CUDA bank
                   in which the fused clouds of the current and the previous frame live.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cross_view_gap import GapAssociation, GapSettings
from .data_types import FrameResult, MultiViewInstance, ProcessedInstance
from .slots import max_cross_frame_instances, tracked_prompt_capacities

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


logger = logging.getLogger(__name__)


def gpu_alignment_enabled(config) -> bool:
    """GPU-resident cross-view fusion and cloud bank (production default).

    Requires the CUDA geometry backend and visualization off (visualization
    consumes NumPy clouds). ``postprocess.gpu_alignment: false`` restores the
    legacy path where voxel representatives are copied to the host, fused in
    NumPy and uploaded again into the cloud bank.
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
    # CUDA view of the same unique representatives (GPU-alignment mode).
    points_gpu: Any | None = None


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
        # The cross-view association (cross_view_gap.py). Earlier tests (world-box containment, voxel coverage)
        # were removed; a config that still names one is refused instead of silently getting another test.
        association = str(cfg.get("association", "gap"))
        if association != "gap":
            raise ValueError(f"shared_voxel_grid.association {association!r} was removed; the only association is 'gap'")
        self.gap_settings = GapSettings.from_config(cfg.get("gap"))

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


class CrossViewAligner:
    """Per class: surface gap under the instance count (cross_view_gap.py) -> groups with at most one
    observation per camera -> fusion of each group's clouds."""

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
        try:
            capacities = dict(tracked_prompt_capacities(config))
        except (AttributeError, KeyError, ValueError):
            capacities = {}                          # no prompts configured: no count to hold
        self.gap = GapAssociation(self.voxelizer.gap_settings, self.voxelizer.voxel_size_m, capacities)
        self._last_stamp_ns = 0

    def _same_object_edges(self, observations: list[_VoxelObservation], results: list[FrameResult], counters: dict[str, int]):
        """-> edges (gap m, index a, index b): the two observations are the same object."""
        stamp_ns = max((int(getattr(result.frame, "timestamp_ns", 0) or 0) for result in results), default=0)
        dt_s = (stamp_ns - self._last_stamp_ns) * 1.0e-9 if stamp_ns and self._last_stamp_ns else 0.0
        self._last_stamp_ns = stamp_ns
        if not 0.0 < dt_s < 1.0:
            dt_s = 1.0 / 30.0
        edges = self.gap.associate(observations, dt_s)
        counters["num_cross_view_candidate_pairs"] += self.gap.candidate_pairs
        counters["num_cross_view_gap_measurements"] = self.gap.measured_pairs
        counters["num_cross_view_forced_merges"] = self.gap.forced_merges
        counters["num_cross_view_over_capacity"] = self.gap.over_capacity
        counters["num_cross_view_matches"] = len(edges)
        return edges

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
        # With one camera no cross-view matching can happen, so GPU-alignment
        # mode skips CPU voxel data entirely (the geometry stage did not even
        # copy it back). Multi-camera GPU mode matches on the voxel indices copied to the host.
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

        match_started = time.perf_counter()
        edges = self._same_object_edges(observations, results, counters)
        if profiler is not None:
            profiler.record("cross_view_match", 1000.0 * (time.perf_counter() - match_started))

        # The edges unite at most one observation per camera.
        parent = list(range(len(observations)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for _, index_a, index_b in edges:
            parent[find(index_b)] = find(index_a)

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
        unique-representative slice. The clouds of a multi-member group are
        concatenated when the bank is staged (no cross-view voxel dedup: the
        scene-flow stage dedups on its own grid). Centroids come from the
        per-record means of the geometry stage.
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


class _CloudBank:
    """The fused clouds of the current and of the previous frame, resident on CUDA.

    The current frame's clouds are packed/uploaded (or copied device to device) exactly once into
    one of two banks; the scene-flow stage reads them as zero-copy views. At frame end the banks
    swap, so the views handed out for the previous frame stay valid for one more frame.
    """

    def __init__(self, device: str, max_instances: int) -> None:
        self.device = torch.device(device) if torch is not None else None
        self.max_instances = max(1, int(max_instances))
        self.point_capacity = 0

        self.host_current = None
        self.host_previous = None
        self.host_current_count = None
        self.host_previous_count = None
        self.gpu_current = None
        self.gpu_previous = None
        self.gpu_current_count = None
        self.gpu_previous_count = None

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

        self.point_capacity = new_points

        # Point-capacity growth is rare (normally covered by startup reserve).
        # Preserve the resident previous frame with one D2D copy.
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
    """Persistent global ids by the nearest world box.

    A track is its last observed world AABB; there is no motion model. A track and an observation of
    its class can be matched when the containment of the boxes (intersection volume over the smaller
    box) reaches ``min_containment_ratio`` or one box holds the other's centre, expanded by a margin.
    The margin is ``centre_margin_m`` after one nominal frame and grows with the time since the track
    was seen (a speed limit of centre_margin_m / nominal_dt_s, 4.5 m/s), so a lower frame rate or an
    object hidden for a moment does not cost the id. Among the possible matches: as many as possible,
    with the least total distance between the box centres (Hungarian); of two tracks at the same
    distance the older one wins, so the survivor keeps the oldest id when two groups merge. Unmatched
    tracks keep their id for ``max_age_s``, unmatched observations start a new track.

    The CUDA bank of fused clouds is filled here (``_stage_current_bank``): the ScenePredictor adapter
    consumes its views.
    """

    def __init__(self, config, *, num_views: int | None = None) -> None:
        cfg = config.get("cross_frame_alignment", {})
        assignment = str(cfg.get("assignment", "nearest"))
        if assignment != "nearest":
            raise ValueError(f"cross_frame_alignment.assignment {assignment!r} was removed; the only assignment is 'nearest'")
        self.min_containment_ratio = float(cfg.get("min_containment_ratio", 0.5))
        self.centre_margin_m = float(cfg.get("centre_margin_m", 0.15))
        self.max_age_s = float(cfg.get("max_age_s", 0.5))
        self.nominal_dt_s = float(cfg.get("nominal_dt_s", 1.0 / 30.0))
        self.bank = _CloudBank(str(config.runtime.get("device", "cuda")), max_cross_frame_instances(config, num_views=num_views))
        self.bank.reserve_points(int(cfg.get("bank_preallocate_points", 0)))
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
        the other's centre (expanded by the margin)."""
        ratio = self._containment(track.bbox_min, track.bbox_max, lo_o, hi_o)
        # the margin is a speed: it grows with the time since the track was seen
        g = self.centre_margin_m * max(1.0, (self._time_s - track.last_seen_s) / self.nominal_dt_s)
        centre_t = 0.5 * (track.bbox_min + track.bbox_max)
        centre_o = 0.5 * (lo_o + hi_o)
        if np.all(centre_t >= lo_o - g) and np.all(centre_t <= hi_o + g):
            return 1.0
        if np.all(centre_o >= track.bbox_min - g) and np.all(centre_o <= track.bbox_max + g):
            return 1.0
        return ratio

    def _matches(self, by_age: list[_Track], current: list[MultiViewInstance], boxes, counters):
        """-> (track, index into current) of every match."""
        matches = []
        for label in {track.label for track in by_age}:
            tracks = [track for track in by_age if track.label == label]
            indices = [i for i, item in enumerate(current) if boxes[i] is not None and item.semantic_label == label]
            if not indices:
                continue
            counters["num_cross_frame_candidate_pairs"] += len(tracks) * len(indices)
            # a possible match is worth more than any distance: as many matches as possible, then the nearest
            worth = np.zeros((len(tracks), len(indices)))
            for row, track in enumerate(tracks):
                centre = 0.5 * (track.bbox_min + track.bbox_max)
                for col, i in enumerate(indices):
                    if self._affinity(track, *boxes[i]) >= self.min_containment_ratio:
                        distance = float(np.linalg.norm(centre - 0.5 * (boxes[i][0] + boxes[i][1])))
                        worth[row, col] = distance + 1.0e-6 * row - 1.0e3
            if len(tracks) == 1 or len(indices) == 1:
                row, col = np.unravel_index(int(np.argmin(worth)), worth.shape)
                pairs = [(int(row), int(col))]
            else:
                pairs = zip(*linear_sum_assignment(worth))
            matches.extend((tracks[row], indices[col]) for row, col in pairs if worth[row, col] < 0.0)
        return matches

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
        if not self.bank.cuda_enabled:
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
                self.bank.stage_current_gpu(
                    [
                        item.fused_member_clouds_gpu
                        if item.fused_member_clouds_gpu is not None
                        else item.fused_points_gpu
                        for item in current
                    ],
                    [int(item.point_count) for item in current],
                )
            else:
                self.bank.stage_current([item.points_world for item in current])

        # Export zero-copy views of the bank. ScenePredictor reuses these views instead of
        # uploading the same fused cloud a second time.
        views, ready_event = self.bank.current_cloud_views()
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
            by_age = sorted(self.tracks.values(), key=lambda t: (t.created, t.track_id))
            for track, index in self._matches(by_age, current, boxes, counters):
                free.discard(index)
                matched_tracks.add(track.track_id)
                self._update_track(track, current[index], boxes[index])
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

        if current and self.bank.cuda_enabled:
            self.bank.promote_current()
        elif not current:
            self.bank.clear_previous()
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
