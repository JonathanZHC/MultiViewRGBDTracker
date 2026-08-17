from __future__ import annotations

from time import perf_counter
from threading import Lock

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None


if wp is not None:

    @wp.kernel
    def _fused_geometry_kernel(
        # Packed [ray_x, ray_y, z, record_index] tuples. ray_x/ray_y are
        # precomputed from intrinsics on CPU and therefore avoid per-point
        # integer conversion/division on the GPU.
        samples: wp.array(dtype=wp.float32),
        record_views: wp.array(dtype=wp.int32),
        transforms: wp.array(dtype=wp.float32),
        transform_valid: wp.array(dtype=wp.int32),
        origin_world: wp.array(dtype=wp.float32),
        inv_voxel_size: float,
        points_camera: wp.array(dtype=wp.float32),
        points_world: wp.array(dtype=wp.float32),
        point_records: wp.array(dtype=wp.int32),
        voxel_coords: wp.array(dtype=wp.int32),
        voxel_keys: wp.array(dtype=wp.int64),
    ):
        i = wp.tid()
        s = i * 4
        ray_x = samples[s + 0]
        ray_y = samples[s + 1]
        z = samples[s + 2]
        record_index = wp.int32(samples[s + 3])
        view_index = record_views[record_index]

        cx = ray_x * z
        cy = ray_y * z
        cz = z

        p = i * 3
        points_camera[p + 0] = cx
        points_camera[p + 1] = cy
        points_camera[p + 2] = cz

        wx = cx
        wy = cy
        wz = cz
        if transform_valid[view_index] != 0:
            t = view_index * 12
            wx = (
                cx * transforms[t + 0]
                + cy * transforms[t + 1]
                + cz * transforms[t + 2]
                + transforms[t + 3]
            )
            wy = (
                cx * transforms[t + 4]
                + cy * transforms[t + 5]
                + cz * transforms[t + 6]
                + transforms[t + 7]
            )
            wz = (
                cx * transforms[t + 8]
                + cy * transforms[t + 9]
                + cz * transforms[t + 10]
                + transforms[t + 11]
            )

        points_world[p + 0] = wx
        points_world[p + 1] = wy
        points_world[p + 2] = wz
        point_records[i] = record_index

        vx = wp.int32(wp.floor((wx - origin_world[0]) * inv_voxel_size))
        vy = wp.int32(wp.floor((wy - origin_world[1]) * inv_voxel_size))
        vz = wp.int32(wp.floor((wz - origin_world[2]) * inv_voxel_size))
        voxel_coords[p + 0] = vx
        voxel_coords[p + 1] = vy
        voxel_coords[p + 2] = vz

        # 21 signed-ish bits per coordinate after a positive bias.  The CPU
        # compatibility materializer validates the range before exposing keys.
        sx = wp.int64(vx) + wp.int64(1048576)
        sy = wp.int64(vy) + wp.int64(1048576)
        sz = wp.int64(vz) + wp.int64(1048576)
        voxel_keys[i] = (
            (sx << wp.int64(42))
            | (sy << wp.int64(21))
            | sz
        )


    @wp.kernel
    def _mark_unique_sorted_kernel(
        # After the two stable global sorts, entries are contiguous by record
        # and ascending by voxel key inside each record.  Mark the first entry
        # of each (record, voxel_key) run in parallel.
        key_sorted_values: wp.array(dtype=wp.int64),
        record_sorted_values: wp.array(dtype=wp.int32),
        record_sort_order: wp.array(dtype=wp.int64),
        unique_flags: wp.array(dtype=wp.int32),
    ):
        pos = wp.tid()
        key_sorted_pos = record_sort_order[pos]
        key = key_sorted_values[key_sorted_pos]

        unique = wp.int32(1)
        if pos > 0:
            prev_key_sorted_pos = record_sort_order[pos - 1]
            prev_key = key_sorted_values[prev_key_sorted_pos]
            current_record = record_sorted_values[pos]
            prev_record = record_sorted_values[pos - 1]
            if current_record == prev_record and key == prev_key:
                unique = wp.int32(0)

        unique_flags[pos] = unique


    @wp.kernel
    def _scatter_unique_sorted_kernel(
        # unique_prefix is a global inclusive prefix sum of unique_flags.
        # Because each record is a contiguous segment, subtracting the prefix
        # at segment begin-1 gives a deterministic per-record compact rank.
        offsets: wp.array(dtype=wp.int32),
        key_sorted_values: wp.array(dtype=wp.int64),
        key_sort_order: wp.array(dtype=wp.int64),
        record_sorted_values: wp.array(dtype=wp.int32),
        record_sort_order: wp.array(dtype=wp.int64),
        unique_flags: wp.array(dtype=wp.int32),
        unique_prefix: wp.array(dtype=wp.int32),
        points_world: wp.array(dtype=wp.float32),
        voxel_coords: wp.array(dtype=wp.int32),
        unique_counts: wp.array(dtype=wp.int32),
        unique_global_indices: wp.array(dtype=wp.int64),
        unique_keys: wp.array(dtype=wp.int64),
        unique_coords: wp.array(dtype=wp.int32),
        unique_points: wp.array(dtype=wp.float32),
    ):
        pos = wp.tid()
        record_index = record_sorted_values[pos]
        begin = offsets[record_index]
        end = offsets[record_index + 1]

        prefix_base = wp.int32(0)
        if begin > 0:
            prefix_base = unique_prefix[begin - 1]

        # Exactly one thread (the last element of each non-empty record) writes
        # that record's unique count. Empty-record counts are zeroed beforehand.
        if pos == end - 1:
            unique_counts[record_index] = unique_prefix[pos] - prefix_base

        if unique_flags[pos] == 0:
            return

        local_rank = unique_prefix[pos] - prefix_base - wp.int32(1)
        out_pos = begin + local_rank

        key_sorted_pos = record_sort_order[pos]
        source_index = key_sort_order[key_sorted_pos]

        unique_global_indices[out_pos] = source_index
        unique_keys[out_pos] = key_sorted_values[key_sorted_pos]

        src = wp.int32(source_index) * 3
        dst = out_pos * 3
        unique_coords[dst + 0] = voxel_coords[src + 0]
        unique_coords[dst + 1] = voxel_coords[src + 1]
        unique_coords[dst + 2] = voxel_coords[src + 2]
        unique_points[dst + 0] = points_world[src + 0]
        unique_points[dst + 1] = points_world[src + 1]
        unique_points[dst + 2] = points_world[src + 2]


    @wp.kernel
    def _direct_mask_geometry_kernel(
        masks: wp.array(dtype=wp.uint8),
        depths: wp.array(dtype=wp.float32),
        record_views: wp.array(dtype=wp.int32),
        strides: wp.array(dtype=wp.int32),
        intrinsics: wp.array(dtype=wp.float32),
        transforms: wp.array(dtype=wp.float32),
        transform_valid: wp.array(dtype=wp.int32),
        origin_world: wp.array(dtype=wp.float32),
        inv_voxel_size: float,
        height: int,
        width: int,
        max_points: int,
        counts: wp.array(dtype=wp.int32),
        points_camera: wp.array(dtype=wp.float32),
        points_world: wp.array(dtype=wp.float32),
        voxel_coords: wp.array(dtype=wp.int32),
        voxel_keys: wp.array(dtype=wp.int64),
        source_pixels: wp.array(dtype=wp.int32),
    ):
        """Direct mask/depth -> sparse geometry with fixed per-record segments.

        The adaptive stride is computed from the already-returned CPU mask metadata.
        A single CUDA kernel then performs lattice sampling, depth validation,
        backprojection, world transform and voxel-key generation. Atomic compaction
        keeps the sampling front-end fast; deterministic representative selection
        later uses the minimum original raster pixel for each voxel.
        """
        i = wp.tid()
        pixels = height * width
        record_index = i // pixels
        pixel = i - record_index * pixels
        y = pixel // width
        x = pixel - y * width

        stride = strides[record_index]
        if stride > 1 and ((x % stride) != 0 or (y % stride) != 0):
            return
        if masks[i] == wp.uint8(0):
            return

        view_index = record_views[record_index]
        z = depths[view_index * pixels + pixel]
        # min/max depth validity is encoded by the caller through intrinsics slots
        # 4 and 5 for each view: [inv_fx, inv_fy, cx, cy, min_z, max_z].
        # Writing the predicate positively also rejects NaNs because comparisons
        # against NaN are false.
        k = view_index * 6
        if not (z >= intrinsics[k + 4] and z <= intrinsics[k + 5]):
            return

        slot = wp.atomic_add(counts, record_index, 1)
        if slot >= max_points:
            return

        out = record_index * max_points + slot
        inv_fx = intrinsics[k + 0]
        inv_fy = intrinsics[k + 1]
        cx0 = intrinsics[k + 2]
        cy0 = intrinsics[k + 3]
        cx = (wp.float32(x) - cx0) * inv_fx * z
        cy = (wp.float32(y) - cy0) * inv_fy * z
        cz = z

        p = out * 3
        points_camera[p + 0] = cx
        points_camera[p + 1] = cy
        points_camera[p + 2] = cz

        wx = cx
        wy = cy
        wz = cz
        if transform_valid[view_index] != 0:
            t = view_index * 12
            wx = cx * transforms[t + 0] + cy * transforms[t + 1] + cz * transforms[t + 2] + transforms[t + 3]
            wy = cx * transforms[t + 4] + cy * transforms[t + 5] + cz * transforms[t + 6] + transforms[t + 7]
            wz = cx * transforms[t + 8] + cy * transforms[t + 9] + cz * transforms[t + 10] + transforms[t + 11]

        points_world[p + 0] = wx
        points_world[p + 1] = wy
        points_world[p + 2] = wz

        vx = wp.int32(wp.floor((wx - origin_world[0]) * inv_voxel_size))
        vy = wp.int32(wp.floor((wy - origin_world[1]) * inv_voxel_size))
        vz = wp.int32(wp.floor((wz - origin_world[2]) * inv_voxel_size))
        voxel_coords[p + 0] = vx
        voxel_coords[p + 1] = vy
        voxel_coords[p + 2] = vz

        sx = wp.int64(vx) + wp.int64(1048576)
        sy = wp.int64(vy) + wp.int64(1048576)
        sz = wp.int64(vz) + wp.int64(1048576)
        voxel_keys[out] = (sx << wp.int64(42)) | (sy << wp.int64(21)) | sz
        # Atomic compaction order is not deterministic; preserve row-major source
        # identity so later voxel representative selection can be deterministic.
        source_pixels[out] = wp.int32(pixel)


    @wp.kernel
    def _compact_direct_segments_kernel(
        lengths: wp.array(dtype=wp.int32),
        offsets: wp.array(dtype=wp.int32),
        max_points: int,
        src_points_camera: wp.array(dtype=wp.float32),
        src_points_world: wp.array(dtype=wp.float32),
        src_voxel_coords: wp.array(dtype=wp.int32),
        src_voxel_keys: wp.array(dtype=wp.int64),
        src_source_pixels: wp.array(dtype=wp.int32),
        dst_points_camera: wp.array(dtype=wp.float32),
        dst_points_world: wp.array(dtype=wp.float32),
        dst_point_records: wp.array(dtype=wp.int32),
        dst_voxel_coords: wp.array(dtype=wp.int32),
        dst_voxel_keys: wp.array(dtype=wp.int64),
        dst_source_pixels: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        record_index = i // max_points
        local = i - record_index * max_points
        if local >= lengths[record_index]:
            return
        src = record_index * max_points + local
        dst = offsets[record_index] + local
        sp = src * 3
        dp = dst * 3
        dst_points_camera[dp + 0] = src_points_camera[sp + 0]
        dst_points_camera[dp + 1] = src_points_camera[sp + 1]
        dst_points_camera[dp + 2] = src_points_camera[sp + 2]
        dst_points_world[dp + 0] = src_points_world[sp + 0]
        dst_points_world[dp + 1] = src_points_world[sp + 1]
        dst_points_world[dp + 2] = src_points_world[sp + 2]
        dst_voxel_coords[dp + 0] = src_voxel_coords[sp + 0]
        dst_voxel_coords[dp + 1] = src_voxel_coords[sp + 1]
        dst_voxel_coords[dp + 2] = src_voxel_coords[sp + 2]
        dst_voxel_keys[dst] = src_voxel_keys[src]
        dst_source_pixels[dst] = src_source_pixels[src]
        dst_point_records[dst] = record_index


    @wp.kernel
    def _mark_unique_sorted_direct_kernel(
        offsets: wp.array(dtype=wp.int32),
        record_count: int,
        key_sorted_values: wp.array(dtype=wp.int64),
        record_sorted_values: wp.array(dtype=wp.int32),
        record_sort_order: wp.array(dtype=wp.int64),
        unique_flags: wp.array(dtype=wp.int32),
    ):
        """Direct-path unique marker with a device-side valid-length boundary.

        The no-sync direct path sorts a fixed upper-bound capacity. Valid entries
        are compacted into [0, offsets[record_count]); invalid tail entries carry
        an INT32_MAX record sentinel and are ignored here.
        """
        pos = wp.tid()
        valid_total = offsets[record_count]
        if pos >= valid_total:
            unique_flags[pos] = wp.int32(0)
            return

        key_sorted_pos = record_sort_order[pos]
        key = key_sorted_values[key_sorted_pos]

        unique = wp.int32(1)
        if pos > 0:
            prev_key_sorted_pos = record_sort_order[pos - 1]
            prev_key = key_sorted_values[prev_key_sorted_pos]
            current_record = record_sorted_values[pos]
            prev_record = record_sorted_values[pos - 1]
            if current_record == prev_record and key == prev_key:
                unique = wp.int32(0)
        unique_flags[pos] = unique


    @wp.kernel
    def _scatter_unique_sorted_direct_kernel(
        offsets: wp.array(dtype=wp.int32),
        record_count: int,
        key_sorted_values: wp.array(dtype=wp.int64),
        key_sort_order: wp.array(dtype=wp.int64),
        record_sorted_values: wp.array(dtype=wp.int32),
        record_sort_order: wp.array(dtype=wp.int64),
        unique_flags: wp.array(dtype=wp.int32),
        unique_prefix: wp.array(dtype=wp.int32),
        source_pixels: wp.array(dtype=wp.int32),
        points_world: wp.array(dtype=wp.float32),
        voxel_coords: wp.array(dtype=wp.int32),
        unique_counts: wp.array(dtype=wp.int32),
        unique_global_indices: wp.array(dtype=wp.int64),
        unique_keys: wp.array(dtype=wp.int64),
        unique_coords: wp.array(dtype=wp.int32),
        unique_points: wp.array(dtype=wp.float32),
    ):
        """Scatter one deterministic representative per (record, voxel key).

        Sampling uses atomic compaction, so compacted source order can change
        across launches.  For each duplicate voxel group choose the sample with
        the minimum original row-major pixel index.  That matches the CPU
        np.nonzero/row-major "first point wins" semantics.
        """
        pos = wp.tid()
        valid_total = offsets[record_count]
        if pos >= valid_total:
            return

        record_index = record_sorted_values[pos]
        begin = offsets[record_index]
        end = offsets[record_index + 1]

        prefix_base = wp.int32(0)
        if begin > 0:
            prefix_base = unique_prefix[begin - 1]

        if pos == end - 1:
            unique_counts[record_index] = unique_prefix[pos] - prefix_base

        if unique_flags[pos] == 0:
            return

        local_rank = unique_prefix[pos] - prefix_base - wp.int32(1)
        out_pos = begin + local_rank

        key_sorted_pos = record_sort_order[pos]
        key = key_sorted_values[key_sorted_pos]
        best_source_index = key_sort_order[key_sorted_pos]
        best_pixel = source_pixels[best_source_index]

        # Warp 1.15: explicitly create dynamic loop variable.
        scan = int(pos + 1)
        while scan < valid_total:
            scan_key_sorted_pos = record_sort_order[scan]
            scan_record = record_sorted_values[scan]
            scan_key = key_sorted_values[scan_key_sorted_pos]
            if scan_record != record_index or scan_key != key:
                break
            candidate_source = key_sort_order[scan_key_sorted_pos]
            candidate_pixel = source_pixels[candidate_source]
            if candidate_pixel < best_pixel:
                best_pixel = candidate_pixel
                best_source_index = candidate_source
            scan = scan + 1

        unique_global_indices[out_pos] = best_source_index
        unique_keys[out_pos] = key

        src = wp.int32(best_source_index) * 3
        dst = out_pos * 3
        unique_coords[dst + 0] = voxel_coords[src + 0]
        unique_coords[dst + 1] = voxel_coords[src + 1]
        unique_coords[dst + 2] = voxel_coords[src + 2]
        unique_points[dst + 0] = points_world[src + 0]
        unique_points[dst + 1] = points_world[src + 1]
        unique_points[dst + 2] = points_world[src + 2]


    @wp.kernel
    def _init_mask_metadata_kernel(
        width: int,
        height: int,
        bboxes: wp.array(dtype=wp.int32),
        foreground: wp.array(dtype=wp.int32),
    ):
        r = wp.tid()
        base = r * 4
        bboxes[base + 0] = width
        bboxes[base + 1] = height
        bboxes[base + 2] = -1
        bboxes[base + 3] = -1
        foreground[r] = 0


    @wp.kernel
    def _mask_metadata_kernel(
        masks: wp.array(dtype=wp.uint8),
        height: int,
        width: int,
        bboxes: wp.array(dtype=wp.int32),
        foreground: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        if masks[i] == wp.uint8(0):
            return
        pixels = height * width
        record_index = i // pixels
        pixel = i - record_index * pixels
        y = pixel // width
        x = pixel - y * width
        base = record_index * 4
        wp.atomic_add(foreground, record_index, 1)
        wp.atomic_min(bboxes, base + 0, x)
        wp.atomic_min(bboxes, base + 1, y)
        wp.atomic_max(bboxes, base + 2, x)
        wp.atomic_max(bboxes, base + 3, y)


    @wp.kernel
    def _adaptive_stride_kernel(
        foreground: wp.array(dtype=wp.int32),
        base_stride: int,
        max_points: int,
        adaptive: int,
        strides: wp.array(dtype=wp.int32),
    ):
        r = wp.tid()
        stride = base_stride
        fg = foreground[r]
        if adaptive != 0 and max_points > 0 and fg > 0:
            # Exact integer equivalent of ceil(sqrt(fg/(base^2*max_points))):
            # choose the smallest multiplier whose lattice capacity is sufficient.
            # Warp 1.15 treats a plain literal assignment (``mult = 1``) as
            # a compile-time constant.  Because ``mult`` is mutated inside a
            # dynamic while-loop, declare it explicitly as a runtime integer.
            mult = int(1)
            denom = max_points * base_stride * base_stride
            while fg > denom * mult * mult:
                mult = mult + 1
            stride = base_stride * mult
        strides[r] = stride


@dataclass(slots=True)
class GeometrySamples:
    """Sparse image samples selected by the exact CPU mask/CC policy."""

    ys: np.ndarray
    xs: np.ndarray
    z: np.ndarray
    colors_rgb: np.ndarray


@dataclass(slots=True)
class GPUGeometryRecord:
    """One instance of sparse geometry with CPU alignment and CUDA data views."""

    points_camera: np.ndarray
    points_world: np.ndarray | None
    colors_rgb: np.ndarray
    voxel_coords: np.ndarray | None = None
    voxel_keys: np.ndarray | None = None
    voxel_points: np.ndarray | None = None
    voxel_colors: np.ndarray | None = None
    voxel_bbox_min: np.ndarray | None = None
    voxel_bbox_max: np.ndarray | None = None

    points_camera_gpu: Any | None = None
    points_world_gpu: Any | None = None
    voxel_coords_gpu: Any | None = None
    voxel_keys_gpu: Any | None = None
    voxel_points_gpu: Any | None = None
    bbox_2d: tuple[int, int, int, int] | None = None
    foreground_pixels: int = 0
    geometry_stride: int = 1


@dataclass(slots=True)
class _PendingBatch:
    lengths: list[int]
    offsets: list[int]
    colors_flat: np.ndarray
    points_camera: Any
    points_world: Any
    unique_counts: Any
    unique_global_indices: Any
    unique_coords: Any
    unique_keys: Any
    unique_points: Any
    # Direct no-sync mode intentionally defers the tiny count readback until
    # materialize(), so the GPU geometry stage never blocks on CPU.
    deferred_lengths_gpu: Any | None = None
    direct_bbox_gpu: Any | None = None
    direct_foreground_gpu: Any | None = None
    direct_strides_gpu: Any | None = None


class GPUSparseGeometryBackend:
    """Fused CUDA sparse RGB-D geometry with deterministic voxel representatives.

    The hot path uses:

      1) one packed H2D copy of [ray_x, ray_y, depth, record_id],
      2) one Warp kernel for camera XYZ + world transform + voxel keys,
      3) two *global* stable radix-style torch sorts to obtain lexicographic
         (record_id, voxel_key) order without one sort per object,
      4) one fully parallel mark -> prefix-sum -> scatter dedup path.

    The CUDA tensors are persistent and reused across frames.  This is designed
    for the small fixed-capacity tabletop workload where launch/allocation
    overhead, not arithmetic throughput, dominated the previous implementation.
    """

    _BITS = 21
    _BIAS = 1 << 20
    _MASK = (1 << _BITS) - 1

    def __init__(self, device: Any, voxelizer: Any) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for GPUSparseGeometryBackend")
        if wp is None:
            raise RuntimeError(
                "warp-lang is required for fused GPU geometry; the tracker Docker "
                "image installs warp-lang by default"
            )

        self.device = torch.device(device)

        if self.device.type != "cuda":
            raise ValueError(
                "GPUSparseGeometryBackend requires a CUDA device"
            )

        # Warp 1.15 requires an explicit CUDA device index in
        # wp.device_from_torch(). torch.device("cuda") has index=None,
        # so resolve it to PyTorch's current CUDA device.
        if self.device.index is None:
            self.device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )

        self.voxelizer = voxelizer

        wp.init()
        self._wp_device = wp.device_from_torch(self.device)
        self._warp_stream = None
        self._warp_stream_ptr: int | None = None

        self._capacity = 0
        self._record_capacity = 0

        # Sparse sample staging: one float4 per point.  x/y are represented as
        # precomputed rays, so small integer image coordinates never reach CUDA.
        self._host_samples = None
        self._device_samples = None

        # Raw fused geometry outputs.
        self._points_camera = None
        self._points_world = None
        self._point_records = None
        self._voxel_coords = None
        self._voxel_keys = None

        # Persistent global-sort workspaces.
        self._key_sorted_values = None
        self._key_sort_order = None
        self._records_in_key_order = None
        self._record_sorted_values = None
        self._record_sort_order = None

        # Persistent parallel unique workspaces.
        self._unique_flags = None
        self._unique_prefix = None

        # Persistent segmented unique outputs.  Each record writes into its raw
        # [offset[r], offset[r+1]) segment, so no dynamic-size compaction/sync is
        # required in the GPU stage.
        self._unique_global_indices = None
        self._unique_keys = None
        self._unique_coords = None
        self._unique_points = None

        # Small per-record metadata.
        self._host_record_views = None
        self._device_record_views = None
        self._host_offsets = None
        self._device_offsets = None
        self._unique_counts = None

        # Packed camera transforms and static world-grid constants.
        self._view_capacity = 0
        self._host_transforms = None
        self._device_transforms = None
        self._host_transform_valid = None
        self._device_transform_valid = None
        self._transform_cache: dict[int, tuple[str, np.ndarray | None]] = {}

        self._origin_world = torch.as_tensor(
            np.asarray(voxelizer.origin_world, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).contiguous()
        self._inv_voxel_size = float(voxelizer.inv_voxel_size)

        # CPU precomputed ray tables.  They are indexed while filling the pinned
        # float4 sample buffer, avoiding GPU divisions without another CUDA read.
        self._ray_cache: dict[
            tuple[float, float, float, float, int, int],
            tuple[np.ndarray, np.ndarray],
        ] = {}

        # Cached Warp zero-copy descriptors. Rebuilt only when a backing Torch
        # allocation grows; this follows Warp's interop performance guidance.
        self._wp: dict[str, Any] = {}
        self._wp_origin = wp.from_torch(self._origin_world, requires_grad=False)

        # Optional CUDA-event profiling used only by the validation/benchmark script.
        # Disabled in the real pipeline, so it has zero steady-state overhead there.
        self._gpu_profile_enabled = False
        self._last_gpu_profile_events = None
        self._last_gpu_profile_names: tuple[str, ...] | None = None
        self._last_gpu_profile_pairs: dict[str, tuple[Any, Any]] | None = None
        self._last_cpu_profile_ms: dict[str, float] | None = None

        # Direct GPU postprocess front-end. Persistent buffers let
        # mask + depth feed the geometry kernel without constructing CPU
        # GeometrySamples or copying sparse samples back to CUDA.
        self._direct_shape: tuple[int, int, int, int] | None = None
        self._direct_max_points = 0
        self._direct_masks = None
        self._direct_host_depths = None
        self._direct_depths = None
        self._direct_host_strides = None
        self._direct_strides = None
        self._direct_host_intrinsics = None
        self._direct_intrinsics = None
        self._direct_counts = None
        self._direct_lengths = None
        self._direct_points_camera = None
        self._direct_points_world = None
        self._direct_voxel_coords = None
        self._direct_voxel_keys = None
        self._direct_bbox = None
        self._direct_foreground = None
        self._direct_wp: dict[str, Any] = {}

        # Depth is staged independently so it can be copied to CUDA on a dedicated
        # stream while EfficientTAM is still running.
        self._direct_depth_shape: tuple[int, int, int] | None = None
        self._depth_stream = torch.cuda.Stream(device=self.device)
        self._depth_ready_event = torch.cuda.Event(enable_timing=False)
        self._depth_profile_start = torch.cuda.Event(enable_timing=True)
        self._depth_profile_end = torch.cuda.Event(enable_timing=True)
        self._prefetched_depth_key: tuple[Any, ...] | None = None
        self._last_depth_prefetch_cpu_ms: float | None = None
        self._depth_buffer_lock = Lock()

        # Pinned compact CPU-alignment transfer workspaces.
        self._host_compact_lengths = None
        self._host_compact_counts = None
        self._host_compact_bbox = None
        self._host_compact_foreground = None
        self._host_compact_strides = None
        self._host_compact_keys = None
        self._host_compact_coords = None
        self._host_compact_points = None

    @property
    def capacity(self) -> int:
        return int(self._capacity)

    def enable_gpu_profile(self, enabled: bool = True) -> None:
        self._gpu_profile_enabled = bool(enabled)

        if not self._gpu_profile_enabled:
            self._last_gpu_profile_events = None
            self._last_gpu_profile_names = None
            self._last_gpu_profile_pairs = None
            self._last_cpu_profile_ms = None

    def gpu_profile_ms(self) -> dict[str, float] | None:
        pairs = self._last_gpu_profile_pairs
        if pairs is not None:
            return {
                name: float(start.elapsed_time(end))
                for name, (start, end) in pairs.items()
            }
        events = self._last_gpu_profile_events
        names = self._last_gpu_profile_names
        if events is None or names is None:
            return None
        return {
            name: float(events[i].elapsed_time(events[i + 1]))
            for i, name in enumerate(names)
        }

    def cpu_profile_ms(self) -> dict[str, float] | None:
        profile = self._last_cpu_profile_ms
        return None if profile is None else dict(profile)

    @staticmethod
    def _offsets(lengths: list[int]) -> list[int]:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + int(length))
        return offsets

    def _current_warp_stream(self):
        torch_stream = torch.cuda.current_stream(device=self.device)
        stream_ptr = int(torch_stream.cuda_stream)
        if self._warp_stream is None or stream_ptr != self._warp_stream_ptr:
            self._warp_stream = wp.stream_from_torch(torch_stream)
            self._warp_stream_ptr = stream_ptr
        return self._warp_stream

    def _rays(self, frame: Any) -> tuple[np.ndarray, np.ndarray]:
        k = frame.intrinsics
        key = (
            float(k.fx),
            float(k.fy),
            float(k.cx),
            float(k.cy),
            int(k.width),
            int(k.height),
        )
        cached = self._ray_cache.get(key)
        if cached is not None:
            return cached
        x_ray = (
            np.arange(int(k.width), dtype=np.float32) - np.float32(k.cx)
        ) / np.float32(k.fx)
        y_ray = (
            np.arange(int(k.height), dtype=np.float32) - np.float32(k.cy)
        ) / np.float32(k.fy)
        cached = (x_ray, y_ray)
        self._ray_cache[key] = cached
        return cached

    def _cache_point_descriptors(self) -> None:
        # Flatten vectors to scalar arrays so the same Warp kernels work across
        # Warp versions without relying on vec3 tensor-shape conventions.
        self._wp.update(
            samples=wp.from_torch(self._device_samples.view(-1), requires_grad=False),
            points_camera=wp.from_torch(self._points_camera.view(-1), requires_grad=False),
            points_world=wp.from_torch(self._points_world.view(-1), requires_grad=False),
            point_records=wp.from_torch(self._point_records, requires_grad=False),
            voxel_coords=wp.from_torch(self._voxel_coords.view(-1), requires_grad=False),
            voxel_keys=wp.from_torch(self._voxel_keys, requires_grad=False),
            source_pixels=wp.from_torch(self._source_pixels, requires_grad=False),
            key_sorted_values=wp.from_torch(self._key_sorted_values, requires_grad=False),
            key_sort_order=wp.from_torch(self._key_sort_order, requires_grad=False),
            record_sorted_values=wp.from_torch(
                self._record_sorted_values, requires_grad=False
            ),
            record_sort_order=wp.from_torch(self._record_sort_order, requires_grad=False),
            unique_flags=wp.from_torch(self._unique_flags, requires_grad=False),
            unique_prefix=wp.from_torch(self._unique_prefix, requires_grad=False),
            unique_global_indices=wp.from_torch(
                self._unique_global_indices, requires_grad=False
            ),
            unique_keys=wp.from_torch(self._unique_keys, requires_grad=False),
            unique_coords=wp.from_torch(
                self._unique_coords.view(-1), requires_grad=False
            ),
            unique_points=wp.from_torch(
                self._unique_points.view(-1), requires_grad=False
            ),
        )

    def _ensure_capacity(self, required: int) -> None:
        required = max(1, int(required))
        if required <= self._capacity:
            return
        capacity = max(required, max(4096, self._capacity * 2))

        self._host_samples = torch.empty(
            (capacity, 4), dtype=torch.float32, pin_memory=True
        )
        self._device_samples = torch.empty(
            (capacity, 4), dtype=torch.float32, device=self.device
        )
        self._points_camera = torch.empty(
            (capacity, 3), dtype=torch.float32, device=self.device
        )
        self._points_world = torch.empty(
            (capacity, 3), dtype=torch.float32, device=self.device
        )
        self._point_records = torch.empty(
            capacity, dtype=torch.int32, device=self.device
        )
        self._voxel_coords = torch.empty(
            (capacity, 3), dtype=torch.int32, device=self.device
        )
        self._voxel_keys = torch.empty(
            capacity, dtype=torch.int64, device=self.device
        )
        self._source_pixels = torch.empty(
            capacity, dtype=torch.int32, device=self.device
        )

        self._key_sorted_values = torch.empty(
            capacity, dtype=torch.int64, device=self.device
        )
        self._key_sort_order = torch.empty(
            capacity, dtype=torch.int64, device=self.device
        )
        self._records_in_key_order = torch.empty(
            capacity, dtype=torch.int32, device=self.device
        )
        self._record_sorted_values = torch.empty(
            capacity, dtype=torch.int32, device=self.device
        )
        self._record_sort_order = torch.empty(
            capacity, dtype=torch.int64, device=self.device
        )

        self._unique_flags = torch.empty(
            capacity, dtype=torch.int32, device=self.device
        )
        self._unique_prefix = torch.empty(
            capacity, dtype=torch.int32, device=self.device
        )

        self._unique_global_indices = torch.empty(
            capacity, dtype=torch.int64, device=self.device
        )
        self._unique_keys = torch.empty(
            capacity, dtype=torch.int64, device=self.device
        )
        self._unique_coords = torch.empty(
            (capacity, 3), dtype=torch.int32, device=self.device
        )
        self._unique_points = torch.empty(
            (capacity, 3), dtype=torch.float32, device=self.device
        )
        self._host_compact_keys = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
        self._host_compact_coords = torch.empty((capacity, 3), dtype=torch.int32, pin_memory=True)
        self._host_compact_points = torch.empty((capacity, 3), dtype=torch.float32, pin_memory=True)

        self._capacity = capacity
        self._cache_point_descriptors()

    def _cache_record_descriptors(self) -> None:
        self._wp.update(
            record_views=wp.from_torch(
                self._device_record_views, requires_grad=False
            ),
            offsets=wp.from_torch(self._device_offsets, requires_grad=False),
            unique_counts=wp.from_torch(self._unique_counts, requires_grad=False),
        )

    def _ensure_record_capacity(self, required: int) -> None:
        required = max(1, int(required))
        if required <= self._record_capacity:
            return
        capacity = max(required, max(8, self._record_capacity * 2))
        self._host_record_views = torch.empty(
            capacity, dtype=torch.int32, pin_memory=True
        )
        self._device_record_views = torch.empty(
            capacity, dtype=torch.int32, device=self.device
        )
        self._host_offsets = torch.empty(
            capacity + 1, dtype=torch.int32, pin_memory=True
        )
        self._device_offsets = torch.empty(
            capacity + 1, dtype=torch.int32, device=self.device
        )
        self._unique_counts = torch.empty(
            capacity, dtype=torch.int32, device=self.device
        )
        self._host_compact_lengths = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self._host_compact_counts = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self._host_compact_bbox = torch.empty((capacity, 4), dtype=torch.int32, pin_memory=True)
        self._host_compact_foreground = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self._host_compact_strides = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self._record_capacity = capacity
        self._cache_record_descriptors()

    def _cache_view_descriptors(self) -> None:
        self._wp.update(
            transforms=wp.from_torch(
                self._device_transforms.view(-1), requires_grad=False
            ),
            transform_valid=wp.from_torch(
                self._device_transform_valid, requires_grad=False
            ),
        )

    def _ensure_view_capacity(self, required: int) -> None:
        required = max(1, int(required))
        if required <= self._view_capacity:
            return
        capacity = max(required, max(2, self._view_capacity * 2))
        self._host_transforms = torch.empty(
            (capacity, 12), dtype=torch.float32, pin_memory=True
        )
        self._device_transforms = torch.empty(
            (capacity, 12), dtype=torch.float32, device=self.device
        )
        self._host_transform_valid = torch.empty(
            capacity, dtype=torch.int32, pin_memory=True
        )
        self._device_transform_valid = torch.empty(
            capacity, dtype=torch.int32, device=self.device
        )
        self._view_capacity = capacity
        # Force a complete refresh after reallocating the packed view table.
        self._transform_cache = {}
        self._cache_view_descriptors()

    def _update_transforms(self, frames: list[Any]) -> None:
        self._ensure_view_capacity(len(frames))
        host_t = self._host_transforms[: len(frames)].numpy()
        host_valid = self._host_transform_valid[: len(frames)].numpy()
        changed = False

        for view_index, frame in enumerate(frames):
            transform = frame.world_from_camera
            value = (
                None
                if transform is None
                else np.asarray(transform, dtype=np.float32).reshape(4, 4)
            )
            name = str(frame.camera_name)
            cached_entry = self._transform_cache.get(view_index)
            same = False
            if cached_entry is not None and cached_entry[0] == name:
                cached = cached_entry[1]
                if cached is None and value is None:
                    same = True
                elif isinstance(cached, np.ndarray) and value is not None:
                    same = np.array_equal(cached, value)
            if same:
                continue

            changed = True
            if value is None:
                host_t[view_index].fill(0.0)
                host_valid[view_index] = 0
                self._transform_cache[view_index] = (name, None)
            else:
                r = value[:3, :3]
                tr = value[:3, 3]
                host_t[view_index] = np.asarray(
                    [
                        r[0, 0], r[0, 1], r[0, 2], tr[0],
                        r[1, 0], r[1, 1], r[1, 2], tr[1],
                        r[2, 0], r[2, 1], r[2, 2], tr[2],
                    ],
                    dtype=np.float32,
                )
                host_valid[view_index] = 1
                self._transform_cache[view_index] = (name, value.copy())

        if changed:
            self._device_transforms[: len(frames)].copy_(
                self._host_transforms[: len(frames)], non_blocking=True
            )
            self._device_transform_valid[: len(frames)].copy_(
                self._host_transform_valid[: len(frames)], non_blocking=True
            )

    def _ensure_depth_buffers(self, view_count: int, height: int, width: int) -> None:
        shape = (max(1, int(view_count)), int(height), int(width))
        if self._direct_depth_shape == shape:
            return
        with self._depth_buffer_lock:
            if self._direct_depth_shape == shape:
                return
            v, h, w = shape
            self._direct_host_depths = torch.empty((v, h, w), dtype=torch.float32, pin_memory=True)
            self._direct_depths = torch.empty((v, h, w), dtype=torch.float32, device=self.device)
            self._direct_depth_shape = shape
            if self._direct_wp:
                self._direct_wp["depths"] = wp.from_torch(
                    self._direct_depths.view(-1), requires_grad=False
                )

    def _ensure_direct_buffers(
        self,
        record_count: int,
        view_count: int,
        height: int,
        width: int,
        max_points: int,
    ) -> None:
        self._ensure_depth_buffers(view_count, height, width)
        shape = (int(record_count), int(view_count), int(height), int(width))
        if self._direct_shape == shape and self._direct_max_points == int(max_points):
            return

        r = max(1, int(record_count))
        v = max(1, int(view_count))
        h = int(height)
        w = int(width)
        m = max(1, int(max_points))
        fixed = r * m

        self._direct_masks = torch.empty((r, h, w), dtype=torch.uint8, device=self.device)
        self._direct_host_strides = torch.empty(r, dtype=torch.int32, pin_memory=True)
        self._direct_strides = torch.empty(r, dtype=torch.int32, device=self.device)
        self._direct_host_intrinsics = torch.empty((v, 6), dtype=torch.float32, pin_memory=True)
        self._direct_intrinsics = torch.empty((v, 6), dtype=torch.float32, device=self.device)
        self._direct_counts = torch.empty(r, dtype=torch.int32, device=self.device)
        self._direct_lengths = torch.empty(r, dtype=torch.int32, device=self.device)
        self._direct_points_camera = torch.empty((fixed, 3), dtype=torch.float32, device=self.device)
        self._direct_points_world = torch.empty((fixed, 3), dtype=torch.float32, device=self.device)
        self._direct_voxel_coords = torch.empty((fixed, 3), dtype=torch.int32, device=self.device)
        self._direct_voxel_keys = torch.empty(fixed, dtype=torch.int64, device=self.device)
        self._direct_source_pixels = torch.empty(fixed, dtype=torch.int32, device=self.device)
        self._direct_bbox = torch.empty((r, 4), dtype=torch.int32, device=self.device)
        self._direct_foreground = torch.empty(r, dtype=torch.int32, device=self.device)

        self._direct_wp = {
            "masks": wp.from_torch(self._direct_masks.view(-1), requires_grad=False),
            "depths": wp.from_torch(self._direct_depths.view(-1), requires_grad=False),
            "strides": wp.from_torch(self._direct_strides, requires_grad=False),
            "intrinsics": wp.from_torch(self._direct_intrinsics.view(-1), requires_grad=False),
            "counts": wp.from_torch(self._direct_counts, requires_grad=False),
            "lengths": wp.from_torch(self._direct_lengths, requires_grad=False),
            "points_camera": wp.from_torch(self._direct_points_camera.view(-1), requires_grad=False),
            "points_world": wp.from_torch(self._direct_points_world.view(-1), requires_grad=False),
            "voxel_coords": wp.from_torch(self._direct_voxel_coords.view(-1), requires_grad=False),
            "voxel_keys": wp.from_torch(self._direct_voxel_keys, requires_grad=False),
            "source_pixels": wp.from_torch(self._direct_source_pixels, requires_grad=False),
            "bbox": wp.from_torch(self._direct_bbox.view(-1), requires_grad=False),
            "foreground": wp.from_torch(self._direct_foreground, requires_grad=False),
        }
        self._direct_shape = shape
        self._direct_max_points = m

    @staticmethod
    def _depth_frame_key(frames: list[Any]) -> tuple[Any, ...]:
        return tuple(
            (str(frame.camera_name), int(frame.frame_index), int(frame.timestamp_ns))
            for frame in frames
        )

    def prefetch_depth(self, frames: list[Any]) -> dict[str, float]:
        """Pack and asynchronously upload depth on a dedicated CUDA stream."""
        if not frames:
            return {"cpu_pack_ms": 0.0, "gpu_h2d_ms": 0.0}
        torch.cuda.set_device(self.device)
        h, w = frames[0].depth_m.shape
        if any(tuple(frame.depth_m.shape) != (h, w) for frame in frames):
            raise ValueError("Depth prefetch requires equal depth image shapes")
        self._ensure_depth_buffers(len(frames), h, w)
        started = perf_counter()
        host = self._direct_host_depths[: len(frames)].numpy()
        for view_index, frame in enumerate(frames):
            np.copyto(host[view_index], np.asarray(frame.depth_m, dtype=np.float32))
        cpu_pack_ms = (perf_counter() - started) * 1000.0
        with torch.cuda.stream(self._depth_stream):
            self._depth_profile_start.record(self._depth_stream)
            self._direct_depths[: len(frames)].copy_(
                self._direct_host_depths[: len(frames)], non_blocking=True
            )
            self._depth_profile_end.record(self._depth_stream)
            self._depth_ready_event.record(self._depth_stream)
        self._prefetched_depth_key = self._depth_frame_key(frames)
        self._last_depth_prefetch_cpu_ms = float(cpu_pack_ms)
        return {"cpu_pack_ms": float(cpu_pack_ms)}

    def depth_prefetch_profile_ms(self) -> dict[str, float] | None:
        if self._prefetched_depth_key is None:
            return None
        try:
            gpu_ms = float(self._depth_profile_start.elapsed_time(self._depth_profile_end))
        except RuntimeError:
            return None
        return {
            "cpu_pack_ms": float(self._last_depth_prefetch_cpu_ms or 0.0),
            "gpu_h2d_ms": gpu_ms,
        }

    def _consume_prefetched_depth(self, frames: list[Any]) -> bool:
        if self._prefetched_depth_key != self._depth_frame_key(frames):
            return False
        torch.cuda.current_stream(device=self.device).wait_event(self._depth_ready_event)
        return True

    def prepare_direct_masks(
        self,
        records: list[Any],
        masks_gpu: list[Any],
        *,
        view_count: int,
        base_stride: int,
        max_points: int,
        adaptive_sampling: bool,
    ) -> None:
        """Keep masks and their bbox/count/stride metadata entirely on CUDA."""
        record_count = len(records)
        if record_count == 0:
            return
        h, w = tuple(masks_gpu[0].shape[-2:])
        self._ensure_record_capacity(record_count)
        self._ensure_direct_buffers(record_count, view_count, h, w, max_points)
        torch.stack(
            [mask.to(dtype=torch.uint8) if mask.dtype != torch.uint8 else mask for mask in masks_gpu],
            dim=0,
            out=self._direct_masks[:record_count],
        )
        stream = self._current_warp_stream()
        wp.launch(
            _init_mask_metadata_kernel,
            dim=record_count,
            inputs=[w, h],
            outputs=[self._direct_wp["bbox"], self._direct_wp["foreground"]],
            device=self._wp_device,
            stream=stream,
        )
        wp.launch(
            _mask_metadata_kernel,
            dim=record_count * h * w,
            inputs=[self._direct_wp["masks"], h, w],
            outputs=[self._direct_wp["bbox"], self._direct_wp["foreground"]],
            device=self._wp_device,
            stream=stream,
        )
        wp.launch(
            _adaptive_stride_kernel,
            dim=record_count,
            inputs=[
                self._direct_wp["foreground"],
                max(1, int(base_stride)),
                max(1, int(max_points)),
                1 if adaptive_sampling else 0,
            ],
            outputs=[self._direct_wp["strides"]],
            device=self._wp_device,
            stream=stream,
        )

    def compute_from_masks(
        self,
        records: list[Any],
        frames: list[Any],
        masks_gpu: list[Any],
        strides: list[int] | None,
        *,
        max_points: int,
        min_depth: float,
        max_depth: float,
        masks_prepared: bool = False,
        use_prefetched_depth: bool = False,
    ) -> _PendingBatch:
        """Direct CUDA mask/depth -> sparse geometry without a mid-pipeline sync.

        The previous direct path copied six per-record counts back to CPU, called
        stream.synchronize(), built offsets on CPU, and copied them back to CUDA
        before compaction. Profiling showed that CPU/GPU bubble dominated p95.

        This version keeps counts, capped lengths and prefix offsets entirely on
        CUDA. The only count readback happens later inside materialize(), which is
        already the explicit GPU->CPU boundary required by CPU alignment.

        To avoid needing the device-computed total as a Python integer, the global
        sort/dedup works over the known upper bound record_count * max_points.
        Invalid tail entries carry an INT32_MAX record sentinel and are ignored by
        the direct-path unique kernels.
        """
        record_count = len(records)
        if record_count != len(masks_gpu):
            raise ValueError("Direct GPU geometry record/mask count mismatch")
        if not masks_prepared and (strides is None or record_count != len(strides)):
            raise ValueError("Direct GPU geometry stride count mismatch")
        if record_count == 0:
            self._ensure_capacity(1)
            self._ensure_record_capacity(1)
            self._unique_counts[:0].zero_()
            return _PendingBatch(
                lengths=[],
                offsets=[0],
                colors_flat=np.empty((0, 3), dtype=np.uint8),
                points_camera=self._points_camera[:0],
                points_world=self._points_world[:0],
                unique_counts=self._unique_counts[:0],
                unique_global_indices=self._unique_global_indices[:0],
                unique_coords=self._unique_coords[:0],
                unique_keys=self._unique_keys[:0],
                unique_points=self._unique_points[:0],
            )

        h, w = frames[0].depth_m.shape
        if any(tuple(frame.depth_m.shape) != (h, w) for frame in frames):
            raise ValueError("Direct GPU geometry currently requires equal depth image shapes")
        if any(tuple(mask.shape[-2:]) != (h, w) for mask in masks_gpu):
            raise ValueError("Direct GPU geometry mask/depth shape mismatch")

        max_points = max(1, int(max_points))
        fixed_total = record_count * max_points

        self._ensure_record_capacity(record_count)
        self._ensure_view_capacity(len(frames))
        self._ensure_direct_buffers(record_count, len(frames), h, w, max_points)
        # We now sort the fixed upper bound, so all canonical sort/dedup buffers
        # must be able to hold the complete fixed direct workspace.
        self._ensure_capacity(fixed_total)
        self._update_transforms(frames)

        direct_pairs = None
        if self._gpu_profile_enabled:
            direct_pairs = {
                name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for name in (
                    "input_h2d", "mask_geometry", "device_offsets", "compact",
                    "sort_keys", "index_select", "sort_records",
                    "mark_unique", "prefix_sum", "scatter_unique",
                )
            }
        direct_cpu: dict[str, float] | None = {} if self._gpu_profile_enabled else None
        direct_cpu_t0 = perf_counter() if direct_cpu is not None else 0.0
        torch_stream = torch.cuda.current_stream(device=self.device)

        if not masks_prepared:
            torch.stack(
                [mask.to(dtype=torch.uint8) if mask.dtype != torch.uint8 else mask for mask in masks_gpu],
                dim=0,
                out=self._direct_masks[:record_count],
            )
        if direct_cpu is not None:
            now = perf_counter()
            direct_cpu["mask_stack_submit"] = (now - direct_cpu_t0) * 1000.0
            direct_cpu_t0 = now

        prefetched = bool(use_prefetched_depth and self._consume_prefetched_depth(frames))
        intr_host = self._direct_host_intrinsics[: len(frames)].numpy()
        if not prefetched:
            depth_host = self._direct_host_depths[: len(frames)].numpy()
        for view_index, frame in enumerate(frames):
            if not prefetched:
                np.copyto(depth_host[view_index], np.asarray(frame.depth_m, dtype=np.float32))
            k = frame.intrinsics
            intr_host[view_index] = np.asarray(
                [
                    1.0 / float(k.fx),
                    1.0 / float(k.fy),
                    float(k.cx),
                    float(k.cy),
                    float(min_depth),
                    float(max_depth),
                ],
                dtype=np.float32,
            )
        if not masks_prepared:
            self._direct_host_strides[:record_count].numpy()[:] = np.asarray(
                strides, dtype=np.int32
            )

        host_views = self._host_record_views[:record_count].numpy()
        for i, record in enumerate(records):
            host_views[i] = int(record.view_index)

        if direct_cpu is not None:
            now = perf_counter()
            direct_cpu["depth_metadata_pack"] = (now - direct_cpu_t0) * 1000.0
            direct_cpu_t0 = now

        if direct_pairs is not None:
            direct_pairs["input_h2d"][0].record(torch_stream)
        if not prefetched:
            self._direct_depths[: len(frames)].copy_(
                self._direct_host_depths[: len(frames)], non_blocking=True
            )
        self._direct_intrinsics[: len(frames)].copy_(
            self._direct_host_intrinsics[: len(frames)], non_blocking=True
        )
        if not masks_prepared:
            self._direct_strides[:record_count].copy_(
                self._direct_host_strides[:record_count], non_blocking=True
            )
        self._device_record_views[:record_count].copy_(
            self._host_record_views[:record_count], non_blocking=True
        )
        self._direct_counts[:record_count].zero_()
        if direct_pairs is not None:
            direct_pairs["input_h2d"][1].record(torch_stream)
        if direct_cpu is not None:
            now = perf_counter()
            direct_cpu["input_h2d_submit"] = (now - direct_cpu_t0) * 1000.0
            direct_cpu_t0 = now

        stream = self._current_warp_stream()
        if direct_pairs is not None:
            direct_pairs["mask_geometry"][0].record(torch_stream)
        wp.launch(
            _direct_mask_geometry_kernel,
            dim=record_count * h * w,
            inputs=[
                self._direct_wp["masks"],
                self._direct_wp["depths"],
                self._wp["record_views"],
                self._direct_wp["strides"],
                self._direct_wp["intrinsics"],
                self._wp["transforms"],
                self._wp["transform_valid"],
                self._wp_origin,
                self._inv_voxel_size,
                h,
                w,
                max_points,
            ],
            outputs=[
                self._direct_wp["counts"],
                self._direct_wp["points_camera"],
                self._direct_wp["points_world"],
                self._direct_wp["voxel_coords"],
                self._direct_wp["voxel_keys"],
                self._direct_wp["source_pixels"],
            ],
            device=self._wp_device,
            stream=stream,
        )
        if direct_pairs is not None:
            direct_pairs["mask_geometry"][1].record(torch_stream)

        # Device-only counts -> capped lengths -> exclusive offsets.
        if direct_pairs is not None:
            direct_pairs["device_offsets"][0].record(torch_stream)
        torch.clamp(
            self._direct_counts[:record_count],
            min=0,
            max=max_points,
            out=self._direct_lengths[:record_count],
        )
        self._device_offsets[0].zero_()
        torch.cumsum(
            self._direct_lengths[:record_count],
            dim=0,
            dtype=torch.int32,
            out=self._device_offsets[1 : record_count + 1],
        )
        # The compact kernel writes only valid entries. Mark every remaining
        # upper-bound slot as an invalid record before global sorting.
        self._point_records[:fixed_total].fill_(2147483647)
        if direct_pairs is not None:
            direct_pairs["device_offsets"][1].record(torch_stream)
        if direct_cpu is not None:
            now = perf_counter()
            direct_cpu["device_offsets_submit"] = (now - direct_cpu_t0) * 1000.0
            direct_cpu_t0 = now

        if direct_pairs is not None:
            direct_pairs["compact"][0].record(torch_stream)
        wp.launch(
            _compact_direct_segments_kernel,
            dim=fixed_total,
            inputs=[
                self._direct_wp["lengths"],
                self._wp["offsets"],
                max_points,
                self._direct_wp["points_camera"],
                self._direct_wp["points_world"],
                self._direct_wp["voxel_coords"],
                self._direct_wp["voxel_keys"],
                self._direct_wp["source_pixels"],
            ],
            outputs=[
                self._wp["points_camera"],
                self._wp["points_world"],
                self._wp["point_records"],
                self._wp["voxel_coords"],
                self._wp["voxel_keys"],
                self._wp["source_pixels"],
            ],
            device=self._wp_device,
            stream=stream,
        )
        if direct_pairs is not None:
            direct_pairs["compact"][1].record(torch_stream)

        # Sort the known fixed upper bound. Invalid tail entries can have any key:
        # the second stable record sort moves their INT32_MAX sentinels behind all
        # real records, and the direct unique kernels ignore positions >= total.
        if direct_pairs is not None:
            direct_pairs["sort_keys"][0].record(torch_stream)
        torch.sort(
            self._voxel_keys[:fixed_total],
            stable=True,
            out=(
                self._key_sorted_values[:fixed_total],
                self._key_sort_order[:fixed_total],
            ),
        )
        if direct_pairs is not None:
            direct_pairs["sort_keys"][1].record(torch_stream)

        if direct_pairs is not None:
            direct_pairs["index_select"][0].record(torch_stream)
        torch.index_select(
            self._point_records[:fixed_total],
            0,
            self._key_sort_order[:fixed_total],
            out=self._records_in_key_order[:fixed_total],
        )
        if direct_pairs is not None:
            direct_pairs["index_select"][1].record(torch_stream)

        if direct_pairs is not None:
            direct_pairs["sort_records"][0].record(torch_stream)
        torch.sort(
            self._records_in_key_order[:fixed_total],
            stable=True,
            out=(
                self._record_sorted_values[:fixed_total],
                self._record_sort_order[:fixed_total],
            ),
        )
        if direct_pairs is not None:
            direct_pairs["sort_records"][1].record(torch_stream)

        self._unique_counts[:record_count].zero_()

        if direct_pairs is not None:
            direct_pairs["mark_unique"][0].record(torch_stream)
        wp.launch(
            _mark_unique_sorted_direct_kernel,
            dim=fixed_total,
            inputs=[
                self._wp["offsets"],
                record_count,
                self._wp["key_sorted_values"],
                self._wp["record_sorted_values"],
                self._wp["record_sort_order"],
            ],
            outputs=[self._wp["unique_flags"]],
            device=self._wp_device,
            stream=stream,
        )
        if direct_pairs is not None:
            direct_pairs["mark_unique"][1].record(torch_stream)

        if direct_pairs is not None:
            direct_pairs["prefix_sum"][0].record(torch_stream)
        torch.cumsum(
            self._unique_flags[:fixed_total],
            dim=0,
            dtype=torch.int32,
            out=self._unique_prefix[:fixed_total],
        )
        if direct_pairs is not None:
            direct_pairs["prefix_sum"][1].record(torch_stream)

        if direct_pairs is not None:
            direct_pairs["scatter_unique"][0].record(torch_stream)
        wp.launch(
            _scatter_unique_sorted_direct_kernel,
            dim=fixed_total,
            inputs=[
                self._wp["offsets"],
                record_count,
                self._wp["key_sorted_values"],
                self._wp["key_sort_order"],
                self._wp["record_sorted_values"],
                self._wp["record_sort_order"],
                self._wp["unique_flags"],
                self._wp["unique_prefix"],
                self._wp["source_pixels"],
                self._wp["points_world"],
                self._wp["voxel_coords"],
            ],
            outputs=[
                self._wp["unique_counts"],
                self._wp["unique_global_indices"],
                self._wp["unique_keys"],
                self._wp["unique_coords"],
                self._wp["unique_points"],
            ],
            device=self._wp_device,
            stream=stream,
        )
        if direct_pairs is not None:
            direct_pairs["scatter_unique"][1].record(torch_stream)
            self._last_gpu_profile_pairs = direct_pairs
            self._last_gpu_profile_events = None
            self._last_gpu_profile_names = None
        if direct_cpu is not None:
            now = perf_counter()
            direct_cpu["post_submit"] = (now - direct_cpu_t0) * 1000.0
            self._last_cpu_profile_ms = direct_cpu

        return _PendingBatch(
            lengths=[],
            offsets=[],
            colors_flat=np.empty((0, 3), dtype=np.uint8),
            points_camera=self._points_camera[:fixed_total],
            points_world=self._points_world[:fixed_total],
            unique_counts=self._unique_counts[:record_count],
            unique_global_indices=self._unique_global_indices[:fixed_total],
            unique_coords=self._unique_coords[:fixed_total],
            unique_keys=self._unique_keys[:fixed_total],
            unique_points=self._unique_points[:fixed_total],
            deferred_lengths_gpu=self._direct_lengths[:record_count],
            direct_bbox_gpu=(self._direct_bbox[:record_count] if masks_prepared else None),
            direct_foreground_gpu=(self._direct_foreground[:record_count] if masks_prepared else None),
            direct_strides_gpu=(self._direct_strides[:record_count] if masks_prepared else None),
        )

    def compute(
        self,
        records: list[Any],
        frames: list[Any],
        samples: list[GeometrySamples],
    ) -> _PendingBatch:
        if len(records) != len(samples):
            raise ValueError("GPU geometry record/sample count mismatch")

        lengths = [int(len(sample.z)) for sample in samples]
        offsets = self._offsets(lengths)
        total = offsets[-1]
        record_count = len(records)

        self._ensure_capacity(total)
        self._ensure_record_capacity(record_count)
        self._update_transforms(frames)

        if total == 0:
            self._unique_counts[:record_count].zero_()
            return _PendingBatch(
                lengths=lengths,
                offsets=offsets,
                colors_flat=np.empty((0, 3), dtype=np.uint8),
                points_camera=self._points_camera[:0],
                points_world=self._points_world[:0],
                unique_counts=self._unique_counts[:record_count],
                unique_global_indices=self._unique_global_indices[:0],
                unique_coords=self._unique_coords[:0],
                unique_keys=self._unique_keys[:0],
                unique_points=self._unique_points[:0],
            )

        # Fill one contiguous pinned float4 buffer.  Rays are cached per camera;
        # this moves integer->float/sub/div work out of the CUDA hot path without
        # adding another device lookup table or copy.
        host_samples = self._host_samples[:total]
        host_np = host_samples.numpy()
        colors_chunks: list[np.ndarray] = []
        cursor = 0
        for record_index, (record, sample) in enumerate(zip(records, samples)):
            count = lengths[record_index]
            if count <= 0:
                continue
            end = cursor + count
            frame = frames[int(record.view_index)]
            x_ray, y_ray = self._rays(frame)
            xs = np.asarray(sample.xs, dtype=np.intp)
            ys = np.asarray(sample.ys, dtype=np.intp)
            host_np[cursor:end, 0] = x_ray[xs]
            host_np[cursor:end, 1] = y_ray[ys]
            host_np[cursor:end, 2] = np.asarray(sample.z, dtype=np.float32)
            host_np[cursor:end, 3] = np.float32(record_index)
            if sample.colors_rgb.size:
                colors_chunks.append(
                    np.ascontiguousarray(sample.colors_rgb, dtype=np.uint8)
                )
            else:
                colors_chunks.append(np.empty((count, 0), dtype=np.uint8))
            cursor = end

        # Tiny per-record metadata. Offsets fit int32 because every individual
        # point count is bounded by the configured per-instance cap.
        host_views = self._host_record_views[:record_count].numpy()
        host_offsets = self._host_offsets[: record_count + 1].numpy()
        for i, record in enumerate(records):
            host_views[i] = int(record.view_index)
        host_offsets[:] = np.asarray(offsets, dtype=np.int32)

        profile_events = None

        if self._gpu_profile_enabled:
            profile_events = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(9)
            ]

            profile_events[0].record(
                torch.cuda.current_stream(device=self.device)
            )

        self._device_samples[:total].copy_(host_samples, non_blocking=True)
        self._device_record_views[:record_count].copy_(
            self._host_record_views[:record_count], non_blocking=True
        )
        self._device_offsets[: record_count + 1].copy_(
            self._host_offsets[: record_count + 1], non_blocking=True
        )

        if profile_events is not None:
            profile_events[1].record(
                torch.cuda.current_stream(device=self.device)
            )

        stream = self._current_warp_stream()
        wp.launch(
            _fused_geometry_kernel,
            dim=total,
            inputs=[
                self._wp["samples"],
                self._wp["record_views"],
                self._wp["transforms"],
                self._wp["transform_valid"],
                self._wp_origin,
                self._inv_voxel_size,
            ],
            outputs=[
                self._wp["points_camera"],
                self._wp["points_world"],
                self._wp["point_records"],
                self._wp["voxel_coords"],
                self._wp["voxel_keys"],
            ],
            device=self._wp_device,
            stream=stream,
        )

        if profile_events is not None:
            profile_events[2].record(
                torch.cuda.current_stream(device=self.device)
            )

        # One global lexicographic (record, key) ordering instead of one CUDA
        # stable sort per object.  Because the source is record-contiguous and
        # both sorts are stable, the first duplicate of each voxel remains the
        # first occurrence in the original point order, matching np.unique(...,
        # return_index=True).
        torch.sort(
            self._voxel_keys[:total],
            stable=True,
            out=(
                self._key_sorted_values[:total],
                self._key_sort_order[:total],
            ),
        )

        if profile_events is not None:
            profile_events[3].record(
                torch.cuda.current_stream(device=self.device)
            )

        torch.index_select(
            self._point_records[:total],
            0,
            self._key_sort_order[:total],
            out=self._records_in_key_order[:total],
        )

        if profile_events is not None:
            profile_events[4].record(
                torch.cuda.current_stream(device=self.device)
            )

        torch.sort(
            self._records_in_key_order[:total],
            stable=True,
            out=(
                self._record_sorted_values[:total],
                self._record_sort_order[:total],
            ),
        )

        if profile_events is not None:
            profile_events[5].record(
                torch.cuda.current_stream(device=self.device)
            )

        # Empty records have no element that can write their count in the scatter
        # kernel, so clear the small count array before the parallel dedup path.
        self._unique_counts[:record_count].zero_()

        # 1) Mark the first entry of every (record, voxel_key) run in parallel.
        wp.launch(
            _mark_unique_sorted_kernel,
            dim=total,
            inputs=[
                self._wp["key_sorted_values"],
                self._wp["record_sorted_values"],
                self._wp["record_sort_order"],
            ],
            outputs=[
                self._wp["unique_flags"],
            ],
            device=self._wp_device,
            stream=stream,
        )

        if profile_events is not None:
            profile_events[6].record(
                torch.cuda.current_stream(device=self.device)
            )

        # 2) One global inclusive scan is sufficient because the record segments
        # are contiguous after the stable record sort.  The scatter kernel turns
        # the global prefix into a per-record rank by subtracting prefix[begin-1].
        torch.cumsum(
            self._unique_flags[:total],
            dim=0,
            dtype=torch.int32,
            out=self._unique_prefix[:total],
        )

        if profile_events is not None:
            profile_events[7].record(
                torch.cuda.current_stream(device=self.device)
            )

        # 3) Scatter each unique representative to the front of its raw record
        # segment. This preserves sorted voxel-key order and the first-source
        # representative semantics of np.unique(..., return_index=True).
        wp.launch(
            _scatter_unique_sorted_kernel,
            dim=total,
            inputs=[
                self._wp["offsets"],
                self._wp["key_sorted_values"],
                self._wp["key_sort_order"],
                self._wp["record_sorted_values"],
                self._wp["record_sort_order"],
                self._wp["unique_flags"],
                self._wp["unique_prefix"],
                self._wp["points_world"],
                self._wp["voxel_coords"],
            ],
            outputs=[
                self._wp["unique_counts"],
                self._wp["unique_global_indices"],
                self._wp["unique_keys"],
                self._wp["unique_coords"],
                self._wp["unique_points"],
            ],
            device=self._wp_device,
            stream=stream,
        )

        if profile_events is not None:
            profile_events[8].record(
                torch.cuda.current_stream(device=self.device)
            )
            self._last_gpu_profile_events = profile_events
            self._last_gpu_profile_pairs = None
            self._last_gpu_profile_names = (
                "h2d", "fused_warp", "sort_keys", "index_select",
                "sort_records", "mark_unique", "prefix_sum", "scatter_unique",
            )
            self._last_cpu_profile_ms = None

        if colors_chunks and all(
            chunk.ndim == 2 and chunk.shape[1] == 3 for chunk in colors_chunks
        ):
            colors_flat = np.ascontiguousarray(
                np.concatenate(colors_chunks, axis=0), dtype=np.uint8
            )
        else:
            colors_flat = np.empty((0, 3), dtype=np.uint8)

        return _PendingBatch(
            lengths=lengths,
            offsets=offsets,
            colors_flat=colors_flat,
            points_camera=self._points_camera[:total],
            points_world=self._points_world[:total],
            unique_counts=self._unique_counts[:record_count],
            unique_global_indices=self._unique_global_indices[:total],
            unique_coords=self._unique_coords[:total],
            unique_keys=self._unique_keys[:total],
            unique_points=self._unique_points[:total],
        )

    def materialize_compact(
        self,
        pending: _PendingBatch,
        records: list[Any],
        frames: list[Any],
    ) -> list[GPUGeometryRecord]:
        """Materialize the exact CPU-alignment voxel packet.

        The direct no-sync backend stores each record's unique representatives in
        the raw compacted record segment::

            [raw_offsets[r], raw_offsets[r] + unique_counts[r])

        To preserve CPU-alignment semantics exactly, this compact path copies
        those exact representative slices to pinned host memory. Raw per-pixel
        camera/world clouds remain GPU-only.
        """
        record_count = len(records)
        if record_count == 0:
            return []
        if pending.deferred_lengths_gpu is None:
            return self.materialize(pending, records, frames)

        current = torch.cuda.current_stream(device=self.device)

        # Phase 1: tiny per-record metadata only.
        self._host_compact_lengths[:record_count].copy_(
            pending.deferred_lengths_gpu[:record_count], non_blocking=True
        )
        self._host_compact_counts[:record_count].copy_(
            pending.unique_counts[:record_count], non_blocking=True
        )
        if pending.direct_bbox_gpu is not None:
            self._host_compact_bbox[:record_count].copy_(
                pending.direct_bbox_gpu[:record_count], non_blocking=True
            )
            self._host_compact_foreground[:record_count].copy_(
                pending.direct_foreground_gpu[:record_count], non_blocking=True
            )
            self._host_compact_strides[:record_count].copy_(
                pending.direct_strides_gpu[:record_count], non_blocking=True
            )
        current.synchronize()

        lengths_np = self._host_compact_lengths[:record_count].numpy()
        counts_np = self._host_compact_counts[:record_count].numpy()
        lengths = [int(v) for v in lengths_np]
        counts = [int(v) for v in counts_np]
        raw_offsets = self._offsets(lengths)
        unique_offsets = self._offsets(counts)
        unique_total = int(unique_offsets[-1])

        # Phase 2: exact unique slices, preserving key->representative mapping.
        if unique_total > 0:
            for record_index in range(record_count):
                count = counts[record_index]
                if count <= 0:
                    continue
                src_begin = raw_offsets[record_index]
                src_end = src_begin + count
                dst_begin = unique_offsets[record_index]
                dst_end = dst_begin + count

                self._host_compact_keys[dst_begin:dst_end].copy_(
                    pending.unique_keys[src_begin:src_end], non_blocking=True
                )
                self._host_compact_coords[dst_begin:dst_end].copy_(
                    pending.unique_coords[src_begin:src_end], non_blocking=True
                )
                self._host_compact_points[dst_begin:dst_end].copy_(
                    pending.unique_points[src_begin:src_end], non_blocking=True
                )

            current.synchronize()
            keys_cpu = self._host_compact_keys[:unique_total].numpy()
            coords_cpu = self._host_compact_coords[:unique_total].numpy()
            points_cpu = self._host_compact_points[:unique_total].numpy()
        else:
            keys_cpu = np.empty((0,), dtype=np.int64)
            coords_cpu = np.empty((0, 3), dtype=np.int32)
            points_cpu = np.empty((0, 3), dtype=np.float32)

        bbox_cpu = (
            self._host_compact_bbox[:record_count].numpy()
            if pending.direct_bbox_gpu is not None else None
        )
        foreground_cpu = (
            self._host_compact_foreground[:record_count].numpy()
            if pending.direct_foreground_gpu is not None else None
        )
        strides_cpu = (
            self._host_compact_strides[:record_count].numpy()
            if pending.direct_strides_gpu is not None else None
        )

        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        output: list[GPUGeometryRecord] = []

        for record_index, record in enumerate(records):
            ubegin = unique_offsets[record_index]
            uend = unique_offsets[record_index + 1]
            rbegin = raw_offsets[record_index]
            rend = raw_offsets[record_index + 1]
            frame = frames[int(record.view_index)]

            unique_coords = np.array(
                coords_cpu[ubegin:uend], dtype=np.int64, copy=True, order="C"
            )
            unique_keys = np.array(
                keys_cpu[ubegin:uend], dtype=np.int64, copy=True, order="C"
            )
            unique_points = np.array(
                points_cpu[ubegin:uend], dtype=np.float32, copy=True, order="C"
            )

            has_world = frame.world_from_camera is not None
            if has_world and unique_coords.size:
                shifted = unique_coords + int(self._BIAS)
                if np.any(shifted < 0) or np.any(shifted > int(self._MASK)):
                    raise ValueError(
                        "World voxel coordinate exceeded the compact 21-bit/key "
                        "range. Move shared_voxel_grid.origin_world closer to the "
                        "workspace or increase voxel_size_m."
                    )
                voxel_bbox_min = unique_coords.min(axis=0)
                voxel_bbox_max = unique_coords.max(axis=0)
            else:
                voxel_bbox_min = voxel_bbox_max = None

            bbox_2d = None
            fg = 0
            stride = 1
            if bbox_cpu is not None:
                x0, y0, x1, y1 = [int(v) for v in bbox_cpu[record_index]]
                if x1 >= x0 and y1 >= y0 and x1 >= 0 and y1 >= 0:
                    bbox_2d = (x0, y0, x1, y1)
                fg = int(foreground_cpu[record_index])
                stride = int(strides_cpu[record_index])

            cpu_world = unique_points if has_world else None
            output.append(
                GPUGeometryRecord(
                    points_camera=empty_points,
                    points_world=cpu_world,
                    colors_rgb=empty_colors,
                    voxel_coords=(
                        unique_coords
                        if has_world and len(unique_coords)
                        else None
                    ),
                    voxel_keys=(
                        unique_keys
                        if has_world and len(unique_keys)
                        else None
                    ),
                    voxel_points=(
                        unique_points
                        if has_world and len(unique_points)
                        else None
                    ),
                    voxel_colors=None,
                    voxel_bbox_min=voxel_bbox_min,
                    voxel_bbox_max=voxel_bbox_max,
                    points_camera_gpu=pending.points_camera[rbegin:rend],
                    points_world_gpu=(
                        pending.points_world[rbegin:rend]
                        if has_world else None
                    ),
                    voxel_coords_gpu=(
                        pending.unique_coords[
                            rbegin : rbegin + counts[record_index]
                        ]
                        if has_world and counts[record_index] > 0 else None
                    ),
                    voxel_keys_gpu=(
                        pending.unique_keys[
                            rbegin : rbegin + counts[record_index]
                        ]
                        if has_world and counts[record_index] > 0 else None
                    ),
                    voxel_points_gpu=(
                        pending.unique_points[
                            rbegin : rbegin + counts[record_index]
                        ]
                        if has_world and counts[record_index] > 0 else None
                    ),
                    bbox_2d=bbox_2d,
                    foreground_pixels=fg,
                    geometry_stride=stride,
                )
            )
        return output

    def materialize(
        self,
        pending: _PendingBatch,
        records: list[Any],
        frames: list[Any],
    ) -> list[GPUGeometryRecord]:
        """Materialize the full NumPy fallback/visualization interface.

        CPU alignment/visualization still needs NumPy arrays in this fallback
        path; matching CUDA slices are retained for downstream consumers.
        """
        # In direct no-sync mode, resolve the six-ish raw lengths only here at
        # the intentional GPU->CPU compatibility boundary. This keeps the GPU
        # compute stage fully asynchronous while still letting CPU alignment see
        # exactly the same packed variable-length representation.
        if pending.deferred_lengths_gpu is not None:
            lengths_cpu = (
                pending.deferred_lengths_gpu.detach().cpu().numpy().astype(
                    np.int32, copy=False
                )
            )
            lengths = [int(v) for v in lengths_cpu]
            offsets = self._offsets(lengths)
        else:
            lengths = pending.lengths
            offsets = pending.offsets

        total = offsets[-1] if offsets else 0

        pc_cpu = pending.points_camera[:total].detach().cpu().numpy()
        pw_cpu = pending.points_world[:total].detach().cpu().numpy()
        counts_cpu = pending.unique_counts.detach().cpu().numpy().astype(
            np.int32, copy=False
        )
        unique_coords_cpu = pending.unique_coords[:total].detach().cpu().numpy()
        unique_keys_cpu = pending.unique_keys[:total].detach().cpu().numpy()
        unique_points_cpu = pending.unique_points[:total].detach().cpu().numpy()
        unique_global_cpu = pending.unique_global_indices[:total].detach().cpu().numpy()
        bbox_cpu = (
            pending.direct_bbox_gpu.detach().cpu().numpy().astype(np.int32, copy=False)
            if pending.direct_bbox_gpu is not None else None
        )
        foreground_cpu = (
            pending.direct_foreground_gpu.detach().cpu().numpy().astype(np.int32, copy=False)
            if pending.direct_foreground_gpu is not None else None
        )
        strides_cpu = (
            pending.direct_strides_gpu.detach().cpu().numpy().astype(np.int32, copy=False)
            if pending.direct_strides_gpu is not None else None
        )

        output: list[GPUGeometryRecord] = []
        for record_index, record in enumerate(records):
            begin, end = offsets[record_index], offsets[record_index + 1]
            unique_count = int(counts_cpu[record_index])
            ubegin = begin
            uend = begin + unique_count
            frame = frames[int(record.view_index)]

            points_camera = np.ascontiguousarray(pc_cpu[begin:end], dtype=np.float32)
            points_world = (
                np.ascontiguousarray(pw_cpu[begin:end], dtype=np.float32)
                if frame.world_from_camera is not None
                else None
            )
            unique_coords = np.ascontiguousarray(
                unique_coords_cpu[ubegin:uend], dtype=np.int64
            )
            unique_keys = np.ascontiguousarray(
                unique_keys_cpu[ubegin:uend], dtype=np.int64
            )
            unique_points = np.ascontiguousarray(
                unique_points_cpu[ubegin:uend], dtype=np.float32
            )

            if unique_coords.size:
                shifted = unique_coords + int(self._BIAS)
                if np.any(shifted < 0) or np.any(shifted > int(self._MASK)):
                    raise ValueError(
                        "World voxel coordinate exceeded the compact 21-bit/key "
                        "range. Move shared_voxel_grid.origin_world closer to the "
                        "workspace or increase voxel_size_m."
                    )

            if pending.colors_flat.size:
                colors = np.ascontiguousarray(
                    pending.colors_flat[begin:end], dtype=np.uint8
                )
                keep = unique_global_cpu[ubegin:uend] - begin
                unique_colors = np.ascontiguousarray(colors[keep], dtype=np.uint8)
            else:
                colors = np.empty((0, 3), dtype=np.uint8)
                unique_colors = np.empty((0, 3), dtype=np.uint8)

            has_world = frame.world_from_camera is not None
            if has_world and len(unique_coords):
                voxel_bbox_min = unique_coords.min(axis=0)
                voxel_bbox_max = unique_coords.max(axis=0)
            else:
                voxel_bbox_min = None
                voxel_bbox_max = None

            bbox_2d = None
            fg = 0
            geometry_stride = 1
            if bbox_cpu is not None:
                bx0, by0, bx1, by1 = [int(v) for v in bbox_cpu[record_index]]
                if bx1 >= bx0 and by1 >= by0 and bx1 >= 0 and by1 >= 0:
                    bbox_2d = (bx0, by0, bx1, by1)
                fg = int(foreground_cpu[record_index])
                geometry_stride = int(strides_cpu[record_index])

            output.append(
                GPUGeometryRecord(
                    points_camera=points_camera,
                    points_world=points_world,
                    colors_rgb=colors,
                    voxel_coords=(
                        unique_coords if has_world and len(unique_coords) else None
                    ),
                    voxel_keys=(
                        unique_keys if has_world and len(unique_keys) else None
                    ),
                    voxel_points=(
                        unique_points if has_world and len(unique_points) else None
                    ),
                    voxel_colors=(
                        unique_colors if has_world and len(unique_colors) else None
                    ),
                    voxel_bbox_min=voxel_bbox_min,
                    voxel_bbox_max=voxel_bbox_max,
                    points_camera_gpu=pending.points_camera[begin:end],
                    points_world_gpu=(
                        pending.points_world[begin:end] if has_world else None
                    ),
                    voxel_coords_gpu=(
                        pending.unique_coords[ubegin:uend]
                        if has_world and unique_count > 0
                        else None
                    ),
                    voxel_keys_gpu=(
                        pending.unique_keys[ubegin:uend]
                        if has_world and unique_count > 0
                        else None
                    ),
                    voxel_points_gpu=(
                        pending.unique_points[ubegin:uend]
                        if has_world and unique_count > 0
                        else None
                    ),
                    bbox_2d=bbox_2d,
                    foreground_pixels=fg,
                    geometry_stride=geometry_stride,
                )
            )
        return output