from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

from .data_types import (
    FrameResult,
    ProcessedInstance,
    RGBDFrame,
    TrackerPrediction,
    VisibilityState,
)
from .processing import _erosion_kernel
from .gpu_geometry import GeometrySamples, GPUSparseGeometryBackend


@dataclass(slots=True)
class _MaskRecord:
    view_index: int
    channel: int
    track_id: int
    raw_mask: np.ndarray | None = None
    final_mask: np.ndarray | None = None
    final_mask_gpu: Any | None = None
    raw_bbox_2d: tuple[int, int, int, int] | None = None
    bbox_2d: tuple[int, int, int, int] | None = None
    # Geometry coordinates are extracted once while the cleaned CC ROI is hot.
    # Geometry consumes these directly instead of rescanning the mask.
    geometry_y: np.ndarray | None = None
    geometry_x: np.ndarray | None = None
    foreground_pixels: int = 0
    geometry_stride: int = 1


@dataclass(slots=True)
class _GeometryRecord:
    points_camera: np.ndarray
    points_world: np.ndarray | None
    colors_rgb: np.ndarray
    centroid_camera: np.ndarray | None = None
    centroid_world: np.ndarray | None = None
    bbox_min: np.ndarray | None = None
    bbox_max: np.ndarray | None = None
    voxel_coords: np.ndarray | None = None
    voxel_keys: np.ndarray | None = None
    voxel_points: np.ndarray | None = None
    voxel_colors: np.ndarray | None = None
    voxel_bbox_min: np.ndarray | None = None
    voxel_bbox_max: np.ndarray | None = None
    # CUDA views retained alongside the CPU alignment representation.
    points_camera_gpu: Any | None = None
    points_world_gpu: Any | None = None
    voxel_coords_gpu: Any | None = None
    voxel_keys_gpu: Any | None = None
    voxel_points_gpu: Any | None = None


class BatchedPostprocessor:
    """Batched mask cleanup and sparse RGB-D geometry for all camera views.

    The production CUDA path keeps masks/depth on GPU, emits compact CPU voxel
    views for alignment, and materializes full CPU masks only when visualization
    or SAM3 refresh actually needs them. CPU fallbacks are retained for tracker
    standalone/debug use.
    """

    def __init__(self, config, num_views: int, *, voxelizer: Any | None = None) -> None:
        self.config = config
        self.num_views = int(num_views)
        self.voxelizer = voxelizer
        workers = int(
            config.postprocess.get(
                "cpu_workers", config.runtime.get("postprocess_workers", 0)
            )
        )
        if workers <= 0:
            workers = min(8, max(1, os.cpu_count() or 1))
        self._pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="tracking-postprocess",
        )
        # We parallelize independent masks/ROIs explicitly; nested OpenCV worker
        # pools only oversubscribe the CPU in this path.
        cv2.setNumThreads(1)
        requested_gpu = bool(
            config.postprocess.get(
                "gpu_batch", config.runtime.get("gpu_postprocess", True)
            )
        )
        device_name = str(config.runtime.get("device", "cuda"))
        self.mask_stage_cuda = bool(
            requested_gpu
            and torch is not None
            and torch.cuda.is_available()
            and device_name.startswith("cuda")
        )
        self.device = torch.device(device_name) if self.mask_stage_cuda else None
        self._ray_cache: dict[
            tuple[float, float, float, float, int, int],
            tuple[np.ndarray, np.ndarray],
        ] = {}
        # Frozen production fast path. ``gpu_geometry`` is the only geometry
        # switch that remains: when CUDA geometry is enabled, direct mask/depth
        # geometry, compact alignment D2H, depth prefetch, and lazy CPU masks are
        # always enabled together. This preserves the benchmarked runtime path and
        # removes configuration combinations that are no longer supported.
        self.gpu_geometry_enabled = bool(
            self.mask_stage_cuda
            and self.voxelizer is not None
            and config.postprocess.get("gpu_geometry", True)
        )
        self._gpu_geometry = (
            GPUSparseGeometryBackend(self.device, self.voxelizer)
            if self.gpu_geometry_enabled
            else None
        )
        self._depth_prefetch_future = None
        self._depth_prefetch_key = None

    def submit(self, function, /, *args):
        """Submit one small CPU job to the persistent postprocess pool."""
        return self._pool.submit(function, *args)

    @staticmethod
    def _frame_key(frames: list[RGBDFrame]) -> tuple[Any, ...]:
        return tuple(
            (str(frame.camera_name), int(frame.frame_index), int(frame.timestamp_ns))
            for frame in frames
        )

    def prefetch_depth_async(self, frames: list[RGBDFrame]) -> None:
        if (
            not self.gpu_geometry_enabled
            or self._gpu_geometry is None
            or not frames
            or bool(self.config.runtime.get("enable_visualization", True))
        ):
            return
        key = self._frame_key(frames)
        self._depth_prefetch_key = key
        self._depth_prefetch_future = self._pool.submit(
            self._gpu_geometry.prefetch_depth, frames
        )

    def _finish_depth_prefetch(self, frames: list[RGBDFrame]) -> bool:
        if self._depth_prefetch_future is None:
            return False
        if self._depth_prefetch_key != self._frame_key(frames):
            return False
        self._depth_prefetch_future.result()
        self._depth_prefetch_future = None
        return True

    def depth_prefetch_profile_ms(self) -> dict[str, float] | None:
        if self._gpu_geometry is None:
            return None
        return self._gpu_geometry.depth_prefetch_profile_ms()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _rays(self, frame: RGBDFrame) -> tuple[np.ndarray, np.ndarray]:
        k = frame.intrinsics
        key = (k.fx, k.fy, k.cx, k.cy, k.width, k.height)
        cached = self._ray_cache.get(key)
        if cached is not None:
            return cached
        x_ray = (
            np.arange(k.width, dtype=np.float32) - np.float32(k.cx)
        ) / np.float32(k.fx)
        y_ray = (
            np.arange(k.height, dtype=np.float32) - np.float32(k.cy)
        ) / np.float32(k.fy)
        cached = (x_ray, y_ray)
        self._ray_cache[key] = cached
        return cached

    @staticmethod
    def _filter_component_roi_inplace(
        mask_u8: np.ndarray,
        min_component_pixels: int,
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None, int]:
        """Filter connected components only inside the nonzero ROI.

        Returns the exact number of retained foreground pixels from CC statistics so
        later geometry sampling can choose its stride before extracting coordinates.
        """
        x, y, width, height = cv2.boundingRect(mask_u8)
        if width <= 0 or height <= 0:
            mask_u8.fill(0)
            return mask_u8, None, 0

        if min_component_pixels <= 1:
            foreground_pixels = int(cv2.countNonZero(mask_u8[y : y + height, x : x + width]))
            return mask_u8, (x, y, x + width - 1, y + height - 1), foreground_pixels

        roi = mask_u8[y : y + height, x : x + width]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            roi, connectivity=8
        )
        if count <= 1:
            mask_u8.fill(0)
            return mask_u8, None, 0

        component_ids = np.arange(1, count, dtype=np.int32)
        valid = component_ids[
            stats[1:, cv2.CC_STAT_AREA] >= int(min_component_pixels)
        ]
        if valid.size == 0:
            mask_u8.fill(0)
            return mask_u8, None, 0

        keep_lut = np.zeros(count, dtype=np.uint8)
        keep_lut[valid] = 1
        filtered_roi = keep_lut[labels]
        mask_u8.fill(0)
        mask_u8[y : y + height, x : x + width] = filtered_roi

        valid_stats = stats[valid]
        x0 = x + int(valid_stats[:, cv2.CC_STAT_LEFT].min())
        y0 = y + int(valid_stats[:, cv2.CC_STAT_TOP].min())
        x1 = x + int(
            (
                valid_stats[:, cv2.CC_STAT_LEFT]
                + valid_stats[:, cv2.CC_STAT_WIDTH]
                - 1
            ).max()
        )
        y1 = y + int(
            (
                valid_stats[:, cv2.CC_STAT_TOP]
                + valid_stats[:, cv2.CC_STAT_HEIGHT]
                - 1
            ).max()
        )
        foreground_pixels = int(valid_stats[:, cv2.CC_STAT_AREA].sum())
        return mask_u8, (x0, y0, x1, y1), foreground_pixels

    @staticmethod
    def _adaptive_geometry_stride(
        base_stride: int,
        foreground_pixels: int,
        max_points: int,
        enabled: bool,
    ) -> int:
        """Choose a larger global-lattice stride before coordinate extraction.

        The cleaned mask itself is unchanged.  The returned stride is always an
        integer multiple of ``base_stride``, so adaptive sampling remains a subset
        of the configured global image sampling lattice.
        """
        base_stride = max(1, int(base_stride))
        if not enabled or max_points <= 0 or foreground_pixels <= 0:
            return base_stride
        estimated_base_samples = foreground_pixels / float(base_stride * base_stride)
        if estimated_base_samples <= max_points:
            return base_stride
        multiplier = int(np.ceil(np.sqrt(estimated_base_samples / float(max_points))))
        return base_stride * max(1, multiplier)

    @staticmethod
    def _sample_geometry_coordinates(
        mask: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
        stride: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract compact global (y,x) foreground coordinates once per CC ROI."""
        if bbox is None:
            empty = np.empty((0,), dtype=np.int32)
            return empty, empty.copy()
        x0, y0, x1, y1 = bbox
        stride = max(1, int(stride))
        xs0 = ((x0 + stride - 1) // stride) * stride
        ys0 = ((y0 + stride - 1) // stride) * stride
        if xs0 > x1 or ys0 > y1:
            empty = np.empty((0,), dtype=np.int32)
            return empty, empty.copy()
        roi = np.asarray(
            mask[ys0 : y1 + 1 : stride, xs0 : x1 + 1 : stride],
            dtype=bool,
        )
        local_y, local_x = np.nonzero(roi)
        if local_y.size == 0:
            empty = np.empty((0,), dtype=np.int32)
            return empty, empty.copy()
        ys = np.asarray(ys0 + local_y * stride, dtype=np.int32)
        xs = np.asarray(xs0 + local_x * stride, dtype=np.int32)
        return ys, xs

    def _collect_records(
        self,
        views: list[Any],
        frames: list[RGBDFrame],
        predictions: list[TrackerPrediction],
    ) -> tuple[
        list[_MaskRecord],
        dict[tuple[int, int, int, int], list[int]],
        list[Any],
    ]:
        records: list[_MaskRecord] = []
        resize_groups: dict[tuple[int, int, int, int], list[int]] = {}
        logits_per_view: list[Any] = []

        for view_index, (view, frame, prediction) in enumerate(
            zip(views, frames, predictions)
        ):
            logits = prediction.mask_logits
            if torch is not None and torch.is_tensor(logits):
                if logits.ndim == 2:
                    logits = logits[None]
                if logits.ndim != 3:
                    raise ValueError(
                        f"Unexpected tracker mask shape: {tuple(logits.shape)}"
                    )
            else:
                logits = np.asarray(logits, dtype=np.float32)
                if logits.ndim == 2:
                    logits = logits[None]
                if logits.ndim != 3:
                    raise ValueError(
                        f"Unexpected tracker mask shape: {logits.shape}"
                    )
            logits_per_view.append(logits)
            dst_h, dst_w = frame.depth_m.shape
            src_h = int(logits.shape[-2]) if int(logits.shape[0]) else dst_h
            src_w = int(logits.shape[-1]) if int(logits.shape[0]) else dst_w
            key = (src_h, src_w, dst_h, dst_w)
            for channel, value in enumerate(prediction.track_ids):
                track_id = int(value)
                track = view.tracks.get(track_id)
                if (
                    track is None
                    or not track.active
                    or channel >= int(logits.shape[0])
                ):
                    continue
                index = len(records)
                records.append(_MaskRecord(view_index, channel, track_id))
                resize_groups.setdefault(key, []).append(index)
        return records, resize_groups, logits_per_view

    @staticmethod
    def _resize_threshold_erode_cpu(
        logits: np.ndarray,
        width: int,
        height: int,
        threshold: float,
        erosion_pixels: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if logits.shape != (height, width):
            resized = cv2.resize(
                np.asarray(logits, dtype=np.float32),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized = np.asarray(logits, dtype=np.float32)
        raw = resized > threshold
        if erosion_pixels > 0 and np.any(raw):
            eroded = cv2.erode(
                raw.astype(np.uint8, copy=False),
                _erosion_kernel(erosion_pixels),
                iterations=1,
            )
        else:
            eroded = raw.astype(np.uint8, copy=False)
        return raw, eroded

    def _batch_masks_gpu(
        self,
        records: list[_MaskRecord],
        resize_groups: dict[tuple[int, int, int, int], list[int]],
        logits_per_view: list[Any],
        *,
        need_raw_masks: bool,
        need_final_masks: bool,
    ) -> list[tuple[_MaskRecord, np.ndarray | None]]:
        assert torch is not None and F is not None and self.device is not None
        threshold = float(self.config.postprocess.mask_threshold)
        erosion_pixels = int(self.config.postprocess.erosion_pixels)
        pending: list[tuple[_MaskRecord, np.ndarray]] = []

        for (_, _, dst_h, dst_w), indices in resize_groups.items():
            tensors = []
            for record_index in indices:
                record = records[record_index]
                source = logits_per_view[record.view_index][record.channel]
                if torch.is_tensor(source):
                    value = source.detach()
                    same_device = (
                        value.device.type == self.device.type
                        and (
                            self.device.index is None
                            or value.device.index == self.device.index
                        )
                    )
                    if not same_device:
                        value = value.to(self.device, non_blocking=True)
                    if not value.is_floating_point():
                        value = value.float()
                else:
                    value = torch.as_tensor(
                        np.asarray(source, dtype=np.float32),
                        device=self.device,
                    )
                tensors.append(value)
            batch = torch.stack(tensors, dim=0).unsqueeze(1)
            if tuple(batch.shape[-2:]) != (dst_h, dst_w):
                batch = F.interpolate(
                    batch,
                    size=(dst_h, dst_w),
                    mode="bilinear",
                    align_corners=False,
                )
            raw = batch[:, 0] > threshold
            if erosion_pixels > 0:
                k = 2 * erosion_pixels + 1
                background = (~raw).to(torch.float32).unsqueeze(1)
                eroded = (
                    F.max_pool2d(
                        background,
                        kernel_size=k,
                        stride=1,
                        padding=erosion_pixels,
                    )[:, 0]
                    < 0.5
                )
            else:
                eroded = raw

            # Normal tracking needs only the cleaned/eroded mask on CPU.  Raw
            # full-resolution masks are transferred only when they are actually
            # consumed: debug-image publication or the exact frame submitted to
            # asynchronous SAM3 as its fallback reference.
            eroded_u8_gpu = eroded.to(torch.uint8)
            if need_raw_masks:
                packed = torch.cat((raw.to(torch.uint8), eroded_u8_gpu), dim=0).cpu().numpy()
                count = len(indices)
                raw_cpu = packed[:count]
                eroded_cpu = packed[count:]
            elif need_final_masks:
                raw_cpu = None
                eroded_cpu = eroded_u8_gpu.cpu().numpy()
            else:
                raw_cpu = None
                eroded_cpu = None
            for local_index, record_index in enumerate(indices):
                record = records[record_index]
                # Keep a zero-copy view of the eroded mask on CUDA for the direct
                # mask/depth -> geometry fast path. The parent tensor stays alive
                # through this view for the duration of process().
                record.final_mask_gpu = eroded_u8_gpu[local_index]
                if raw_cpu is not None:
                    record.raw_mask = raw_cpu[local_index].view(np.bool_)
                pending.append((
                    record,
                    None if eroded_cpu is None else eroded_cpu[local_index],
                ))
        return pending

    def _batch_masks_cpu(
        self,
        records: list[_MaskRecord],
        resize_groups: dict[tuple[int, int, int, int], list[int]],
        logits_per_view: list[np.ndarray],
        *,
        need_raw_masks: bool,
        need_final_masks: bool,
    ) -> list[tuple[_MaskRecord, np.ndarray | None]]:
        """CPU fallback: submit all independent masks as one persistent task batch."""
        threshold = float(self.config.postprocess.mask_threshold)
        erosion_pixels = int(self.config.postprocess.erosion_pixels)
        jobs = []
        for (_, _, dst_h, dst_w), indices in resize_groups.items():
            for record_index in indices:
                record = records[record_index]
                jobs.append(
                    (
                        record,
                        self._pool.submit(
                            self._resize_threshold_erode_cpu,
                            (
                                logits_per_view[record.view_index][record.channel]
                                .detach().float().cpu().numpy()
                                if torch is not None
                                and torch.is_tensor(
                                    logits_per_view[record.view_index][record.channel]
                                )
                                else logits_per_view[record.view_index][record.channel]
                            ),
                            dst_w,
                            dst_h,
                            threshold,
                            erosion_pixels,
                        ),
                    )
                )
        pending: list[tuple[_MaskRecord, np.ndarray]] = []
        for record, future in jobs:
            raw, eroded = future.result()
            if need_raw_masks:
                record.raw_mask = raw
            pending.append((record, eroded))
        return pending

    def _batch_masks(
        self,
        records: list[_MaskRecord],
        resize_groups: dict[tuple[int, int, int, int], list[int]],
        logits_per_view: list[np.ndarray],
        *,
        need_raw_masks: bool,
        need_final_masks: bool,
    ) -> list[tuple[_MaskRecord, np.ndarray | None]]:
        if self.mask_stage_cuda:
            return self._batch_masks_gpu(
                records, resize_groups, logits_per_view,
                need_raw_masks=need_raw_masks, need_final_masks=need_final_masks
            )
        return self._batch_masks_cpu(
            records, resize_groups, logits_per_view,
            need_raw_masks=need_raw_masks, need_final_masks=need_final_masks
        )

    @classmethod
    def _filter_record_masks(
        cls,
        raw_mask: np.ndarray | None,
        eroded_u8: np.ndarray,
        min_component_pixels: int,
        base_stride: int,
        max_points: int,
        adaptive_sampling: bool,
    ) -> tuple[
        np.ndarray,
        tuple[int, int, int, int] | None,
        tuple[int, int, int, int] | None,
        np.ndarray,
        np.ndarray,
        int,
        int,
    ]:
        raw_bbox = None
        if raw_mask is not None:
            raw_u8 = np.asarray(raw_mask).view(np.uint8)
            x, y, width, height = cv2.boundingRect(raw_u8)
            if width and height:
                raw_bbox = (x, y, x + width - 1, y + height - 1)

        filtered, bbox, foreground_pixels = cls._filter_component_roi_inplace(
            eroded_u8, min_component_pixels
        )
        geometry_stride = cls._adaptive_geometry_stride(
            base_stride, foreground_pixels, max_points, adaptive_sampling
        )
        ys, xs = cls._sample_geometry_coordinates(filtered, bbox, geometry_stride)
        return (
            filtered,
            bbox,
            raw_bbox,
            ys,
            xs,
            foreground_pixels,
            geometry_stride,
        )

    @classmethod
    def _filter_record_masks_fast(
        cls,
        raw_mask: np.ndarray | None,
        eroded_u8: np.ndarray,
        base_stride: int,
        max_points: int,
        adaptive_sampling: bool,
    ) -> tuple[
        np.ndarray,
        tuple[int, int, int, int] | None,
        tuple[int, int, int, int] | None,
        int,
        int,
    ]:
        """Cheap common path: erosion already removed tiny speckles.

        This intentionally skips general connected-components. It only computes
        bbox + foreground count, which are sufficient to choose the same adaptive
        global-lattice stride used by geometry. Exact connected-components remains available in the CPU fallback path.
        """
        raw_bbox = None
        if raw_mask is not None:
            raw_u8 = np.asarray(raw_mask).view(np.uint8)
            x, y, width, height = cv2.boundingRect(raw_u8)
            if width and height:
                raw_bbox = (x, y, x + width - 1, y + height - 1)

        x, y, width, height = cv2.boundingRect(eroded_u8)
        if width <= 0 or height <= 0:
            eroded_u8.fill(0)
            return eroded_u8, None, raw_bbox, 0, max(1, int(base_stride))

        bbox = (x, y, x + width - 1, y + height - 1)
        foreground_pixels = int(cv2.countNonZero(eroded_u8[y : y + height, x : x + width]))
        geometry_stride = cls._adaptive_geometry_stride(
            base_stride, foreground_pixels, max_points, adaptive_sampling
        )
        return eroded_u8, bbox, raw_bbox, foreground_pixels, geometry_stride

    def _batch_components(
        self,
        pending: list[tuple[_MaskRecord, np.ndarray | None]],
    ) -> None:
        """Run CPU CC or the lazy GPU bbox/count/stride metadata path."""
        min_component_pixels = int(self.config.postprocess.min_component_pixels)
        base_stride = max(1, int(self.config.pointcloud.stride))
        max_points = int(self.config.pointcloud.max_points_per_instance)
        adaptive_sampling = bool(
            self.config.postprocess.get("adaptive_geometry_sampling", True)
        )
        if (
            self.gpu_geometry_enabled
            and self._gpu_geometry is not None
            and pending
            and all(mask is None for _, mask in pending)
        ):
            masks_gpu = [record.final_mask_gpu for record, _ in pending]
            if all(mask is not None for mask in masks_gpu):
                self._gpu_geometry.prepare_direct_masks(
                    [record for record, _ in pending],
                    masks_gpu,
                    view_count=self.num_views,
                    base_stride=base_stride,
                    max_points=max_points,
                    adaptive_sampling=adaptive_sampling,
                )
                for record, _ in pending:
                    record.final_mask = None
                    record.geometry_y = None
                    record.geometry_x = None
                    record.geometry_stride = 0
                return
        if self.gpu_geometry_enabled:
            jobs = [
                (
                    record,
                    self._pool.submit(
                        self._filter_record_masks_fast,
                        record.raw_mask,
                        mask_u8,
                        base_stride,
                        max_points,
                        adaptive_sampling,
                    ),
                )
                for record, mask_u8 in pending
            ]
            for record, future in jobs:
                filtered, bbox, raw_bbox, foreground_pixels, geometry_stride = future.result()
                record.final_mask = filtered.view(np.bool_)
                record.bbox_2d = bbox
                record.raw_bbox_2d = raw_bbox

                # This branch is used when a CPU mask is intentionally
                # materialized (standalone RViz/debug and occasional refresh
                # frames).  The fast mask filter only computes bbox/count/stride,
                # so recover the sparse global-lattice coordinates once here for
                # the legacy/full geometry materializer.  Normal headless
                # production frames return through prepare_direct_masks() above
                # and therefore pay none of this CPU work.
                record.geometry_y, record.geometry_x = (
                    self._sample_geometry_coordinates(
                        record.final_mask,
                        record.bbox_2d,
                        int(geometry_stride),
                    )
                )
                record.foreground_pixels = int(foreground_pixels)
                record.geometry_stride = int(geometry_stride)
            return

        jobs = [
            (
                record,
                self._pool.submit(
                    self._filter_record_masks,
                    record.raw_mask,
                    mask_u8,
                    min_component_pixels,
                    base_stride,
                    max_points,
                    adaptive_sampling,
                ),
            )
            for record, mask_u8 in pending
        ]
        for record, future in jobs:
            (
                filtered,
                bbox,
                raw_bbox,
                ys,
                xs,
                foreground_pixels,
                geometry_stride,
            ) = future.result()
            record.final_mask = filtered.view(np.bool_)
            record.bbox_2d = bbox
            record.raw_bbox_2d = raw_bbox
            record.geometry_y = ys
            record.geometry_x = xs
            record.foreground_pixels = int(foreground_pixels)
            record.geometry_stride = int(geometry_stride)

    def _geometry_samples_one(
        self,
        record: _MaskRecord,
        frame: RGBDFrame,
        *,
        need_colors: bool,
    ) -> GeometrySamples:
        """Apply the exact existing depth-validity and max-point selection policy.

        Connected-components already chose the sparse image lattice.  Keeping this
        tiny selection step on CPU preserves the old ordering/capping semantics;
        all XYZ/world/voxel arithmetic after it is handled by the packed CUDA path.
        """
        ys = record.geometry_y
        xs = record.geometry_x
        if ys is None or xs is None or ys.size == 0:
            return GeometrySamples(
                ys=np.empty((0,), dtype=np.int64),
                xs=np.empty((0,), dtype=np.int64),
                z=np.empty((0,), dtype=np.float32),
                colors_rgb=np.empty((0, 3), dtype=np.uint8),
            )

        ys = np.asarray(ys, dtype=np.intp)
        xs = np.asarray(xs, dtype=np.intp)
        z = np.asarray(frame.depth_m[ys, xs], dtype=np.float32)
        min_depth = float(self.config.postprocess.min_valid_depth_m)
        max_depth = float(self.config.postprocess.max_valid_depth_m)
        valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
        if not np.all(valid):
            ys, xs, z = ys[valid], xs[valid], z[valid]
        if z.size == 0:
            return GeometrySamples(
                ys=np.empty((0,), dtype=np.int64),
                xs=np.empty((0,), dtype=np.int64),
                z=np.empty((0,), dtype=np.float32),
                colors_rgb=np.empty((0, 3), dtype=np.uint8),
            )

        max_points = int(self.config.pointcloud.max_points_per_instance)
        if max_points > 0 and len(z) > max_points:
            keep = np.linspace(0, len(z) - 1, max_points, dtype=np.int64)
            ys, xs, z = ys[keep], xs[keep], z[keep]

        colors = (
            np.ascontiguousarray(frame.rgb[ys, xs, :3], dtype=np.uint8)
            if need_colors
            else np.empty((0, 3), dtype=np.uint8)
        )
        return GeometrySamples(
            ys=np.ascontiguousarray(ys, dtype=np.int64),
            xs=np.ascontiguousarray(xs, dtype=np.int64),
            z=np.ascontiguousarray(z, dtype=np.float32),
            colors_rgb=colors,
        )

    def _batch_geometry_gpu(
        self,
        records: list[_MaskRecord],
        frames: list[RGBDFrame],
        *,
        need_colors: bool,
        visualization_enabled: bool,
        profiler: Any,
    ) -> list[_GeometryRecord]:
        assert self._gpu_geometry is not None

        if self.gpu_geometry_enabled and not visualization_enabled:
            masks_gpu = [record.final_mask_gpu for record in records]
            if all(mask is not None for mask in masks_gpu):
                # No CPU GeometrySamples are built in this path. The direct backend
                # performs mask lattice sampling + depth validity + geometry on GPU.
                with profiler.stage("postprocess_geometry_prepare", cuda=False):
                    pass
                masks_prepared = all(record.final_mask is None for record in records)
                prefetched = self._finish_depth_prefetch(frames)
                with profiler.stage("postprocess_geometry_gpu", cuda=True):
                    pending = self._gpu_geometry.compute_from_masks(
                        records,
                        frames,
                        masks_gpu,
                        None if masks_prepared else [record.geometry_stride for record in records],
                        max_points=int(self.config.pointcloud.max_points_per_instance),
                        min_depth=float(self.config.postprocess.min_valid_depth_m),
                        max_depth=float(self.config.postprocess.max_valid_depth_m),
                        masks_prepared=masks_prepared,
                        use_prefetched_depth=prefetched,
                    )
                with profiler.stage("postprocess_geometry_d2h", cuda=False):
                    gpu_records = self._gpu_geometry.materialize_compact(
                        pending, records, frames
                    )

                for record, item in zip(records, gpu_records):
                    if record.bbox_2d is None and item.bbox_2d is not None:
                        record.bbox_2d = item.bbox_2d
                        record.foreground_pixels = int(item.foreground_pixels)
                        record.geometry_stride = int(item.geometry_stride)

                return [
                    _GeometryRecord(
                        points_camera=item.points_camera,
                        points_world=item.points_world,
                        colors_rgb=item.colors_rgb,
                        voxel_coords=item.voxel_coords,
                        voxel_keys=(
                            np.ascontiguousarray(item.voxel_keys, dtype=np.uint64)
                            if item.voxel_keys is not None else None
                        ),
                        voxel_points=item.voxel_points,
                        voxel_colors=item.voxel_colors,
                        voxel_bbox_min=item.voxel_bbox_min,
                        voxel_bbox_max=item.voxel_bbox_max,
                        points_camera_gpu=item.points_camera_gpu,
                        points_world_gpu=item.points_world_gpu,
                        voxel_coords_gpu=item.voxel_coords_gpu,
                        voxel_keys_gpu=item.voxel_keys_gpu,
                        voxel_points_gpu=item.voxel_points_gpu,
                    )
                    for item in gpu_records
                ]

        with profiler.stage("postprocess_geometry_prepare", cuda=False):
            jobs = [
                self._pool.submit(
                    self._geometry_samples_one,
                    record,
                    frames[record.view_index],
                    need_colors=need_colors,
                )
                for record in records
            ]
            samples = [future.result() for future in jobs]

        with profiler.stage("postprocess_geometry_gpu", cuda=True):
            pending = self._gpu_geometry.compute(records, frames, samples)

        # Visualization/fallback geometry still materializes the full NumPy
        # representation while retaining CUDA views in the geometry record.
        with profiler.stage("postprocess_geometry_d2h", cuda=False):
            gpu_records = self._gpu_geometry.materialize(pending, records, frames)

        output: list[_GeometryRecord] = []
        for item in gpu_records:
            geometry = _GeometryRecord(
                points_camera=item.points_camera,
                points_world=item.points_world,
                colors_rgb=item.colors_rgb,
                voxel_coords=item.voxel_coords,
                voxel_keys=(
                    np.ascontiguousarray(item.voxel_keys, dtype=np.uint64)
                    if item.voxel_keys is not None
                    else None
                ),
                voxel_points=item.voxel_points,
                voxel_colors=item.voxel_colors,
                voxel_bbox_min=item.voxel_bbox_min,
                voxel_bbox_max=item.voxel_bbox_max,
                points_camera_gpu=item.points_camera_gpu,
                points_world_gpu=item.points_world_gpu,
                voxel_coords_gpu=item.voxel_coords_gpu,
                voxel_keys_gpu=item.voxel_keys_gpu,
                voxel_points_gpu=item.voxel_points_gpu,
            )

            # Preserve the exact current metadata policy.  These reductions stay
            # on the compatibility CPU view for step 1 and are moved with the
            # cross-view data plane in step 2.
            if visualization_enabled:
                if geometry.points_world is not None and geometry.points_world.size:
                    vis_points = (
                        geometry.voxel_points
                        if geometry.voxel_points is not None
                        else geometry.points_world
                    )
                    geometry.centroid_world = np.median(
                        vis_points, axis=0
                    ).astype(np.float32)
                    if (
                        geometry.voxel_bbox_min is not None
                        and geometry.voxel_bbox_max is not None
                        and self.voxelizer is not None
                    ):
                        geometry.bbox_min = (
                            self.voxelizer.origin_world
                            + geometry.voxel_bbox_min.astype(np.float32)
                            * np.float32(self.voxelizer.voxel_size_m)
                        )
                        geometry.bbox_max = (
                            self.voxelizer.origin_world
                            + (geometry.voxel_bbox_max.astype(np.float32) + 1.0)
                            * np.float32(self.voxelizer.voxel_size_m)
                        )
                    else:
                        geometry.bbox_min = geometry.points_world.min(axis=0).astype(
                            np.float32
                        )
                        geometry.bbox_max = geometry.points_world.max(axis=0).astype(
                            np.float32
                        )
                if geometry.points_camera.size:
                    geometry.centroid_camera = np.median(
                        geometry.points_camera, axis=0
                    ).astype(np.float32)
                    if geometry.points_world is None:
                        geometry.bbox_min = geometry.points_camera.min(axis=0).astype(
                            np.float32
                        )
                        geometry.bbox_max = geometry.points_camera.max(axis=0).astype(
                            np.float32
                        )
            output.append(geometry)
        return output

    def _geometry_one(
        self,
        record: _MaskRecord,
        frame: RGBDFrame,
        x_ray: np.ndarray,
        y_ray: np.ndarray,
        *,
        need_colors: bool,
        visualization_enabled: bool,
    ) -> _GeometryRecord:
        """Sparse ROI backprojection for one instance.

        The previous batch implementation stacked every full-resolution mask and
        called one global ``np.nonzero``.  This version keeps the batched scheduling
        but only touches each cleaned instance ROI, which is much cheaper for the
        small tabletop objects in this pipeline.
        """
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        bbox = record.bbox_2d
        mask = record.final_mask
        if bbox is None or mask is None:
            return _GeometryRecord(
                points_camera=empty_points,
                points_world=(empty_points.copy() if frame.world_from_camera is not None else None),
                colors_rgb=empty_colors,
            )

        ys = record.geometry_y
        xs = record.geometry_x
        if ys is None or xs is None or ys.size == 0:
            return _GeometryRecord(
                points_camera=empty_points,
                points_world=(empty_points.copy() if frame.world_from_camera is not None else None),
                colors_rgb=empty_colors,
            )

        # CC already extracted these global coordinates using the configured/adaptive
        # image lattice. Geometry therefore does no second mask/ROI scan.
        ys = np.asarray(ys, dtype=np.intp)
        xs = np.asarray(xs, dtype=np.intp)

        z = np.asarray(frame.depth_m[ys, xs], dtype=np.float32)
        min_depth = float(self.config.postprocess.min_valid_depth_m)
        max_depth = float(self.config.postprocess.max_valid_depth_m)
        valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
        if not np.all(valid):
            ys, xs, z = ys[valid], xs[valid], z[valid]
        if z.size == 0:
            return _GeometryRecord(
                points_camera=empty_points,
                points_world=(empty_points.copy() if frame.world_from_camera is not None else None),
                colors_rgb=empty_colors,
            )

        max_points = int(self.config.pointcloud.max_points_per_instance)
        if max_points > 0 and len(z) > max_points:
            keep = np.linspace(0, len(z) - 1, max_points, dtype=np.int64)
            ys, xs, z = ys[keep], xs[keep], z[keep]

        points_camera = np.empty((len(z), 3), dtype=np.float32)
        points_camera[:, 0] = x_ray[xs] * z
        points_camera[:, 1] = y_ray[ys] * z
        points_camera[:, 2] = z

        colors = (
            np.ascontiguousarray(frame.rgb[ys, xs, :3], dtype=np.uint8)
            if need_colors
            else empty_colors
        )
        points_world = None
        transform = frame.world_from_camera
        if transform is not None:
            transform = np.asarray(transform, dtype=np.float32)
            points_world = np.ascontiguousarray(
                points_camera @ transform[:3, :3].T + transform[:3, 3][None, :],
                dtype=np.float32,
            )

        geometry = _GeometryRecord(
            points_camera=np.ascontiguousarray(points_camera, dtype=np.float32),
            points_world=points_world,
            colors_rgb=colors,
        )

        # Quantize once while the freshly generated cloud is hot in cache.  The
        # exact sparse keys/representatives are reused by cross-view matching and
        # final fusion, removing the old second quantize+unique pass.
        if self.voxelizer is not None and points_world is not None and points_world.size:
            prepared = self.voxelizer.prepare_points(
                points_world, colors if need_colors else None
            )
            if prepared is not None:
                (
                    geometry.voxel_coords,
                    geometry.voxel_keys,
                    geometry.voxel_points,
                    geometry.voxel_colors,
                    geometry.voxel_bbox_min,
                    geometry.voxel_bbox_max,
                ) = prepared

        if visualization_enabled:
            # Marker-only geometry uses the already deduplicated cloud when
            # available.  Min/max voxel bounds replace expensive 1/99% quantiles.
            if points_world is not None and points_world.size:
                vis_points = (
                    geometry.voxel_points
                    if geometry.voxel_points is not None
                    else points_world
                )
                geometry.centroid_world = np.median(vis_points, axis=0).astype(np.float32)
                if (
                    geometry.voxel_bbox_min is not None
                    and geometry.voxel_bbox_max is not None
                    and self.voxelizer is not None
                ):
                    geometry.bbox_min = (
                        self.voxelizer.origin_world
                        + geometry.voxel_bbox_min.astype(np.float32)
                        * np.float32(self.voxelizer.voxel_size_m)
                    )
                    geometry.bbox_max = (
                        self.voxelizer.origin_world
                        + (geometry.voxel_bbox_max.astype(np.float32) + 1.0)
                        * np.float32(self.voxelizer.voxel_size_m)
                    )
                else:
                    geometry.bbox_min = points_world.min(axis=0).astype(np.float32)
                    geometry.bbox_max = points_world.max(axis=0).astype(np.float32)
            if points_camera.size:
                geometry.centroid_camera = np.median(points_camera, axis=0).astype(np.float32)
                if points_world is None:
                    geometry.bbox_min = points_camera.min(axis=0).astype(np.float32)
                    geometry.bbox_max = points_camera.max(axis=0).astype(np.float32)
        return geometry

    def _batch_geometry(
        self,
        records: list[_MaskRecord],
        frames: list[RGBDFrame],
        *,
        need_colors: bool,
        visualization_enabled: bool,
        profiler: Any,
    ) -> list[_GeometryRecord]:
        if self.gpu_geometry_enabled:
            return self._batch_geometry_gpu(
                records,
                frames,
                need_colors=need_colors,
                visualization_enabled=visualization_enabled,
                profiler=profiler,
            )

        """Process all view×instance ROIs as one persistent CPU task batch."""
        rays = [self._rays(frame) for frame in frames]
        jobs = []
        for record in records:
            frame = frames[record.view_index]
            x_ray, y_ray = rays[record.view_index]
            jobs.append(
                self._pool.submit(
                    self._geometry_one,
                    record,
                    frame,
                    x_ray,
                    y_ray,
                    need_colors=need_colors,
                    visualization_enabled=visualization_enabled,
                )
            )
        return [future.result() for future in jobs]

    def process(
        self,
        views: list[Any],
        frames: list[RGBDFrame],
        predictions: list[TrackerPrediction],
        *,
        keyframe: bool,
        trigger_reasons: list[list[str]],
        extra_metadata_per_view: list[dict[str, Any]],
        profiler: Any,
    ) -> list[FrameResult]:
        visualization_enabled = bool(
            self.config.runtime.get("enable_visualization", True)
        )
        debug_images_enabled = visualization_enabled and bool(
            self.config.runtime.get("publish_debug_images", True)
        )
        # Raw masks are only required by debug rasters or by the exact frame that
        # is submitted as an asynchronous SAM3 fallback reference. Normal frames
        # transfer only the eroded mask from CUDA.
        need_raw_masks = debug_images_enabled or any(
            bool(metadata.get("sam3_refresh_due", False))
            for metadata in extra_metadata_per_view
        )
        build_owner_map = bool(
            self.config.postprocess.get("build_owner_map", False)
        )
        lazy_common_frame = bool(
            self.gpu_geometry_enabled
            and not visualization_enabled
            and not build_owner_map
            and not need_raw_masks
        )
        need_final_masks = not lazy_common_frame

        with profiler.stage("postprocess_total", cuda=False):
            with profiler.stage(
                "postprocess_masks", cuda=self.mask_stage_cuda
            ):
                records, resize_groups, logits_per_view = self._collect_records(
                    views, frames, predictions
                )
                pending_components = self._batch_masks(
                    records,
                    resize_groups,
                    logits_per_view,
                    need_raw_masks=need_raw_masks,
                    need_final_masks=need_final_masks,
                )

            with profiler.stage(
                "postprocess_components", cuda=bool(lazy_common_frame and self.mask_stage_cuda)
            ):
                self._batch_components(pending_components)

            with profiler.stage("postprocess_geometry", cuda=False):
                geometry = self._batch_geometry(
                    records,
                    frames,
                    need_colors=visualization_enabled,
                    visualization_enabled=visualization_enabled,
                    profiler=profiler,
                )

            with profiler.stage("postprocess_finalize", cuda=False):
                instances_per_view: list[list[ProcessedInstance]] = [[] for _ in views]
                owner_per_view: list[np.ndarray | None] = []
                raw_maps: list[np.ndarray | None] = [None] * len(views)
                filtered_maps: list[np.ndarray | None] = [None] * len(views)

                records_per_view: list[list[int]] = [[] for _ in views]
                for record_index, record in enumerate(records):
                    records_per_view[record.view_index].append(record_index)

                # Build owner/debug rasters only inside object ROIs.  The old
                # implementation stacked every full-frame mask and rescanned the
                # entire H×W image in finalize.  -1 is a temporary conflict
                # sentinel; it is converted back to the public 0=unowned value.
                for view_index, frame in enumerate(frames):
                    h, w = frame.depth_m.shape
                    owner = (
                        np.zeros((h, w), dtype=np.int32)
                        if build_owner_map
                        else None
                    )
                    raw_map = (
                        np.zeros((h, w), dtype=np.uint8)
                        if debug_images_enabled
                        else None
                    )
                    filtered_map = (
                        np.zeros((h, w), dtype=np.uint8)
                        if debug_images_enabled
                        else None
                    )
                    for code, record_index in enumerate(
                        records_per_view[view_index], start=1
                    ):
                        record = records[record_index]
                        bbox = record.bbox_2d
                        if bbox is not None and record.final_mask is not None:
                            x0, y0, x1, y1 = bbox
                            rows = slice(y0, y1 + 1)
                            cols = slice(x0, x1 + 1)
                            mask_roi = np.asarray(
                                record.final_mask[rows, cols], dtype=bool
                            )
                            if owner is not None:
                                owner_roi = owner[rows, cols]
                                empty = owner_roi == 0
                                owner_roi[mask_roi & empty] = record.track_id
                                owner_roi[mask_roi & ~empty] = -1
                            if filtered_map is not None:
                                filtered_map[rows, cols][mask_roi] = np.uint8(
                                    min(code, 255)
                                )
                        if (
                            raw_map is not None
                            and record.raw_bbox_2d is not None
                            and record.raw_mask is not None
                        ):
                            x0, y0, x1, y1 = record.raw_bbox_2d
                            rows = slice(y0, y1 + 1)
                            cols = slice(x0, x1 + 1)
                            raw_roi = np.asarray(record.raw_mask[rows, cols], dtype=bool)
                            raw_map[rows, cols][raw_roi] = np.uint8(min(code, 255))
                    if owner is not None:
                        owner[owner < 0] = 0
                    owner_per_view.append(owner)
                    raw_maps[view_index] = raw_map
                    filtered_maps[view_index] = filtered_map

                for record_index, record in enumerate(records):
                    view = views[record.view_index]
                    frame = frames[record.view_index]
                    prediction = predictions[record.view_index]
                    track = view.tracks[record.track_id]
                    geom = geometry[record_index]

                    if record.final_mask is not None:
                        final_mask = np.asarray(record.final_mask, dtype=bool)
                        raw_mask = (
                            np.asarray(record.raw_mask, dtype=bool)
                            if record.raw_mask is not None
                            else final_mask
                        )
                        track.last_raw_mask = raw_mask
                        track.last_mask = final_mask
                    else:
                        # Lazy common frame: masks remain canonical on CUDA. CPU masks
                        # are materialized on refresh/debug/visualization frames only.
                        final_mask = np.empty((0, 0), dtype=bool)
                        raw_mask = final_mask
                    track.centroid_camera = geom.centroid_camera
                    track.centroid_world = geom.centroid_world
                    if record.channel < prediction.presence_scores.size:
                        track.tracking_confidence = float(
                            prediction.presence_scores[record.channel]
                        )

                    if record.bbox_2d is not None:
                        track.last_seen_frame = frame.frame_index
                        track.missing_frames = 0
                        status = VisibilityState.VISIBLE
                    else:
                        track.missing_frames += 1
                        status = VisibilityState.LOST
                        if track.missing_frames >= view.release_after_missing_frames:
                            track.active = False

                    motion_conf = min(max(float(track.tracking_confidence), 0.0), 1.0)
                    instances_per_view[record.view_index].append(
                        ProcessedInstance(
                            track_id=record.track_id,
                            label=track.label,
                            semantic_confidence=track.semantic_confidence,
                            tracking_confidence=track.tracking_confidence,
                            motion_prediction_confidence=motion_conf,
                            raw_mask=raw_mask,
                            mask=final_mask,
                            points_camera=geom.points_camera,
                            points_world=geom.points_world,
                            colors_rgb=geom.colors_rgb,
                            centroid_camera=geom.centroid_camera,
                            centroid_world=geom.centroid_world,
                            bbox_min=geom.bbox_min,
                            bbox_max=geom.bbox_max,
                            bbox_2d_xyxy=record.bbox_2d,
                            status=status,
                            tracker_slot=track.tracker_slot,
                            class_slot=track.class_slot,
                            voxel_coords=geom.voxel_coords,
                            voxel_keys=geom.voxel_keys,
                            voxel_points=geom.voxel_points,
                            voxel_colors=geom.voxel_colors,
                            voxel_bbox_min=geom.voxel_bbox_min,
                            voxel_bbox_max=geom.voxel_bbox_max,
                            points_camera_gpu=geom.points_camera_gpu,
                            points_world_gpu=geom.points_world_gpu,
                            voxel_coords_gpu=geom.voxel_coords_gpu,
                            voxel_keys_gpu=geom.voxel_keys_gpu,
                            voxel_points_gpu=geom.voxel_points_gpu,
                            mask_gpu=record.final_mask_gpu,
                        )
                    )

        results: list[FrameResult] = []
        for view_index, (view, frame) in enumerate(zip(views, frames)):
            metadata = {
                "trigger_reasons": list(trigger_reasons[view_index]),
                "num_active_instances_per_view": sum(
                    1 for track in view.tracks.values() if track.active
                ),
                "num_dummy_slots_per_view": sum(
                    1 for track in view.tracks.values() if not track.active
                ),
            }
            metadata.update(extra_metadata_per_view[view_index])
            results.append(
                FrameResult(
                    frame=frame,
                    instances=instances_per_view[view_index],
                    owner_track_map=owner_per_view[view_index],
                    keyframe=bool(keyframe),
                    timings_ms={},
                    raw_instance_map=raw_maps[view_index],
                    filtered_instance_map=filtered_maps[view_index],
                    metadata=metadata,
                )
            )
        return results
