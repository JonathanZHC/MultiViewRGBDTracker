from __future__ import annotations

from array import array
from dataclasses import dataclass
import time
from typing import Any

import cv2
import numpy as np

from .data_types import FrameResult, MultiViewInstance
from .slots import object_slots_per_view


_PALETTE = np.asarray(
    [
        [230, 80, 80],
        [80, 210, 120],
        [80, 150, 240],
        [235, 190, 70],
        [180, 90, 225],
        [70, 210, 215],
    ],
    dtype=np.uint8,
)
_PALETTE_F32 = _PALETTE.astype(np.float32) / 255.0


def _color(track_id: int) -> np.ndarray:
    return _PALETTE[(int(track_id) - 1) % len(_PALETTE)]


def _color_f32(track_id: int) -> np.ndarray:
    return _PALETTE_F32[(int(track_id) - 1) % len(_PALETTE_F32)]



def _instance_color_key(instance: Any) -> int:
    global_id = getattr(instance, "global_track_id", None)
    return int(global_id) if global_id is not None else int(instance.track_id)


def _instance_label(instance: Any) -> str:
    global_id = getattr(instance, "global_track_id", None)
    group_id = getattr(instance, "multiview_group_id", None)
    suffix = f" G{global_id}" if global_id is not None else ""
    if group_id is not None:
        suffix += f" M{group_id}"
    return f"{instance.label} [slot {instance.tracker_slot}]{suffix}"


def instance_mask_cpu(instance: Any, shape: tuple[int, int]) -> np.ndarray:
    """Return the final instance mask as a CPU bool array on demand.

    Lazy postprocess keeps the canonical final mask on CUDA and leaves the
    compatibility NumPy field empty on normal frames. Visualization/output calls
    this helper only after a subscriber check, so the D2H transfer stays outside
    the numerical tracking critical path.
    """
    mask = getattr(instance, "mask", None)
    if mask is not None:
        value = np.asarray(mask, dtype=bool)
        if value.shape == shape:
            return value

    mask_gpu = getattr(instance, "mask_gpu", None)
    if mask_gpu is not None:
        value = mask_gpu.detach().cpu().numpy()
        value = np.asarray(value, dtype=bool)
        if value.shape == shape:
            return value

    return np.zeros(shape, dtype=bool)


def make_overlay(result: FrameResult) -> np.ndarray:
    """Compatibility helper for callers outside ``RvizPublisher``.

    The live ROS path uses persistent buffers in ``RvizPublisher`` instead.
    """
    image = result.frame.rgb.copy()
    for instance in result.instances:
        color = _color(_instance_color_key(instance))
        if instance.bbox_2d_xyxy is None:
            continue
        mask = instance_mask_cpu(instance, result.frame.depth_m.shape)
        blended = (
            image[mask].astype(np.float32) * 0.55
            + color.astype(np.float32) * 0.45
        )
        image[mask] = np.clip(blended, 0, 255).astype(np.uint8)
        x0, y0, x1, y1 = instance.bbox_2d_xyxy
        cv2.rectangle(image, (x0, y0), (x1, y1), color.tolist(), 2)
        text = (
            f"{_instance_label(instance)} "
            f"det={instance.semantic_confidence:.2f} "
            f"trk={instance.tracking_confidence:.2f}"
        )
        cv2.putText(
            image,
            text,
            (x0, max(18, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color.tolist(),
            1,
            cv2.LINE_AA,
        )
    return image


def make_mask_debug(result: FrameResult, *, raw: bool) -> np.ndarray:
    """Compatibility helper using the same fixed-LUT fast path as RViz."""
    h, w = result.frame.depth_m.shape
    instance_map = (
        result.raw_instance_map if raw else result.filtered_instance_map
    )

    # Cached debug maps use compact uint8 instance codes (1..N), not global
    # track IDs.  This keeps the raster at one byte/pixel even if track IDs
    # grow during a long run.
    if instance_map is None:
        instance_map = np.zeros((h, w), dtype=np.uint8)
        for code, instance in enumerate(result.instances, start=1):
            if raw:
                mask = np.asarray(instance.raw_mask, dtype=bool)
                if mask.shape != (h, w):
                    continue
            else:
                mask = instance_mask_cpu(instance, (h, w))
            instance_map[mask] = np.uint8(min(code, 255))
    else:
        instance_map = np.asarray(instance_map, dtype=np.uint8)

    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for code, instance in enumerate(result.instances, start=1):
        if code >= 256:
            break
        lut[code, 0] = _color(_instance_color_key(instance))
    return cv2.applyColorMap(instance_map, lut)


@dataclass(slots=True)
class _MessageSet:
    points: Any
    markers: Any
    overlay: Any
    raw: Any
    filtered: Any
    overlay_view: np.ndarray | None = None
    raw_view: np.ndarray | None = None
    filtered_view: np.ndarray | None = None
    overlay_direct: bool = False
    raw_direct: bool = False
    filtered_direct: bool = False


class RvizPublisher:
    """Low-allocation synchronous ROS/RViz publisher.

    Visualization intentionally remains on the main tracking worker critical
    path.  The optimization target is therefore to make that path cheap, not to
    hide it on another thread.

    Main implementation choices:
      * BEST_EFFORT / KEEP_LAST depth=1 visualization QoS.
      * Two pre-created ROS message sets, alternated every frame.
      * Pre-created PointField and Marker objects.
      * Persistent/double-buffered ROS Image storage exposed as NumPy views.
      * Fixed 256-entry OpenCV color LUT + uint8 debug instance maps.
      * Persistent NumPy blend scratch and structured PointCloud2 storage.
      * Cached 2-D bounding boxes and instance-code maps from postprocess.
      * When debug publication is enabled, all configured visualization messages
        are built and published every frame.
    """

    _CLOUD_DTYPE = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("rgb", "<u4"),
        ],
        align=False,
    )

    def __init__(self, node: Any, camera_name: str, config) -> None:
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import Image, PointCloud2
        from visualization_msgs.msg import MarkerArray

        self.node = node
        self.camera_name = camera_name
        self.config = config
        self._optical_frame = f"{camera_name}_optical_frame"
        self._world_frame = str(config.ros.world_frame)
        self._debug_images = bool(
            config.runtime.get("publish_debug_images", True)
        )
        self._profile_enabled = bool(config.profiling.get("enabled", True))
        self.last_build_timings_ms: dict[str, float] = {}
        self.last_publish_timings_ms: dict[str, float] = {}

        # Visualization is a latest-state stream.  BEST_EFFORT/depth=1 prevents
        # stale RViz frames from creating reliable-history backpressure.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        prefix = str(config.ros.output_prefix).format(camera=camera_name)
        self.points_pub = node.create_publisher(
            PointCloud2,
            f"{prefix}/visible_points",
            qos,
        )
        self.markers_pub = node.create_publisher(
            MarkerArray,
            f"{prefix}/markers",
            qos,
        )
        self.overlay_pub = node.create_publisher(
            Image,
            f"{prefix}/rgb_overlay",
            qos,
        )
        self.raw_pub = node.create_publisher(
            Image,
            f"{prefix}/raw_masks",
            qos,
        )
        self.filtered_pub = node.create_publisher(
            Image,
            f"{prefix}/filtered_masks",
            qos,
        )

        self._object_slots = max(1, object_slots_per_view(config))
        self._message_sets = [
            self._create_message_set(),
            self._create_message_set(),
        ]
        self._next_message_set = 0

        max_points_per_instance = int(
            config.pointcloud.max_points_per_instance
        )
        initial_cloud_capacity = (
            max_points_per_instance * self._object_slots
            if max_points_per_instance > 0
            else 4096 * self._object_slots
        )
        self._cloud_buffer = np.empty(
            max(1, initial_cloud_capacity),
            dtype=self._CLOUD_DTYPE,
        )

        # Image-dependent buffers are allocated once when the first frame reveals
        # the actual RGB-D resolution, and only reallocated on a resolution change.
        self._image_shape: tuple[int, int] | None = None
        self._blend_buffer: np.ndarray | None = None
        self._fallback_raw_map: np.ndarray | None = None
        self._fallback_filtered_map: np.ndarray | None = None

        # Fixed 256-entry custom OpenCV colormap.  Index 0 is background;
        # indices 1..N are refreshed from the current instance ordering.
        # No per-frame max() scan or dynamic LUT growth is required.
        self._color_lut = np.zeros((256, 1, 3), dtype=np.uint8)

    @staticmethod
    def _elapsed_ms(start_s: float) -> float:
        return 1000.0 * (time.perf_counter() - start_s)

    def _timed(self, timings: dict[str, float], name: str, start_s: float) -> None:
        if self._profile_enabled:
            timings[name] = self._elapsed_ms(start_s)

    def _create_pointcloud(self):
        from sensor_msgs.msg import PointCloud2, PointField

        cloud = PointCloud2()
        cloud.height = 1
        cloud.is_bigendian = False
        cloud.is_dense = False
        cloud.point_step = 16
        cloud.row_step = 0
        cloud.width = 0
        cloud.data = b""
        cloud.fields = [
            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="rgb",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]
        return cloud

    def _create_marker_array(self):
        from visualization_msgs.msg import Marker, MarkerArray

        array = MarkerArray()
        for slot in range(self._object_slots):
            text = Marker()
            text.ns = f"{self.camera_name}_labels"
            text.id = int(slot + 1)
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.DELETE
            text.pose.orientation.w = 1.0
            text.scale.z = 0.08
            text.color.a = 1.0
            array.markers.append(text)

            box = Marker()
            box.ns = f"{self.camera_name}_boxes"
            box.id = int(slot + 1)
            box.type = Marker.CUBE
            box.action = Marker.DELETE
            box.pose.orientation.w = 1.0
            box.color.a = 0.14
            array.markers.append(box)
        return array

    def _create_image_message(self):
        from sensor_msgs.msg import Image

        image = Image()
        image.header.frame_id = self._optical_frame
        image.encoding = "rgb8"
        image.is_bigendian = 0
        image.height = 0
        image.width = 0
        image.step = 0
        image.data = b""
        return image

    def _create_message_set(self) -> _MessageSet:
        return _MessageSet(
            points=self._create_pointcloud(),
            markers=self._create_marker_array(),
            overlay=self._create_image_message(),
            raw=self._create_image_message(),
            filtered=self._create_image_message(),
        )

    def _ensure_cloud_capacity(self, required: int) -> None:
        if required <= self._cloud_buffer.shape[0]:
            return
        new_capacity = max(required, self._cloud_buffer.shape[0] * 2)
        self._cloud_buffer = np.empty(new_capacity, dtype=self._CLOUD_DTYPE)

    @staticmethod
    def _prepare_image_storage(
        image_message: Any,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, bool]:
        """Allocate Image.data once and expose it as a writable NumPy view.

        ROS 2 Python's uint8 sequence is normally backed by ``array('B')``.
        Mutating that array in place avoids a per-frame ``ndarray.tobytes()``
        allocation and a second Python-side copy.  A conservative NumPy scratch
        fallback is retained for unusual message implementations that do not
        expose a writable buffer.
        """
        image_message.height = int(height)
        image_message.width = int(width)
        image_message.step = 3 * int(width)
        nbytes = int(height) * int(width) * 3
        image_message.data = array("B", [0]) * nbytes
        try:
            view = np.frombuffer(image_message.data, dtype=np.uint8)
            if not view.flags.writeable or view.size != nbytes:
                raise ValueError("Image.data is not a writable fixed-size buffer")
            return view.reshape(height, width, 3), True
        except (TypeError, ValueError, BufferError):
            return np.empty((height, width, 3), dtype=np.uint8), False

    def _ensure_image_buffers(self, height: int, width: int) -> None:
        shape = (int(height), int(width))
        if self._image_shape == shape:
            return
        self._image_shape = shape
        h, w = shape
        self._blend_buffer = np.empty((h, w, 3), dtype=np.uint8)
        self._fallback_raw_map = np.empty((h, w), dtype=np.uint8)
        self._fallback_filtered_map = np.empty((h, w), dtype=np.uint8)

        for message_set in self._message_sets:
            overlay_view, overlay_direct = self._prepare_image_storage(
                message_set.overlay, h, w
            )
            raw_view, raw_direct = self._prepare_image_storage(
                message_set.raw, h, w
            )
            filtered_view, filtered_direct = self._prepare_image_storage(
                message_set.filtered, h, w
            )
            message_set.overlay_view = overlay_view
            message_set.raw_view = raw_view
            message_set.filtered_view = filtered_view
            message_set.overlay_direct = bool(overlay_direct)
            message_set.raw_direct = bool(raw_direct)
            message_set.filtered_direct = bool(filtered_direct)

    @staticmethod
    def _visual_code(code: int) -> np.uint8:
        return np.uint8(min(max(int(code), 0), 255))

    def _refresh_color_lut(self, result: FrameResult) -> None:
        # Only N tiny rows are touched.  The full 256-entry LUT never grows and
        # no full-image max() scan is needed.
        self._color_lut.fill(0)
        for code, instance in enumerate(result.instances, start=1):
            if code >= 256:
                break
            self._color_lut[code, 0] = _color(_instance_color_key(instance))

    def _instance_map(self, result: FrameResult, *, raw: bool) -> np.ndarray:
        supplied = (
            result.raw_instance_map if raw else result.filtered_instance_map
        )
        if supplied is not None:
            supplied = np.asarray(supplied)
            if supplied.dtype == np.uint8:
                return supplied
            # Compatibility with an older producer.  This conversion is not used
            # by the updated postprocess path.
            fallback = (
                self._fallback_raw_map if raw else self._fallback_filtered_map
            )
            if fallback is None:
                raise RuntimeError("Visualization buffers have not been initialized")
            fallback.fill(0)
            for code, instance in enumerate(result.instances, start=1):
                fallback[
                    instance.raw_mask if raw else instance.mask
                ] = self._visual_code(code)
            return fallback

        # Compatibility path for FrameResult producers without cached maps.
        fallback = self._fallback_raw_map if raw else self._fallback_filtered_map
        if fallback is None:
            raise RuntimeError("Visualization buffers have not been initialized")
        fallback.fill(0)
        for code, instance in enumerate(result.instances, start=1):
            fallback[instance.raw_mask if raw else instance.mask] = self._visual_code(
                code
            )
        return fallback

    def _update_pointcloud(
        self,
        cloud: Any,
        result: FrameResult,
        stamp: Any,
    ) -> None:
        use_world = result.frame.world_from_camera is not None
        frame_id = self._world_frame if use_world else self._optical_frame
        cloud.header.stamp = stamp
        cloud.header.frame_id = frame_id

        total_points = 0
        for instance in result.instances:
            points = instance.points_world if use_world else instance.points_camera
            if points is not None:
                total_points += int(points.shape[0])
        self._ensure_cloud_capacity(total_points)

        offset = 0
        for instance in result.instances:
            points = instance.points_world if use_world else instance.points_camera
            if points is None or points.size == 0:
                continue
            points = np.asarray(points, dtype=np.float32)
            colors = np.asarray(instance.colors_rgb, dtype=np.uint8)
            count = int(points.shape[0])
            destination = self._cloud_buffer[offset : offset + count]

            np.copyto(destination["x"], points[:, 0], casting="unsafe")
            np.copyto(destination["y"], points[:, 1], casting="unsafe")
            np.copyto(destination["z"], points[:, 2], casting="unsafe")

            # Pack RGB in-place without allocating uint32 channel temporaries.
            packed_rgb = destination["rgb"]
            np.copyto(packed_rgb, colors[:, 0], casting="unsafe")
            packed_rgb <<= np.uint32(8)
            packed_rgb |= colors[:, 1]
            packed_rgb <<= np.uint32(8)
            packed_rgb |= colors[:, 2]
            offset += count

        cloud.width = offset
        cloud.row_step = cloud.point_step * offset
        if offset == 0:
            cloud.data = b""
        else:
            # rclpy still owns the final message serialization/copy.  All larger
            # intermediate concatenate/pack allocations have already been removed.
            cloud.data = self._cloud_buffer[:offset].tobytes(order="C")

    def _update_markers(
        self,
        array: Any,
        result: FrameResult,
        stamp: Any,
    ) -> None:
        from visualization_msgs.msg import Marker

        use_world = result.frame.world_from_camera is not None
        frame_id = self._world_frame if use_world else self._optical_frame
        instances = result.instances

        for slot in range(self._object_slots):
            text = array.markers[2 * slot]
            box = array.markers[2 * slot + 1]
            text.header.frame_id = frame_id
            text.header.stamp = stamp
            box.header.frame_id = frame_id
            box.header.stamp = stamp

            if slot >= len(instances):
                text.action = Marker.DELETE
                box.action = Marker.DELETE
                continue

            instance = instances[slot]
            centroid = (
                instance.centroid_world
                if use_world
                else instance.centroid_camera
            )
            color = _color_f32(_instance_color_key(instance))

            if centroid is None:
                text.action = Marker.DELETE
            else:
                text.action = Marker.ADD
                text.pose.position.x = float(centroid[0])
                text.pose.position.y = float(centroid[1])
                text.pose.position.z = float(centroid[2]) + 0.12
                text.color.r = float(color[0])
                text.color.g = float(color[1])
                text.color.b = float(color[2])
                text.text = (
                    f"{_instance_label(instance)} "
                    f"[{instance.status.value}]"
                )

            if instance.bbox_min is None or instance.bbox_max is None:
                box.action = Marker.DELETE
            else:
                box.action = Marker.ADD
                minimum = instance.bbox_min
                maximum = instance.bbox_max
                box.pose.position.x = 0.5 * (
                    float(minimum[0]) + float(maximum[0])
                )
                box.pose.position.y = 0.5 * (
                    float(minimum[1]) + float(maximum[1])
                )
                box.pose.position.z = 0.5 * (
                    float(minimum[2]) + float(maximum[2])
                )
                box.scale.x = max(float(maximum[0] - minimum[0]), 1e-3)
                box.scale.y = max(float(maximum[1] - minimum[1]), 1e-3)
                box.scale.z = max(float(maximum[2] - minimum[2]), 1e-3)
                box.color.r = float(color[0])
                box.color.g = float(color[1])
                box.color.b = float(color[2])

    def _update_overlay(
        self,
        result: FrameResult,
        output: np.ndarray,
    ) -> None:
        if self._blend_buffer is None:
            raise RuntimeError("Visualization buffers have not been initialized")

        # The full RGB background must appear in the overlay, so one image copy is
        # unavoidable.  Render directly into the ROS message-owned output buffer.
        np.copyto(output, result.frame.rgb)

        # Blend only each cached 2-D bounding ROI.  ``cv2.addWeighted`` performs
        # the arithmetic in optimized C++/SIMD; the boolean mask is reinterpreted
        # as uint8 without a copy before ``cv2.copyTo``.
        for instance in result.instances:
            bbox = instance.bbox_2d_xyxy
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            rows = slice(y0, y1 + 1)
            cols = slice(x0, x1 + 1)
            image_roi = output[rows, cols]
            blend_roi = self._blend_buffer[rows, cols]
            color = _color(_instance_color_key(instance))

            blend_roi[...] = color
            cv2.addWeighted(
                image_roi,
                0.55,
                blend_roi,
                0.45,
                0.0,
                dst=blend_roi,
            )
            mask_roi = np.asarray(instance.mask[rows, cols], dtype=bool).view(
                np.uint8
            )
            cv2.copyTo(blend_roi, mask_roi, image_roi)

            color_tuple = tuple(int(value) for value in color)
            cv2.rectangle(
                output,
                (x0, y0),
                (x1, y1),
                color_tuple,
                2,
            )
            text = (
                f"{_instance_label(instance)} "
                f"det={instance.semantic_confidence:.2f} "
                f"trk={instance.tracking_confidence:.2f}"
            )
            cv2.putText(
                output,
                text,
                (x0, max(18, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color_tuple,
                1,
                cv2.LINE_AA,
            )

    def _colorize_map(self, instance_map: np.ndarray, output: np.ndarray) -> None:
        # Custom OpenCV LUT executes in optimized C++ and writes directly into
        # the final ROS message-owned RGB buffer.
        if instance_map.dtype != np.uint8:
            raise TypeError(
                "Updated debug instance maps must use uint8 compact codes"
            )
        cv2.applyColorMap(instance_map, self._color_lut, dst=output)

    @staticmethod
    def _commit_image_message(
        image_message: Any,
        image: np.ndarray,
        stamp: Any,
        *,
        direct_storage: bool,
    ) -> None:
        image_message.header.stamp = stamp
        if not direct_storage:
            # Conservative compatibility fallback only.  Normal ROS 2 generated
            # Image messages expose mutable array('B') data and never take this path.
            image_message.data = image.tobytes(order="C")

    def build_messages(self, result: FrameResult, stamp: Any) -> _MessageSet:
        """Refresh one pre-created message set and return it for publication."""
        h, w = result.frame.depth_m.shape
        self._ensure_image_buffers(h, w)
        message_set = self._message_sets[self._next_message_set]
        self._next_message_set = (self._next_message_set + 1) % len(
            self._message_sets
        )
        if message_set.overlay_view is None:
            raise RuntimeError("Visualization image storage was not initialized")

        self._refresh_color_lut(result)
        timings: dict[str, float] = {}

        started = time.perf_counter()
        self._update_pointcloud(message_set.points, result, stamp)
        self._timed(timings, "rviz_pointcloud_build_cpu", started)

        started = time.perf_counter()
        self._update_markers(message_set.markers, result, stamp)
        self._timed(timings, "rviz_markers_build_cpu", started)

        started = time.perf_counter()
        self._update_overlay(result, message_set.overlay_view)
        self._commit_image_message(
            message_set.overlay,
            message_set.overlay_view,
            stamp,
            direct_storage=message_set.overlay_direct,
        )
        self._timed(timings, "rviz_overlay_build_cpu", started)

        if self._debug_images:
            if message_set.raw_view is None or message_set.filtered_view is None:
                raise RuntimeError("Visualization image storage was not initialized")

            started = time.perf_counter()
            raw_map = self._instance_map(result, raw=True)
            self._colorize_map(raw_map, message_set.raw_view)
            self._commit_image_message(
                message_set.raw,
                message_set.raw_view,
                stamp,
                direct_storage=message_set.raw_direct,
            )
            self._timed(timings, "rviz_raw_mask_build_cpu", started)

            started = time.perf_counter()
            filtered_map = self._instance_map(result, raw=False)
            self._colorize_map(filtered_map, message_set.filtered_view)
            self._commit_image_message(
                message_set.filtered,
                message_set.filtered_view,
                stamp,
                direct_storage=message_set.filtered_direct,
            )
            self._timed(timings, "rviz_filtered_mask_build_cpu", started)

        self.last_build_timings_ms = timings
        return message_set

    def publish_messages(self, messages: _MessageSet) -> None:
        """Publish every enabled visualization output synchronously."""
        timings: dict[str, float] = {}

        started = time.perf_counter()
        self.points_pub.publish(messages.points)
        self._timed(timings, "publish_points_cpu", started)

        started = time.perf_counter()
        self.markers_pub.publish(messages.markers)
        self._timed(timings, "publish_markers_cpu", started)

        started = time.perf_counter()
        self.overlay_pub.publish(messages.overlay)
        self._timed(timings, "publish_overlay_cpu", started)

        if self._debug_images:
            started = time.perf_counter()
            self.raw_pub.publish(messages.raw)
            self._timed(timings, "publish_raw_mask_cpu", started)

            started = time.perf_counter()
            self.filtered_pub.publish(messages.filtered)
            self._timed(timings, "publish_filtered_mask_cpu", started)

        self.last_publish_timings_ms = timings

    def publish(self, result: FrameResult, stamp: Any) -> None:
        """Compatibility wrapper preserving the original public API."""
        self.publish_messages(self.build_messages(result, stamp))


@dataclass(slots=True)
class _FusedMessageSet:
    points: Any
    markers: Any


class FusedRvizPublisher:
    """World-frame fused-instance view with stable global-ID colors."""

    _CLOUD_DTYPE = RvizPublisher._CLOUD_DTYPE

    def __init__(self, node: Any, config) -> None:
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import PointCloud2
        from visualization_msgs.msg import MarkerArray

        self.node = node
        self.config = config
        self._world_frame = str(config.ros.world_frame)
        self._profile_enabled = bool(config.profiling.get("enabled", True))
        self.last_build_timings_ms: dict[str, float] = {}
        self.last_publish_timings_ms: dict[str, float] = {}
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        prefix = str(config.ros.get("fused_output_prefix", "/tracking/fused"))
        self.points_pub = node.create_publisher(PointCloud2, f"{prefix}/points", qos)
        self.markers_pub = node.create_publisher(MarkerArray, f"{prefix}/markers", qos)

        slots = object_slots_per_view(config)
        views = len(list(config.runtime.camera_names))
        self._max_groups = max(1, slots * views)
        self._message_sets = [self._create_message_set(), self._create_message_set()]
        self._next_message_set = 0
        max_points = int(config.pointcloud.max_points_per_instance)
        initial_capacity = max(1, (max_points if max_points > 0 else 4096) * slots * views)
        self._cloud_buffer = np.empty(initial_capacity, dtype=self._CLOUD_DTYPE)

    def _create_pointcloud(self):
        from sensor_msgs.msg import PointCloud2, PointField

        cloud = PointCloud2()
        cloud.height = 1
        cloud.is_bigendian = False
        cloud.is_dense = False
        cloud.point_step = 16
        cloud.row_step = 0
        cloud.width = 0
        cloud.data = b""
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        return cloud

    def _create_markers(self):
        from visualization_msgs.msg import Marker, MarkerArray

        array_msg = MarkerArray()
        for slot in range(self._max_groups):
            text = Marker()
            text.ns = "fused_instance_labels"
            text.id = slot + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.DELETE
            text.pose.orientation.w = 1.0
            text.scale.z = 0.09
            text.color.a = 1.0
            array_msg.markers.append(text)

            box = Marker()
            box.ns = "fused_instance_boxes"
            box.id = slot + 1
            box.type = Marker.CUBE
            box.action = Marker.DELETE
            box.pose.orientation.w = 1.0
            box.color.a = 0.12
            array_msg.markers.append(box)
        return array_msg

    def _create_message_set(self) -> _FusedMessageSet:
        return _FusedMessageSet(
            points=self._create_pointcloud(),
            markers=self._create_markers(),
        )

    def _ensure_cloud_capacity(self, required: int) -> None:
        if required <= self._cloud_buffer.shape[0]:
            return
        self._cloud_buffer = np.empty(
            max(required, self._cloud_buffer.shape[0] * 2),
            dtype=self._CLOUD_DTYPE,
        )

    def build_messages(
        self,
        groups: list[MultiViewInstance],
        stamp: Any,
    ) -> _FusedMessageSet:
        from visualization_msgs.msg import Marker

        started = time.perf_counter()
        message_set = self._message_sets[self._next_message_set]
        self._next_message_set = (self._next_message_set + 1) % len(self._message_sets)
        cloud = message_set.points
        cloud.header.stamp = stamp
        cloud.header.frame_id = self._world_frame

        total_points = sum(int(group.points_world.shape[0]) for group in groups)
        self._ensure_cloud_capacity(total_points)
        offset = 0
        for group in groups:
            points = np.asarray(group.points_world, dtype=np.float32)
            if points.size == 0:
                continue
            count = int(points.shape[0])
            destination = self._cloud_buffer[offset : offset + count]
            np.copyto(destination["x"], points[:, 0], casting="unsafe")
            np.copyto(destination["y"], points[:, 1], casting="unsafe")
            np.copyto(destination["z"], points[:, 2], casting="unsafe")
            color_key = group.global_track_id or group.group_id
            color = _color(color_key)
            packed = destination["rgb"]
            packed.fill(np.uint32(color[0]))
            packed <<= np.uint32(8)
            packed |= np.uint32(color[1])
            packed <<= np.uint32(8)
            packed |= np.uint32(color[2])
            offset += count
        cloud.width = offset
        cloud.row_step = cloud.point_step * offset
        cloud.data = b"" if offset == 0 else self._cloud_buffer[:offset].tobytes(order="C")

        for slot in range(self._max_groups):
            text = message_set.markers.markers[2 * slot]
            box = message_set.markers.markers[2 * slot + 1]
            text.header.frame_id = self._world_frame
            text.header.stamp = stamp
            box.header.frame_id = self._world_frame
            box.header.stamp = stamp
            if slot >= len(groups):
                text.action = Marker.DELETE
                box.action = Marker.DELETE
                continue
            group = groups[slot]
            color_key = group.global_track_id or group.group_id
            color = _color_f32(color_key)
            if group.centroid_world is None:
                text.action = Marker.DELETE
            else:
                text.action = Marker.ADD
                text.pose.position.x = float(group.centroid_world[0])
                text.pose.position.y = float(group.centroid_world[1])
                text.pose.position.z = float(group.centroid_world[2]) + 0.14
                text.color.r, text.color.g, text.color.b = map(float, color)
                member_text = ",".join(
                    f"{camera}:s{instance.tracker_slot}"
                    for camera, instance in group.members
                )
                gid = group.global_track_id if group.global_track_id is not None else "?"
                text.text = f"G{gid} {group.semantic_label} [{member_text}]"

            if group.bbox_min is None or group.bbox_max is None:
                box.action = Marker.DELETE
            else:
                box.action = Marker.ADD
                minimum = group.bbox_min
                maximum = group.bbox_max
                center = 0.5 * (minimum + maximum)
                size = np.maximum(maximum - minimum, 1e-3)
                box.pose.position.x = float(center[0])
                box.pose.position.y = float(center[1])
                box.pose.position.z = float(center[2])
                box.scale.x = float(size[0])
                box.scale.y = float(size[1])
                box.scale.z = float(size[2])
                box.color.r, box.color.g, box.color.b = map(float, color)

        self.last_build_timings_ms = (
            {"rviz_fused_build_cpu": 1000.0 * (time.perf_counter() - started)}
            if self._profile_enabled
            else {}
        )
        return message_set

    def publish_messages(self, messages: _FusedMessageSet) -> None:
        started = time.perf_counter()
        self.points_pub.publish(messages.points)
        self.markers_pub.publish(messages.markers)
        self.last_publish_timings_ms = (
            {"publish_fused_cpu": 1000.0 * (time.perf_counter() - started)}
            if self._profile_enabled
            else {}
        )
