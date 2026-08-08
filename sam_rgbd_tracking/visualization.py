from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .data_types import FrameResult


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


def _color(track_id: int) -> np.ndarray:
    return _PALETTE[(track_id - 1) % len(_PALETTE)]


def make_overlay(result: FrameResult) -> np.ndarray:
    image = result.frame.rgb.copy()
    for instance in result.instances:
        color = _color(instance.track_id)
        mask = instance.mask
        if mask.any():
            blended = (
                image[mask].astype(np.float32) * 0.55
                + color.astype(np.float32) * 0.45
            )
            image[mask] = np.clip(blended, 0, 255).astype(np.uint8)
            ys, xs = np.nonzero(mask)
            x0, y0, x1, y1 = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            )
            cv2.rectangle(image, (x0, y0), (x1, y1), color.tolist(), 2)
            text = (
                f"#{instance.track_id} {instance.label} "
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
    h, w = result.frame.depth_m.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for instance in result.instances:
        canvas[instance.raw_mask if raw else instance.mask] = _color(
            instance.track_id
        )
    return canvas


class RvizPublisher:
    """ROS/RViz adapter kept separate from the reusable tracking component.

    ``build_messages`` and ``publish_messages`` are intentionally separate so
    the ROS worker can profile CPU message construction independently from the
    actual ``publisher.publish(...)`` calls. ``publish`` remains as the old
    one-call API for compatibility with external users.
    """

    def __init__(self, node: Any, camera_name: str, config) -> None:
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image, PointCloud2
        from visualization_msgs.msg import MarkerArray

        self.node = node
        self.camera_name = camera_name
        self.config = config
        self.bridge = CvBridge()
        prefix = str(config.ros.output_prefix).format(camera=camera_name)
        self.points_pub = node.create_publisher(
            PointCloud2,
            f"{prefix}/visible_points",
            2,
        )
        self.markers_pub = node.create_publisher(
            MarkerArray,
            f"{prefix}/markers",
            2,
        )
        self.overlay_pub = node.create_publisher(
            Image,
            f"{prefix}/rgb_overlay",
            2,
        )
        self.raw_pub = node.create_publisher(
            Image,
            f"{prefix}/raw_masks",
            2,
        )
        self.filtered_pub = node.create_publisher(
            Image,
            f"{prefix}/filtered_masks",
            2,
        )

    @staticmethod
    def _pack_rgb(colors: np.ndarray) -> np.ndarray:
        rgb_u32 = (
            (colors[:, 0].astype(np.uint32) << 16)
            | (colors[:, 1].astype(np.uint32) << 8)
            | colors[:, 2].astype(np.uint32)
        )
        return rgb_u32.view(np.float32)

    def _pointcloud(self, result: FrameResult, stamp: Any):
        from sensor_msgs.msg import PointCloud2, PointField

        parts = []
        colors = []
        use_world = result.frame.world_from_camera is not None
        for instance in result.instances:
            points = (
                instance.points_world
                if use_world
                else instance.points_camera
            )
            if points is not None and points.size:
                parts.append(points.astype(np.float32, copy=False))
                colors.append(instance.colors_rgb)
        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = (
            str(self.config.ros.world_frame)
            if use_world
            else f"{self.camera_name}_optical_frame"
        )
        cloud.height = 1
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
        if not parts:
            cloud.width = 0
            cloud.is_bigendian = False
            cloud.is_dense = False
            cloud.point_step = 16
            cloud.row_step = 0
            cloud.data = b""
            return cloud
        xyz = np.concatenate(parts, axis=0)
        rgb = np.concatenate(colors, axis=0)
        packed = np.empty((xyz.shape[0], 4), dtype=np.float32)
        packed[:, :3] = xyz
        packed[:, 3] = self._pack_rgb(rgb)
        cloud.width = xyz.shape[0]
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = False
        cloud.data = packed.tobytes()
        return cloud

    def _markers(self, result: FrameResult, stamp: Any):
        from visualization_msgs.msg import Marker, MarkerArray

        array = MarkerArray()
        frame_id = (
            str(self.config.ros.world_frame)
            if result.frame.world_from_camera is not None
            else f"{self.camera_name}_optical_frame"
        )
        for instance in result.instances:
            centroid = (
                instance.centroid_world
                if result.frame.world_from_camera is not None
                else instance.centroid_camera
            )
            if centroid is None:
                continue
            color = _color(instance.track_id).astype(np.float32) / 255.0
            text = Marker()
            text.header.frame_id = frame_id
            text.header.stamp = stamp
            text.ns = f"{self.camera_name}_labels"
            text.id = int(instance.track_id)
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            (
                text.pose.position.x,
                text.pose.position.y,
                text.pose.position.z,
            ) = map(float, centroid)
            text.pose.position.z += 0.12
            text.pose.orientation.w = 1.0
            text.scale.z = 0.08
            (
                text.color.r,
                text.color.g,
                text.color.b,
                text.color.a,
            ) = float(color[0]), float(color[1]), float(color[2]), 1.0
            text.text = (
                f"#{instance.track_id} {instance.label} "
                f"[{instance.status.value}]"
            )
            array.markers.append(text)

            if instance.bbox_min is not None and instance.bbox_max is not None:
                box = Marker()
                box.header.frame_id = frame_id
                box.header.stamp = stamp
                box.ns = f"{self.camera_name}_boxes"
                box.id = int(instance.track_id)
                box.type = Marker.CUBE
                box.action = Marker.ADD
                center = 0.5 * (instance.bbox_min + instance.bbox_max)
                size = np.maximum(
                    instance.bbox_max - instance.bbox_min,
                    1e-3,
                )
                (
                    box.pose.position.x,
                    box.pose.position.y,
                    box.pose.position.z,
                ) = map(float, center)
                box.pose.orientation.w = 1.0
                box.scale.x, box.scale.y, box.scale.z = map(float, size)
                (
                    box.color.r,
                    box.color.g,
                    box.color.b,
                    box.color.a,
                ) = float(color[0]), float(color[1]), float(color[2]), 0.14
                array.markers.append(box)
        return array

    def build_messages(self, result: FrameResult, stamp: Any) -> dict[str, Any]:
        overlay = self.bridge.cv2_to_imgmsg(
            make_overlay(result),
            encoding="rgb8",
        )
        overlay.header.stamp = stamp
        overlay.header.frame_id = f"{self.camera_name}_optical_frame"

        messages: dict[str, Any] = {
            "points": self._pointcloud(result, stamp),
            "markers": self._markers(result, stamp),
            "overlay": overlay,
            "raw": None,
            "filtered": None,
        }
        if bool(self.config.runtime.publish_debug_images):
            raw = self.bridge.cv2_to_imgmsg(
                make_mask_debug(result, raw=True),
                encoding="rgb8",
            )
            filtered = self.bridge.cv2_to_imgmsg(
                make_mask_debug(result, raw=False),
                encoding="rgb8",
            )
            raw.header = overlay.header
            filtered.header = overlay.header
            messages["raw"] = raw
            messages["filtered"] = filtered
        return messages

    def publish_messages(self, messages: dict[str, Any]) -> None:
        self.points_pub.publish(messages["points"])
        self.markers_pub.publish(messages["markers"])
        self.overlay_pub.publish(messages["overlay"])
        if messages.get("raw") is not None:
            self.raw_pub.publish(messages["raw"])
        if messages.get("filtered") is not None:
            self.filtered_pub.publish(messages["filtered"])

    def publish(self, result: FrameResult, stamp: Any) -> None:
        """Compatibility wrapper preserving the original public API."""
        self.publish_messages(self.build_messages(result, stamp))
