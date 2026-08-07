from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from .data_types import FrameResult
from .ros_utils import color_for_track


class RosVisualizer:
    def __init__(self, node: Any, camera_name: str, output_prefix: str, world_frame: str) -> None:
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image, PointCloud2
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray

        self.node = node
        self.camera_name = camera_name
        self.output_prefix = output_prefix
        self.world_frame = world_frame
        self.bridge = CvBridge()
        self.overlay_pub = node.create_publisher(Image, f"{output_prefix}/rgb_overlay", 2)
        self.raw_pub = node.create_publisher(Image, f"{output_prefix}/raw_masks", 2)
        self.filtered_pub = node.create_publisher(Image, f"{output_prefix}/filtered_masks", 2)
        self.rejected_pub = node.create_publisher(Image, f"{output_prefix}/depth_rejected", 2)
        self.cloud_pub = node.create_publisher(PointCloud2, f"{output_prefix}/visible_points", 2)
        self.marker_pub = node.create_publisher(MarkerArray, f"{output_prefix}/markers", 2)
        self.state_pub = node.create_publisher(String, f"{output_prefix}/state", 2)
        self.profile_pub = node.create_publisher(String, f"{output_prefix}/profiling", 2)

    def publish(self, result: FrameResult) -> None:
        stamp = result.frame.stamp_ns
        overlay, raw_image, filtered_image, rejected_image = self._render_images(result)
        self._publish_image(self.overlay_pub, overlay, stamp)
        self._publish_image(self.raw_pub, raw_image, stamp)
        self._publish_image(self.filtered_pub, filtered_image, stamp)
        self._publish_image(self.rejected_pub, rejected_image, stamp)
        self.cloud_pub.publish(self._pointcloud_message(result))
        self.marker_pub.publish(self._marker_message(result))

        from std_msgs.msg import String

        states = [
            {
                "track_id": item.track_id,
                "label": item.label,
                "status": item.status.value,
                "visible_ratio": item.visible_ratio,
                "depth_consistency": item.depth_consistency,
                "tracking_confidence": item.tracking_confidence,
                "motion_prediction_confidence": item.motion_prediction_confidence,
                "point_count": int(item.points_world.shape[0]),
            }
            for item in result.instances
        ]
        state_msg = String()
        state_msg.data = json.dumps(
            {
                "camera": self.camera_name,
                "frame_index": result.frame.frame_index,
                "keyframe": result.keyframe,
                "dropped_frames": int(result.metadata.get("dropped_frames", 0)),
                "instances": states,
            }
        )
        self.state_pub.publish(state_msg)
        profile_msg = String()
        profile_msg.data = json.dumps(
            {
                **result.timings_ms,
                "dropped_frames": int(result.metadata.get("dropped_frames", 0)),
                "keyframe": result.keyframe,
            }
        )
        self.profile_pub.publish(profile_msg)

    def _render_images(self, result: FrameResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rgb = result.frame.rgb.astype(np.uint8)
        overlay = rgb.astype(np.float32)
        raw_image = np.zeros_like(rgb)
        filtered_image = np.zeros_like(rgb)
        rejected_image = np.zeros_like(rgb)
        for instance in result.instances:
            color = np.array(color_for_track(instance.track_id), dtype=np.uint8)
            raw_image[instance.raw_mask] = color
            filtered_image[instance.depth_filtered_mask] = color
            rejected_image[instance.depth_rejected_mask] = (255, 0, 255)
            overlay[instance.depth_filtered_mask] = (
                0.55 * overlay[instance.depth_filtered_mask] + 0.45 * color
            )
        return overlay.astype(np.uint8), raw_image, filtered_image, rejected_image

    def _publish_image(self, publisher: Any, image: np.ndarray, stamp_ns: int) -> None:
        message = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
        message.header.frame_id = f"{self.camera_name}_optical_frame"
        message.header.stamp.sec = stamp_ns // 1_000_000_000
        message.header.stamp.nanosec = stamp_ns % 1_000_000_000
        publisher.publish(message)

    def _pointcloud_message(self, result: FrameResult) -> Any:
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header

        chunks: list[np.ndarray] = []
        for instance in result.instances:
            if instance.points_world.shape[0] == 0:
                continue
            color = np.tile(np.array(color_for_track(instance.track_id), np.uint8), (instance.points_world.shape[0], 1))
            rgb_uint = (
                color[:, 0].astype(np.uint32) << 16
                | color[:, 1].astype(np.uint32) << 8
                | color[:, 2].astype(np.uint32)
            )
            packed = np.empty(
                instance.points_world.shape[0],
                dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4"), ("track_id", "<i4")],
            )
            packed["x"] = instance.points_world[:, 0]
            packed["y"] = instance.points_world[:, 1]
            packed["z"] = instance.points_world[:, 2]
            packed["rgb"] = rgb_uint
            packed["track_id"] = instance.track_id
            chunks.append(packed)
        data = np.concatenate(chunks) if chunks else np.empty(0, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4"), ("track_id", "<i4")])
        header = Header()
        header.frame_id = self.world_frame
        header.stamp.sec = result.frame.stamp_ns // 1_000_000_000
        header.stamp.nanosec = result.frame.stamp_ns % 1_000_000_000
        message = PointCloud2()
        message.header = header
        message.height = 1
        message.width = len(data)
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            PointField(name="track_id", offset=16, datatype=PointField.INT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = data.dtype.itemsize
        message.row_step = message.point_step * message.width
        message.is_dense = False
        message.data = data.tobytes()
        return message

    def _marker_message(self, result: FrameResult) -> Any:
        from visualization_msgs.msg import Marker, MarkerArray

        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        marker_id = 1
        for instance in result.instances:
            color = color_for_track(instance.track_id)
            if instance.centroid_world is not None:
                sphere = Marker()
                sphere.header.frame_id = self.world_frame
                sphere.ns = "centroids"
                sphere.id = marker_id
                marker_id += 1
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = map(float, instance.centroid_world)
                sphere.pose.orientation.w = 1.0
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.045
                sphere.color.r, sphere.color.g, sphere.color.b = [value / 255.0 for value in color]
                sphere.color.a = 1.0
                array.markers.append(sphere)

                text = Marker()
                text.header.frame_id = self.world_frame
                text.ns = "labels"
                text.id = marker_id
                marker_id += 1
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose.position.x = float(instance.centroid_world[0])
                text.pose.position.y = float(instance.centroid_world[1])
                text.pose.position.z = float(instance.centroid_world[2] + 0.10)
                text.pose.orientation.w = 1.0
                text.scale.z = 0.055
                text.color.r = text.color.g = text.color.b = text.color.a = 1.0
                text.text = (
                    f"{instance.track_id}: {instance.label} | {instance.status.value}\n"
                    f"vis={instance.visible_ratio:.2f} conf={instance.motion_prediction_confidence:.2f}"
                )
                array.markers.append(text)
            if instance.bbox_3d_min is not None and instance.bbox_3d_max is not None:
                box = Marker()
                box.header.frame_id = self.world_frame
                box.ns = "boxes"
                box.id = marker_id
                marker_id += 1
                box.type = Marker.CUBE
                box.action = Marker.ADD
                center = (instance.bbox_3d_min + instance.bbox_3d_max) * 0.5
                extent = np.maximum(instance.bbox_3d_max - instance.bbox_3d_min, 1e-3)
                box.pose.position.x, box.pose.position.y, box.pose.position.z = map(float, center)
                box.pose.orientation.w = 1.0
                box.scale.x, box.scale.y, box.scale.z = map(float, extent)
                box.color.r, box.color.g, box.color.b = [value / 255.0 for value in color]
                box.color.a = 0.18
                array.markers.append(box)
        return array
