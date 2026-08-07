from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np

from camera_math import rotation_matrix_to_quaternion_xyzw
from camera_settings import CameraFrame, CameraRigConfig, CameraRuntime


class RosCameraPublisher:
    """Publish synchronized RGB, depth, calibration, GT and camera TF topics."""

    def __init__(
        self,
        cameras: list[CameraRuntime],
        rig: CameraRigConfig,
        depth_noise_enabled: bool = False,
        depth_noise_sigma_m: float = 0.0,
        depth_dropout_probability: float = 0.0,
    ) -> None:
        import rclpy
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String
        from tf2_ros import StaticTransformBroadcaster

        if not rclpy.ok():
            rclpy.init(args=None)
        self.rclpy = rclpy
        self.node = rclpy.create_node("sam_rgbd_isaac_camera_publisher")
        self.cameras = {camera.spec.name: camera for camera in cameras}
        self.rig = rig
        self.depth_noise_enabled = bool(depth_noise_enabled)
        self.depth_noise_sigma_m = float(depth_noise_sigma_m)
        self.depth_dropout_probability = float(depth_dropout_probability)

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=3,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        metadata_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publishers: dict[str, dict[str, Any]] = {}
        for name in self.cameras:
            prefix = f"/{name}"
            self.publishers[name] = {
                "color": self.node.create_publisher(
                    Image, f"{prefix}/color/image_raw", image_qos
                ),
                "depth": self.node.create_publisher(
                    Image, f"{prefix}/depth/image_raw", image_qos
                ),
                "info": self.node.create_publisher(
                    CameraInfo, f"{prefix}/camera_info", metadata_qos
                ),
                "gt": self.node.create_publisher(
                    Image, f"{prefix}/gt/instance", image_qos
                ),
                "metadata": self.node.create_publisher(
                    String, f"{prefix}/gt/metadata", metadata_qos
                ),
            }
        self.static_tf = StaticTransformBroadcaster(self.node)
        self._publish_static_transforms()
        self.node.get_logger().info(
            "Publishing synchronized RGB8, 32FC1 depth, CameraInfo, "
            "32SC1 instance GT and metadata for cameras="
            f"{list(self.cameras)}"
        )

    @staticmethod
    def _image_message(
        array: np.ndarray,
        encoding: str,
        frame_id: str,
        stamp: Any,
    ) -> Any:
        from sensor_msgs.msg import Image

        contiguous = np.ascontiguousarray(array)
        message = Image()
        message.header.frame_id = frame_id
        message.header.stamp = stamp
        message.height = int(contiguous.shape[0])
        message.width = int(contiguous.shape[1])
        message.encoding = encoding
        message.is_bigendian = int(sys.byteorder == "big")
        channels = 1 if contiguous.ndim == 2 else int(contiguous.shape[2])
        message.step = int(
            contiguous.shape[1] * channels * contiguous.dtype.itemsize
        )
        message.data = contiguous.tobytes()
        return message

    def _camera_info(self, runtime: CameraRuntime, stamp: Any) -> Any:
        from sensor_msgs.msg import CameraInfo

        message = CameraInfo()
        message.header.frame_id = f"{runtime.spec.name}_optical_frame"
        message.header.stamp = stamp
        message.width = int(self.rig.width)
        message.height = int(self.rig.height)
        fx = float(runtime.K[0, 0])
        fy = float(runtime.K[1, 1])
        cx = float(runtime.K[0, 2])
        cy = float(runtime.K[1, 2])
        message.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        message.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        message.distortion_model = "plumb_bob"
        message.d = []
        return message

    def _publish_static_transforms(self) -> None:
        from geometry_msgs.msg import TransformStamped

        stamp = self.node.get_clock().now().to_msg()
        messages = []
        for runtime in self.cameras.values():
            transform = runtime.T_world_from_camera_optical
            quaternion = rotation_matrix_to_quaternion_xyzw(
                transform[:3, :3]
            )
            message = TransformStamped()
            message.header.stamp = stamp
            message.header.frame_id = self.rig.world_frame_id
            message.child_frame_id = f"{runtime.spec.name}_optical_frame"
            message.transform.translation.x = float(transform[0, 3])
            message.transform.translation.y = float(transform[1, 3])
            message.transform.translation.z = float(transform[2, 3])
            message.transform.rotation.x = float(quaternion[0])
            message.transform.rotation.y = float(quaternion[1])
            message.transform.rotation.z = float(quaternion[2])
            message.transform.rotation.w = float(quaternion[3])
            messages.append(message)
        self.static_tf.sendTransform(messages)

    def _apply_depth_noise(self, depth_m: np.ndarray) -> np.ndarray:
        if not self.depth_noise_enabled:
            return depth_m
        depth = depth_m.copy()
        finite = np.isfinite(depth)
        if self.depth_noise_sigma_m > 0.0 and finite.any():
            depth[finite] += np.random.normal(
                0.0,
                self.depth_noise_sigma_m,
                int(finite.sum()),
            ).astype(np.float32)
        if self.depth_dropout_probability > 0.0:
            dropout = np.random.random(depth.shape) < self.depth_dropout_probability
            depth[dropout] = np.nan
        return depth

    def publish(self, frames: dict[str, CameraFrame]) -> None:
        from std_msgs.msg import String

        stamp = self.node.get_clock().now().to_msg()
        for name, frame in frames.items():
            runtime = self.cameras[name]
            publishers = self.publishers[name]
            frame_id = f"{name}_optical_frame"
            depth = self._apply_depth_noise(frame.depth_m)
            publishers["color"].publish(
                self._image_message(frame.rgb, "rgb8", frame_id, stamp)
            )
            publishers["depth"].publish(
                self._image_message(depth, "32FC1", frame_id, stamp)
            )
            publishers["gt"].publish(
                self._image_message(
                    frame.instance_map,
                    "32SC1",
                    frame_id,
                    stamp,
                )
            )
            publishers["info"].publish(self._camera_info(runtime, stamp))
            metadata = String()
            metadata.data = json.dumps(
                {str(key): value for key, value in frame.instance_metadata.items()},
                separators=(",", ":"),
            )
            publishers["metadata"].publish(metadata)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self) -> None:
        try:
            self.node.destroy_node()
        finally:
            if self.rclpy.ok():
                self.rclpy.shutdown()
