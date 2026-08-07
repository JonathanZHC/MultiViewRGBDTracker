from __future__ import annotations

import traceback

import argparse
import queue
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from .component import SAMTrackingComponent
from .config import load_config
from .visualization import RvizPublisher


@dataclass
class _Packet:
    color: Any
    depth: Any
    info: Any


class _CameraWorker:
    """Keep only the newest synchronized RGB-D frame for one camera."""

    def __init__(self, node: Any, camera_name: str, config) -> None:
        import message_filters
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image
        from rclpy.qos import qos_profile_sensor_data

        self.node = node
        self.camera_name = camera_name
        self.config = config
        self.bridge = CvBridge()
        self.component = SAMTrackingComponent(config, camera_name=camera_name)
        self.visualizer = RvizPublisher(node, camera_name, config)
        self.queue: queue.Queue[_Packet] = queue.Queue(
            maxsize=int(config.runtime.queue_size)
        )
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name=f"tracking-{camera_name}",
            daemon=True,
        )

        color_topic = str(config.ros.color_topic).format(camera=camera_name)
        depth_topic = str(config.ros.depth_topic).format(camera=camera_name)
        info_topic = str(config.ros.camera_info_topic).format(camera=camera_name)

        self.color_sub = message_filters.Subscriber(
            node,
            Image,
            color_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_sub = message_filters.Subscriber(
            node,
            Image,
            depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.info_sub = message_filters.Subscriber(
            node,
            CameraInfo,
            info_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=max(4, int(config.runtime.queue_size) * 2),
            slop=float(config.ros.sync_slop_seconds),
        )
        self.sync.registerCallback(self._sync_callback)
        self.thread.start()

    def _sync_callback(self, color: Any, depth: Any, info: Any) -> None:
        packet = _Packet(color, depth, info)
        try:
            self.queue.put_nowait(packet)
            return
        except queue.Full:
            pass

        if not bool(self.config.runtime.drop_when_busy):
            return

        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            pass

    def _world_from_camera(
        self,
        frame_id: str,
        stamp: Any,
    ) -> np.ndarray | None:
        try:
            from rclpy.duration import Duration
            from rclpy.time import Time
            from tf2_ros import Buffer, TransformListener

            if not hasattr(self, "tf_buffer"):
                self.tf_buffer = Buffer()
                self.tf_listener = TransformListener(
                    self.tf_buffer,
                    self.node,
                    spin_thread=False,
                )
            transform = self.tf_buffer.lookup_transform(
                str(self.config.ros.world_frame),
                frame_id,
                Time.from_msg(stamp),
                timeout=Duration(seconds=0.01),
            )
        except Exception:
            return None

        translation = transform.transform.translation
        quaternion = transform.transform.rotation
        x, y, z, w = (
            float(quaternion.x),
            float(quaternion.y),
            float(quaternion.z),
            float(quaternion.w),
        )
        rotation = np.array(
            [
                [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
            ],
            dtype=np.float32,
        )
        result = np.eye(4, dtype=np.float32)
        result[:3, :3] = rotation
        result[:3, 3] = [
            float(translation.x),
            float(translation.y),
            float(translation.z),
        ]
        return result

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                packet = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                rgb = self.bridge.imgmsg_to_cv2(
                    packet.color,
                    desired_encoding="rgb8",
                )
                depth = np.asarray(
                    self.bridge.imgmsg_to_cv2(
                        packet.depth,
                        desired_encoding="passthrough",
                    )
                )
                if depth.dtype == np.uint16:
                    depth_m = depth.astype(np.float32) * 0.001
                else:
                    depth_m = depth.astype(np.float32, copy=False)

                intrinsics = packet.info.k
                frame_id = (
                    packet.color.header.frame_id
                    or f"{self.camera_name}_optical_frame"
                )
                stamp = packet.color.header.stamp
                world_from_camera = self._world_from_camera(frame_id, stamp)
                timestamp_ns = (
                    int(stamp.sec) * 1_000_000_000
                    + int(stamp.nanosec)
                )

                result = self.component.process_arrays(
                    rgb,
                    depth_m,
                    fx=float(intrinsics[0]),
                    fy=float(intrinsics[4]),
                    cx=float(intrinsics[2]),
                    cy=float(intrinsics[5]),
                    timestamp_ns=timestamp_ns,
                    world_from_camera=world_from_camera,
                )
                self.visualizer.publish(result, stamp)
            except Exception as error:
                self.node.get_logger().error(
                    f"{self.camera_name}: "
                    f"{type(error).__name__}: {error}\n"
                    f"{traceback.format_exc()}"
                )

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.component.print_stats()
        self.component.close()


class TrackingNode:
    def __init__(self, config) -> None:
        from rclpy.node import Node

        class _Node(Node):
            pass

        self.node = _Node("sam_rgbd_tracking")
        self.workers = [
            _CameraWorker(self.node, str(name), config)
            for name in config.runtime.camera_names
        ]
        self.node.get_logger().info(
            "Tracking "
            f"cameras={list(config.runtime.camera_names)} "
            f"backend={config.tracker.backend}"
        )

    def close(self) -> None:
        for worker in self.workers:
            worker.close()
        self.node.destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tracking.yaml")
    parser.add_argument(
        "--tracker",
        choices=("sam_mt", "efficient_tam"),
    )
    return parser.parse_args()


def main() -> None:
    import rclpy

    args = parse_args()
    config = load_config(args.config, tracker=args.tracker)
    rclpy.init()
    wrapper = TrackingNode(config)
    try:
        rclpy.spin(wrapper.node)
    except KeyboardInterrupt:
        pass
    finally:
        wrapper.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
