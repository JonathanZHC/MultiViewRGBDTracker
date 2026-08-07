from __future__ import annotations

import argparse
import json
import sys
import threading
from typing import Any

import numpy as np

from .config import load_config
from .data_types import CameraIntrinsics, RGBDFrame
from .detector import build_detector
from .pipeline import CameraTrackingPipeline
from .ros_utils import depth_message_to_meters, image_to_numpy, matrix_from_transform, metadata_from_json_message, stamp_to_ns
from .visualization import RosVisualizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--tracker", choices=["sam_mt", "efficient_tam", "mock"])
    parser.add_argument("--detector", choices=["sam3", "ground_truth"])
    parser.add_argument("--camera", action="append", dest="cameras")
    parser.add_argument("--set", action="append", default=[], help="Configuration override key=value")
    return parser


class TrackingBenchmarkNode:
    def __init__(self, config: Any) -> None:
        import message_filters
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.duration import Duration
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String
        from tf2_ros import Buffer, TransformListener

        class _Node(Node):
            pass

        self.node = _Node("sam_rgbd_tracking_benchmark")
        self.config = config
        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.gt_images: dict[str, Any] = {}
        self.gt_metadata: dict[str, Any] = {}
        self.frame_indices = {camera: 0 for camera in config.runtime.camera_names}
        self.processing_locks = {camera: threading.Lock() for camera in config.runtime.camera_names}
        self.dropped_frames = {camera: 0 for camera in config.runtime.camera_names}
        shared_detector = build_detector(config)
        self.pipelines = {
            camera: CameraTrackingPipeline(camera, config, detector=shared_detector)
            for camera in config.runtime.camera_names
        }
        self.visualizers = {
            camera: RosVisualizer(
                self.node,
                camera,
                config.ros.output_prefix.format(camera=camera),
                config.ros.world_frame,
            )
            for camera in config.runtime.camera_names
        }
        self.syncs = []
        self.subscribers = []
        for camera in config.runtime.camera_names:
            color = message_filters.Subscriber(
                self.node,
                Image,
                config.ros.color_topic.format(camera=camera),
            )
            depth = message_filters.Subscriber(
                self.node,
                Image,
                config.ros.depth_topic.format(camera=camera),
            )
            info = message_filters.Subscriber(
                self.node,
                CameraInfo,
                config.ros.camera_info_topic.format(camera=camera),
            )
            sync = message_filters.ApproximateTimeSynchronizer(
                [color, depth, info],
                queue_size=int(config.runtime.queue_size),
                slop=float(config.ros.sync_slop_seconds),
                allow_headerless=False,
            )
            sync.registerCallback(lambda color_msg, depth_msg, info_msg, cam=camera: self._callback(cam, color_msg, depth_msg, info_msg))
            self.syncs.append(sync)
            self.subscribers.extend([color, depth, info])
            self.node.create_subscription(
                Image,
                config.ros.gt_instance_topic.format(camera=camera),
                lambda msg, cam=camera: self.gt_images.__setitem__(cam, msg),
                2,
            )
            self.node.create_subscription(
                String,
                config.ros.gt_metadata_topic.format(camera=camera),
                lambda msg, cam=camera: self.gt_metadata.__setitem__(cam, msg),
                2,
            )
        self.node.get_logger().info(
            f"Started benchmark: cameras={list(config.runtime.camera_names)}, "
            f"detector={config.detector.backend}, tracker={config.tracker.backend}"
        )

    def _callback(self, camera: str, color_msg: Any, depth_msg: Any, info_msg: Any) -> None:
        frame_index = self.frame_indices[camera]
        self.frame_indices[camera] += 1
        lock = self.processing_locks[camera]
        if bool(self.config.runtime.drop_when_busy):
            acquired = lock.acquire(blocking=False)
            if not acquired:
                self.dropped_frames[camera] += 1
                dropped = self.dropped_frames[camera]
                if dropped == 1 or dropped % 100 == 0:
                    self.node.get_logger().warning(
                        f"{camera}: dropped {dropped} synchronized RGB-D frames while processing"
                    )
                return
        else:
            lock.acquire()
        try:
            rgb = image_to_numpy(color_msg, self.bridge, "rgb8")
            depth_m = depth_message_to_meters(depth_msg, self.bridge)
            intrinsics = CameraIntrinsics(
                width=int(info_msg.width),
                height=int(info_msg.height),
                fx=float(info_msg.k[0]),
                fy=float(info_msg.k[4]),
                cx=float(info_msg.k[2]),
                cy=float(info_msg.k[5]),
            )
            stamp_ns = stamp_to_ns(color_msg.header.stamp)
            world_from_camera = np.eye(4, dtype=np.float32)
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.config.ros.world_frame,
                    color_msg.header.frame_id,
                    color_msg.header.stamp,
                )
                world_from_camera = matrix_from_transform(transform)
            except Exception:
                pass
            gt_map = None
            gt_message = self.gt_images.get(camera)
            if gt_message is not None and abs(stamp_to_ns(gt_message.header.stamp) - stamp_ns) < 50_000_000:
                gt_map = image_to_numpy(gt_message, self.bridge, "passthrough").astype(np.int32)
            frame = RGBDFrame(
                camera_name=camera,
                frame_index=frame_index,
                stamp_ns=stamp_ns,
                rgb=rgb,
                depth_m=depth_m,
                intrinsics=intrinsics,
                world_from_camera=world_from_camera,
                gt_instance_map=gt_map,
                gt_metadata=metadata_from_json_message(self.gt_metadata.get(camera)),
            )
            result = self.pipelines[camera].process(
                frame,
                dropped_frames=self.dropped_frames[camera],
            )
            result.metadata["dropped_frames"] = self.dropped_frames[camera]
            self.visualizers[camera].publish(result)
        except Exception as exc:
            self.node.get_logger().error(f"{camera} callback failed: {exc}")
        finally:
            lock.release()

    def destroy(self) -> None:
        for pipeline in self.pipelines.values():
            pipeline.close()
        self.node.destroy_node()


def main() -> None:
    parser = _build_parser()
    args, ros_args = parser.parse_known_args()
    overrides = list(args.set)
    if args.tracker:
        overrides.append(f"tracker.backend={args.tracker}")
    if args.detector:
        overrides.append(f"detector.backend={args.detector}")
    if args.cameras:
        overrides.append("runtime.camera_names=" + json.dumps(args.cameras))
    config = load_config(args.config, overrides)

    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

    rclpy.init(args=ros_args)
    wrapper = TrackingBenchmarkNode(config)
    executor = MultiThreadedExecutor(num_threads=max(2, len(config.runtime.camera_names) + 1))
    executor.add_node(wrapper.node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            executor.shutdown(timeout_sec=1.0)
        except Exception:
            pass
        wrapper.destroy()
        # SIGINT may already have shut the default context down. Calling
        # shutdown twice raises RCLError, so guard it explicitly.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
