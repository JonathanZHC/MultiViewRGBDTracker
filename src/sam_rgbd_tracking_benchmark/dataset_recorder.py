from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera", action="append", default=["camera_0", "camera_1"])
    parser.add_argument("--duration", type=float, default=20.0)
    args, ros_args = parser.parse_known_args()

    import message_filters
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.duration import Duration
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener

    from .ros_utils import matrix_from_transform

    class Recorder(Node):
        def __init__(self) -> None:
            super().__init__("sam_rgbd_dataset_recorder")
            self.bridge = CvBridge()
            self.root = Path(args.output)
            self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.counts = {camera: 0 for camera in args.camera}
            self.gt_images: dict[str, Image] = {}
            self.gt_metadata: dict[str, String] = {}
            self.syncs = []
            self.subscribers = []
            for camera in args.camera:
                color = message_filters.Subscriber(self, Image, f"/{camera}/color/image_raw")
                depth = message_filters.Subscriber(self, Image, f"/{camera}/depth/image_raw")
                info = message_filters.Subscriber(self, CameraInfo, f"/{camera}/camera_info")
                sync = message_filters.ApproximateTimeSynchronizer([color, depth, info], 5, 0.02)
                sync.registerCallback(lambda c, d, i, cam=camera: self.callback(cam, c, d, i))
                self.syncs.append(sync)
                self.subscribers.extend([color, depth, info])
                self.create_subscription(Image, f"/{camera}/gt/instance", lambda msg, cam=camera: self.gt_images.__setitem__(cam, msg), 2)
                self.create_subscription(String, f"/{camera}/gt/metadata", lambda msg, cam=camera: self.gt_metadata.__setitem__(cam, msg), 2)

        def callback(self, camera: str, color: Any, depth: Any, info: Any) -> None:
            rgb = np.asarray(self.bridge.imgmsg_to_cv2(color, "rgb8"), dtype=np.uint8)
            depth_array = np.asarray(self.bridge.imgmsg_to_cv2(depth, "passthrough"))
            depth_m = depth_array.astype(np.float32) * (0.001 if depth.encoding == "16UC1" else 1.0)
            gt = None
            if camera in self.gt_images:
                gt = np.asarray(self.bridge.imgmsg_to_cv2(self.gt_images[camera], "passthrough"), dtype=np.int32)
            metadata = self.gt_metadata.get(camera)
            index = self.counts[camera]
            self.counts[camera] += 1
            out = self.root / camera / f"frame_{index:06d}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            world_from_camera = np.eye(4, dtype=np.float32)
            try:
                transform = self.tf_buffer.lookup_transform(
                    "world",
                    color.header.frame_id,
                    color.header.stamp,
                )
                world_from_camera = matrix_from_transform(transform)
            except Exception:
                pass
            payload = dict(
                rgb=rgb,
                depth_m=depth_m,
                intrinsics=np.array([info.k[0], info.k[4], info.k[2], info.k[5]], np.float32),
                stamp_ns=int(color.header.stamp.sec) * 1_000_000_000 + int(color.header.stamp.nanosec),
                world_from_camera=world_from_camera,
                gt_metadata_json=metadata.data if metadata else "{}",
            )
            if gt is not None:
                payload["gt_instance_map"] = gt
            np.savez_compressed(out, **payload)

    rclpy.init(args=ros_args)
    node = Recorder()
    node.create_timer(args.duration, lambda: rclpy.shutdown())
    try:
        rclpy.spin(node)
    finally:
        print(json.dumps(node.counts, indent=2))
        node.destroy_node()


if __name__ == "__main__":
    main()
