from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config
from .data_types import CameraIntrinsics, RGBDFrame
from .pipeline import CameraTrackingPipeline


def make_frame(index: int, width: int = 320, height: int = 240) -> RGBDFrame:
    rgb = np.full((height, width, 3), 35, dtype=np.uint8)
    depth = np.full((height, width), 3.0, dtype=np.float32)
    gt = np.zeros((height, width), dtype=np.int32)
    objects = [
        (1, "bottle", 0.8, 45 + index, 80, 85 + index, 170, (235, 190, 20)),
        (2, "box", 1.1, 210 - index, 100, 275 - index, 175, (30, 145, 240)),
        (3, "shelf", 1.4, 125, 45 + index // 3, 205, 115 + index // 3, (185, 55, 50)),
    ]
    metadata = {}
    # Draw rear to front so the GT map follows visible depth ownership.
    for object_id, label, z, x0, y0, x1, y1, color in sorted(objects, key=lambda item: item[2], reverse=True):
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        rgb[y0:y1, x0:x1] = color
        depth[y0:y1, x0:x1] = z
        gt[y0:y1, x0:x1] = object_id
        metadata[object_id] = {"label": label}
    return RGBDFrame(
        camera_name="camera_0",
        frame_index=index,
        stamp_ns=index * 33_333_333,
        rgb=rgb,
        depth_m=depth,
        intrinsics=CameraIntrinsics(width, height, 260.0, 260.0, width / 2, height / 2),
        gt_instance_map=gt,
        gt_metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    config = load_config(
        args.config,
        [
            "runtime.camera_names=[camera_0]",
            "runtime.log_dir=logs/synthetic",
            "detector.backend=ground_truth",
            "tracker.backend=mock",
            "detector.refresh_seconds=0.5",
        ],
    )
    pipeline = CameraTrackingPipeline("camera_0", config)
    for index in range(args.frames):
        result = pipeline.process(make_frame(index))
        if index % 10 == 0:
            summary = [(item.track_id, item.label, item.status.value, item.points_world.shape[0]) for item in result.instances]
            print(index, summary, f"{result.timings_ms['pipeline_total']:.2f} ms")
    pipeline.close()
    print("Synthetic end-to-end smoke test completed.")


if __name__ == "__main__":
    main()
