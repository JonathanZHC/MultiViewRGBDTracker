from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .data_types import CameraIntrinsics, RGBDFrame
from .pipeline import CameraTrackingPipeline


def load_frame(path: Path, camera_name: str, frame_index: int) -> RGBDFrame:
    data = np.load(path, allow_pickle=True)
    intrinsics_array = np.asarray(data["intrinsics"], dtype=np.float32).reshape(-1)
    intrinsics = CameraIntrinsics(
        width=int(data["rgb"].shape[1]),
        height=int(data["rgb"].shape[0]),
        fx=float(intrinsics_array[0]),
        fy=float(intrinsics_array[1]),
        cx=float(intrinsics_array[2]),
        cy=float(intrinsics_array[3]),
    )
    metadata_raw = data.get("gt_metadata_json", np.array("{}", dtype=object)).item()
    metadata = {int(key): value for key, value in json.loads(str(metadata_raw)).items()}
    return RGBDFrame(
        camera_name=camera_name,
        frame_index=frame_index,
        stamp_ns=int(data.get("stamp_ns", frame_index * 33_333_333)),
        rgb=np.asarray(data["rgb"], dtype=np.uint8),
        depth_m=np.asarray(data["depth_m"], dtype=np.float32),
        intrinsics=intrinsics,
        world_from_camera=np.asarray(data.get("world_from_camera", np.eye(4)), dtype=np.float32),
        gt_instance_map=np.asarray(data["gt_instance_map"], dtype=np.int32) if "gt_instance_map" in data else None,
        gt_metadata=metadata,
    )


def save_result(path: Path, result) -> None:
    shape = result.owner_track_map.shape
    masks = (
        np.stack([item.depth_filtered_mask for item in result.instances])
        if result.instances
        else np.empty((0, *shape), bool)
    )
    track_ids = np.array([item.track_id for item in result.instances], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        owner_track_map=result.owner_track_map,
        masks=masks,
        track_ids=track_ids,
        statuses=np.asarray([str(item.status) for item in result.instances], dtype=object),
        point_counts=np.asarray([item.points_camera.shape[0] for item in result.instances], dtype=np.int32),
        visible_ratios=np.asarray([item.visible_ratio for item in result.instances], dtype=np.float32),
        depth_consistency=np.asarray([item.depth_consistency for item in result.instances], dtype=np.float32),
        gt_instance_map=(
            result.frame.gt_instance_map
            if result.frame.gt_instance_map is not None
            else np.empty((0, 0), np.int32)
        ),
        gt_metadata_json=json.dumps(result.frame.gt_metadata),
        timings_json=json.dumps(result.timings_ms),
        keyframe=result.keyframe,
        anomaly_triggered=result.anomaly_triggered,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--tracker", choices=["sam_mt", "efficient_tam", "mock"], required=True)
    parser.add_argument("--detector", choices=["sam3", "ground_truth"], default="ground_truth")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output or f"logs/replay/{args.tracker}")
    config = load_config(
        args.config,
        [
            f"tracker.backend={args.tracker}",
            f"detector.backend={args.detector}",
            "runtime.camera_names=[camera_0]",
            f"runtime.log_dir={str(output / 'profiling')}",
        ],
    )
    camera_dirs = [path for path in dataset.iterdir() if path.is_dir()]
    if not camera_dirs:
        camera_dirs = [dataset]
    for camera_dir in camera_dirs:
        camera_name = camera_dir.name if camera_dir != dataset else "camera_0"
        pipeline = CameraTrackingPipeline(camera_name, config)
        frame_paths = sorted(camera_dir.glob("frame_*.npz"))
        for index, frame_path in enumerate(frame_paths):
            result = pipeline.process(load_frame(frame_path, camera_name, index))
            save_result(output / camera_name / f"result_{index:06d}.npz", result)
        pipeline.close()
        print(f"{camera_name}: replayed {len(frame_paths)} frames with {args.tracker}")


if __name__ == "__main__":
    main()
