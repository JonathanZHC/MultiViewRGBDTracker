import numpy as np

from sam_rgbd_tracking_benchmark.config import load_config
from sam_rgbd_tracking_benchmark.data_types import CameraIntrinsics, RGBDFrame
from sam_rgbd_tracking_benchmark.detector import GroundTruthDetector
from sam_rgbd_tracking_benchmark.pipeline import CameraTrackingPipeline
from sam_rgbd_tracking_benchmark.trackers.mock import MockOpticalFlowTracker


def test_owner_map_matches_final_masks(tmp_path) -> None:
    config = load_config(
        "configs/benchmark.yaml",
        [
            "detector.backend=ground_truth",
            "tracker.backend=mock",
            f"runtime.log_dir={tmp_path}",
            "profiling.cuda_events=false",
        ],
    )
    h, w = 48, 64
    gt = np.zeros((h, w), dtype=np.int32)
    gt[10:34, 12:32] = 1
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.ones((h, w), dtype=np.float32)
    frame = RGBDFrame(
        camera_name="camera_0",
        frame_index=0,
        stamp_ns=0,
        rgb=rgb,
        depth_m=depth,
        intrinsics=CameraIntrinsics(w, h, 60.0, 60.0, w / 2, h / 2),
        gt_instance_map=gt,
        gt_metadata={1: {"label": "box"}},
    )
    pipeline = CameraTrackingPipeline(
        "camera_0", config, detector=GroundTruthDetector(), tracker=MockOpticalFlowTracker()
    )
    result = pipeline.process(frame)
    pipeline.close()
    union = np.zeros((h, w), dtype=bool)
    for instance in result.instances:
        union |= instance.depth_filtered_mask
        assert np.all(result.owner_track_map[instance.depth_filtered_mask] == instance.track_id)
    assert np.array_equal(result.owner_track_map > 0, union)
