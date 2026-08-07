"""Checkpoint-free smoke test for the reusable component boundary."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam_rgbd_tracking.component import SAMTrackingComponent
from sam_rgbd_tracking.config import Config
from sam_rgbd_tracking.data_types import DetectionInstance, TrackerPrediction
from sam_rgbd_tracking.processing import color_embedding
from sam_rgbd_tracking.trackers.base import MultiObjectTracker


class FakeDetector:
    def detect(self, frame):
        mask = np.zeros(frame.depth_m.shape, dtype=bool)
        mask[12:36, 18:46] = True
        return [
            DetectionInstance(
                detection_id=1,
                label="object",
                score=0.9,
                mask=mask,
                embedding=color_embedding(frame.rgb, mask),
            )
        ]


class FakeTracker(MultiObjectTracker):
    def __init__(self):
        self.ids = []
        self.masks = np.empty((0, 0, 0), np.float32)

    def initialize(self, frame, seeds):
        self.ids = [seed.track_id for seed in seeds]
        self.masks = (
            np.stack([seed.mask.astype(np.float32) * 2.0 - 1.0 for seed in seeds])
            if seeds
            else np.empty((0, *frame.depth_m.shape), np.float32)
        )
        return TrackerPrediction(
            list(self.ids),
            self.masks.copy(),
            np.ones(len(self.ids), np.float32),
        )

    def track(self, frame):
        return TrackerPrediction(
            list(self.ids),
            self.masks.copy(),
            np.ones(len(self.ids), np.float32),
        )


def config() -> Config:
    return Config(
        {
            "runtime": {
                "target_hz": 30.0,
                "enable_tf32": False,
            },
            "detector": {
                "refresh_seconds": 10.0,
                "phase_offsets_seconds": {"camera_0": 0.0},
                "min_frames_between_triggers": 5,
                "trigger_on_anomaly": False,
            },
            "tracker": {"backend": "fake"},
            "association": {
                "max_centroid_distance_m": 0.35,
                "min_mask_iou": 0.05,
                "weight_centroid": 2.0,
                "weight_iou": 1.0,
                "weight_depth": 0.5,
                "weight_embedding": 0.35,
                "label_mismatch_cost": 10.0,
                "lost_ttl_frames": 45,
            },
            "postprocess": {
                "mask_threshold": 0.0,
                "overlap_depth_only": True,
                "depth_model_min_pixels": 10,
                "depth_gate_mad_scale": 4.0,
                "depth_gate_min_m": 0.035,
                "depth_gate_max_m": 0.25,
                "logit_weight": 0.15,
                "erosion_pixels": 1,
                "depth_edge_threshold_m": 0.04,
                "min_component_pixels": 10,
                "min_valid_depth_m": 0.10,
                "max_valid_depth_m": 6.0,
                "visible_ratio_visible": 0.60,
                "visible_ratio_partial": 0.15,
            },
            "pointcloud": {
                "stride": 1,
                "max_points_per_instance": 30000,
                "transform_to_world": True,
            },
            "profiling": {
                "enabled": True,
                "summary_interval_frames": 0,
                "cuda_events": False,
                "csv_path": "",
            },
        }
    )


def main() -> None:
    component = SAMTrackingComponent(
        config(),
        camera_name="camera_0",
        detector=FakeDetector(),
        tracker=FakeTracker(),
    )
    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[..., 1] = 160
    depth = np.full((48, 64), 1.2, np.float32)
    T = np.eye(4, dtype=np.float32)

    first = component.process_arrays(
        rgb, depth, fx=60.0, fy=60.0, cx=31.5, cy=23.5,
        world_from_camera=T,
    )
    second = component.process_arrays(
        rgb, depth, fx=60.0, fy=60.0, cx=31.5, cy=23.5,
        world_from_camera=T,
    )

    assert first.keyframe
    assert not second.keyframe
    assert len(first.instances) == 1
    assert first.instances[0].track_id == second.instances[0].track_id == 1
    assert first.instances[0].points_world is not None
    assert first.instances[0].points_world.shape[0] > 0
    assert "pipeline_total" in first.timings_ms
    assert "postprocess_cpu" in first.timings_ms
    print("[OK] component smoke test")


if __name__ == "__main__":
    main()
