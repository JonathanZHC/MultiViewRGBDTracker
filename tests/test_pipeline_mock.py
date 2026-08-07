from sam_rgbd_tracking_benchmark.config import load_config
from sam_rgbd_tracking_benchmark.pipeline import CameraTrackingPipeline
from sam_rgbd_tracking_benchmark.synthetic_demo import make_frame


def test_pipeline_mock_end_to_end(tmp_path):
    config = load_config(
        "configs/benchmark.yaml",
        [
            "runtime.camera_names=[camera_0]",
            f"runtime.log_dir={tmp_path.as_posix()}",
            "detector.backend=ground_truth",
            "tracker.backend=mock",
            "detector.refresh_seconds=0.2",
            "profiling.summary_interval_frames=0",
        ],
    )
    pipeline = CameraTrackingPipeline("camera_0", config)
    outputs = [pipeline.process(make_frame(index)) for index in range(6)]
    pipeline.close()
    assert outputs[0].keyframe
    assert len(outputs[-1].instances) == 3
    assert all(instance.points_world.ndim == 2 for instance in outputs[-1].instances)
    assert (tmp_path / "camera_0" / "mock" / "timing.csv").exists()
