from sam_rgbd_tracking_benchmark.config import load_config


def test_config_override():
    config = load_config(
        "configs/benchmark.yaml",
        ["tracker.backend=mock", "runtime.camera_names=[camera_0]"],
    )
    assert config.tracker.backend == "mock"
    assert config.runtime.camera_names == ["camera_0"]
    assert config.postprocess.erosion_pixels == 3
