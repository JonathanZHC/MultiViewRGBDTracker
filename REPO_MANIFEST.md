# Repository manifest

## Runtime environment

- `Dockerfile`: Isaac Sim 6.0.1, ROS 2 Jazzy, tracking Python venv, SAM3,
  SAM-MT and EfficientTAM sources.
- `scripts/docker_common.sh`: common GPU/X11/network/bind-mount arguments.
- `scripts/run_in_container.sh`: one-shot or persistent-container execution.
- `scripts/run_isaac.sh`: system-Jazzy + Isaac standalone launcher.

## Isaac Sim

- `isaac_sim/run_isaacsim.py`: standalone lifecycle, warmup, capture, timing.
- `isaac_sim/scene_settings.py`: static, dynamic and occlusion scenes.
- `isaac_sim/camera_settings.py`: USD cameras and Replicator annotators.
- `isaac_sim/ros_camera_publisher.py`: RGB-D, calibration, GT and TF topics.
- `isaac_sim/camera_math.py`: ROS-optical/USD camera transforms.

## Tracking benchmark

- `src/sam_rgbd_tracking_benchmark/`: detector, tracker adapters,
  association, depth ownership, point-cloud extraction, profiling, evaluation
  and ROS visualization.
- `configs/benchmark.yaml`: tracker/detector/post-processing parameters.
- `configs/isaac.yaml`: camera, scene and RGB-D publication settings.

## Testing and tools

- `tests/`: unit and synthetic pipeline tests.
- `scripts/smoke_test.sh`: tests plus synthetic end-to-end run.
- `scripts/record_dataset.sh`: deterministic dataset recording.
- `scripts/benchmark_all.sh`: same-input SAM-MT/EfficientTAM comparison.
