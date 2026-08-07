# Changelog

## 0.3.0

- Rebuilt the complete Isaac Sim side around the ScenePredictor YOLOE standalone pattern.
- Fixed camera orientation type mismatch by using `Gf.Quatf`.
- Added CUDA RGB/depth annotators plus non-colorized instance GT annotation.
- Added periodic static, dynamic, and occlusion scenes.
- Added synchronized RGB/depth/CameraInfo/GT/metadata publishing and static TF.
- Removed the superseded World/ReplicatorCamera implementation.
- Removed all checks for nonexistent internal Jazzy directories.
- Kept live host bind mounts; simulator script updates do not require rebuilding the image.
- Added camera-convention unit tests.

# Changelog

## 0.2.1

- Run containers with the Isaac Sim image-native UID/GID 1234 so `/isaac-sim` remains accessible.
- Keep the repository bind-mounted for zero-rebuild script and source updates.
- Earlier v0.2.1 attempted to use internal Jazzy paths; v0.3.0 supersedes that approach with system Jazzy as used by Isaac Sim 6.0 in this image.
- Keep system ROS 2 Jazzy isolated to tracking and RViz processes.
- Guard `rclpy.shutdown()` after SIGINT to avoid duplicate-shutdown RCLError.
- Make runtime cache, checkpoint, dataset, and log directories writable across host/container UIDs.

## 0.2.0

- Bind-mount the complete host repository into every runtime container so source, scripts and configuration changes are immediately visible without rebuilding.
- Add host/container-transparent wrappers for Isaac Sim, tracking, RViz, recording, evaluation, Hugging Face login and checkpoint downloads.
- Separate Isaac Sim's bundled Python/ROS ABI from the external Python 3.12/system ROS 2 Jazzy environment.
- Launch Isaac as `python -m isaac_sim.run_isaacsim` and explicitly expose the repository root to Isaac Python.
- Enable `isaacsim.ros2.bridge` from the standalone application before importing internal `rclpy`.
- Publish RGB, depth and instance images without `cv_bridge` inside Isaac Sim.
- Persist Hugging Face, Torch, Warp, Kit and NVIDIA caches outside disposable containers.
- Mount host passwd/group databases and preserve video/render supplementary groups.
- Correct package smoke-test imports and ROS setup handling under `set -u`.
- Add deterministic checkpoint downloader for SAM3, SAM-MT and EfficientTAM.

## 0.1.0

- Initial standalone benchmark with three Isaac scenes, SAM3 keyframes, SAM-MT/EfficientTAM adapters, RGB-D depth ownership, RViz, profiling, recording, replay and evaluation.
