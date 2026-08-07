# Isaac Sim publisher

This directory follows the standalone layout proven in the `YOLOE` branch of
`JonathanZHC/ScenePredictor`:

1. parse arguments and configure ROS;
2. create `SimulationApp` before importing Omniverse modules;
3. define cameras as `UsdGeom.Camera` prims;
4. write camera orientation with `Gf.Quatf`;
5. attach CUDA RGB/depth Replicator annotators;
6. attach a non-colorized instance-segmentation annotator for benchmark GT;
7. warm up until all annotators return valid frames;
8. update object root transforms, call `simulation_app.update()`, capture, and
   publish in a permanent loop.

The active files are:

- `run_isaacsim.py` — standalone lifecycle and timed publishing loop;
- `scene_settings.py` — static, dynamic, and periodic occlusion scenes;
- `camera_settings.py` — camera creation and annotator capture;
- `ros_camera_publisher.py` — synchronized ROS 2 messages and static TF;
- `camera_math.py` — dependency-free camera convention conversions.

The simulator publishes, per camera:

- `/{camera}/color/image_raw` (`rgb8`)
- `/{camera}/depth/image_raw` (`32FC1`, metres)
- `/{camera}/camera_info`
- `/{camera}/gt/instance` (`32SC1`)
- `/{camera}/gt/metadata` (JSON)
- `/tf_static`
