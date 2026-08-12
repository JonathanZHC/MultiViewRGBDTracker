# isaacscene

五个主模块：

- `scene_settings.py`：桌面和日常物体。
- `occlusion_scene.py`：确定性的前景遮挡测试面板与 clear/partial/full 遮挡轨迹。
- `camera_settings.py`：两台相机、标定、RGB-D、点云和可选 corruption。
- `ros_camera_publisher.py`：ROS 2 Image、CameraInfo、PointCloud2、Pose、TF。
- `run_isaacsim.py`：启动 Isaac Sim 并组合前三个模块。

默认话题：

- `/camera_0/color/image_raw`
- `/camera_0/depth/image_raw`
- `/camera_0/camera_info`
- `/camera_0/points`
- `/camera_0/pose`
- `/camera_1/color/image_raw`
- `/camera_1/depth/image_raw`
- `/camera_1/camera_info`
- `/camera_1/points`
- `/camera_1/pose`
- `/cameras/fused_points`

静态 TF：

- `world -> camera_0_optical_frame`
- `world -> camera_1_optical_frame`

## Occlusion test scene

Use `--scene occlusion` for a deterministic visibility stress test. The scene
contains a rear-center mustard bottle plus a food can and ball. A large opaque
panel moves laterally in front of them. Each half-cycle follows:

`clear -> partial -> full -> partial -> clear`

The panel is a test fixture and is not part of the semantic target vocabulary.
Its default full motion period is 12 s; `--motion-speed-scale` changes the
trajectory speed without changing the geometry. Runtime logs emit
`[OCCLUSION] state=clear|partial|full` at phase transitions.

Example:

```bash
./run_isaacsim.py --scene occlusion --rgbd-hz 30 --pointcloud-hz 0
```

## Occlusion benchmark scene

`--scene occlusion` keeps the mustard bottle and food can static, moves the
opaque foreground panel through clear/partial/full occlusion phases, and moves
the ball with exactly the same bounce/translation trajectory used by
`--scene dynamic`.  `--motion-speed-scale` scales both the ball and panel
motion together.
