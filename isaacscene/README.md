# isaacscene

四个主模块：

- `scene_settings.py`：桌面和日常物体。
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
