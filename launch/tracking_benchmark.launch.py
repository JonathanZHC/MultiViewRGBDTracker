from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("tracker", default_value="sam_mt"),
            DeclareLaunchArgument("detector", default_value="sam3"),
            ExecuteProcess(
                cmd=[
                    "python3.12",
                    "-m",
                    "sam_rgbd_tracking_benchmark.node",
                    "--config",
                    "configs/benchmark.yaml",
                    "--tracker",
                    LaunchConfiguration("tracker"),
                    "--detector",
                    LaunchConfiguration("detector"),
                ],
                output="screen",
            ),
        ]
    )
