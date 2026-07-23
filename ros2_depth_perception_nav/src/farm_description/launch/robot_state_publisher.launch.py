"""Publish the Husky+RealSense URDF to /robot_description and TF (static links)."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('farm_description')
    xacro_file = os.path.join(pkg, 'urdf', 'husky_realsense.urdf.xacro')

    use_sim_time = LaunchConfiguration('use_sim_time')

    # NOTE (Humble): the Command substitution returns a *string*; it MUST be wrapped
    # in ParameterValue(..., value_type=str) or robot_state_publisher dies parsing URDF.
    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', xacro_file]),
        value_type=str,
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': use_sim_time,
            }],
        ),
    ])
