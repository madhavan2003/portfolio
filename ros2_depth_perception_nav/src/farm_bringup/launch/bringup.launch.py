"""Top-level bringup: Gazebo + Husky/RealSense + depth perception + Nav2 + autonomous
coverage mission (unmapped/partially-known environment navigation).

Args:
  use_rviz:=true|false      open RViz with the farm config
  use_slam:=true|false      run RTAB-Map (decoupled: publish_tf:=false, viz/map only)
  capture:=true|false       run annotated video + path-overlay recorders
  headless:=true|false      Gazebo without the gzclient GUI
  spawn_x/spawn_y/spawn_yaw the blue-marker pose the robot is PLACED at (sim setup;
                            the nav goal/start are still read at runtime from markers)

Ordering note (hard-won): Nav2 must be up and ACTIVATING before RTAB-Map starts, or
RTAB's first heavy map build starves the lifecycle activation and the stack hangs.
The mission node is also delayed until Nav2 is active.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction, GroupAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_bringup = get_package_share_directory('farm_bringup')
    pkg_desc = get_package_share_directory('farm_description')
    pkg_nav = get_package_share_directory('farm_navigation')
    pkg_gazebo = get_package_share_directory('gazebo_ros')

    world = os.path.join(pkg_bringup, 'worlds', 'five_lane_farm.world')
    nav2_params = os.path.join(pkg_nav, 'config', 'nav2_params.yaml')
    rviz_cfg = os.path.join(pkg_bringup, 'rviz', 'farm.rviz')

    use_rviz = LaunchConfiguration('use_rviz')
    use_slam = LaunchConfiguration('use_slam')
    capture = LaunchConfiguration('capture')
    headless = LaunchConfiguration('headless')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    args = [
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_slam', default_value='true'),
        DeclareLaunchArgument('capture', default_value='true'),
        DeclareLaunchArgument('headless', default_value='false'),
        # blue-marker pose (robot placement = sim setup)
        DeclareLaunchArgument('spawn_x', default_value='5.8'),
        DeclareLaunchArgument('spawn_y', default_value='-3.0'),
        DeclareLaunchArgument('spawn_yaw', default_value='3.14159'),
    ]

    # ---- Gazebo ----
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world, 'verbose': 'true',
                          'init': 'true', 'factory': 'true', 'force_system': 'true'}.items(),
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo, 'launch', 'gzclient.launch.py')),
        condition=UnlessCondition(headless),
    )

    # robot description / TF
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_desc, 'launch', 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    spawn = Node(
        package='gazebo_ros', executable='spawn_entity.py', output='screen',
        arguments=['-topic', 'robot_description', '-entity', 'husky',
                   '-x', spawn_x, '-y', spawn_y, '-z', '0.18', '-Y', spawn_yaw],
    )

    # identity static map->odom (planar_move odom is already world-aligned; this just
    # decouples the nav frame from RTAB-Map's jumpy SLAM correction)
    static_map_odom = Node(
        package='farm_navigation', executable='static_map_odom_node', output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # ---- Perception ----
    # (plant_counter_node intentionally not launched here -- this demo is scoped to
    # autonomous navigation of an unmapped/partially-known environment; the depth-based
    # lane/obstacle perception below is what drives Nav2 goal generation.)
    lane = Node(package='farm_perception', executable='lane_perception_node',
                output='screen', parameters=[{'use_sim_time': True}])

    # ---- runtime marker reader ----
    marker = Node(package='farm_navigation', executable='marker_reader_node',
                  output='screen', parameters=[{'use_sim_time': True}])

    # ---- Nav2 (no AMCL / map_server; camera-only costmap) ----
    nav2_nodes = ['controller_server', 'smoother_server', 'planner_server',
                  'behavior_server', 'bt_navigator', 'velocity_smoother']
    nav2 = GroupAction([
        Node(package='nav2_controller', executable='controller_server', output='screen',
             parameters=[nav2_params]),
        Node(package='nav2_planner', executable='planner_server', output='screen',
             parameters=[nav2_params]),
        Node(package='nav2_behaviors', executable='behavior_server', output='screen',
             parameters=[nav2_params]),
        Node(package='nav2_bt_navigator', executable='bt_navigator', output='screen',
             parameters=[nav2_params]),
        Node(package='nav2_velocity_smoother', executable='velocity_smoother', output='screen',
             parameters=[nav2_params]),
        Node(package='nav2_smoother', executable='smoother_server', output='screen',
             parameters=[nav2_params]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', output='screen',
             name='lifecycle_manager_navigation',
             parameters=[{'use_sim_time': True, 'autostart': True,
                          'node_names': nav2_nodes}]),
    ])

    # ---- mission (delayed until Nav2 active) ----
    mission = TimerAction(period=18.0, actions=[
        Node(package='farm_navigation', executable='mission_node', output='screen',
             parameters=[{'use_sim_time': True}]),
    ])

    # ---- recorders ----
    recorders = GroupAction(condition=IfCondition(capture), actions=[
        Node(package='farm_bringup', executable='video_recorder_node', output='screen',
             parameters=[{'use_sim_time': True}]),
        Node(package='farm_bringup', executable='path_recorder_node', output='screen',
             parameters=[{'use_sim_time': True}]),
    ])

    # ---- RViz ----
    rviz = Node(package='rviz2', executable='rviz2', output='screen',
                condition=IfCondition(use_rviz), arguments=['-d', rviz_cfg],
                parameters=[{'use_sim_time': True}])

    # ---- RTAB-Map (decoupled, started LAST) ----
    rtab = TimerAction(period=26.0, actions=[GroupAction(
        condition=IfCondition(use_slam), actions=[
            Node(package='rtabmap_slam', executable='rtabmap', output='screen',
                 name='rtabmap', parameters=[{
                     'use_sim_time': True,
                     'frame_id': 'base_footprint',
                     'odom_frame_id': 'odom',
                     'map_frame_id': 'map_slam',     # do NOT fight the nav map frame
                     'publish_tf': False,            # decoupled from nav TF
                     'subscribe_depth': True,
                     'subscribe_rgb': True,
                     'approx_sync': True,
                     'Rtabmap/DetectionRate': '1.0',
                 }],
                 remappings=[
                     ('rgb/image', '/camera/image_raw'),
                     ('rgb/camera_info', '/camera/camera_info'),
                     ('depth/image', '/camera/depth/image_raw'),
                     ('odom', '/odom'),
                 ],
                 arguments=['--delete_db_on_start']),
        ])])

    ld = LaunchDescription(args)
    for a in [gzserver, gzclient, rsp, spawn, static_map_odom, lane, marker,
              nav2, mission, recorders, rviz, rtab]:
        ld.add_action(a)
    return ld
