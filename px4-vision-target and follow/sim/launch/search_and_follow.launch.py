"""Minimal launch file for Senior Challenge 3.

Scope of what this file does vs. what you run yourself:

  Terminal 1 (you run this, per PX4's own SITL workflow):
      cd <your PX4-Autopilot checkout>
      make px4_sitl gz_x500_mono_cam     # EDIT ME if your checkout uses a
                                          # different camera-equipped model
                                          # name -- see README "Sim setup".

  Terminal 2 (you run this, standard PX4 ROS 2 bridge):
      MicroXRCEAgent udp4 -p 8888

  Terminal 3 (this file):
      ros2 launch sim/launch/search_and_follow.launch.py

This file only starts the two things that are actually part of this
challenge's deliverable: the Gazebo->ROS 2 camera bridge, and the mission
executor. It deliberately does not try to also launch PX4/Gazebo itself --
that boot sequence is owned by PX4's own `make px4_sitl` tooling and
reimplementing it here would just be a second, more fragile copy of logic
that already exists and that you already have working PX4 SITL habits around.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Confirmed against a live `gz topic -l` on this checkout: PX4-Autopilot
# main + Gazebo Sim 8.14.0 (Harmonic), model x500_mono_cam_0, world "default".
# Sensor name is "camera", not "imager" -- differs from some older
# tutorials/PX4 versions, so re-check with `gz topic -l` if you switch PX4
# branch/tag or model.
GZ_CAMERA_TOPIC = "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"
ROS_CAMERA_TOPIC = "/camera"
# ---------------------------------------------------------------------------
#
# GOTCHA #1 (ROS 2 Humble + Gazebo Harmonic specifically): a `ros_gz_bridge`
# built from source against Harmonic (GZ_VERSION=harmonic, correct libs
# confirmed via ldd -- gz-sim8/gz-transport13/gz-msgs10) still fails to
# create a GZ->ROS Image bridge via `parameter_bridge`, throwing
# std::runtime_error out of BridgeHandle's get_factory(ros_type, gz_type)
# lookup -- i.e. the (sensor_msgs/msg/Image, gz.msgs.Image) pair isn't
# resolving in that binary's factory table, for reasons not fully root
# caused (not a version/build/domain issue -- all three verified fine).
# `/clock` and other basic types were not confirmed broken the same way, so
# this looks Image-specific rather than a broken `parameter_bridge` build
# overall.
#
# WORKAROUND: use the purpose-built `ros_gz_image` package instead of
# `ros_gz_bridge`'s generic `parameter_bridge` for the camera topic only --
# confirmed working end-to-end (~28Hz) on this same checkout. It takes the
# raw Gazebo topic name as a positional CLI arg (no "@type[type" syntax) and
# publishes on a topic with the *same name* as the Gazebo topic by default,
# which is why we still need the `remappings=` below to land it on
# ROS_CAMERA_TOPIC.


def generate_launch_description():
    mission_arg = DeclareLaunchArgument(
        "mission", default_value="mission/mission_example.json",
        description="Path to the mission JSON to execute",
    )
    vision_config_arg = DeclareLaunchArgument(
        "vision_config", default_value="config/target_config.yaml",
    )

    camera_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        # image_bridge takes the raw Gazebo image topic name as a positional
        # arg -- see GOTCHA #1 above for why this isn't ros_gz_bridge's
        # generic parameter_bridge.
        arguments=[GZ_CAMERA_TOPIC],
        remappings=[(GZ_CAMERA_TOPIC, ROS_CAMERA_TOPIC)],
        output="screen",
    )

    # Run as a plain module rather than a colcon-installed executable so this
    # works straight out of the repo with no ament package build step --
    # you only need ROS 2 sourced and px4_msgs built/importable.
    executor_process = ExecuteProcess(
        cmd=[
            "python3", "-m", "ros2_nodes.mission_executor_node",
            "--mission", LaunchConfiguration("mission"),
            "--vision-config", LaunchConfiguration("vision_config"),
            "--camera-topic", ROS_CAMERA_TOPIC,
        ],
        output="screen",
    )

    return LaunchDescription([mission_arg, vision_config_arg, camera_bridge, executor_process])
