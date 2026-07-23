"""ROS 2 entry point: wires up the PX4/Gazebo-specific pieces (camera
subscriber, PX4 offboard wrapper) and hands them to the stack-agnostic
MissionExecutor from executor/executor.py.

Run (after sourcing ROS 2 + your workspace, with PX4 SITL + Gazebo already
up -- see sim/launch/search_and_follow.launch.py and the README):

    python3 -m ros2_nodes.mission_executor_node \
        --mission mission/mission_example.json \
        --vision-config config/target_config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import rclpy
from rclpy.node import Node

from executor.executor import ExecutorContext, MissionExecutor
from executor.registry import ACTION_REGISTRY
from ros2_nodes.camera_subscriber import CameraSubscriber
from ros2_nodes.vehicle_offboard import VehicleOffboard
from vision.config import VisionFollowConfig


def load_mission(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        mission = json.load(f)
    # Accept either a bare list of steps or {"steps": [...]}; your upstream
    # pipeline's validator decides the canonical shape, this just accepts
    # either since it's convenient for hand-written example missions.
    if isinstance(mission, dict) and "steps" in mission:
        mission = mission["steps"]
    return mission


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic mission executor (PX4 + ROS 2 + Gazebo)")
    parser.add_argument("--mission", required=True, help="Path to mission JSON")
    parser.add_argument("--vision-config", default="config/target_config.yaml")
    parser.add_argument("--camera-topic", default="/camera")
    args = parser.parse_args()

    rclpy.init()
    node = Node("mission_executor")
    exit_code = 0
    try:
        vision_config = VisionFollowConfig.load(args.vision_config)
        camera = CameraSubscriber(node, topic=args.camera_topic)
        vehicle = VehicleOffboard(node)

        context = ExecutorContext(vehicle=vehicle, camera=camera, vision_config=vision_config, node=node)
        mission = load_mission(args.mission)

        results = MissionExecutor(mission, ACTION_REGISTRY, context).run()
        for i, result in enumerate(results):
            node.get_logger().info(f"step {i}: success={result.success} message={result.message}")

        if not all(r.success for r in results):
            exit_code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
