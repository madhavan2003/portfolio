"""Deterministic mission executor.

Reads a validated mission (list of step dicts, each with an "action" field)
and runs each step's registered handler in order. This file has no
ROS/PX4/YOLO imports -- it is pure dispatch logic, which is what makes it
possible to unit test the sequencing/error-handling behavior without a
simulator, and to reuse this exact executor if the underlying vehicle stack
ever changes (e.g. MAVSDK instead of ROS 2) by only swapping what's inside
ExecutorContext.

This module is intentionally small: it is the boundary between "validated
JSON" and "concrete commands" described in your system prompt. All the
stack-specific work (talking to PX4 offboard topics, running YOLO, running
the follow controller) lives in the action handlers and in
ros2_nodes/, not here.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Type

from executor.actions.base import ActionHandler, ActionResult

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ExecutorContext:
    """Everything an ActionHandler might need to act, injected from outside.

    Kept as a plain dataclass of loosely-typed fields (rather than a big
    abstract interface) because different actions need very different
    subsets: `takeoff`/`land` only need `vehicle`; `search_and_follow` needs
    `vehicle`, `camera`, and `vision_config`. Using `Any` here avoids forcing
    every handler to depend on ros2_nodes' concrete types, so
    executor/executor.py and executor/actions/*.py stay importable without
    ROS installed (see tests/).
    """
    vehicle: Any = None            # ros2_nodes.vehicle_offboard.VehicleOffboard
    camera: Any = None             # ros2_nodes.camera_subscriber.CameraSubscriber
    vision_config: Any = None      # vision.config.VisionFollowConfig
    node: Any = None               # the underlying rclpy.Node, for logging/spin/clock
    snapshot_dir_override: Optional[str] = None


class MissionExecutor:
    def __init__(self, mission: List[Dict[str, Any]], registry: Dict[str, Type[ActionHandler]], context: ExecutorContext):
        self.mission = mission
        self.registry = registry
        self.context = context

    def run(self) -> List[ActionResult]:
        results: List[ActionResult] = []
        for i, step in enumerate(self.mission):
            action_name = step.get("action")
            handler_cls = self.registry.get(action_name)
            if handler_cls is None:
                raise ValueError(
                    f"Mission step {i} has unknown action '{action_name}'. "
                    f"Registered actions: {sorted(self.registry)}"
                )
            logger.info("executor: running step %d/%d -> %s", i + 1, len(self.mission), action_name)
            result = handler_cls().execute(step, self.context)
            results.append(result)
            logger.info("executor: step %d/%d -> %s: %s", i + 1, len(self.mission), action_name, result.message)
            if not result.success:
                logger.error("executor: aborting mission, step %d (%s) failed: %s", i + 1, action_name, result.message)
                break
        return results
