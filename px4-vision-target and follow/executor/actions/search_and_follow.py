"""The `search_and_follow` mission mode -- the integration point requested by
this challenge.

Mission JSON step shape:

    {"action": "search_and_follow", "target_class": "person", "return_after_seconds": 30}

What happens when the executor reaches this step:
  1. Build a Detector scoped to exactly `target_class` (never hardcoded --
     it's read straight from the step) and a fresh FollowController /
     SnapshotSender for this target.
  2. Engage PX4 offboard mode and arm (if not already flying).
  3. Run a fixed-rate control loop for up to `return_after_seconds`:
       - pull the latest camera frame
       - run YOLO, pick the best matching detection
       - on first acquisition, fire a snapshot to the operator
       - feed the detection into the deterministic FollowController
       - publish the resulting body-frame velocity command to PX4
  4. On timeout (or if the target is LOST for the full duration), hover and
     hand control back to the executor, which moves on to the next mission
     step (e.g. `land`).

No LLM call happens anywhere in this loop -- steps 3's control loop runs
entirely on FollowController's proportional-control math (vision/follow_controller.py).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import rclpy

from executor.actions.base import ActionHandler, ActionResult
from executor.executor import ExecutorContext
from executor.schema import SEARCH_AND_FOLLOW_REQUIRED, require_fields
from vision.detector import Detector, pick_target
from vision.follow_controller import FollowController, FollowState
from vision.snapshot_sender import SnapshotSender

logger = logging.getLogger(__name__)


class SearchAndFollowAction(ActionHandler):
    def execute(self, step: Dict[str, Any], context: ExecutorContext) -> ActionResult:
        require_fields(step, "search_and_follow", SEARCH_AND_FOLLOW_REQUIRED)
        target_class = step["target_class"]
        return_after_seconds = float(step["return_after_seconds"])

        vision_cfg = context.vision_config
        node = context.node
        camera = context.camera
        vehicle = context.vehicle

        detector = Detector(
            model_path=vision_cfg.detector.model_path,
            target_classes=[target_class],
            confidence_threshold=vision_cfg.detector.confidence_threshold,
        )
        controller = FollowController(vision_cfg.follow_controller)
        controller.reset()

        snapshot_dir = context.snapshot_dir_override or vision_cfg.snapshot.output_dir
        snapshot_sender = SnapshotSender(snapshot_dir, vision_cfg.snapshot.operator_endpoint)
        snapshot_sender.reset()

        if not vehicle.engage_offboard_and_arm():
            return ActionResult(False, "search_and_follow: failed to engage offboard mode / arm")

        control_period_s = 1.0 / vision_cfg.follow_controller.control_rate_hz
        start_time_s = _now_seconds(node)
        last_time_s = start_time_s
        last_state = None

        while True:
            # Pumps the node's subscription callbacks (camera frames, PX4
            # telemetry) for up to one control period, then returns control
            # here so we run the loop at a bounded rate instead of spinning
            # as fast as possible.
            rclpy.spin_once(node, timeout_sec=control_period_s)

            now_s = _now_seconds(node)
            if now_s - start_time_s >= return_after_seconds:
                break
            dt = max(now_s - last_time_s, 1e-3)
            last_time_s = now_s

            detection = None
            frame = camera.get_latest_frame()
            if frame is not None:
                detections = detector.infer(frame)
                detection = pick_target(detections)
                if detection is not None:
                    snapshot_sender.maybe_send(
                        frame, detection,
                        mission_context={"target_class": target_class, "action": "search_and_follow"},
                    )

            command = controller.update(detection, dt)
            if command.state == FollowState.LOST and last_state != FollowState.LOST:
                snapshot_sender.notify_lost()
            last_state = command.state

            vehicle.publish_body_velocity(
                forward=command.forward_speed,
                vertical_down=command.vertical_speed,
                yaw_rate=command.yaw_rate,
            )

        vehicle.publish_body_velocity(forward=0.0, vertical_down=0.0, yaw_rate=0.0)  # hover before returning control
        return ActionResult(
            success=True,
            message=f"search_and_follow completed after {return_after_seconds}s (final state: "
                    f"{last_state.value if last_state else 'no_detections'})",
            data={"final_state": last_state.value if last_state else None, "target_class": target_class},
        )


def _now_seconds(node) -> float:
    return node.get_clock().now().nanoseconds / 1e9
