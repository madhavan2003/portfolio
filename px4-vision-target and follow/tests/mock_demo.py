#!/usr/bin/env python3
"""Runs the real vision/ module (Detector -> SnapshotSender -> FollowController)
against a webcam or a video file, with no ROS 2 / PX4 / Gazebo involved.

This exists because getting a full PX4 SITL + Gazebo + ROS 2 stack up is
heavy and environment-specific, while the actual thing this challenge is
grading -- detect -> snapshot -> deterministic closed-loop follow -- can be
demonstrated end-to-end against any camera source in under a minute. It is
the exact same detector.py / follow_controller.py / snapshot_sender.py code
that ros2_nodes/mission_executor_node.py drives; only the frame source and
the "send a velocity command" sink differ (here we just print the command
instead of publishing a PX4 TrajectorySetpoint).

Usage:
    python3 tests/mock_demo.py --source 0 --target-class person
    python3 tests/mock_demo.py --source path/to/video.mp4 --target-class car
    python3 tests/mock_demo.py --source 0 --target-class person --headless

Press 'q' to quit (when not running --headless).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from vision.config import VisionFollowConfig
from vision.detector import Detector, pick_target
from vision.follow_controller import FollowController, FollowState
from vision.snapshot_sender import SnapshotSender


def parse_args():
    parser = argparse.ArgumentParser(description="Vision follow-loop demo, no sim required")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--target-class", default=None, help="Overrides detector.default_target_classes from config")
    parser.add_argument("--config", default="config/target_config.yaml")
    parser.add_argument("--headless", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    # A bare integer string means "webcam index"; anything else is a file path.
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def main():
    args = parse_args()
    config = VisionFollowConfig.load(args.config)
    target_classes = [args.target_class] if args.target_class else config.detector.default_target_classes

    print(f"[mock_demo] loading YOLO model '{config.detector.model_path}', target class(es)={target_classes}")
    detector = Detector(
        model_path=config.detector.model_path,
        target_classes=target_classes,
        confidence_threshold=config.detector.confidence_threshold,
    )
    controller = FollowController(config.follow_controller)
    controller.reset()
    snapshot_sender = SnapshotSender(config.snapshot.output_dir, config.snapshot.operator_endpoint)
    snapshot_sender.reset()

    cap = open_source(args.source)
    last_time = time.time()
    last_state = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[mock_demo] end of stream")
                break

            now = time.time()
            dt = max(now - last_time, 1e-3)
            last_time = now

            detections = detector.infer(frame)
            detection = pick_target(detections)

            if detection is not None:
                record = snapshot_sender.maybe_send(
                    frame, detection, mission_context={"target_class": target_classes, "source": "mock_demo"}
                )
                if record is not None:
                    print(f"[mock_demo] ACQUIRED target -> snapshot saved to {record.path}")

            command = controller.update(detection, dt)
            if command.state == FollowState.LOST and last_state != FollowState.LOST:
                snapshot_sender.notify_lost()
                print("[mock_demo] target LOST -> hovering")
            last_state = command.state

            print(
                f"state={command.state.value:9s} "
                f"forward={command.forward_speed:+.2f} m/s  "
                f"vertical={command.vertical_speed:+.2f} m/s  "
                f"yaw_rate={command.yaw_rate:+.2f} rad/s"
            )

            if not args.headless:
                display = frame.copy()
                if detection is not None:
                    cv2.rectangle(display, (int(detection.x1), int(detection.y1)),
                                  (int(detection.x2), int(detection.y2)), (0, 255, 0), 2)
                cv2.putText(display, f"state={command.state.value}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("mock_demo (search_and_follow, no sim)", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
