"""Configuration loading for the vision/follow module.

Design intent
-------------
Everything that is "tuning" (gains, thresholds, model path, snapshot folder)
lives in a YAML file so it can be tweaked without touching code. The one
thing that is deliberately *not* defaulted here is the target class: it is
meant to come from the mission JSON step (`target_class`) so the same running
system can be told "follow a person" one time and "follow a car" the next
without a code change or restart. A YAML-level `default_target_classes` is
still provided purely as a convenience for running the vision module
standalone (see tests/mock_demo.py) when there is no mission JSON at hand.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional

import yaml


@dataclasses.dataclass
class DetectorConfig:
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    default_target_classes: List[str] = dataclasses.field(default_factory=lambda: ["person"])


@dataclasses.dataclass
class FollowControllerConfig:
    # Desired apparent size of the target, expressed as (bbox height / frame
    # height). This is the standoff-distance setpoint: a bigger ratio means
    # "get closer", a smaller ratio means "back off". See follow_controller.py
    # for how this turns into a forward-speed command.
    desired_bbox_height_ratio: float = 0.35

    # Proportional gains. Each axis of motion is controlled independently by
    # its own P controller (see follow_controller.py docstring for the full
    # derivation).
    kp_yaw: float = 1.2
    kp_vertical: float = 0.8
    kp_forward: float = 1.5

    # Output clamps (safety limits) so a bad detection can never command a
    # violent maneuver.
    max_yaw_rate: float = 0.6          # rad/s
    max_vertical_speed: float = 0.5    # m/s
    max_forward_speed: float = 1.5     # m/s

    # Deadbands: errors smaller than these are treated as "close enough" and
    # produce zero command. Without this a P controller chatters forever
    # around the setpoint because pixel-level detector noise never lets the
    # error hit exactly zero.
    deadband_center_norm: float = 0.05   # fraction of half-frame
    deadband_size_ratio: float = 0.03    # fraction of frame height

    # Exponential moving average smoothing factor applied to the bounding box
    # before it is fed to the controller. 0 < alpha <= 1; lower = smoother but
    # more lag. Filters frame-to-frame detector jitter.
    ema_alpha: float = 0.4

    # If no detection arrives for longer than this, the controller drops out
    # of TRACKING into LOST and commands a safe hover instead of coasting on a
    # stale bounding box.
    lost_target_timeout_s: float = 2.0

    # Control loop rate assumed by the caller (used only for documentation /
    # sanity checks, the controller itself is driven by whatever dt it is
    # given each call).
    control_rate_hz: float = 10.0


@dataclasses.dataclass
class SnapshotConfig:
    output_dir: str = "snapshots"
    # Placeholder for a real "send to operator" integration (webhook URL,
    # email relay, MQTT topic, etc). Left as None means "log locally only".
    operator_endpoint: Optional[str] = None


@dataclasses.dataclass
class VisionFollowConfig:
    detector: DetectorConfig = dataclasses.field(default_factory=DetectorConfig)
    follow_controller: FollowControllerConfig = dataclasses.field(default_factory=FollowControllerConfig)
    snapshot: SnapshotConfig = dataclasses.field(default_factory=SnapshotConfig)

    @classmethod
    def load(cls, path: str | Path) -> "VisionFollowConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            detector=DetectorConfig(**raw.get("detector", {})),
            follow_controller=FollowControllerConfig(**raw.get("follow_controller", {})),
            snapshot=SnapshotConfig(**raw.get("snapshot", {})),
        )
