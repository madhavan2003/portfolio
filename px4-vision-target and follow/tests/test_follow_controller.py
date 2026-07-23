"""Unit tests for the deterministic follow controller.

Pure Python + stdlib only (no numpy/torch/ROS/YOLO needed) -- these should
run on any machine with just `pytest` installed, which is the point: the
control-loop math should be reviewable/verifiable independent of the whole
sim stack.

Run with:  python3 -m pytest tests/test_follow_controller.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.config import FollowControllerConfig
from vision.detector import Detection
from vision.follow_controller import FollowController, FollowState

FRAME_W, FRAME_H = 640, 480


def make_config(**overrides) -> FollowControllerConfig:
    return FollowControllerConfig(**overrides)


def make_detection(cx, cy, height, width=None) -> Detection:
    width = width or height * 0.5
    return Detection(
        class_name="person", confidence=0.9,
        x1=cx - width / 2, y1=cy - height / 2, x2=cx + width / 2, y2=cy + height / 2,
        frame_w=FRAME_W, frame_h=FRAME_H,
    )


def test_centered_correct_size_produces_near_zero_command():
    cfg = make_config(desired_bbox_height_ratio=0.35)
    controller = FollowController(cfg)
    detection = make_detection(cx=FRAME_W / 2, cy=FRAME_H / 2, height=0.35 * FRAME_H)

    command = controller.update(detection, dt=0.1)

    assert command.state == FollowState.TRACKING
    assert command.yaw_rate == 0.0
    assert command.vertical_speed == 0.0
    assert abs(command.forward_speed) < 1e-6


def test_target_left_of_center_yaws_left_negative():
    cfg = make_config(ema_alpha=1.0)  # alpha=1 -> no smoothing lag, one call converges fully
    controller = FollowController(cfg)
    detection = make_detection(cx=FRAME_W * 0.2, cy=FRAME_H / 2, height=0.35 * FRAME_H)

    command = controller.update(detection, dt=0.1)

    assert command.yaw_rate < 0, "target left of center should command a negative (turn-left) yaw rate"


def test_target_right_of_center_yaws_right_positive():
    cfg = make_config(ema_alpha=1.0)
    controller = FollowController(cfg)
    detection = make_detection(cx=FRAME_W * 0.8, cy=FRAME_H / 2, height=0.35 * FRAME_H)

    command = controller.update(detection, dt=0.1)

    assert command.yaw_rate > 0, "target right of center should command a positive (turn-right) yaw rate"


def test_small_bbox_far_target_moves_forward():
    cfg = make_config(ema_alpha=1.0, desired_bbox_height_ratio=0.35)
    controller = FollowController(cfg)
    # Bbox much smaller than the desired ratio => target reads as "far away".
    detection = make_detection(cx=FRAME_W / 2, cy=FRAME_H / 2, height=0.10 * FRAME_H)

    command = controller.update(detection, dt=0.1)

    assert command.forward_speed > 0, "an apparently-distant target should command forward motion"


def test_large_bbox_close_target_backs_off():
    cfg = make_config(ema_alpha=1.0, desired_bbox_height_ratio=0.35)
    controller = FollowController(cfg)
    detection = make_detection(cx=FRAME_W / 2, cy=FRAME_H / 2, height=0.60 * FRAME_H)

    command = controller.update(detection, dt=0.1)

    assert command.forward_speed < 0, "an apparently-close target should command backward motion"


def test_commands_are_clamped_to_configured_limits():
    cfg = make_config(ema_alpha=1.0, kp_yaw=100.0, max_yaw_rate=0.6)
    controller = FollowController(cfg)
    detection = make_detection(cx=FRAME_W, cy=FRAME_H / 2, height=0.35 * FRAME_H)  # extreme off-center

    command = controller.update(detection, dt=0.1)

    assert command.yaw_rate == cfg.max_yaw_rate


def test_deadband_suppresses_small_errors():
    cfg = make_config(ema_alpha=1.0, deadband_center_norm=0.10, kp_yaw=5.0)
    controller = FollowController(cfg)
    # 3% off center, inside the 10% deadband -> should be fully suppressed.
    detection = make_detection(cx=FRAME_W / 2 + 0.03 * FRAME_W / 2, cy=FRAME_H / 2, height=0.35 * FRAME_H)

    command = controller.update(detection, dt=0.1)

    assert command.yaw_rate == 0.0


def test_losing_target_past_timeout_reports_lost_and_hovers():
    cfg = make_config(lost_target_timeout_s=1.0)
    controller = FollowController(cfg)
    controller.update(make_detection(cx=FRAME_W / 2, cy=FRAME_H / 2, height=0.35 * FRAME_H), dt=0.1)

    # Feed "no detection" past the timeout.
    command = controller.update(None, dt=1.5)

    assert command.state == FollowState.LOST
    assert command.forward_speed == 0.0
    assert command.vertical_speed == 0.0
    assert command.yaw_rate == 0.0


def test_brief_dropout_within_grace_period_keeps_tracking():
    cfg = make_config(lost_target_timeout_s=2.0)
    controller = FollowController(cfg)
    controller.update(make_detection(cx=FRAME_W * 0.8, cy=FRAME_H / 2, height=0.35 * FRAME_H), dt=0.1)

    # One dropped frame, well within the timeout -- YOLO can legitimately
    # miss a single frame of a real target.
    command = controller.update(None, dt=0.2)

    assert command.state == FollowState.TRACKING


def test_reset_clears_state_between_missions():
    cfg = make_config()
    controller = FollowController(cfg)
    controller.update(make_detection(cx=FRAME_W / 2, cy=FRAME_H / 2, height=0.35 * FRAME_H), dt=0.1)
    assert controller.state == FollowState.TRACKING

    controller.reset()

    assert controller.state == FollowState.SEARCHING
