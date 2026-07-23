"""Deterministic visual-servoing controller for the "follow" behavior.

No LLM, no network call, no randomness: this is a plain proportional (P)
controller that runs every frame on the executor's control-loop thread. Given
a bounding box (or "no detection"), it outputs a body-frame velocity command.
This module has zero ROS / PX4 / Ultralytics imports on purpose -- it only
knows about pixels and simple kinematics, which is what makes it unit
testable without a simulator (see tests/test_follow_controller.py).

------------------------------------------------------------------------
The control law (three independent P loops)
------------------------------------------------------------------------

We control three axes independently off two pieces of information the
detector gives us each frame: where the bbox center is, and how big the bbox
is.

1) Yaw rate <- horizontal pixel error
   error_x_norm = (bbox_center_x - frame_center_x) / (frame_w / 2)   in [-1, 1]
   yaw_rate = clamp(kp_yaw * error_x_norm, -max_yaw_rate, max_yaw_rate)

   If the target drifts right of center, error_x_norm > 0, so we command a
   positive yaw rate to turn right and re-center it. This is a standard image
   -based P controller: pixel error is already roughly linear in angular
   error for a narrow FOV, so proportional gain on pixel error approximates
   proportional gain on angular error without needing the camera's intrinsic
   matrix.

2) Vertical speed <- vertical pixel error
   error_y_norm = (bbox_center_y - frame_center_y) / (frame_h / 2)
   vertical_speed = clamp(kp_vertical * error_y_norm, -max_v, max_v)

   Same idea on the vertical axis: target below image center => descend
   towards it (positive = down, NED-style, see ControlCommand docstring).

3) Forward speed <- bounding-box size error (the standoff-distance loop)
   size_ratio = bbox_height / frame_h                     (apparent size, 0..1)
   error_size = desired_bbox_height_ratio - size_ratio
   forward_speed = clamp(kp_forward * error_size, -max_fwd, max_fwd)

   Apparent size is inversely proportional to real distance for a fixed
   real-world target height, so it's a cheap monocular distance proxy that
   needs no depth sensor or stereo. If the target looks smaller than our
   desired ratio (it's far), error_size > 0 and we move forward; if it looks
   bigger than desired (too close), error_size < 0 and we back off.

Deadbands: each error is zeroed out inside a small band around zero before
the gain is applied. Detector bounding boxes jitter a few pixels frame to
frame even for a static target; without a deadband the P term never settles
exactly to zero and the vehicle chatters (small oscillating corrections)
forever. The deadband trades a small amount of steady-state centering
accuracy for a controller that actually goes quiet at the setpoint.

Smoothing: bbox center/size are run through an exponential moving average
(EMA) before the error is computed: filtered = alpha*new + (1-alpha)*old.
This reduces frame-to-frame detector jitter feeding into the derivative-free
P loop (we have no D term, so unfiltered noise would otherwise show up
directly as commanded velocity noise).

State machine: SEARCHING (no target ever acquired) -> TRACKING (detection
present, or missing for less than lost_target_timeout_s) -> LOST (missing
longer than the timeout). LOST commands a safe hover (all zero velocities)
rather than coasting on a stale bounding box or continuing to integrate an
old error. This is the entire "fallback" behavior and it is deterministic
code, not a re-query of the LLM.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Optional

from vision.config import FollowControllerConfig
from vision.detector import Detection


class FollowState(enum.Enum):
    SEARCHING = "searching"   # never acquired a target yet
    TRACKING = "tracking"     # actively following
    LOST = "lost"             # had a target, lost it for too long -> hover


@dataclasses.dataclass
class ControlCommand:
    forward_speed: float    # m/s, body-frame +x (nose-forward)
    vertical_speed: float   # m/s, positive = descend (NED convention, matches PX4)
    yaw_rate: float         # rad/s, positive = turn right (clockwise from above)
    state: FollowState


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _deadband(value: float, band: float) -> float:
    return 0.0 if abs(value) < band else value


class FollowController:
    def __init__(self, config: FollowControllerConfig):
        self.config = config
        self._state = FollowState.SEARCHING
        self._time_since_last_detection = 0.0
        # EMA filter state, in normalized units so it's independent of frame size.
        self._filtered_center_x_norm: Optional[float] = None
        self._filtered_center_y_norm: Optional[float] = None
        self._filtered_size_ratio: Optional[float] = None

    @property
    def state(self) -> FollowState:
        return self._state

    def reset(self) -> None:
        """Call when starting a fresh search_and_follow step (new target)."""
        self._state = FollowState.SEARCHING
        self._time_since_last_detection = 0.0
        self._filtered_center_x_norm = None
        self._filtered_center_y_norm = None
        self._filtered_size_ratio = None

    def update(self, detection: Optional[Detection], dt: float) -> ControlCommand:
        """Advance the controller by one frame.

        `detection` is the (already-selected, e.g. via detector.pick_target)
        best match for this frame, or None if nothing matched.
        `dt` is the wall-clock seconds since the previous call.
        """
        cfg = self.config

        if detection is None:
            self._time_since_last_detection += dt
            if self._time_since_last_detection > cfg.lost_target_timeout_s:
                self._state = FollowState.LOST
                return ControlCommand(0.0, 0.0, 0.0, self._state)
            # Still within the grace period: hold the last known command by
            # falling through with the last filtered values, rather than
            # snapping straight to LOST on a single dropped frame (YOLO can
            # legitimately miss one frame of a real target).
            if self._filtered_center_x_norm is None:
                # We were never tracking anything to hold onto.
                return ControlCommand(0.0, 0.0, 0.0, self._state)
        else:
            self._time_since_last_detection = 0.0
            self._state = FollowState.TRACKING

            center_x_norm = (detection.center_x - detection.frame_w / 2.0) / (detection.frame_w / 2.0)
            center_y_norm = (detection.center_y - detection.frame_h / 2.0) / (detection.frame_h / 2.0)
            size_ratio = detection.height / detection.frame_h

            alpha = cfg.ema_alpha
            self._filtered_center_x_norm = self._ema(self._filtered_center_x_norm, center_x_norm, alpha)
            self._filtered_center_y_norm = self._ema(self._filtered_center_y_norm, center_y_norm, alpha)
            self._filtered_size_ratio = self._ema(self._filtered_size_ratio, size_ratio, alpha)

        error_x = _deadband(self._filtered_center_x_norm, cfg.deadband_center_norm)
        error_y = _deadband(self._filtered_center_y_norm, cfg.deadband_center_norm)
        error_size = _deadband(cfg.desired_bbox_height_ratio - self._filtered_size_ratio, cfg.deadband_size_ratio)

        yaw_rate = _clamp(cfg.kp_yaw * error_x, cfg.max_yaw_rate)
        vertical_speed = _clamp(cfg.kp_vertical * error_y, cfg.max_vertical_speed)
        forward_speed = _clamp(cfg.kp_forward * error_size, cfg.max_forward_speed)

        return ControlCommand(forward_speed, vertical_speed, yaw_rate, self._state)

    @staticmethod
    def _ema(previous: Optional[float], new_value: float, alpha: float) -> float:
        if previous is None:
            return new_value
        return alpha * new_value + (1 - alpha) * previous
