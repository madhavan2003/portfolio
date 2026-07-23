"""Thin wrapper around PX4's ROS 2 offboard-control API (uXRCE-DDS bridge,
px4_msgs), so action handlers can say "go forward at 0.5 m/s and turn right
at 0.2 rad/s" without knowing about PX4 topic names, message fields, or the
arm/offboard-engage handshake.

Everything below follows PX4's documented ROS 2 offboard control pattern
(see PX4 docs: "ROS 2 Offboard Control"). Three gotchas worth knowing cold if
asked to defend this live:

1. Heartbeat requirement: PX4 will only accept (and stay in) offboard mode
   while it is receiving OffboardControlMode messages at a rate above ~2Hz.
   If that stream stops, PX4 auto-switches out of offboard for safety (it
   assumes the companion computer has died). We publish this on our own
   timer (`_on_heartbeat`), decoupled from the vision control loop's own
   rate, so a slow YOLO inference on one frame can never accidentally starve
   the heartbeat and trigger a mode fallback mid-follow.

2. Body-frame to NED rotation: TrajectorySetpoint.velocity is in the local
   NED frame, but our follow controller reasons in body-frame terms
   ("forward", "yaw rate") because that's what's natural for a camera-based
   controller. We only ever command pure forward speed (no lateral/strafe
   component -- centering is handled by yaw, not by moving sideways), so the
   rotation collapses to:
       v_north = forward * cos(yaw)
       v_east  = forward * sin(yaw)
   using the vehicle's current heading from VehicleLocalPosition.

3. NaN convention: PX4 messages use NaN to mean "ignore this field". We set
   `yaw = NaN` and drive heading purely through `yawspeed` (yaw-rate control)
   instead, since our controller outputs a rate, not a target heading.

Not implemented here: failsafes beyond what PX4 itself provides, geofencing,
RC override handling. This wrapper is scoped to what the search_and_follow /
takeoff / land actions need for this challenge.
"""
from __future__ import annotations

import math
from typing import Optional

import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class VehicleOffboard:
    PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

    def __init__(self, node: Node, heartbeat_rate_hz: float = 10.0):
        self.node = node

        # PX4's uXRCE-DDS bridge publishes/expects best-effort, volatile,
        # keep-last-1 QoS on the /fmu/* topics -- matching it here is
        # required for the subscriptions to actually receive anything.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._offboard_mode_pub = node.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self._trajectory_pub = node.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
        self._vehicle_command_pub = node.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)

        # NOTE: PX4 versions these two topics (_v1 suffix) because their wire
        # format changed in a breaking way; trajectory_setpoint/vehicle_command
        # below did not get versioned. Confirmed against a live `ros2 topic
        # list` on this checkout -- re-check if you change PX4-Autopilot
        # branch/tag, this is exactly the kind of thing that drifts across
        # PX4 releases (see SOURCES.md).
        node.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1", self._on_status, qos)
        node.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self._on_local_position, qos)

        self._latest_status: Optional[VehicleStatus] = None
        self._latest_local_position: Optional[VehicleLocalPosition] = None
        # (forward m/s, vertical_down m/s, yaw_rate rad/s), body-frame.
        # Republished every heartbeat tick, see class docstring point 1.
        self._current_command = (0.0, 0.0, 0.0)

        self._heartbeat_timer = node.create_timer(1.0 / heartbeat_rate_hz, self._on_heartbeat)

    # ---- subscriptions -----------------------------------------------
    def _on_status(self, msg: VehicleStatus) -> None:
        self._latest_status = msg

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        self._latest_local_position = msg

    # ---- streaming setpoints -------------------------------------------
    def _on_heartbeat(self) -> None:
        self._publish_offboard_control_mode()
        self._publish_trajectory_setpoint(*self._current_command)

    def _publish_offboard_control_mode(self) -> None:
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self._timestamp_us()
        self._offboard_mode_pub.publish(msg)

    def _publish_trajectory_setpoint(self, forward: float, vertical_down: float, yaw_rate: float) -> None:
        yaw = self._current_yaw()
        v_north = forward * math.cos(yaw)
        v_east = forward * math.sin(yaw)

        msg = TrajectorySetpoint()
        msg.position = [float("nan")] * 3
        msg.velocity = [v_north, v_east, vertical_down]
        msg.acceleration = [float("nan")] * 3
        msg.yaw = float("nan")     # rate control instead of heading hold, see docstring point 3
        msg.yawspeed = yaw_rate
        msg.timestamp = self._timestamp_us()
        self._trajectory_pub.publish(msg)

    def _current_yaw(self) -> float:
        if self._latest_local_position is not None:
            return self._latest_local_position.heading
        return 0.0

    def _timestamp_us(self) -> int:
        return int(self.node.get_clock().now().nanoseconds / 1000)

    # ---- public API used by action handlers -----------------------------
    def publish_body_velocity(self, forward: float, vertical_down: float, yaw_rate: float) -> None:
        """Called every control-loop tick from search_and_follow. Just
        stores the command -- the actual PX4 publish happens on the next
        heartbeat tick (see _on_heartbeat), keeping the >2Hz stream alive
        independent of the caller's own loop timing."""
        self._current_command = (forward, vertical_down, yaw_rate)

    def engage_offboard_and_arm(self) -> bool:
        """PX4 refuses OFFBOARD until it's already receiving the
        OffboardControlMode stream, so we prime it for a few ticks before
        requesting the mode switch."""
        for _ in range(10):
            self._on_heartbeat()
            rclpy.spin_once(self.node, timeout_sec=0.05)

        self._send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=float(self.PX4_CUSTOM_MAIN_MODE_OFFBOARD)
        )
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

        for _ in range(20):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self._is_offboard_and_armed():
                return True
        return False

    def _is_offboard_and_armed(self) -> bool:
        status = self._latest_status
        if status is None:
            return False
        return (
            status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            and status.arming_state == VehicleStatus.ARMING_STATE_ARMED
        )

    def arm_and_takeoff(self, altitude_m: float, timeout_s: float = 15.0, climb_speed: float = 1.0) -> bool:
        if not self.engage_offboard_and_arm():
            return False

        target_z = -abs(altitude_m)  # NED: up is negative z
        start_s = self.node.get_clock().now().nanoseconds / 1e9
        while True:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            now_s = self.node.get_clock().now().nanoseconds / 1e9
            if now_s - start_s > timeout_s:
                return False

            current_z = self._latest_local_position.z if self._latest_local_position else 0.0
            if current_z <= target_z + 0.3:  # within 30cm of target altitude
                self.publish_body_velocity(0.0, 0.0, 0.0)
                return True
            self.publish_body_velocity(forward=0.0, vertical_down=-climb_speed, yaw_rate=0.0)

    def land(self) -> bool:
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        return True

    def _send_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self._timestamp_us()
        self._vehicle_command_pub.publish(msg)
