#!/usr/bin/env python3
"""Perception-driven coverage mission (no hardcoded trajectory).

The robot sweeps the farm corridor-by-corridor, south -> north, then stops at the
green goal marker. Nothing about the route is a fixed list of world waypoints:

  * Each corridor's CENTER comes from depth perception (/farm/lane_center).
  * Each corridor is driven END-TO-END by issuing a single Nav2 NavigateToPose
    goal at the far margin; Nav2 + the local planner produce the real-time,
    collision-free path on the depth-built costmap (DWB hugs the perceived
    obstacle rows, so the robot stays centered in the ~1 m lanes).
  * Stepping to the NEXT corridor uses the perceived bed-row pitch, then SNAPS to
    the next perceived corridor center. A corridor the planner cannot traverse
    (e.g. the 0.4 m narrow lane -> beds block it in the costmap) is detected by
    lack of progress and SKIPPED -- no width constant is hardcoded.
  * Sweeping stops when the goal-marker latitude is reached; the final leg drives
    to the exact green marker pose (read at runtime), retrying on abort.

States: INIT -> COVER -> SHIFT -> COVER ... -> GO_GOAL -> DONE
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

import tf2_ros
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class Mission(Node):
    def __init__(self):
        super().__init__('mission_node')
        # ---- tunable params (documented in WRITEUP hardcoded table) ----
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('default_pitch', 1.5)      # m bed-row pitch fallback
        self.declare_parameter('default_margin_x', 6.0)   # m field half-extent + margin
        self.declare_parameter('min_pass_spacing', 1.2)   # m bed-center spacing -> passable lane
        self.declare_parameter('lane_match_tol', 0.6)     # m, "reached goal latitude"
        self.declare_parameter('arrive_tol', 0.6)         # m, far-end arrival radius
        self.declare_parameter('impassable_progress', 1.5)  # m min progress else skip
        self.declare_parameter('per_lane_timeout', 70.0)  # s safety per lane
        self.declare_parameter('goal_retries', 12)

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.pitch = float(self.get_parameter('default_pitch').value)
        self.margin_x = float(self.get_parameter('default_margin_x').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1)
        latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        latched.reliability = QoSReliabilityPolicy.RELIABLE
        self.create_subscription(PoseStamped, '/farm/goal_pose', self.on_goal, latched)
        self.create_subscription(PoseStamped, '/farm/start_pose', self.on_start, latched)
        self.create_subscription(PoseStamped, '/farm/lane_center', self.on_lane, 5)
        self.create_subscription(Float64MultiArray, '/farm/bed_rows', self.on_beds, 5)

        self.pub_status = self.create_publisher(String, '/mission/status', 5)
        self.pub_summary = self.create_publisher(String, '/mission/summary', 5)
        self.pub_route = self.create_publisher(MarkerArray, '/mission/route', 5)

        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # state
        self.state = 'INIT'
        self.goal_pose = None
        self.start_pose = None
        self.lane_center_y = None
        self.bed_ys = []           # accumulated perceived bed-row map-y's
        self.cur_lane_y = None
        self.lanes_covered = 0
        self.skips = 0
        self.path_len = 0.0
        self._last_xy = None

        self.nav_state = 'idle'    # idle|active|succeeded|aborted
        self._goal_handle = None
        self.lane_start_time = None
        self.lane_start_x = None
        self.shift_start_time = 0.0
        self.cover_retries = 0
        self.last_send = 0.0
        self.goal_attempts = 0
        self.summary_sent = False

        self.create_timer(0.5, self.tick)
        self.create_timer(0.2, self.track_path)
        self.get_logger().info('mission_node up (perception-driven coverage).')

    # ---- subscriptions ----
    def on_goal(self, msg):
        self.goal_pose = msg

    def on_start(self, msg):
        self.start_pose = msg

    def on_lane(self, msg):
        # /farm/lane_center is in odom; transform its y to map
        p = self._to_map(msg.pose.position.x, msg.pose.position.y, msg.header.frame_id)
        if p is not None:
            self.lane_center_y = p[1]

    def on_beds(self, msg):
        # layout: [yaw_odom, x_robot_odom, x_far_odom, bed_y0_odom, ...]
        d = list(msg.data)
        if len(d) < 4:
            return
        # accept only when robot is row-aligned (heading near 0 or pi in map)
        rp = self._robot_pose()
        if rp is None:
            return
        h = rp[2]
        if min(abs(math.sin(h)), abs(math.sin(h))) > 0.30:  # |sin(heading)|>0.3 -> skewed
            return
        for by in d[3:]:
            mp = self._odom_y_to_map(d[1], by)  # transform an odom point to map
            if mp is not None:
                self.bed_ys.append(mp[1])
        # refine pitch from accumulated bed rows
        self._refine_pitch()

    # ---- helpers ----
    def _to_map(self, x, y, frame):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, frame, rclpy.time.Time())
        except Exception:
            return None
        from farm_navigation._tf import apply_xy
        return apply_xy(t, x, y)

    def _odom_y_to_map(self, x_odom, y_odom):
        return self._to_map(x_odom, y_odom, 'odom')

    def _robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        return (t.transform.translation.x, t.transform.translation.y, yaw)

    def _refine_pitch(self):
        if len(self.bed_ys) < 20:
            return
        ys = sorted(self.bed_ys)
        # cluster bed-y's (0.4 m): centers
        centers = []
        cur = [ys[0]]
        for y in ys[1:]:
            if y - cur[-1] < 0.4:
                cur.append(y)
            else:
                centers.append(sum(cur) / len(cur)); cur = [y]
        centers.append(sum(cur) / len(cur))
        if len(centers) >= 2:
            gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
            gaps = [g for g in gaps if g > 0.6]
            if gaps:
                med = sorted(gaps)[len(gaps) // 2]         # median spacing
                self.pitch = min(max(med, 1.2), 1.8)       # clamp to plausible bed pitch

    def _perceived_lane_centers(self):
        """Cluster accumulated bed-row y's -> bed centers -> passable lane centers
        (midpoints of adjacent bed rows whose spacing >= min_pass_spacing)."""
        if len(self.bed_ys) < 8:
            return []
        ys = sorted(self.bed_ys)
        centers = []
        cur = [ys[0]]
        for y in ys[1:]:
            if y - cur[-1] < 0.40:
                cur.append(y)
            else:
                centers.append(sum(cur) / len(cur)); cur = [y]
        centers.append(sum(cur) / len(cur))
        min_pass = float(self.get_parameter('min_pass_spacing').value)
        lanes = []
        for i in range(len(centers) - 1):
            if centers[i + 1] - centers[i] >= min_pass:        # skip narrow lane
                lanes.append(0.5 * (centers[i] + centers[i + 1]))
        return lanes

    def track_path(self):
        rp = self._robot_pose()
        if rp is None:
            return
        if self._last_xy is not None:
            self.path_len += math.hypot(rp[0] - self._last_xy[0], rp[1] - self._last_xy[1])
        self._last_xy = (rp[0], rp[1])

    # ---- nav action ----
    def send_nav(self, x, y, yaw):
        if not self.nav.server_is_ready():
            self.nav.wait_for_server(timeout_sec=1.0)
            if not self.nav.server_is_ready():
                return False
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x); ps.pose.position.y = float(y)
        qz = yaw_to_quat(yaw)
        ps.pose.orientation.z = qz[2]; ps.pose.orientation.w = qz[3]
        goal.pose = ps
        self.nav_state = 'active'
        self._goal_handle = None
        self.last_send = self.get_clock().now().nanoseconds * 1e-9
        fut = self.nav.send_goal_async(goal)
        fut.add_done_callback(self._on_goal_resp)
        self._publish_route_marker(x, y)
        return True

    def _settled(self, grace=4.0):
        """True once the current nav goal has had real time to plan+drive, so an
        'aborted' is meaningful rather than a transient submit/preempt race."""
        return (self.get_clock().now().nanoseconds * 1e-9 - self.last_send) > grace

    def _on_goal_resp(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.nav_state = 'aborted'
            return
        self._goal_handle = gh
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, fut):
        status = fut.result().status
        self.nav_state = 'succeeded' if status == 4 else 'aborted'  # 4=SUCCEEDED

    def cancel_nav(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
        self.nav_state = 'idle'

    # ---- FSM ----
    def set_state(self, s):
        self.state = s
        self.pub_status.publish(String(data=s))
        self.get_logger().info(f'STATE -> {s}')

    def tick(self):
        self.pub_status.publish(String(
            data=f'{self.state} lane_y={self.cur_lane_y} covered={self.lanes_covered} '
                 f'pitch={self.pitch:.2f}'))
        rp = self._robot_pose()
        if rp is None:
            return

        if self.state == 'INIT':
            if self.goal_pose is None:
                return
            # start corridor = perceived lane center if available, else current y
            self.cur_lane_y = self.lane_center_y if self.lane_center_y is not None else rp[1]
            # update field extent from perceived beds x if any
            self._begin_cover(rp)
            self.set_state('COVER')

        elif self.state == 'COVER':
            far_x = self._far_x(rp)
            arrived = abs(rp[0] - far_x) < float(self.get_parameter('arrive_tol').value)
            timed_out = (self.get_clock().now().nanoseconds * 1e-9 - self.lane_start_time
                         > float(self.get_parameter('per_lane_timeout').value))
            progressed = abs(rp[0] - self.lane_start_x)
            imp = float(self.get_parameter('impassable_progress').value)
            if self.nav_state == 'succeeded' or arrived:
                self.lanes_covered += 1
                self.set_state('SHIFT')
            elif timed_out:
                if progressed >= imp:
                    self.lanes_covered += 1
                else:
                    self.skips += 1
                self.set_state('SHIFT')
            elif self.nav_state == 'aborted' and self._settled():
                if progressed >= imp:
                    self.lanes_covered += 1
                    self.set_state('SHIFT')
                elif self.cover_retries < 2:
                    # Nav2 recovery (spin/backup) may unstick the tight lane -> retry
                    self.cover_retries += 1
                    yaw = 0.0 if far_x > rp[0] else math.pi
                    self.get_logger().info(f'COVER retry {self.cover_retries} lane y={self.cur_lane_y:.2f}')
                    self.send_nav(far_x, self.cur_lane_y, yaw)
                else:
                    self.skips += 1
                    self.get_logger().warn(
                        f'lane y={self.cur_lane_y:.2f} not traversable (progress {progressed:.1f}m) -> skip')
                    self.set_state('SHIFT')

        elif self.state == 'SHIFT':
            # Cover lanes south->north UP TO AND INCLUDING the goal's lane, then GO_GOAL.
            # next corridor = nearest PERCEIVED lane centre (midpoint between two detected
            # bed rows) north of the current lane and not past the goal latitude. This
            # lands the target IN a gap (not on a bed -> instant Nav2 reject) and
            # auto-skips the narrow lane (bed spacing < min_pass_spacing). Pitch-steps as a
            # fallback until the accumulated bed map is populated.
            tol = float(self.get_parameter('lane_match_tol').value)
            goal_y = self.goal_pose.pose.position.y
            lane_centers = self._perceived_lane_centers()
            north = [c for c in lane_centers
                     if c > self.cur_lane_y + 0.4 and c <= goal_y + tol]
            if north:
                next_y = min(north)
            elif self.cur_lane_y < goal_y - tol:
                # below the goal latitude but no perceived lane yet -> step by pitch
                next_y = self.cur_lane_y + self.pitch
                if self.lane_center_y is not None and abs(self.lane_center_y - next_y) < self.pitch * 0.6:
                    next_y = self.lane_center_y
                next_y = min(next_y, goal_y + tol)
            else:
                # goal lane covered -> final precise approach to the green marker
                self.set_state('GO_GOAL'); self.goal_attempts = 0; self.nav_state = 'idle'
                return
            self.cur_lane_y = next_y
            # LATERAL move in the open margin first (stay at current x), then COVER.
            self.shift_start_time = self.get_clock().now().nanoseconds * 1e-9
            self.send_nav(rp[0], next_y, math.pi / 2 if next_y > rp[1] else -math.pi / 2)
            self.set_state('SHIFTING')

        elif self.state == 'SHIFTING':
            reached = abs(rp[1] - self.cur_lane_y) < float(self.get_parameter('arrive_tol').value)
            timed_out = (self.get_clock().now().nanoseconds * 1e-9 - self.shift_start_time > 25.0)
            if self.nav_state == 'succeeded' or reached or self.nav_state == 'aborted' or timed_out:
                self._begin_cover(rp)
                self.set_state('COVER')

        elif self.state == 'GO_GOAL':
            gx = self.goal_pose.pose.position.x
            gy = self.goal_pose.pose.position.y
            d = math.hypot(rp[0] - gx, rp[1] - gy)
            if self.nav_state == 'succeeded' or d < 0.5:
                self.set_state('DONE')
            elif self.nav_state in ('idle', 'aborted') and self._settled():
                if self.goal_attempts >= int(self.get_parameter('goal_retries').value) and d < 0.9:
                    self.set_state('DONE')
                    return
                self.goal_attempts += 1
                self.get_logger().info(f'GO_GOAL attempt {self.goal_attempts} -> ({gx:.2f},{gy:.2f})')
                self.send_nav(gx, gy, math.atan2(gy - rp[1], gx - rp[0]))

        elif self.state == 'DONE':
            self.cancel_nav()
            if not self.summary_sent:
                self._emit_summary()
                self.summary_sent = True

    def _begin_cover(self, rp):
        self.lane_start_time = self.get_clock().now().nanoseconds * 1e-9
        self.lane_start_x = rp[0]
        self.cover_retries = 0
        far_x = self._far_x(rp)
        yaw = 0.0 if far_x > rp[0] else math.pi
        self.send_nav(far_x, self.cur_lane_y, yaw)

    def _far_x(self, rp):
        # drive to the opposite field margin from current x
        return -self.margin_x if rp[0] > 0 else self.margin_x

    def _publish_route_marker(self, x, y):
        ma = MarkerArray()
        mk = Marker()
        mk.header.frame_id = self.map_frame
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = 'nav_goal'; mk.id = 0; mk.type = Marker.ARROW; mk.action = Marker.ADD
        mk.pose.position.x = float(x); mk.pose.position.y = float(y); mk.pose.position.z = 0.2
        mk.pose.orientation.w = 1.0
        mk.scale.x = 0.6; mk.scale.y = 0.15; mk.scale.z = 0.15
        mk.color.b = 1.0; mk.color.r = 1.0; mk.color.a = 1.0
        ma.markers.append(mk)
        self.pub_route.publish(ma)

    def _emit_summary(self):
        s = (f'MISSION COMPLETE | '
             f'lanes_covered={self.lanes_covered} | lanes_skipped={self.skips} | '
             f'path_length_m={self.path_len:.1f} | '
             f'goal=({self.goal_pose.pose.position.x:.2f},{self.goal_pose.pose.position.y:.2f})')
        self.get_logger().info(s)
        for _ in range(5):
            self.pub_summary.publish(String(data=s))


def main(args=None):
    rclpy.init(args=args)
    node = Mission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
