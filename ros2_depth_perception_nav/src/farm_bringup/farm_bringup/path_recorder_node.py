#!/usr/bin/env python3
"""Path / coverage visualization recorder (deliverable #4).

Saves a PNG overlaying, in the map frame:
  * the planned Nav2 path(s)  (/plan)            -- blue
  * the ACTUAL driven trajectory (TF map->base)  -- red
  * start (blue marker) and goal (green marker)
  * a swept-area hull approximating covered lane area, with a coverage estimate.

Autosaves every few seconds (so an artifact survives SIGINT) and keeps the LONGEST
planned path seen (not the near-goal stub).
"""
import math
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

import tf2_ros
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PathRecorder(Node):
    def __init__(self):
        super().__init__('path_recorder_node')
        self.declare_parameter('out_path', '/farm_nav_ws/output/path_overlay.png')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('cell', 0.25)        # m, coverage raster
        self.declare_parameter('swath', 0.67)       # m, robot width -> covered strip

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.actual = []
        self.best_plan = []
        self.start = None
        self.goal = None
        self.covered_cells = set()

        latched = QoSProfile(depth=1)
        latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        latched.reliability = QoSReliabilityPolicy.RELIABLE
        self.create_subscription(Path, '/plan', self.on_plan, 5)
        self.create_subscription(PoseStamped, '/farm/start_pose', self.on_start, latched)
        self.create_subscription(PoseStamped, '/farm/goal_pose', self.on_goal, latched)

        self.create_timer(0.25, self.sample)
        self.create_timer(5.0, self.save)
        self.get_logger().info('path_recorder_node up.')

    def on_plan(self, msg):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(pts) > len(self.best_plan):
            self.best_plan = pts

    def on_start(self, m): self.start = (m.pose.position.x, m.pose.position.y)
    def on_goal(self, m): self.goal = (m.pose.position.x, m.pose.position.y)

    def sample(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame,
                                                rclpy.time.Time())
        except Exception:
            return
        x, y = t.transform.translation.x, t.transform.translation.y
        if not self.actual or math.hypot(x - self.actual[-1][0], y - self.actual[-1][1]) > 0.05:
            self.actual.append((x, y))
            cell = float(self.get_parameter('cell').value)
            self.covered_cells.add((round(x / cell), round(y / cell)))

    def save(self):
        if not self.actual:
            return
        cell = float(self.get_parameter('cell').value)
        covered_area = len(self.covered_cells) * cell * cell
        path_len = sum(math.hypot(self.actual[i + 1][0] - self.actual[i][0],
                                  self.actual[i + 1][1] - self.actual[i][1])
                       for i in range(len(self.actual) - 1))

        fig, ax = plt.subplots(figsize=(11, 7))
        if self.best_plan:
            px, py = zip(*self.best_plan)
            ax.plot(px, py, '-', color='royalblue', lw=2, label='planned (Nav2)')
        if self.actual:
            ax, _ = self._draw_actual(ax)
        if self.start:
            ax.scatter(*self.start, c='blue', s=220, marker='*', zorder=6, label='start (blue)')
        if self.goal:
            ax.scatter(*self.goal, c='green', s=220, marker='X', zorder=6, label='goal (green)')

        ax.set_title(f'Coverage run | path={path_len:.1f} m | covered area={covered_area:.1f} m^2')
        ax.set_xlabel('x (m, map)'); ax.set_ylabel('y (m, map)')
        ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend(loc='upper right')
        out = self.get_parameter('out_path').value
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        self.get_logger().info(
            f'path overlay -> {out} (path {path_len:.1f} m, covered {covered_area:.1f} m^2)',
            throttle_duration_sec=10.0)

    def _draw_actual(self, ax):
        axx, ayy = zip(*self.actual)
        sw = float(self.get_parameter('swath').value)
        ax.plot(axx, ayy, '-', color='crimson', lw=sw * 18, alpha=0.18, solid_capstyle='round')
        ax.plot(axx, ayy, '-', color='crimson', lw=2, label='actual')
        return ax, None


def main(args=None):
    rclpy.init(args=args)
    node = PathRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
