#!/usr/bin/env python3
"""Depth-geometric farm-lane perception from ONE RGB-D camera (no HSV / no color).

Pipeline per depth cloud:
  1. Read the organized point cloud (camera optical frame), downsample.
  2. Transform to base_footprint (gravity-aligned, z up) via TF.
  3. RANSAC fit the dominant near-horizontal plane  ->  the ground.
  4. Classify points by HEIGHT above that plane:
        ground   : |h| < ground_tol               (drivable surface)
        obstacle  : obstacle_min_h <= h <= obstacle_max_h   (beds + plants)
  5. Publish:
        /farm/obstacles  (PointCloud2, odom frame)  -> Nav2 obstacle layer
        /farm/lane_center (PoseStamped, odom)        -> centerline waypoint ahead
        /farm/bed_rows  (Float64MultiArray)          -> bed-row y's (odom) for the
                                                        mission's perception lane-map
        /lane/markers (MarkerArray)                  -> RViz: bed rows, centerline,
                                                        drivable lane region

Everything is geometric / depth-based. The only "color" the node ever touches is
to copy RGB into the obstacle cloud for visualization; no decision uses color.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import tf2_ros
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, Header
from visualization_msgs.msg import Marker, MarkerArray

from farm_perception.geom import (
    tf_to_matrix, transform_points, ransac_ground_plane, find_row_gaps,
)


class LanePerception(Node):
    def __init__(self):
        super().__init__('lane_perception_node')

        # ---- tuned geometric parameters (see WRITEUP hardcoded table) ----
        self.declare_parameter('ground_tol', 0.04)        # m, ground-plane band
        self.declare_parameter('obstacle_min_h', 0.07)    # m, bed top ~0.10
        self.declare_parameter('obstacle_max_h', 0.80)    # m, above plant foliage
        self.declare_parameter('stride', 3)               # cloud downsample stride
        self.declare_parameter('fwd_min', 0.45)           # m, forward window for lanes
        self.declare_parameter('fwd_max', 4.5)
        self.declare_parameter('lat_max', 3.5)            # m, lateral window
        self.declare_parameter('lookahead', 1.8)          # m, centerline waypoint dist
        self.declare_parameter('robot_clear_radius', 0.55)  # m, drop self points
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('odom_frame', 'odom')

        self.gp = self.get_parameter

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            PointCloud2, '/camera/points', self.on_cloud, qos_profile_sensor_data)

        self.pub_obs = self.create_publisher(PointCloud2, '/farm/obstacles', 5)
        self.pub_center = self.create_publisher(PoseStamped, '/farm/lane_center', 5)
        self.pub_beds = self.create_publisher(Float64MultiArray, '/farm/bed_rows', 5)
        self.pub_mk = self.create_publisher(MarkerArray, '/lane/markers', 5)

        self._last = self.get_clock().now()
        self.get_logger().info('lane_perception_node up (depth-geometric, no HSV).')

    # -------------------------------------------------------------------------
    def on_cloud(self, msg: PointCloud2):
        # throttle to ~6 Hz to keep pure-python processing real-time
        now = self.get_clock().now()
        if (now - self._last).nanoseconds < 1.5e8:
            return
        self._last = now

        stride = int(self.gp('stride').value)
        try:
            pts = point_cloud2.read_points_numpy(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception:
            arr = point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)
            pts = np.array([[p[0], p[1], p[2]] for p in arr], dtype=np.float32)
        if pts.shape[0] < 200:
            return
        pts = pts[::stride].astype(np.float64)
        # Gazebo returns NaN/inf for no-return depth pixels; drop them (else they
        # poison RANSAC and the costmap, and flood matmul warnings).
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 200:
            return

        # camera optical -> base_footprint
        try:
            t_cb = self.tf_buffer.lookup_transform(
                self.gp('base_frame').value, msg.header.frame_id,
                rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'TF cam->base unavailable: {e}', throttle_duration_sec=2.0)
            return
        M_cb = tf_to_matrix(t_cb)
        P = transform_points(pts, M_cb)   # in base frame, z up

        # crop to a sensible working volume in front of the robot
        m = (P[:, 0] > 0.0) & (P[:, 0] < 8.0) & (np.abs(P[:, 1]) < 5.0) & (P[:, 2] < 1.2)
        P = P[m]
        if P.shape[0] < 100:
            return

        # ---- RANSAC ground plane ----
        nrm, d, ground_mask = ransac_ground_plane(
            P, tol=float(self.gp('ground_tol').value))
        height = P @ nrm + d   # signed height above ground plane

        omin = float(self.gp('obstacle_min_h').value)
        omax = float(self.gp('obstacle_max_h').value)
        obs_mask = (height > omin) & (height < omax)
        obs = P[obs_mask]

        # drop points on/near the robot body (prevents costmap self-trap)
        rcr = float(self.gp('robot_clear_radius').value)
        if obs.shape[0]:
            keep = (obs[:, 0] ** 2 + obs[:, 1] ** 2) > rcr * rcr
            obs = obs[keep]
        if obs.shape[0] < 10:
            self._publish_markers([], None, None)
            return

        # base -> odom (publish obstacles & waypoint in a fixed frame)
        try:
            t_bo = self.tf_buffer.lookup_transform(
                self.gp('odom_frame').value, self.gp('base_frame').value,
                rclpy.time.Time())
        except Exception:
            return
        M_bo = tf_to_matrix(t_bo)
        obs_odom = transform_points(obs, M_bo)
        self._publish_obstacles(obs_odom)

        # ---- lane / bed-row analysis in base frame (forward window) ----
        fmin = float(self.gp('fwd_min').value)
        fmax = float(self.gp('fwd_max').value)
        lmax = float(self.gp('lat_max').value)
        win = (obs[:, 0] > fmin) & (obs[:, 0] < fmax) & (np.abs(obs[:, 1]) < lmax)
        W = obs[win]
        bed_centers, lane_centers = [], []
        if W.shape[0] > 30:
            # weight near points more (denser, more reliable)
            wts = 1.0 / (W[:, 0] + 0.5)
            bed_centers, lane_centers = find_row_gaps(W[:, 1], wts)

        # current corridor = lane center nearest y=0
        center_y = None
        if lane_centers:
            center_y = min(lane_centers, key=lambda y: abs(y))

        # publish centerline waypoint (pose ahead at lookahead distance)
        if center_y is not None:
            la = float(self.gp('lookahead').value)
            p_base = np.array([[la, center_y, 0.0]])
            p_odom = transform_points(p_base, M_bo)[0]
            ps = PoseStamped()
            ps.header.frame_id = self.gp('odom_frame').value
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = float(p_odom[0])
            ps.pose.position.y = float(p_odom[1])
            # heading along base x, expressed in odom
            yaw = np.arctan2(M_bo[1, 0], M_bo[0, 0])
            ps.pose.orientation.z = float(np.sin(yaw / 2))
            ps.pose.orientation.w = float(np.cos(yaw / 2))
            self.pub_center.publish(ps)

        # publish bed-row y's transformed into odom (for the mission lane-map).
        # Only meaningful when robot is roughly row-aligned; mission filters on heading.
        if bed_centers:
            beds_base = np.array([[2.0, y, 0.0] for y in bed_centers])
            beds_odom = transform_points(beds_base, M_bo)
            arr = Float64MultiArray()
            # layout: [yaw, x_robot, x_far,  bed_y0, bed_y1, ...] in odom
            yaw = float(np.arctan2(M_bo[1, 0], M_bo[0, 0]))
            x_far = float(transform_points(np.array([[fmax, 0, 0]]), M_bo)[0][0])
            arr.data = [yaw, float(M_bo[0, 3]), x_far] + [float(v) for v in beds_odom[:, 1]]
            self.pub_beds.publish(arr)

        self._publish_markers(bed_centers, center_y, lane_centers)

    # -------------------------------------------------------------------------
    def _publish_obstacles(self, pts_odom):
        header = Header()
        header.frame_id = self.gp('odom_frame').value
        header.stamp = self.get_clock().now().to_msg()
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        data = pts_odom.astype(np.float32)
        cloud = point_cloud2.create_cloud(header, fields, data)
        self.pub_obs.publish(cloud)

    def _publish_markers(self, bed_centers, center_y, lane_centers):
        ma = MarkerArray()
        base = self.gp('base_frame').value
        # bed rows (red lines along x)
        for i, y in enumerate(bed_centers or []):
            mk = Marker()
            mk.header.frame_id = base
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns = 'bed_rows'; mk.id = i; mk.type = Marker.LINE_STRIP; mk.action = Marker.ADD
            mk.scale.x = 0.05
            mk.color.r = 1.0; mk.color.a = 0.9
            for x in (0.4, 4.5):
                from geometry_msgs.msg import Point
                mk.points.append(Point(x=float(x), y=float(y), z=0.05))
            ma.markers.append(mk)
        # detected drivable centerline (green)
        if center_y is not None:
            mk = Marker()
            mk.header.frame_id = base
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns = 'lane_center'; mk.id = 0; mk.type = Marker.LINE_STRIP; mk.action = Marker.ADD
            mk.scale.x = 0.10
            mk.color.g = 1.0; mk.color.a = 1.0
            from geometry_msgs.msg import Point
            for x in (0.3, 5.0):
                mk.points.append(Point(x=float(x), y=float(center_y), z=0.06))
            ma.markers.append(mk)
        if ma.markers:
            self.pub_mk.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = LanePerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
