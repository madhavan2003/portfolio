#!/usr/bin/env python3
"""Plant detection + de-duplicated counting from depth geometry (no color).

Consumes the height-classified obstacle cloud (/farm/obstacles, odom frame) from
lane_perception_node. Plants are isolated from the 0.10 m bed boxes purely by
HEIGHT (foliage / upper stem sit at 0.20-0.55 m), then clustered. Each cluster is
a candidate plant; clusters are tracked across viewpoints in the fixed odom frame
and merged within an association radius so a plant seen from several angles is
counted ONCE.

  obstacle cloud --(z band)--> plant points --(grid cluster)--> centroids
                 --(nearest-track association in odom)--> persistent plant tracks
                 --(>= min_hits)--> confirmed, counted once
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node

import tf2_ros
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from farm_perception.geom import cluster_2d


class PlantTrack:
    __slots__ = ('x', 'y', 'hits', 'confirmed')

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.hits = 1
        self.confirmed = False


class PlantCounter(Node):
    def __init__(self):
        super().__init__('plant_counter_node')
        self.declare_parameter('plant_min_z', 0.20)   # m above ground: clears 0.10 beds
        self.declare_parameter('plant_max_z', 0.75)
        self.declare_parameter('cluster_cell', 0.25)  # m grid cell
        self.declare_parameter('min_cluster_pts', 6)
        self.declare_parameter('max_extent', 1.0)     # m: reject elongated bed walls
        self.declare_parameter('assoc_radius', 0.75)  # m: dedup / RTAB-drift tolerance
        self.declare_parameter('min_hits', 2)         # observations before confirming
        self.declare_parameter('max_range', 5.5)      # m: only count nearby (accurate) clusters
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.gp = self.get_parameter

        self.tracks = []
        self.count = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            PointCloud2, '/farm/obstacles', self.on_obstacles, 5)
        self.pub_count = self.create_publisher(Int32, '/farm/plant_count', 5)
        self.pub_mk = self.create_publisher(MarkerArray, '/farm/plant_markers', 5)
        # republish count+markers at 1 Hz so the topic is never silent at the goal
        self.create_timer(1.0, self.publish)
        self.get_logger().info('plant_counter_node up (depth height-cluster, dedup).')

    def on_obstacles(self, msg: PointCloud2):
        try:
            pts = point_cloud2.read_points_numpy(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception:
            return
        if pts.shape[0] < 10:
            return
        pts = pts.astype(np.float64)

        zmin = float(self.gp('plant_min_z').value)
        zmax = float(self.gp('plant_max_z').value)
        band = pts[(pts[:, 2] > zmin) & (pts[:, 2] < zmax)]
        if band.shape[0] < int(self.gp('min_cluster_pts').value):
            return

        # robot pose in odom (range-gate detections)
        rx, ry = None, None
        try:
            t = self.tf_buffer.lookup_transform(
                self.gp('odom_frame').value, self.gp('base_frame').value,
                rclpy.time.Time())
            rx = t.transform.translation.x
            ry = t.transform.translation.y
        except Exception:
            pass

        clusters = cluster_2d(
            band[:, :2],
            cell=float(self.gp('cluster_cell').value),
            min_pts=int(self.gp('min_cluster_pts').value))

        max_ext = float(self.gp('max_extent').value)
        max_rng = float(self.gp('max_range').value)
        for centroid, _cnt in clusters:
            cx, cy = float(centroid[0]), float(centroid[1])
            # compactness guard: reject bed walls (elongated clusters)
            member = band[(np.abs(band[:, 0] - cx) < max_ext) &
                          (np.abs(band[:, 1] - cy) < max_ext)]
            ext_x = member[:, 0].ptp() if member.shape[0] else 0.0
            ext_y = member[:, 1].ptp() if member.shape[0] else 0.0
            if ext_x > max_ext or ext_y > max_ext:
                continue
            if rx is not None and math.hypot(cx - rx, cy - ry) > max_rng:
                continue
            self._associate(cx, cy)

        self.publish()

    def _associate(self, cx, cy):
        ar = float(self.gp('assoc_radius').value)
        best, best_d = None, ar
        for tr in self.tracks:
            d = math.hypot(tr.x - cx, tr.y - cy)
            if d < best_d:
                best_d, best = d, tr
        if best is None:
            self.tracks.append(PlantTrack(cx, cy))
            return
        # running-average update; do not let two real plants merge beyond min spacing
        best.x = 0.8 * best.x + 0.2 * cx
        best.y = 0.8 * best.y + 0.2 * cy
        best.hits += 1
        if not best.confirmed and best.hits >= int(self.gp('min_hits').value):
            best.confirmed = True
            self.count += 1

    def publish(self):
        self.pub_count.publish(Int32(data=self.count))
        ma = MarkerArray()
        idx = 0
        for tr in self.tracks:
            if not tr.confirmed:
                continue
            mk = Marker()
            mk.header.frame_id = self.gp('odom_frame').value
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns = 'plants'; mk.id = idx; mk.type = Marker.CYLINDER; mk.action = Marker.ADD
            mk.pose.position.x = tr.x; mk.pose.position.y = tr.y; mk.pose.position.z = 0.3
            mk.pose.orientation.w = 1.0
            mk.scale.x = 0.35; mk.scale.y = 0.35; mk.scale.z = 0.6
            mk.color.g = 1.0; mk.color.b = 0.2; mk.color.a = 0.55
            ma.markers.append(mk); idx += 1
        # count label
        txt = Marker()
        txt.header.frame_id = self.gp('odom_frame').value
        txt.header.stamp = self.get_clock().now().to_msg()
        txt.ns = 'plant_count'; txt.id = 0; txt.type = Marker.TEXT_VIEW_FACING; txt.action = Marker.ADD
        txt.pose.position.z = 2.0; txt.pose.orientation.w = 1.0
        txt.scale.z = 0.6; txt.color.r = txt.color.g = txt.color.b = 1.0; txt.color.a = 1.0
        txt.text = f'plants: {self.count}'
        ma.markers.append(txt)
        self.pub_mk.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = PlantCounter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
