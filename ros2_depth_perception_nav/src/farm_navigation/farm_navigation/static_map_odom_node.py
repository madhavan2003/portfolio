#!/usr/bin/env python3
"""Static IDENTITY map->odom transform.

gazebo_ros_planar_move publishes odometry as the model's ABSOLUTE world pose, so the
`odom` frame is already world-aligned with its origin at the world origin (verified:
plant/obstacle clouds transformed through odom land at true world coordinates). The
`map` frame therefore equals `odom` equals the world, and map->odom is IDENTITY.

(An earlier version published map->odom = the spawn pose, which DOUBLE-transformed the
nav frame -- the robot's trajectory then started at the world origin instead of the
blue marker. Identity is correct.)

We still publish our own static map->odom (rather than letting RTAB-Map own it) to
DECOUPLE visual SLAM from the navigation TF tree: RTAB-Map runs for mapping / RViz
(publish_tf:=false, map_frame_id:=map_slam), while Nav2 localizes on the drift-free
planar_move odometry. RTAB's map->odom correction periodically jumps ~0.8 m on the
low-texture farm floor, which would teleport the path and topple the robot in the
~1 m lanes -- so it must not drive the nav frame.
"""
import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped


class StaticMapOdom(Node):
    def __init__(self):
        super().__init__('static_map_odom_node')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')

        self.br = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter('map_frame').value
        t.child_frame_id = self.get_parameter('odom_frame').value
        # identity: odom is already world-aligned (planar_move reports world pose)
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)
        self.get_logger().info('Published IDENTITY static map->odom (map=odom=world).')


def main(args=None):
    rclpy.init(args=args)
    node = StaticMapOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
