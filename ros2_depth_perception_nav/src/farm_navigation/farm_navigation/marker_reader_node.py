#!/usr/bin/env python3
"""Read the blue (start) and green (goal) markers from the world AT RUNTIME.

Satisfies the hard constraint: start/goal are NOT hardcoded in the navigation
logic. We query Gazebo's /gazebo/get_entity_state service for the two reference
models' `footprint_link` (querying the bare model name returns the origin, so the
link must be named explicitly) and republish them as latched PoseStamped in the
`map` frame. Because map is world-aligned (see static_map_odom_node), the world
pose is the map pose directly.

Published (transient_local / latched):
  /farm/start_pose (geometry_msgs/PoseStamped, map)
  /farm/goal_pose  (geometry_msgs/PoseStamped, map)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import GetEntityState


class MarkerReader(Node):
    def __init__(self):
        super().__init__('marker_reader_node')
        self.declare_parameter('start_entity', 'husky_footprint_start_reference::footprint_link')
        self.declare_parameter('goal_entity', 'husky_footprint_end_reference::footprint_link')
        self.declare_parameter('map_frame', 'map')

        latched = QoSProfile(depth=1)
        latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        latched.reliability = QoSReliabilityPolicy.RELIABLE

        self.pub_start = self.create_publisher(PoseStamped, '/farm/start_pose', latched)
        self.pub_goal = self.create_publisher(PoseStamped, '/farm/goal_pose', latched)

        self.cli = self.create_client(GetEntityState, '/gazebo/get_entity_state')
        self.start_pose = None
        self.goal_pose = None
        self.timer = self.create_timer(1.0, self.tick)

    def tick(self):
        if not self.cli.service_is_ready():
            self.get_logger().info('waiting for /gazebo/get_entity_state...', throttle_duration_sec=5.0)
            return
        if self.start_pose is None:
            self._request(self.get_parameter('start_entity').value, 'start')
        if self.goal_pose is None:
            self._request(self.get_parameter('goal_entity').value, 'goal')
        # keep republishing latched (cheap) so late subscribers always get them
        if self.start_pose:
            self.pub_start.publish(self.start_pose)
        if self.goal_pose:
            self.pub_goal.publish(self.goal_pose)
        if self.start_pose and self.goal_pose:
            self.get_logger().info(
                'start=(%.2f,%.2f) goal=(%.2f,%.2f) read from markers.' % (
                    self.start_pose.pose.position.x, self.start_pose.pose.position.y,
                    self.goal_pose.pose.position.x, self.goal_pose.pose.position.y),
                throttle_duration_sec=10.0)

    def _request(self, entity, which):
        req = GetEntityState.Request()
        req.name = entity
        req.reference_frame = 'world'
        fut = self.cli.call_async(req)
        fut.add_done_callback(lambda f, w=which: self._done(f, w))

    def _done(self, fut, which):
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().warn(f'{which} marker query failed: {e}')
            return
        if not res.success:
            self.get_logger().warn(f'{which} marker not found yet', throttle_duration_sec=5.0)
            return
        ps = PoseStamped()
        ps.header.frame_id = self.get_parameter('map_frame').value
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = res.state.pose
        if which == 'start':
            self.start_pose = ps
        else:
            self.goal_pose = ps


def main(args=None):
    rclpy.init(args=args)
    node = MarkerReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
