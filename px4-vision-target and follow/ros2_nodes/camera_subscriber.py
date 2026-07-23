"""Thin wrapper turning a ROS 2 sensor_msgs/Image topic into "give me the
latest frame as a numpy array", which is all vision.detector.Detector needs.

Expects the Gazebo camera sensor already bridged into ROS 2 as
sensor_msgs/Image, e.g. via ros_gz_bridge/ros_gz_image
(see sim/launch/search_and_follow.launch.py for the bridge setup).
"""
from __future__ import annotations

from typing import Optional

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraSubscriber:
    def __init__(self, node: Node, topic: str = "/camera"):
        self._bridge = CvBridge()
        self._latest_frame = None
        # depth=1: we only ever want the most recent frame, an old queued
        # frame is worse than useless for a real-time follow loop.
        self._sub = node.create_subscription(Image, topic, self._on_image, 1)

    def _on_image(self, msg: Image) -> None:
        self._latest_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def get_latest_frame(self) -> Optional["numpy.ndarray"]:
        """Returns None until the first frame has arrived."""
        return self._latest_frame
