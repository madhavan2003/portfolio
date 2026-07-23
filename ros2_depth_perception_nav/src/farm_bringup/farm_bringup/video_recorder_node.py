#!/usr/bin/env python3
"""Annotated demo-video recorder (deliverable #4).

Writes an MP4 that overlays, on the live RGB feed:
  * the detected DRIVABLE LANE region (green) and obstacle pixels (orange) --
    computed per-pixel from depth height, the same geometric rule the navigation
    perception uses (NO color thresholding);
  * the NAV / mission status (/mission/status);
  * a HUD banner.

Per-pixel lane region is obtained by deprojecting the depth image with the camera
intrinsics, classifying each pixel by its height above the ground plane, exactly
mirroring lane_perception_node. This makes the "detected lane region" in the video
faithful to what the planner actually sees.
"""
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import message_filters

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String


class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder_node')
        self.declare_parameter('out_path', '/farm_nav_ws/output/run_video.mp4')
        self.declare_parameter('fps', 10.0)
        self.declare_parameter('ground_tol', 0.05)
        self.declare_parameter('obstacle_min_h', 0.07)
        self.declare_parameter('cam_height', 0.165)   # m, camera above ground
        self.declare_parameter('cam_pitch', 0.26)     # rad, down-pitch

        self.bridge = CvBridge()
        self.K = None
        self.status = 'INIT'
        self.writer = None
        self.size = None
        self._finalized = False
        self.fps = float(self.get_parameter('fps').value)
        self._last_t = self.get_clock().now()

        self.create_subscription(String, '/mission/status', self.on_status, 5)
        self.create_subscription(String, '/mission/summary', self.on_summary, 5)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, 5)

        color = message_filters.Subscriber(self, Image, '/camera/image_raw',
                                           qos_profile=qos_profile_sensor_data)
        depth = message_filters.Subscriber(self, Image, '/camera/depth/image_raw',
                                           qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer([color, depth], 10, 0.1)
        self.sync.registerCallback(self.on_pair)
        self.get_logger().info('video_recorder_node up.')

    def on_status(self, m): self.status = m.data

    def on_summary(self, m):
        # Mission is done -- finalize the MP4 (write the moov atom) NOW, so the
        # file is valid on disk regardless of how/when the container is later
        # stopped (a hard `docker kill`/`docker rm -f` gives this process no
        # chance to run its shutdown handler, which otherwise corrupts the
        # video: frames are on disk but the header is never written).
        if self._finalized:
            return
        self._finalized = True
        self.get_logger().info(f'mission summary received, finalizing video: {m.data}')
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.get_logger().info('video finalized early (mission complete).')

    def on_info(self, m): self.K = np.array(m.k).reshape(3, 3)

    def on_pair(self, color_msg, depth_msg):
        now = self.get_clock().now()
        if (now - self._last_t).nanoseconds < 1e9 / self.fps:
            return
        self._last_t = now
        try:
            rgb = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough').astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}', throttle_duration_sec=5.0)
            return
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        overlay = self._classify_overlay(rgb, depth)
        frame = self._hud(overlay)
        self._write(frame)

    def _classify_overlay(self, rgb, depth):
        h, w = depth.shape
        out = rgb.copy()
        if self.K is None:
            return out
        # deproject (optical frame): X right, Y down, Z forward
        ys, xs = np.mgrid[0:h:4, 0:w:4]
        z = depth[::4, ::4].astype(np.float32)
        valid = np.isfinite(z) & (z > 0.1) & (z < 8.0)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        Y = (ys - cy) * z / fy
        Z = z
        # optical -> ground height: account for camera pitch + mount height.
        p = float(self.get_parameter('cam_pitch').value)
        cam_h = float(self.get_parameter('cam_height').value)
        height = cam_h - (Y * np.cos(p) - Z * np.sin(p))   # [optical Y is down]
        gt = float(self.get_parameter('ground_tol').value)
        omin = float(self.get_parameter('obstacle_min_h').value)

        ground = valid & (np.abs(height) < gt)
        obstacle = valid & (height >= omin) & (height < 1.0)

        # vectorized tint at downsampled resolution, then upsample (no per-pixel loop)
        small = np.zeros((ground.shape[0], ground.shape[1], 3), dtype=np.uint8)
        small[ground] = (0, 180, 0)       # drivable lane -> green
        small[obstacle] = (0, 140, 255)   # obstacle -> orange
        tint = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        out = cv2.addWeighted(out, 1.0, tint, 0.45, 0)
        return out

    def _hud(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(frame, f'NAV: {self.status[:56]}', (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.rectangle(frame, (0, h - 26), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, 'lane=green  obstacle=orange  (depth-geometric, no HSV)',
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def _write(self, frame):
        if self._finalized:
            return
        if self.writer is None:
            out = self.get_parameter('out_path').value
            os.makedirs(os.path.dirname(out), exist_ok=True)
            self.size = (frame.shape[1], frame.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(out, fourcc, self.fps, self.size)
            self.get_logger().info(f'recording video -> {out}')
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)

    def destroy_node(self):
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info('video saved.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
