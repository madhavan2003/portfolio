"""Tiny TF helper: apply a TransformStamped to a 2D point (z=0)."""
import math


def apply_xy(t, x, y):
    q = t.transform.rotation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    c, s = math.cos(yaw), math.sin(yaw)
    mx = c * x - s * y + t.transform.translation.x
    my = s * x + c * y + t.transform.translation.y
    return (mx, my)
