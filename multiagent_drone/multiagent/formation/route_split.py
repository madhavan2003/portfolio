"""
formation.route_split
-----------------------
Splits a single shared path (a patrol route, or a sweep lane through an
area) into N contiguous segments, one per drone, for "split" formations.

Splitting is done by cumulative path DISTANCE, not by waypoint count -- a
route like [(0,0), (10,0), (10,100)] should not be split into "first
waypoint" / "everything else", it should be split into roughly equal-length
flying segments. Each segment is then resampled to a fixed number of points
so every drone ends up with an equal-length waypoint list, which is required
by schema.validate_formation_separation's lockstep-index assumption and is
convenient for the executor's per-step sim loop.
"""

from __future__ import annotations

import math
from typing import List, Tuple

Point = Tuple[float, float]

# Fixed resample density per split segment. This is an implementation detail
# of the formation layer (how finely we track each drone's assigned
# segment), not something the schema or the operator prompt controls.
RESAMPLE_POINTS_PER_SEGMENT = 6


def _cumulative_lengths(path: List[Point]) -> List[float]:
    lengths = [0.0]
    for i in range(1, len(path)):
        (x1, y1), (x2, y2) = path[i - 1], path[i]
        lengths.append(lengths[-1] + math.hypot(x2 - x1, y2 - y1))
    return lengths


def _point_at_distance(path: List[Point], cum_lengths: List[float], target: float) -> Point:
    """Linearly interpolates the point on `path` at cumulative distance
    `target` measured along it."""
    if target <= 0:
        return path[0]
    if target >= cum_lengths[-1]:
        return path[-1]
    for i in range(1, len(cum_lengths)):
        if cum_lengths[i] >= target:
            seg_start, seg_end = cum_lengths[i - 1], cum_lengths[i]
            seg_len = seg_end - seg_start
            t = 0.0 if seg_len < 1e-9 else (target - seg_start) / seg_len
            (x1, y1), (x2, y2) = path[i - 1], path[i]
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return path[-1]  # pragma: no cover -- unreachable given the loop above


def resample_polyline(path: List[Point], num_points: int) -> List[Point]:
    """Resamples `path` to exactly `num_points` points, evenly spaced by
    distance along the polyline, preserving its start and end points."""
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    if len(path) < 2:
        raise ValueError("path must have at least 2 points to resample")
    cum_lengths = _cumulative_lengths(path)
    total = cum_lengths[-1]
    return [
        _point_at_distance(path, cum_lengths, total * i / (num_points - 1))
        for i in range(num_points)
    ]


def split_path(path: List[Point], n: int) -> List[List[Point]]:
    """Divides `path` into `n` contiguous segments of equal length (by
    distance, not waypoint count), each resampled to
    RESAMPLE_POINTS_PER_SEGMENT points.

    Any original waypoints that fall strictly inside a segment's distance
    range are preserved before resampling, so a segment that contains an
    interior bend in the source route still tracks that bend rather than
    being flattened into a straight line between its cut points.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(path) < 2:
        raise ValueError("path must have at least 2 points to split")

    cum_lengths = _cumulative_lengths(path)
    total = cum_lengths[-1]
    if total < 1e-9:
        raise ValueError("path has zero length -- cannot split a degenerate path")
    segment_length = total / n

    segments: List[List[Point]] = []
    for i in range(n):
        start_dist = i * segment_length
        end_dist = total if i == n - 1 else (i + 1) * segment_length

        interior = [
            p for p, d in zip(path, cum_lengths)
            if start_dist < d < end_dist
        ]
        ordered = (
            [_point_at_distance(path, cum_lengths, start_dist)]
            + interior
            + [_point_at_distance(path, cum_lengths, end_dist)]
        )
        segments.append(resample_polyline(ordered, RESAMPLE_POINTS_PER_SEGMENT))

    return segments
