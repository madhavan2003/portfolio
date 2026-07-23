"""
formation.geometry
-------------------
Formation-hold geometry: given a single reference path (the "virtual
leader's" path) and a formation type, compute each drone's own path by
applying a fixed lateral/longitudinal offset at every point along the
reference path.

Offsets are expressed in a LOCAL frame relative to the path's direction of
travel at each point (lateral = perpendicular to travel, longitudinal =
along travel, negative meaning "behind"), then rotated into world XY. This
means the formation shape holds even as the reference path curves, not just
on dead-straight paths -- a wedge stays a wedge through a turn.

Reference path derivation lives in formation_planner.py, not here. This
module only knows how to offset an already-computed path; it has no opinion
about where that path comes from (a patrol route vs. a sweep line).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

Point = Tuple[float, float]


def sweep_path_for_area(polygon: List[Point]) -> List[Point]:
    """Reference sweep path for an area_sweep task: a single straight pass
    through the area's bounding-box centerline, along its longer axis.

    Scope note: this is a single-pass sweep, not full lawnmower/boustrophedon
    multi-lane coverage. A wedge/line/column formation flown along this line
    still covers a swath roughly (num_agents * spacing_m) wide, which is a
    reasonable interpretation of "sweep this area in a wedge" for a system
    this size -- clipping multiple parallel lanes to the polygon boundary is
    a natural extension point, isolated to this one function, that we chose
    not to build out for this challenge. See docs/design_decisions.md.
    """
    if len(polygon) < 3:
        raise ValueError("area_of_interest must have at least 3 points")

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width, height = max_x - min_x, max_y - min_y

    if width >= height:
        mid_y = (min_y + max_y) / 2.0
        return [(min_x, mid_y), (max_x, mid_y)]
    else:
        mid_x = (min_x + max_x) / 2.0
        return [(mid_x, min_y), (mid_x, max_y)]


def _tangents(path: List[Point]) -> List[Point]:
    """Unit tangent (direction of travel) at each point, via central
    differences with one-sided differences at the endpoints."""
    n = len(path)
    tangents: List[Point] = []
    for i in range(n):
        if n == 1:
            tangents.append((1.0, 0.0))
            continue
        prev_i = max(i - 1, 0)
        next_i = min(i + 1, n - 1)
        dx = path[next_i][0] - path[prev_i][0]
        dy = path[next_i][1] - path[prev_i][1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            # Degenerate (repeated point): reuse the previous tangent so the
            # offset path doesn't produce a NaN/zero direction.
            tangents.append(tangents[-1] if tangents else (1.0, 0.0))
        else:
            tangents.append((dx / length, dy / length))
    return tangents


def _offset_path(path: List[Point], lateral_m: float, longitudinal_m: float) -> List[Point]:
    """Shifts every point in `path` by a fixed lateral/longitudinal offset,
    expressed relative to the path's own local direction of travel at each
    point, then rotated into world XY."""
    tangents = _tangents(path)
    offset: List[Point] = []
    for (x, y), (tx, ty) in zip(path, tangents):
        # "right" is the tangent rotated -90 degrees: (tx, ty) -> (ty, -tx).
        rx, ry = ty, -tx
        offset.append((
            x + lateral_m * rx + longitudinal_m * tx,
            y + lateral_m * ry + longitudinal_m * ty,
        ))
    return offset


def _rank_offsets(formation_type: str, rank: int, spacing_m: float) -> Tuple[float, float]:
    """Returns (lateral_m, longitudinal_m) for the drone at this rank
    (0 = leader, flies the reference path unmodified) in the given
    formation type."""
    if rank == 0:
        return (0.0, 0.0)

    if formation_type == "line":
        # Side by side with the leader, alternating left/right, no
        # longitudinal offset: rank 1 -> left, 2 -> right, 3 -> further left...
        side = -1 if rank % 2 == 1 else 1
        depth = (rank + 1) // 2
        return (side * depth * spacing_m, 0.0)

    if formation_type == "column":
        # Every drone trails directly behind the previous one, nose to tail.
        return (0.0, -rank * spacing_m)

    if formation_type == "wedge":
        # Classic V: alternate left/right, each rank further out AND
        # further behind than the last -- like a flock of geese, or a
        # squadron holding a wedge.
        side = -1 if rank % 2 == 1 else 1
        depth = (rank + 1) // 2
        return (side * depth * spacing_m, -depth * spacing_m)

    raise ValueError(f"Unknown formation type: {formation_type!r}")


def apply_formation_offsets(
    reference_path: List[Point],
    formation_type: str,
    spacing_m: float,
    agent_ids: List[str],
) -> Dict[str, List[Point]]:
    """Computes each agent's own path by applying a rank-based offset to the
    shared reference path. agent_ids[0] is the virtual leader and flies the
    reference path unmodified; every other agent's offset is a function of
    its rank (its index in agent_ids) and the formation type.

    Returned lists are all the same length as reference_path -- callers
    (formation_planner, the executor's lockstep sim loop, and
    schema.validate_formation_separation) all depend on that invariant.
    """
    if formation_type not in ("wedge", "line", "column"):
        raise ValueError(
            f"apply_formation_offsets does not handle formation_type={formation_type!r} "
            "-- 'split' is handled by route_split.split_path via formation_planner, not here."
        )
    result: Dict[str, List[Point]] = {}
    for rank, agent_id in enumerate(agent_ids):
        lateral_m, longitudinal_m = _rank_offsets(formation_type, rank, spacing_m)
        result[agent_id] = _offset_path(reference_path, lateral_m, longitudinal_m)
    return result
