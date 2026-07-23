"""
formation.formation_planner
-----------------------------
Top-level entry point for the coordination layer: turns a validated mission
dict into a concrete per-drone waypoint plan.

    mission (validated dict) --> {agent_id: [(x, y), ...], ...}

Two families of formation, handled differently:
  - wedge / line / column: every agent flies a variation of ONE shared
    reference path, offset by a fixed geometric relationship (see
    formation.geometry). This is "hold formation."
  - split: the shared task is divided into N distinct segments, one per
    agent, each ending at a common regroup point. This is "divide and
    reconverge," not a fixed geometric shape (see formation.route_split).

altitude_m and speed_mps are uniform across the formation and deliberately
are NOT part of this module's output -- they're applied later by the
executor, which turns these XY waypoints into full flight commands. Keeping
the formation layer's output pure 2D geometry means it can be unit tested
completely without touching altitude, speed, or simulator concerns at all.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from . import geometry, route_split

Point = Tuple[float, float]


def agent_ids_for(num_agents: int) -> List[str]:
    """Canonical agent ID scheme used across formation, executor, and sim.
    Centralized here so nothing downstream invents its own naming."""
    return [f"drone_{i}" for i in range(num_agents)]


def _reference_path(task: dict) -> List[Point]:
    if task["type"] == "patrol_route":
        return [(p["x"], p["y"]) for p in task["route"]]
    elif task["type"] == "area_sweep":
        polygon = [(p["x"], p["y"]) for p in task["area_of_interest"]]
        return geometry.sweep_path_for_area(polygon)
    raise ValueError(f"Unknown task type: {task['type']!r}")


def plan_formation(mission: dict) -> Dict[str, List[Point]]:
    """Turns a validated mission dict (the `.mission` of a passing
    schema.validate_mission() result) into {agent_id: [(x, y), ...]} -- one
    waypoint list per drone, all lists the same length. Equal length across
    agents is a precondition of schema.validate_formation_separation and of
    the executor's lockstep sim loop, and both formation strategies below
    guarantee it (apply_formation_offsets by construction; split_path via
    its fixed resample density plus the shared regroup point appended here).
    """
    formation_type = mission["formation"]["type"]
    spacing_m = mission["formation"]["spacing_m"]
    num_agents = mission["num_agents"]
    agent_ids = agent_ids_for(num_agents)
    task = mission["task"]

    if formation_type == "split":
        regroup = task["regroup_point"]
        regroup_point: Point = (regroup["x"], regroup["y"])
        reference_path = _reference_path(task)
        segments = route_split.split_path(reference_path, num_agents)
        return {
            agent_id: segment + [regroup_point]
            for agent_id, segment in zip(agent_ids, segments)
        }

    reference_path = _reference_path(task)
    return geometry.apply_formation_offsets(reference_path, formation_type, spacing_m, agent_ids)
