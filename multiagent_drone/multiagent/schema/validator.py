"""
schema.validator
-----------------
The safety gate between the LLM and everything downstream.

Design: validation happens in TWO stages, because they need different inputs
and different tools.

  Stage 1 -- validate_mission(raw_json)
    Runs on the LLM's raw output, before anything else touches it.
      (a) JSON Schema structural validation against mission_schema.json --
          catches wrong types, missing required fields, unknown properties,
          bad enum values. This is what `jsonschema` is good at.
      (b) Cross-field / numeric sanity checks that JSON Schema *cannot*
          express cleanly (comparing one field to another, checking a point
          against a bounding box). This is what plain Python is good at.
    A mission that fails stage 1 never reaches the formation layer or the
    executor -- full stop.

  Stage 2 -- validate_formation_separation(...) and validate_formation_geofence(...)
    Run AFTER the formation layer has turned the mission's single task
    (one route, one area) into concrete per-drone waypoint sequences. You
    cannot check "do any two drones' paths get too close to each other," or
    "does every drone's ACTUAL flown path stay inside the geofence," before
    that computation exists -- the raw mission JSON only describes one
    shared task and a formation type, not per-drone trajectories. Stage 1's
    geofence check only sees the raw task's points (the route/area/regroup
    point); it cannot see that a wide wedge's outer wingmen fly offset well
    beyond those points. So both stage-2 checks are deliberately separate
    functions with a separate input shape, called by the executor after
    formation-splitting and before the sim spawns any drones.

Both stages return a Result object rather than raising on failure. The CLI
(run.py) is expected to check `.valid` and stop the pipeline with a clear
message rather than letting an exception surface from deep in the stack --
this mirrors how a real flight-safety gate should behave: a rejection is a
normal, expected outcome, not an exceptional one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parent / "mission_schema.json"

# Formation spacing / safety separation are both expressed in meters; this
# margin exists purely to reject degenerate polygons/routes (e.g. an
# "area_of_interest" whose three points are collinear, which is technically
# 3+ points but encloses zero area and would make the formation layer divide
# by zero downstream).
_MIN_POLYGON_AREA_M2 = 1.0
_MIN_WAYPOINT_SPACING_M = 0.1


@dataclass
class ValidationError:
    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


@dataclass
class ValidationResult:
    valid: bool
    mission: Dict[str, Any] | None = None
    errors: List[ValidationError] = field(default_factory=list)

    def raise_if_invalid(self) -> Dict[str, Any]:
        if not self.valid:
            raise MissionValidationError(self.errors)
        assert self.mission is not None
        return self.mission


class MissionValidationError(Exception):
    def __init__(self, errors: Sequence[ValidationError]):
        self.errors = list(errors)
        message = "Mission validation failed:\n" + "\n".join(f"  - {e}" for e in self.errors)
        super().__init__(message)


def _load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _polygon_area(points: List[Tuple[float, float]]) -> float:
    """Shoelace formula. Used only to reject degenerate (zero-area) polygons."""
    n = len(points)
    total = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _point_in_bbox(x: float, y: float, geofence: Dict[str, float]) -> bool:
    return (
        geofence["min_x"] <= x <= geofence["max_x"]
        and geofence["min_y"] <= y <= geofence["max_y"]
    )


def _as_xy(point: Dict[str, float]) -> Tuple[float, float]:
    return (point["x"], point["y"])


def validate_mission(raw: Union[str, Dict[str, Any]]) -> ValidationResult:
    """Stage 1 validation: JSON Schema structure, then safety/sanity checks.

    `raw` may be the raw JSON text from llm_planner.plan_mission(), or an
    already-parsed dict (useful for tests and for loading saved example
    missions from demo/example_missions/).
    """
    errors: List[ValidationError] = []

    # --- Parse -----------------------------------------------------------
    if isinstance(raw, str):
        try:
            mission = json.loads(raw)
        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                errors=[ValidationError("<root>", f"Response is not valid JSON: {e}")],
            )
    else:
        mission = raw

    # --- Stage 1a: JSON Schema structural validation ----------------------
    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    schema_errors = sorted(validator.iter_errors(mission), key=lambda e: list(e.path))
    if schema_errors:
        for e in schema_errors:
            path = ".".join(str(p) for p in e.path) or "<root>"
            errors.append(ValidationError(path, e.message))
        # If the structure itself is broken, cross-field checks below would
        # likely KeyError on missing fields -- stop here rather than produce
        # confusing secondary errors on top of the real problem.
        return ValidationResult(valid=False, errors=errors)

    # --- Stage 1b: safety and sanity checks beyond the schema -------------
    safety = mission["safety"]
    formation = mission["formation"]
    task = mission["task"]

    if mission["altitude_m"] > safety["max_altitude_m"]:
        errors.append(ValidationError(
            "altitude_m",
            f"altitude_m ({mission['altitude_m']}) exceeds safety.max_altitude_m ({safety['max_altitude_m']}).",
        ))

    if mission["speed_mps"] > safety["max_speed_mps"]:
        errors.append(ValidationError(
            "speed_mps",
            f"speed_mps ({mission['speed_mps']}) exceeds safety.max_speed_mps ({safety['max_speed_mps']}).",
        ))

    if formation["spacing_m"] < safety["min_separation_m"]:
        errors.append(ValidationError(
            "formation.spacing_m",
            f"formation.spacing_m ({formation['spacing_m']}) is less than safety.min_separation_m "
            f"({safety['min_separation_m']}) -- the formation would start out of compliance with its "
            "own separation requirement.",
        ))

    if formation["type"] == "split" and task.get("regroup_point") is None:
        errors.append(ValidationError(
            "task.regroup_point",
            "formation.type is 'split' but task.regroup_point is null. A split formation "
            "must specify where the drones rendezvous after their individual segments.",
        ))

    # Collect every point this mission references, tagged with its field
    # name, for the geofence-containment and degeneracy checks below.
    points: List[Tuple[str, Tuple[float, float]]] = []
    if task["type"] == "area_sweep":
        aoi = [_as_xy(p) for p in task["area_of_interest"]]
        points.extend((f"task.area_of_interest[{i}]", p) for i, p in enumerate(aoi))
        area = _polygon_area(aoi)
        if area < _MIN_POLYGON_AREA_M2:
            errors.append(ValidationError(
                "task.area_of_interest",
                f"area_of_interest encloses ~{area:.2f} m^2, which is degenerate "
                f"(below the {_MIN_POLYGON_AREA_M2} m^2 minimum) -- points may be collinear or duplicated.",
            ))
    else:  # patrol_route
        route = [_as_xy(p) for p in task["route"]]
        points.extend((f"task.route[{i}]", p) for i, p in enumerate(route))
        for i in range(len(route) - 1):
            (x1, y1), (x2, y2) = route[i], route[i + 1]
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if dist < _MIN_WAYPOINT_SPACING_M:
                errors.append(ValidationError(
                    f"task.route[{i}]",
                    f"route waypoints {i} and {i + 1} are {dist:.3f} m apart, which is degenerate "
                    f"(below the {_MIN_WAYPOINT_SPACING_M} m minimum) -- likely a duplicated point.",
                ))

    if task.get("regroup_point") is not None:
        points.append(("task.regroup_point", _as_xy(task["regroup_point"])))

    geofence = safety["geofence"]
    if geofence["min_x"] >= geofence["max_x"] or geofence["min_y"] >= geofence["max_y"]:
        errors.append(ValidationError(
            "safety.geofence",
            f"geofence is degenerate or inverted (min_x={geofence['min_x']}, max_x={geofence['max_x']}, "
            f"min_y={geofence['min_y']}, max_y={geofence['max_y']}).",
        ))
    else:
        for name, (x, y) in points:
            if not _point_in_bbox(x, y, geofence):
                errors.append(ValidationError(
                    name,
                    f"point ({x}, {y}) falls outside safety.geofence "
                    f"(x: [{geofence['min_x']}, {geofence['max_x']}], "
                    f"y: [{geofence['min_y']}, {geofence['max_y']}]).",
                ))

    if errors:
        return ValidationResult(valid=False, errors=errors)
    return ValidationResult(valid=True, mission=mission)


def validate_formation_separation(
    agent_waypoints: Dict[str, List[Tuple[float, float]]],
    min_separation_m: float,
) -> ValidationResult:
    """Stage 2 validation: reject overlapping/too-close per-drone paths.

    Only meaningful AFTER the formation layer has computed a concrete
    waypoint sequence per drone (see formation/formation_planner.py). Checks
    every pair of drones at every time-aligned waypoint index -- this
    assumes drones advance through their waypoint lists in lockstep, which
    holds for every formation type this system implements (wedge/line/column
    hold relative position each step; split segments are padded to equal
    length with a shared regroup waypoint at the end -- see
    formation/route_split.py for how that padding is done).

    The FINAL waypoint index is deliberately excluded from this check. For
    'split' formations, every drone's path ends at the exact same
    regroup_point by construction -- that is the intended behavior of
    "regroup at the end," not a near-miss, and flagging it would make every
    split mission unrejectable-but-always-rejected. wedge/line/column
    formations hold a constant non-zero offset at every waypoint including
    the last (spacing_m is schema-constrained to be > 0), so excluding the
    last index doesn't weaken the check for them -- there's nothing there
    that would only fail at that one index. We treat the final waypoint
    generally as "the designated mission-end/rendezvous point, where
    controlled convergence is expected," rather than special-casing on
    formation type, so this function stays agnostic to formation semantics.

    Does not do continuous-time / trajectory-interpolation collision
    checking -- only checks the waypoints themselves. That is a reasonable
    scope for a mission-planning-time safety gate; the simulator's own
    per-step distance is a separate, runtime concern (see executor/executor.py).
    """
    errors: List[ValidationError] = []
    agent_ids = list(agent_waypoints.keys())

    lengths = {aid: len(wps) for aid, wps in agent_waypoints.items()}
    if len(set(lengths.values())) > 1:
        errors.append(ValidationError(
            "agent_waypoints",
            f"per-drone waypoint lists have mismatched lengths {lengths} -- the formation layer "
            "must pad every drone's path to equal length (e.g. with a shared regroup waypoint) "
            "before this check can run.",
        ))
        return ValidationResult(valid=False, errors=errors)

    num_steps = next(iter(lengths.values())) if lengths else 0
    steps_to_check = range(num_steps - 1) if num_steps > 1 else range(num_steps)
    for step in steps_to_check:
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a, b = agent_ids[i], agent_ids[j]
                (x1, y1) = agent_waypoints[a][step]
                (x2, y2) = agent_waypoints[b][step]
                dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if dist < min_separation_m:
                    errors.append(ValidationError(
                        f"waypoint[{step}]",
                        f"drones '{a}' and '{b}' are {dist:.2f} m apart at waypoint index {step}, "
                        f"violating min_separation_m ({min_separation_m}).",
                    ))

    if errors:
        return ValidationResult(valid=False, errors=errors)
    return ValidationResult(valid=True, mission=None)


def validate_formation_geofence(
    agent_waypoints: Dict[str, List[Tuple[float, float]]],
    geofence: Dict[str, float],
) -> ValidationResult:
    """Stage 2 validation: reject a formation whose ACTUAL per-drone paths
    leave the geofence, even if the raw mission's task points (the route,
    the area of interest, the regroup point) were all comfortably inside it.

    This is necessary because stage 1's geofence check (in validate_mission)
    only sees those raw task points -- it has no way to know that a wedge or
    line formation offsets some drones sideways from them. A wide wedge
    (large spacing_m, several ranks back) can push its outer wingmen outside
    a geofence that the shared reference path never leaves. This function
    closes that gap by checking every point of every drone's own computed
    path, not just the shared task.
    """
    errors: List[ValidationError] = []
    for agent_id, waypoints in agent_waypoints.items():
        for i, (x, y) in enumerate(waypoints):
            if not _point_in_bbox(x, y, geofence):
                errors.append(ValidationError(
                    f"{agent_id}.waypoint[{i}]",
                    f"point ({x:.1f}, {y:.1f}) falls outside safety.geofence "
                    f"(x: [{geofence['min_x']}, {geofence['max_x']}], "
                    f"y: [{geofence['min_y']}, {geofence['max_y']}]).",
                ))

    if errors:
        return ValidationResult(valid=False, errors=errors)
    return ValidationResult(valid=True, mission=None)
