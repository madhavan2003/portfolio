"""
executor.executor
--------------------
The deterministic executor: reads a VALIDATED mission dict plus the
formation layer's per-drone XY waypoint lists, turns them into 3D,
time-indexed target trajectories, runs the second (post-formation) safety
check, and drives the simulation.

This module never calls the LLM and never re-derives anything an earlier
pipeline stage already decided -- it is a pure function of
(validated mission, formation waypoints) -> (sim inputs), plus the actual
sim.run_simulation() call. Same mission JSON + same formation output =>
same trajectories => same simulation, every time. No prompt ever reaches
this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from formation.formation_planner import agent_ids_for, plan_formation
from formation.route_split import resample_polyline
from schema.validator import (
    MissionValidationError,
    ValidationError,
    ValidationResult,
    validate_formation_geofence,
    validate_formation_separation,
)
from sim.sim_runner import SimulationResult, run_simulation

Point = Tuple[float, float]

# A trajectory shorter than this can't be flown as a control-step sequence;
# guards the degenerate case of a zero-length or single-point mission path.
MIN_CONTROL_STEPS = 2


@dataclass
class ExecutionResult:
    formation_waypoints: Dict[str, List[Point]]
    separation_check: ValidationResult
    geofence_check: ValidationResult
    simulation: SimulationResult


def _path_length(path: List[Point]) -> float:
    """math.dist works on any-dimensional tuples, but every path here is 2D
    (altitude is constant per-agent and added separately below, so it never
    contributes to path length)."""
    total = 0.0
    for i in range(1, len(path)):
        total += math.dist(path[i - 1], path[i])
    return total


def build_target_trajectories(
    mission: Dict,
    formation_waypoints: Dict[str, List[Point]],
    control_freq_hz: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """Turns 2D formation waypoints + mission altitude/speed/loop_count into
    a per-agent (num_control_steps, 3) array of XYZ targets, one per control
    step, plus each agent's initial XYZ position.

    Every agent's trajectory is resampled to the SAME number of control
    steps, driven by the slowest (longest single-lap-duration) agent's
    travel time at speed_mps. This is what makes drones on the inside of a
    formation turn, or on a shorter split segment, move slower and stay
    synchronized with the rest of the squad -- the same thing real formation
    flying does. loop_count repeats are done by tiling the single-lap
    trajectory, not by re-resampling a loop_count-times-longer path, so a
    looping mission's per-lap timing matches its single-pass timing exactly.
    """
    altitude_m = mission["altitude_m"]
    speed_mps = mission["speed_mps"]
    loop_count = mission["loop_count"]
    agent_ids = agent_ids_for(mission["num_agents"])

    lap_durations = {
        agent_id: _path_length(formation_waypoints[agent_id]) / speed_mps
        for agent_id in agent_ids
    }
    max_lap_duration = max(lap_durations.values())
    num_steps_per_lap = max(MIN_CONTROL_STEPS, round(max_lap_duration * control_freq_hz))

    trajectories: Dict[str, np.ndarray] = {}
    for agent_id in agent_ids:
        single_lap_xy = resample_polyline(formation_waypoints[agent_id], num_steps_per_lap)
        single_lap_xyz = np.array([[x, y, altitude_m] for (x, y) in single_lap_xy])
        trajectories[agent_id] = np.tile(single_lap_xyz, (loop_count, 1))

    initial_xyzs = np.array([trajectories[agent_id][0] for agent_id in agent_ids])
    return trajectories, initial_xyzs, agent_ids


def execute_mission(
    mission: Dict,
    *,
    gui: bool = False,
    output_folder: str = "logs",
    control_freq_hz: int = 48,
    simulation_freq_hz: int = 240,
    chase_agent_id: Optional[str] = None,
) -> ExecutionResult:
    """The full deterministic executor pipeline for an already-validated
    mission (i.e. mission is schema.validate_mission(raw).mission -- this
    function assumes stage-1 validation already passed, it does not repeat
    it):

      1. formation.plan_formation() -> per-drone XY waypoints.
      2. schema.validate_formation_separation() and
         schema.validate_formation_geofence() -> stage-2 validation,
         rejecting overlapping/too-close per-drone paths and paths that
         leave the geofence, BEFORE any sim time is spent. These are the
         second validation stage described in schema/validator.py's module
         docstring -- neither can run any earlier because they need
         concrete per-drone paths, which only exist after step 1. Both run
         even if one already failed, so a rejection reports every problem
         at once rather than one round-trip per issue.
      3. Build 3D, time-indexed target trajectories (build_target_trajectories).
      4. sim.run_simulation() -> drive CtrlAviary, log telemetry.

    Raises MissionValidationError if step 2 fails. Callers (run.py) should
    catch this and stop the pipeline with a clear message, exactly like a
    stage-1 validation failure -- the executor never proceeds to the
    simulator on a rejected plan.
    """
    formation_waypoints = plan_formation(mission)

    separation_check = validate_formation_separation(
        formation_waypoints, mission["safety"]["min_separation_m"]
    )
    geofence_check = validate_formation_geofence(
        formation_waypoints, mission["safety"]["geofence"]
    )
    stage2_errors: List[ValidationError] = list(separation_check.errors) + list(geofence_check.errors)
    if stage2_errors:
        raise MissionValidationError(stage2_errors)

    trajectories, initial_xyzs, agent_ids = build_target_trajectories(
        mission, formation_waypoints, control_freq_hz
    )

    simulation = run_simulation(
        agent_ids,
        trajectories,
        initial_xyzs,
        mission["mission_id"],
        output_folder,
        gui=gui,
        simulation_freq_hz=simulation_freq_hz,
        control_freq_hz=control_freq_hz,
        chase_agent_id=chase_agent_id,
    )

    return ExecutionResult(
        formation_waypoints=formation_waypoints,
        separation_check=separation_check,
        geofence_check=geofence_check,
        simulation=simulation,
    )
