"""
sim.sim_runner
-----------------
Thin wrapper around gym-pybullet-drones' CtrlAviary + DSLPIDControl, driving
a multi-drone position-tracking simulation from precomputed, time-indexed
target trajectories.

Adapted from the control-loop pattern in gym-pybullet-drones' own
`gym_pybullet_drones/examples/pid.py`
(MIT License, Copyright (c) 2020 Jacopo Panerati,
https://github.com/utiasDSL/gym-pybullet-drones).
Reused from that source: the CtrlAviary + DSLPIDControl + Logger
combination, its import paths, the "step() first using the previous
action, then compute the next action from the returned obs" loop ordering,
and the target_pos/target_rpy keyword shape passed into
DSLPIDControl.computeControlFromState. Everything else here -- building
trajectories from mission JSON, sourcing initial positions from the
formation plan instead of an arbitrary circle, and the inter-drone
separation report -- is specific to this project, not copied from upstream.

This module has NO LLM calls, NO JSON parsing, and NO validation logic --
it only knows how to fly a set of already-computed target trajectories.
executor/executor.py is its only caller.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pybullet as p

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48

# How long to keep the GUI window open after the last control step, so the
# final formation state is actually visible instead of vanishing the instant
# the loop ends.
DEFAULT_GUI_HOLD_SECONDS = 4.0

# Close-range follow distance for chase-cam mode. A CF2X drone is only
# ~10cm across, so this has to be close enough to actually resolve it --
# the opposite end of the tradeoff from a whole-mission-fit shot.
CHASE_CAM_DISTANCE_M = 1.2
CHASE_CAM_YAW_DEG = -30.0
CHASE_CAM_PITCH_DEG = -20.0

# Swarm-follow camera (the default GUI view): margin multiplier applied to
# the current spread between drones, and a floor so a tight formation
# doesn't zoom in uncomfortably close.
SWARM_CAM_SPAN_MARGIN = 1.25
SWARM_CAM_MIN_DISTANCE_M = 6.0
SWARM_CAM_YAW_DEG = -30.0
SWARM_CAM_PITCH_DEG = -35.0

# A CF2X drone (~10cm) rendered at swarm-cam range (several meters to tens
# of meters, to fit the whole formation) is 1-2 raw pixels -- easy to lose
# entirely to video compression. addUserDebugPoints' pointSize is
# screen-space (constant pixels regardless of camera distance), so an
# overlay marker per drone stays visible at any zoom level the mission-wide
# view needs. Colors cycle if there are more drones than swatches.
SWARM_MARKER_SIZE_PX = 14.0
SWARM_MARKER_COLORS = [
    [1.0, 0.2, 0.2],
    [0.2, 1.0, 0.2],
    [0.3, 0.5, 1.0],
    [1.0, 0.8, 0.1],
    [0.8, 0.2, 1.0],
]

# cf2x + pure-PyBullet dynamics: no aerodynamic perturbations, no randomized
# disturbances. Given identical inputs, PyBullet's DIRECT-mode physics is
# deterministic -- this is what makes "same mission JSON in => same sim
# behavior out" true rather than aspirational.
DRONE_MODEL = DroneModel.CF2X
PHYSICS = Physics.PYB


def _update_swarm_camera(env: CtrlAviary, positions: np.ndarray) -> None:
    """Reframes the debug-GUI camera to fit every drone's CURRENT position,
    every step -- the default GUI view.

    gym-pybullet-drones' own default (cameraDistance=3, target [0, 0, 0])
    assumes drones stay within a few meters of the origin. This project's
    waypoints routinely span tens to hundreds of meters, so a camera sized
    to fit the WHOLE mission up front (a one-time fit at start) leaves each
    drone a barely-visible dot once it's mid-flight. Recomputing the
    bounding box from live positions every step instead keeps the camera
    tracking the swarm, sized only to the current spread BETWEEN drones
    (a few meters to tens of meters, not the whole route) -- so every
    drone stays close and in frame throughout, not just at t=0.
    """
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = (mins + maxs) / 2
    span = float(np.max(maxs - mins))
    distance = max(span * SWARM_CAM_SPAN_MARGIN, SWARM_CAM_MIN_DISTANCE_M)

    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=SWARM_CAM_YAW_DEG,
        cameraPitch=SWARM_CAM_PITCH_DEG,
        cameraTargetPosition=center.tolist(),
        physicsClientId=env.CLIENT,
    )


def _update_swarm_markers(env: CtrlAviary, positions: np.ndarray, marker_id: int) -> int:
    """Draws (or moves, via replaceItemUniqueId) one fixed-pixel-size debug
    point per drone at its current position, so every drone stays visibly
    distinct in the swarm-cam view no matter how far the camera has pulled
    back. Returns the marker id to pass back in on the next call."""
    colors = [SWARM_MARKER_COLORS[i % len(SWARM_MARKER_COLORS)] for i in range(len(positions))]
    return p.addUserDebugPoints(
        pointPositions=positions.tolist(),
        pointColorsRGB=colors,
        pointSize=SWARM_MARKER_SIZE_PX,
        lifeTime=0,
        replaceItemUniqueId=marker_id,
        physicsClientId=env.CLIENT,
    )


def _update_chase_camera(env: CtrlAviary, position: np.ndarray) -> None:
    """Locks the debug-GUI camera onto a single drone at close range, every
    step, so it stays framed as the drone moves -- trading mission-wide
    coverage for actually being able to see the drone itself."""
    p.resetDebugVisualizerCamera(
        cameraDistance=CHASE_CAM_DISTANCE_M,
        cameraYaw=CHASE_CAM_YAW_DEG,
        cameraPitch=CHASE_CAM_PITCH_DEG,
        cameraTargetPosition=position.tolist(),
        physicsClientId=env.CLIENT,
    )


def _align_logger_arrays_for_csv_export(logger: Logger) -> None:
    """gym_pybullet_drones.utils.Logger.save_as_csv() (v2.1.0) recomputes
    its time axis as `np.arange(0, N/freq, 1/freq)` instead of reusing the
    N samples Logger.log() actually recorded. For some N/freq combinations
    (e.g. our 640-step / 48Hz runs) that float-stepped arange yields N+1
    samples due to floating-point rounding of the stop bound, so
    save_as_csv's internal `np.vstack([t, states])` then raises a
    dimension-mismatch ValueError -- this is a library bug, not a
    logger.log() call-count issue in the control loop above (every step
    logs exactly once per drone, matching env.step() one-to-one).

    Pre-align logger's arrays to whatever length that same arange call
    will actually produce, by padding (or, in the rare undershoot case,
    trimming) with a repeat of the last logged sample. This only affects
    the CSV export below -- the .npy dump from logger.save() has already
    been written with the true, unpadded data.
    """
    freq = logger.LOGGING_FREQ_HZ
    n_logged = logger.timestamps.shape[1]
    n_time_axis = len(np.arange(0, n_logged / freq, 1 / freq))
    diff = n_time_axis - n_logged
    if diff == 0:
        return
    if diff > 0:
        logger.timestamps = np.concatenate(
            [logger.timestamps, np.repeat(logger.timestamps[:, -1:], diff, axis=1)], axis=1
        )
        logger.states = np.concatenate(
            [logger.states, np.repeat(logger.states[:, :, -1:], diff, axis=2)], axis=2
        )
        logger.controls = np.concatenate(
            [logger.controls, np.repeat(logger.controls[:, :, -1:], diff, axis=2)], axis=2
        )
    else:
        logger.timestamps = logger.timestamps[:, :diff]
        logger.states = logger.states[:, :, :diff]
        logger.controls = logger.controls[:, :, :diff]


@dataclass
class SimulationResult:
    agent_ids: List[str]
    output_folder: str
    min_separation_observed_m: float
    num_control_steps: int


def run_simulation(
    agent_ids: List[str],
    target_trajectories: Dict[str, np.ndarray],
    initial_xyzs: np.ndarray,
    mission_id: str,
    output_folder: str,
    *,
    gui: bool = False,
    simulation_freq_hz: int = DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz: int = DEFAULT_CONTROL_FREQ_HZ,
    chase_agent_id: Optional[str] = None,
) -> SimulationResult:
    """Drives CtrlAviary through every drone's target trajectory in
    lockstep: drone j's target at control step i is
    target_trajectories[agent_ids[j]][i]. Logs telemetry via
    gym-pybullet-drones' own Logger and reports the minimum observed
    inter-drone separation over the whole run (informational -- the
    planning-time separation guarantee is schema.validate_formation_separation,
    which already ran before this function was ever called; this is a
    physics-level sanity check that the PID controllers actually held it).

    target_trajectories: {agent_id: (num_control_steps, 3) array of XYZ
    targets in meters}. All arrays must have the same length.
    initial_xyzs: (num_agents, 3) array, ordered to match agent_ids.
    chase_agent_id: if set (and gui=True), the camera follows this agent up
    close instead of framing the whole mission. Must be one of agent_ids.
    """
    num_agents = len(agent_ids)
    lengths = {aid: arr.shape[0] for aid, arr in target_trajectories.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"target_trajectories must all have equal length, got {lengths}")
    num_wp = next(iter(lengths.values()))

    if chase_agent_id is not None and chase_agent_id not in agent_ids:
        raise ValueError(f"chase_agent_id {chase_agent_id!r} not in agent_ids {agent_ids}")
    chase_index = agent_ids.index(chase_agent_id) if chase_agent_id is not None else None

    os.makedirs(output_folder, exist_ok=True)

    # Level, facing +X. The mission schema has no heading concept, so every
    # drone starts and targets zero roll/pitch/yaw throughout -- a
    # deliberate simplification documented in docs/design_decisions.md.
    initial_rpys = np.zeros((num_agents, 3))

    env = CtrlAviary(
        drone_model=DRONE_MODEL,
        num_drones=num_agents,
        initial_xyzs=initial_xyzs,
        initial_rpys=initial_rpys,
        physics=PHYSICS,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )
    swarm_marker_id = -1
    if gui:
        if chase_index is not None:
            _update_chase_camera(env, initial_xyzs[chase_index])
        else:
            _update_swarm_camera(env, initial_xyzs)
            swarm_marker_id = _update_swarm_markers(env, initial_xyzs, swarm_marker_id)
    controllers = [DSLPIDControl(drone_model=DRONE_MODEL) for _ in range(num_agents)]
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_agents,
        output_folder=output_folder,
        colab=False,
    )

    action = np.zeros((num_agents, 4))
    min_separation = float("inf")
    start_time = time.time()

    try:
        for step in range(num_wp):
            obs, _reward, _terminated, _truncated, _info = env.step(action)

            # obs[j][0:3] is drone j's XYZ position -- the first three
            # entries of gym-pybullet-drones' per-drone state vector.
            positions = np.array([obs[j][0:3] for j in range(num_agents)])
            for a in range(num_agents):
                for b in range(a + 1, num_agents):
                    dist = float(np.linalg.norm(positions[a] - positions[b]))
                    if dist < min_separation:
                        min_separation = dist

            for j, agent_id in enumerate(agent_ids):
                target = target_trajectories[agent_id][step]
                action[j, :], _, _ = controllers[j].computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=obs[j],
                    target_pos=target,
                    target_rpy=initial_rpys[j],
                )
                logger.log(
                    drone=j,
                    timestamp=step / env.CTRL_FREQ,
                    state=obs[j],
                    control=np.hstack([target, initial_rpys[j], np.zeros(6)]),
                )

            if gui:
                if chase_index is not None:
                    _update_chase_camera(env, positions[chase_index])
                else:
                    _update_swarm_camera(env, positions)
                    swarm_marker_id = _update_swarm_markers(env, positions, swarm_marker_id)
                sync(step, start_time, env.CTRL_TIMESTEP)

        if gui:
            # Hold on the final formation state instead of the window
            # vanishing the instant the last control step runs.
            time.sleep(DEFAULT_GUI_HOLD_SECONDS)
    finally:
        env.close()

    logger.save()
    _align_logger_arrays_for_csv_export(logger)
    logger.save_as_csv(mission_id)

    return SimulationResult(
        agent_ids=agent_ids,
        output_folder=output_folder,
        min_separation_observed_m=min_separation,
        num_control_steps=num_wp,
    )
