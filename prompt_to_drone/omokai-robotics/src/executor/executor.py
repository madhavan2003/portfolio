"""
executor/executor.py
====================
The deterministic command runner.

CONTRACT (must never be violated):
  - Accepts ONLY a validated Mission object from src.schemas.mission.
  - Never imports from llm_planner.
  - Never makes HTTP calls (MAVSDK UDP is not HTTP).
  - Same Mission in → same flight path out, every run.

The executor logs every command and every telemetry state change to a run log
file so the full decision trail is auditable after landing.

Connection:
  PX4 SITL listens on UDP port 14540 by default.
  MAVSDK-Python starts its own gRPC server on localhost:50051 and bridges to SITL.
  Command: mavsdk_server udp://:14540   (started automatically by mavsdk.System())
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# MAVSDK – simulator control only, no HTTP, no LLM
from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError, PositionNedYaw
from mavsdk.telemetry import FlightMode, LandedState, Position

from src.schemas.mission import (
    GotoWaypointAction,
    LandAction,
    LoopAction,
    Mission,
    RtlAction,
    TakeoffAction,
)

# ── Logging setup ─────────────────────────────────────────────────────────────

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"

def _setup_run_logger(mission_id: str) -> logging.Logger:
    """
    Create a per-run logger that writes to both the console and a log file.
    The file is named logs/run_<mission_id>.log for cross-reference with the
    validated mission JSON in missions/<mission_id>.json.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"executor.run.{mission_id}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(LOGS_DIR / f"run_{mission_id}.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Public API ────────────────────────────────────────────────────────────────

async def run_mission(
    mission: Mission,
    *,
    connection_url: str = "udp://:14540",
    takeoff_timeout_s: float = 30.0,
    waypoint_acceptance_radius_m: float = 2.0,
) -> None:
    """
    Execute a validated Mission end-to-end.

    Args:
        mission:                    A fully validated, frozen Mission object.
        connection_url:             MAVLink UDP URL of the SITL instance.
        takeoff_timeout_s:          Seconds to wait for climb-to-altitude confirmation.
        waypoint_acceptance_radius_m: Radius within which a waypoint is considered reached.

    Raises:
        RuntimeError: If the connection to SITL times out or a critical MAVSDK
                      call fails.  The caller (main.py) catches this and treats
                      the run as failed.
    """
    log = _setup_run_logger(mission.mission_id)
    log.info("═" * 60)
    log.info("MISSION START  mission_id=%s", mission.mission_id)
    log.info("Prompt audit : %s", mission.created_from_prompt)
    log.info("Actions      : %d", len(mission.actions))
    log.info("Connection   : %s", connection_url)
    log.info("═" * 60)

    # ── Connect ───────────────────────────────────────────────────────────────
    drone = System()
    log.info("Connecting to SITL at %s …", connection_url)
    await drone.connect(system_address=connection_url)

    # Wait for heartbeat (up to 30 s).
    log.info("Waiting for heartbeat …")
    async for state in drone.core.connection_state():
        if state.is_connected:
            log.info("Heartbeat received — SITL connected.")
            break

    # Wait until health checks pass (GPS lock, etc.).
    log.info("Waiting for global position estimate …")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            log.info("Global position OK — ready to arm.")
            break

    # ── Build waypoint lookup table ───────────────────────────────────────────
    # Used by LoopAction to resolve waypoint_ids to GotoWaypointAction objects.
    waypoint_map: Dict[str, GotoWaypointAction] = {
        a.action_id: a
        for a in mission.actions
        if isinstance(a, GotoWaypointAction)
    }

    # ── Arm ───────────────────────────────────────────────────────────────────
    log.info("CMD: arm()")
    await drone.action.arm()
    log.info("Vehicle armed.")

    # ── Walk the action list deterministically ────────────────────────────────
    for idx, action in enumerate(mission.actions):
        log.info("─── Action %d/%d: type=%s id=%s", idx + 1, len(mission.actions),
                 action.type, action.action_id)

        if isinstance(action, TakeoffAction):
            await _exec_takeoff(drone, action, log, takeoff_timeout_s)

        elif isinstance(action, GotoWaypointAction):
            await _exec_goto(drone, action, log, waypoint_acceptance_radius_m)

        elif isinstance(action, LoopAction):
            # Resolve waypoint_ids → GotoWaypointAction objects in order.
            # The validator already confirmed every ID exists, so KeyError here
            # would indicate a bug in the validator — let it propagate.
            waypoints = [waypoint_map[wid] for wid in action.waypoint_ids]
            log.info("LOOP: %d waypoints × %d repeats", len(waypoints), action.repeat_count)
            for rep in range(action.repeat_count):
                log.info("  Loop iteration %d/%d", rep + 1, action.repeat_count)
                for wp in waypoints:
                    await _exec_goto(drone, wp, log, waypoint_acceptance_radius_m)

        elif isinstance(action, RtlAction):
            await _exec_rtl(drone, log)

        elif isinstance(action, LandAction):
            await _exec_land(drone, log)

    log.info("═" * 60)
    log.info("MISSION COMPLETE  mission_id=%s", mission.mission_id)
    log.info("═" * 60)


# ── Action executors (private) ────────────────────────────────────────────────

async def _exec_takeoff(
    drone: System,
    action: TakeoffAction,
    log: logging.Logger,
    timeout_s: float,
) -> None:
    """Climb to action.altitude_m and wait until the vehicle is airborne."""
    log.info("CMD: set_takeoff_altitude(%.1f m)", action.altitude_m)
    await drone.action.set_takeoff_altitude(action.altitude_m)

    log.info("CMD: takeoff()")
    await drone.action.takeoff()

    log.debug("Waiting for in_air confirmation (timeout=%.0f s) …", timeout_s)

    async def _wait_for_in_air() -> None:
        async for in_air in drone.telemetry.in_air():
            log.debug("  in_air=%s", in_air)
            if in_air:
                log.info("Airborne — takeoff to %.1f m complete.", action.altitude_m)
                return

    try:
        await asyncio.wait_for(_wait_for_in_air(), timeout=timeout_s)
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"Takeoff timed out after {timeout_s} s — SITL may not be running."
        )


async def _exec_goto(
    drone: System,
    action: GotoWaypointAction,
    log: logging.Logger,
    acceptance_radius_m: float,
) -> None:
    """
    Fly to a geographic waypoint.

    We use goto_location() (Mission mode under the hood in PX4 SITL) which
    accepts NED position rather than raw MAVLink SET_POSITION_TARGET.
    The vehicle flies at action.speed_mps and is considered "arrived" when
    the distance to the target drops below acceptance_radius_m.
    """
    log.info(
        "CMD: goto_location(lat=%.6f, lon=%.6f, alt=%.1f m, speed=%.1f m/s)",
        action.lat, action.lon, action.alt_m, action.speed_mps,
    )
    
    # Set the speed for the upcoming goto command
    await drone.action.set_current_speed(action.speed_mps)

    await drone.action.goto_location(
        action.lat,
        action.lon,
        action.alt_m,   # AMSL — SITL's EKF converts AGL → AMSL internally
        float("nan"),   # yaw: nan = hold current heading
    )

    # Wait until within acceptance radius.
    #
    # IMPORTANT: drone.telemetry.position() is a long-lived subscription that
    # delivers every message the autopilot publishes, in order, much faster
    # than the 0.5 s cadence we want to check at. A naive
    # `async for pos in drone.telemetry.position(): ... await asyncio.sleep(0.5)`
    # only consumes one message per iteration, so a backlog piles up behind
    # the sleep and every iteration reads an increasingly stale, queued
    # message — the loop ends up replaying old telemetry in slow motion
    # while the vehicle has actually long since arrived. To avoid this we
    # run the subscription in a background task that always overwrites a
    # single-slot "latest position" holder, and poll that holder instead.
    log.debug("Waiting for arrival at waypoint %s …", action.action_id)

    latest_position: List[Position] = []

    async def _drain_position_stream() -> None:
        async for pos in drone.telemetry.position():
            if latest_position:
                latest_position[0] = pos
            else:
                latest_position.append(pos)

    drain_task = asyncio.ensure_future(_drain_position_stream())
    try:
        while not latest_position:
            await asyncio.sleep(0.05)

        while True:
            pos = latest_position[0]
            dist = _haversine_m(pos.latitude_deg, pos.longitude_deg, action.lat, action.lon)
            log.debug("  distance to waypoint: %.1f m", dist)
            if dist < acceptance_radius_m:
                log.info("Arrived at waypoint %s (dist=%.1f m).", action.action_id, dist)
                break
            await asyncio.sleep(0.5)
    finally:
        drain_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drain_task


async def _exec_rtl(drone: System, log: logging.Logger) -> None:
    """Return to launch and wait for landing."""
    log.info("CMD: return_to_launch()")
    await drone.action.return_to_launch()

    log.debug("Waiting for LANDED_STATE_ON_GROUND …")
    async for landed in drone.telemetry.landed_state():
        log.debug("  landed_state=%s", landed)
        if landed == LandedState.ON_GROUND:
            log.info("Landed (RTL complete).")
            break


async def _exec_land(drone: System, log: logging.Logger) -> None:
    """Land at the current position and wait for ground contact."""
    log.info("CMD: land()")
    await drone.action.land()

    log.debug("Waiting for LANDED_STATE_ON_GROUND …")
    async for landed in drone.telemetry.landed_state():
        log.debug("  landed_state=%s", landed)
        if landed == LandedState.ON_GROUND:
            log.info("Landed (in-place land complete).")
            break


# ── Geometry helper ───────────────────────────────────────────────────────────

import math

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Return the great-circle distance in metres between two WGS-84 points.
    Used only for waypoint acceptance radius — does NOT need to be precise
    to centimetre level, just good enough to detect arrival within ~2 m.
    """
    R = 6_371_000.0  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
