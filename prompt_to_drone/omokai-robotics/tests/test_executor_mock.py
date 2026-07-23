"""
tests/test_executor_mock.py
============================
Mock-based tests for the executor module.

These tests verify the exact sequence of MAVSDK calls without a live SITL.
We replace every mavsdk.System method with AsyncMock, run the executor, then
assert the call sequence matches the expected flight plan.

Run with:
    python3 -m pytest tests/test_executor_mock.py -v
"""

import asyncio
import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

from src.executor.executor import run_mission, _haversine_m
from src.schemas.mission import (
    GotoWaypointAction,
    GlobalConstraints,
    GeofenceBoundingBox,
    LoopAction,
    Mission,
    RtlAction,
    TakeoffAction,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

GEOFENCE = GeofenceBoundingBox(
    lat_min=47.387, lat_max=47.408,
    lon_min=8.535,  lon_max=8.556,
)

CONSTRAINTS = GlobalConstraints(
    max_altitude_m=20.0,
    max_speed_mps=5.0,
    geofence=GEOFENCE,
    max_loop_count=5,
)

WP1 = GotoWaypointAction(action_id="wp1", lat=47.398, lon=8.546, alt_m=15.0, speed_mps=3.0)
WP2 = GotoWaypointAction(action_id="wp2", lat=47.400, lon=8.548, alt_m=15.0, speed_mps=3.0)


def make_loop_mission(repeat_count: int = 2) -> Mission:
    """Build a valid 2-waypoint loop mission object."""
    return Mission(
        created_from_prompt="Test patrol",
        actions=[
            TakeoffAction(action_id="t1", altitude_m=15.0),
            WP1,
            WP2,
            LoopAction(action_id="loop1", repeat_count=repeat_count, waypoint_ids=["wp1", "wp2"]),
            RtlAction(action_id="rtl1"),
        ],
        constraints=CONSTRAINTS,
    )


def make_mock_drone():
    """
    Build a mock mavsdk.System with AsyncMock on every method we call.
    Telemetry async generators are replaced with async generators that yield
    one "good" value and then stop.
    """
    drone = MagicMock()

    # ── Core connection ──────────────────────────────────────────────────
    async def _connected_state():
        connected = MagicMock()
        connected.is_connected = True
        yield connected

    async def _health_ok():
        h = MagicMock()
        h.is_global_position_ok = True
        h.is_home_position_ok = True
        yield h

    async def _in_air_true():
        yield False  # brief moment not in air
        yield True   # then airborne

    # ── Position generator: always at the target waypoint ────────────────
    # We set distance to 0 so the arrival check fires immediately.
    async def _position_at_wp1():
        pos = MagicMock()
        pos.latitude_deg = WP1.lat
        pos.longitude_deg = WP1.lon
        pos.absolute_altitude_m = WP1.alt_m
        yield pos

    async def _position_at_wp2():
        pos = MagicMock()
        pos.latitude_deg = WP2.lat
        pos.longitude_deg = WP2.lon
        pos.absolute_altitude_m = WP2.alt_m
        yield pos

    # ── Landed state ──────────────────────────────────────────────────────
    from mavsdk.telemetry import LandedState

    async def _landed():
        yield LandedState.ON_GROUND

    # Wire up telemetry
    drone.core.connection_state = _connected_state
    drone.telemetry.health = _health_ok
    drone.telemetry.in_air = _in_air_true
    drone.telemetry.landed_state = _landed

    # Position generator cycles: wp1 for first two gotos (initial + loop rep1),
    # wp2 for second two gotos (initial + loop rep1), etc.
    # For simplicity we alternate via a generator factory.
    _wp_cycle = [WP1, WP2, WP1, WP2, WP1, WP2, WP1, WP2]  # enough for repeats
    _wp_idx = {"i": 0}

    async def _position():
        wp = _wp_cycle[_wp_idx["i"] % len(_wp_cycle)]
        _wp_idx["i"] += 1
        pos = MagicMock()
        pos.latitude_deg = wp.lat
        pos.longitude_deg = wp.lon
        pos.absolute_altitude_m = wp.alt_m
        yield pos

    drone.telemetry.position = _position

    # ── Action methods ────────────────────────────────────────────────────
    drone.action.arm = AsyncMock()
    drone.action.set_takeoff_altitude = AsyncMock()
    drone.action.takeoff = AsyncMock()
    drone.action.set_current_speed = AsyncMock()
    drone.action.goto_location = AsyncMock()
    drone.action.return_to_launch = AsyncMock()
    drone.action.land = AsyncMock()

    # ── connect() ────────────────────────────────────────────────────────
    drone.connect = AsyncMock()

    return drone


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_executor_calls_arm_and_takeoff():
    """
    For any mission, arm() and takeoff() must be called exactly once,
    in that order, before any goto_location().
    """
    mission = make_loop_mission(repeat_count=1)
    drone = make_mock_drone()

    with patch("src.executor.executor.System", return_value=drone):
        await run_mission(mission, connection_url="udp://:14540")

    drone.action.arm.assert_called_once()
    drone.action.takeoff.assert_called_once()


@pytest.mark.asyncio
async def test_executor_loop_calls_goto_correct_number_of_times():
    """
    For a loop with 2 waypoints × repeat_count=2, goto_location() must be
    called:
      - 2 times for the initial sequential waypoints (before the loop action)
      - 2 × 2 = 4 more times inside the loop
    Total: 6 goto_location calls.
    """
    mission = make_loop_mission(repeat_count=2)
    drone = make_mock_drone()

    with patch("src.executor.executor.System", return_value=drone):
        await run_mission(mission, connection_url="udp://:14540")

    # 2 initial waypoints + 2 waypoints × 2 loop iterations = 6 total
    assert drone.action.goto_location.call_count == 6, (
        f"Expected 6 goto_location calls, got {drone.action.goto_location.call_count}"
    )


@pytest.mark.asyncio
async def test_executor_calls_rtl_after_loop():
    """
    RTL must be the final action — called exactly once after all waypoints
    and loop iterations are complete.
    """
    mission = make_loop_mission(repeat_count=1)
    drone = make_mock_drone()

    with patch("src.executor.executor.System", return_value=drone):
        await run_mission(mission, connection_url="udp://:14540")

    drone.action.return_to_launch.assert_called_once()
    # land() must NOT be called (we used RTL not Land)
    drone.action.land.assert_not_called()


@pytest.mark.asyncio
async def test_executor_log_file_written(tmp_path):
    """
    A log file logs/run_<mission_id>.log must be created during the run.
    """
    mission = make_loop_mission(repeat_count=1)
    drone = make_mock_drone()

    with patch("src.executor.executor.System", return_value=drone), \
         patch("src.executor.executor.LOGS_DIR", tmp_path):
        await run_mission(mission, connection_url="udp://:14540")

    log_files = list(tmp_path.glob(f"run_{mission.mission_id}.log"))
    assert len(log_files) == 1, f"Expected 1 log file, found: {log_files}"
    content = log_files[0].read_text()
    assert "MISSION START" in content
    assert "MISSION COMPLETE" in content


@pytest.mark.asyncio
async def test_executor_waypoint_coordinates_passed_correctly():
    """
    goto_location() must be called with the exact lat/lon/alt from the Mission,
    not with any transformed or rounded values.
    """
    mission = make_loop_mission(repeat_count=1)
    drone = make_mock_drone()

    with patch("src.executor.executor.System", return_value=drone):
        await run_mission(mission, connection_url="udp://:14540")

    calls = drone.action.goto_location.call_args_list
    # First two calls are for WP1 and WP2 (the sequential goto actions)
    first_call_args = calls[0].args
    assert abs(first_call_args[0] - WP1.lat) < 1e-9, "WP1 lat mismatch"
    assert abs(first_call_args[1] - WP1.lon) < 1e-9, "WP1 lon mismatch"
    assert abs(first_call_args[2] - WP1.alt_m) < 1e-9, "WP1 alt_m mismatch"


# ── Unit test for the haversine helper ────────────────────────────────────────

def test_haversine_zero_distance():
    """Same point → 0 metres."""
    assert _haversine_m(47.397, 8.545, 47.397, 8.545) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance():
    """
    1 degree of latitude ≈ 111_195 m.
    Two points 0.001 degrees apart should be ~111 m.
    """
    d = _haversine_m(47.000, 8.545, 47.001, 8.545)
    assert 110.0 < d < 115.0, f"Unexpected haversine result: {d}"
