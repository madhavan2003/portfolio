"""
tests/test_validator.py
=======================
Unit tests for the validator module.

These tests run with NO simulator and NO LLM — the validator module is
fully isolated by design.  Run with:

    cd omokai-robotics
    python -m pytest tests/test_validator.py -v

All tests use persist=False so they do not touch the missions/ directory.
"""

import pytest

from src.validator.validator import validate

# ── Shared helpers ────────────────────────────────────────────────────────────

# A minimal valid geofence centred on the PX4 SITL default origin
# (Zurich, approx 47.397742 N, 8.545594 E) with a 0.01-degree (~1 km) margin.
GEOFENCE = {
    "lat_min": 47.387,
    "lat_max": 47.408,
    "lon_min": 8.535,
    "lon_max": 8.556,
}

CONSTRAINTS = {
    "max_altitude_m": 20.0,
    "max_speed_mps": 5.0,
    "geofence": GEOFENCE,
    "max_loop_count": 5,
}

WP1 = {
    "action_id": "wp1",
    "type": "goto_waypoint",
    "lat": 47.398,
    "lon": 8.546,
    "alt_m": 15.0,
    "speed_mps": 3.0,
}

WP2 = {
    "action_id": "wp2",
    "type": "goto_waypoint",
    "lat": 47.400,
    "lon": 8.548,
    "alt_m": 15.0,
    "speed_mps": 3.0,
}

WP3 = {
    "action_id": "wp3",
    "type": "goto_waypoint",
    "lat": 47.400,
    "lon": 8.542,
    "alt_m": 15.0,
    "speed_mps": 3.0,
}

WP4 = {
    "action_id": "wp4",
    "type": "goto_waypoint",
    "lat": 47.396,
    "lon": 8.542,
    "alt_m": 15.0,
    "speed_mps": 3.0,
}


def build_patrol_mission(repeat_count: int = 2) -> dict:
    """Return a fully valid perimeter patrol mission dict."""
    return {
        "created_from_prompt": "Patrol the perimeter loop twice at 15 metres",
        "actions": [
            {"action_id": "t1", "type": "takeoff", "altitude_m": 15.0},
            WP1,
            WP2,
            WP3,
            WP4,
            {
                "action_id": "loop1",
                "type": "loop",
                "repeat_count": repeat_count,
                "waypoint_ids": ["wp1", "wp2", "wp3", "wp4"],
            },
            {"action_id": "rtl1", "type": "rtl"},
        ],
        "constraints": CONSTRAINTS,
    }


# ── Test 1: Valid patrol mission passes ───────────────────────────────────────

def test_valid_patrol_mission_passes():
    """A well-formed perimeter patrol mission must be accepted."""
    result = validate(build_patrol_mission(), persist=False)
    assert result.success, f"Expected success, got errors: {result.errors}"
    assert result.mission is not None
    assert result.mission.created_from_prompt == "Patrol the perimeter loop twice at 15 metres"
    assert len(result.errors) == 0


# ── Test 2: Over-altitude rejected ───────────────────────────────────────────

def test_over_altitude_rejected():
    """
    A waypoint altitude of 50 m must be rejected.
    The schema hard ceiling is 30 m; the mission constraint ceiling is 20 m.
    Both checks must fire.
    """
    mission = build_patrol_mission()
    # Corrupt the first waypoint's altitude to exceed both limits.
    mission["actions"][1] = {**WP1, "alt_m": 50.0}
    result = validate(mission, persist=False)
    assert not result.success
    assert len(result.errors) > 0
    # At least one error must mention the altitude.
    assert any("altitude" in e.lower() or "alt_m" in e.lower() for e in result.errors), (
        f"Expected an altitude-related error, got: {result.errors}"
    )


# ── Test 3: Unknown action type rejected ──────────────────────────────────────

def test_unknown_action_type_rejected():
    """
    An action with type='hover' (not in the schema) must be rejected
    at the Pydantic discriminator level.
    """
    mission = build_patrol_mission()
    mission["actions"].insert(2, {
        "action_id": "bad1",
        "type": "hover",       # <-- not a valid ActionType
        "duration_s": 30,
    })
    result = validate(mission, persist=False)
    assert not result.success
    assert len(result.errors) > 0


# ── Test 4: Geofence violation rejected ───────────────────────────────────────

def test_geofence_violation_rejected():
    """
    A waypoint outside the geofence bounding box must be rejected.
    The geofence is [47.387–47.408 N, 8.535–8.556 E].
    We place a waypoint at 47.450 N — clearly outside.
    """
    mission = build_patrol_mission()
    outside_wp = {
        "action_id": "wp_outside",
        "type": "goto_waypoint",
        "lat": 47.450,   # North of lat_max=47.408
        "lon": 8.546,
        "alt_m": 15.0,
        "speed_mps": 3.0,
    }
    # Replace wp2 with the out-of-bounds waypoint.
    mission["actions"][2] = outside_wp
    # Also update the loop to reference the new id.
    mission["actions"][5]["waypoint_ids"] = ["wp1", "wp_outside", "wp3", "wp4"]
    result = validate(mission, persist=False)
    assert not result.success
    assert any("geofence" in e.lower() or "outside" in e.lower() for e in result.errors), (
        f"Expected a geofence-related error, got: {result.errors}"
    )


# ── Test 5: Excessive loop count rejected ─────────────────────────────────────

def test_excessive_loop_count_rejected():
    """
    A loop with repeat_count=15 must be rejected.
    The schema field validator caps at 10; the global constraint caps at 5.
    Both limits must be enforced.
    """
    mission = build_patrol_mission(repeat_count=15)
    result = validate(mission, persist=False)
    assert not result.success
    assert any(
        "repeat_count" in e.lower() or "loop" in e.lower()
        for e in result.errors
    ), f"Expected a loop-count error, got: {result.errors}"


# ── Test 6: Missing required field rejected ───────────────────────────────────

def test_missing_required_field_rejected():
    """
    A mission missing created_from_prompt must be rejected — the audit trail
    field is non-optional.
    """
    mission = build_patrol_mission()
    del mission["created_from_prompt"]
    result = validate(mission, persist=False)
    assert not result.success


# ── Test 7: Loop referencing nonexistent waypoint_id rejected ─────────────────

def test_loop_with_invalid_waypoint_id_rejected():
    """
    A LoopAction referencing a waypoint_id that does not exist in the action
    list must be rejected.
    """
    mission = build_patrol_mission()
    mission["actions"][5]["waypoint_ids"] = ["wp1", "wp_ghost"]  # wp_ghost doesn't exist
    result = validate(mission, persist=False)
    assert not result.success
    assert any("wp_ghost" in e for e in result.errors), (
        f"Expected an error mentioning 'wp_ghost', got: {result.errors}"
    )
