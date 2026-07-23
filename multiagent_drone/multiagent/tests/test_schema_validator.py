"""
Unit tests for schema/validator.py.

These tests never touch the LLM, the formation layer, or the simulator --
they exercise validate_mission() and validate_formation_separation()
directly against hand-built dicts, which is the whole point of keeping
validation as a pure function of JSON in / ValidationResult out.
"""

from __future__ import annotations

import copy

import pytest

from schema.validator import (
    validate_formation_geofence,
    validate_formation_separation,
    validate_mission,
)


def _base_mission() -> dict:
    """A minimal mission that passes every check. Individual tests mutate a
    deep copy of this rather than repeating the whole structure."""
    return {
        "mission_id": "test-mission-01",
        "operator_prompt": "test prompt",
        "num_agents": 2,
        "formation": {"type": "line", "spacing_m": 5},
        "task": {
            "type": "patrol_route",
            "route": [
                {"x": 0, "y": 0},
                {"x": 50, "y": 0},
                {"x": 50, "y": 50},
            ],
            "regroup_point": None,
        },
        "altitude_m": 10,
        "speed_mps": 3,
        "loop_count": 1,
        "safety": {
            "max_altitude_m": 50,
            "max_speed_mps": 10,
            "min_separation_m": 2,
            "geofence": {"min_x": -100, "max_x": 100, "min_y": -100, "max_y": 100},
        },
    }


def test_valid_mission_passes():
    result = validate_mission(_base_mission())
    assert result.valid
    assert result.mission is not None
    assert not result.errors


def test_valid_mission_passes_as_raw_json_string():
    import json

    result = validate_mission(json.dumps(_base_mission()))
    assert result.valid


def test_not_json_is_rejected():
    result = validate_mission("this is not JSON {{{")
    assert not result.valid
    assert result.errors


def test_missing_required_field_rejected():
    mission = _base_mission()
    del mission["altitude_m"]
    result = validate_mission(mission)
    assert not result.valid


def test_unknown_formation_type_rejected():
    mission = _base_mission()
    mission["formation"]["type"] = "triangle"
    result = validate_mission(mission)
    assert not result.valid


def test_unknown_task_type_rejected():
    mission = _base_mission()
    mission["task"]["type"] = "loiter"
    result = validate_mission(mission)
    assert not result.valid


def test_altitude_exceeds_max_altitude_rejected():
    mission = _base_mission()
    mission["altitude_m"] = 100  # safety.max_altitude_m is 50
    result = validate_mission(mission)
    assert not result.valid
    assert any("altitude_m" in e.field for e in result.errors)


def test_speed_exceeds_max_speed_rejected():
    mission = _base_mission()
    mission["speed_mps"] = 20  # safety.max_speed_mps is 10
    result = validate_mission(mission)
    assert not result.valid
    assert any("speed_mps" in e.field for e in result.errors)


def test_spacing_below_min_separation_rejected():
    mission = _base_mission()
    mission["formation"]["spacing_m"] = 1  # safety.min_separation_m is 2
    result = validate_mission(mission)
    assert not result.valid
    assert any("spacing_m" in e.field for e in result.errors)


def test_split_formation_without_regroup_point_rejected():
    mission = _base_mission()
    mission["formation"]["type"] = "split"
    # task.regroup_point stays None (the schema-level default) -- the
    # validator, not the schema, must catch this.
    result = validate_mission(mission)
    assert not result.valid
    assert any("regroup_point" in e.field for e in result.errors)


def test_split_formation_with_regroup_point_passes():
    mission = _base_mission()
    mission["formation"]["type"] = "split"
    mission["task"]["regroup_point"] = {"x": 25, "y": 25}
    result = validate_mission(mission)
    assert result.valid


def test_point_outside_geofence_rejected():
    mission = _base_mission()
    mission["task"]["route"].append({"x": 500, "y": 500})
    result = validate_mission(mission)
    assert not result.valid
    assert any("geofence" in e.message for e in result.errors)


def test_degenerate_route_waypoints_rejected():
    mission = _base_mission()
    mission["task"]["route"] = [
        {"x": 0, "y": 0},
        {"x": 0, "y": 0},  # duplicate of the first point -- zero-length leg
        {"x": 10, "y": 10},
    ]
    result = validate_mission(mission)
    assert not result.valid


def test_degenerate_area_of_interest_polygon_rejected():
    mission = _base_mission()
    mission["task"] = {
        "type": "area_sweep",
        # Collinear points -- zero enclosed area.
        "area_of_interest": [{"x": 0, "y": 0}, {"x": 5, "y": 0}, {"x": 10, "y": 0}],
        "regroup_point": None,
    }
    result = validate_mission(mission)
    assert not result.valid


def test_valid_area_sweep_passes():
    mission = _base_mission()
    mission["task"] = {
        "type": "area_sweep",
        "area_of_interest": [
            {"x": 0, "y": 0},
            {"x": 20, "y": 0},
            {"x": 20, "y": 20},
            {"x": 0, "y": 20},
        ],
        "regroup_point": None,
    }
    result = validate_mission(mission)
    assert result.valid


def test_task_variant_mismatch_rejected():
    """A patrol_route task carrying an area_sweep-only field must fail --
    this specifically exercises the anyOf discriminated union in the
    schema (additionalProperties: false on each variant)."""
    mission = _base_mission()
    mission["task"]["area_of_interest"] = [
        {"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1},
    ]
    result = validate_mission(mission)
    assert not result.valid


def test_inverted_geofence_rejected():
    mission = _base_mission()
    mission["safety"]["geofence"] = {"min_x": 100, "max_x": -100, "min_y": -100, "max_y": 100}
    result = validate_mission(mission)
    assert not result.valid


def test_original_mission_dict_not_mutated_on_success():
    """validate_mission should not mutate the caller's dict."""
    mission = _base_mission()
    snapshot = copy.deepcopy(mission)
    validate_mission(mission)
    assert mission == snapshot


# --- Stage 2: validate_formation_separation -----------------------------------

def test_formation_separation_passes_when_far_apart():
    waypoints = {
        "drone_0": [(0.0, 0.0), (10.0, 0.0)],
        "drone_1": [(0.0, 5.0), (10.0, 5.0)],
    }
    result = validate_formation_separation(waypoints, min_separation_m=2)
    assert result.valid


def test_formation_separation_fails_when_too_close():
    waypoints = {
        "drone_0": [(0.0, 0.0), (10.0, 0.0)],
        "drone_1": [(0.0, 1.0), (10.0, 1.0)],
    }
    result = validate_formation_separation(waypoints, min_separation_m=2)
    assert not result.valid
    assert result.errors


def test_formation_separation_allows_convergence_at_final_regroup_waypoint():
    """Split formations legitimately converge to the exact same point at
    their final (regroup) waypoint -- that must NOT be flagged as a
    separation violation, or every split mission would be unrejectable."""
    waypoints = {
        "drone_0": [(0.0, 0.0), (10.0, 0.0), (25.0, 25.0)],
        "drone_1": [(0.0, 20.0), (10.0, 20.0), (25.0, 25.0)],
    }
    result = validate_formation_separation(waypoints, min_separation_m=2)
    assert result.valid


def test_formation_separation_still_flags_violation_before_the_final_waypoint():
    """The final-waypoint exemption must not blind the check to a real
    violation earlier in the flight."""
    waypoints = {
        "drone_0": [(0.0, 0.0), (0.5, 0.0), (25.0, 25.0)],
        "drone_1": [(0.0, 20.0), (0.5, 0.0), (25.0, 25.0)],
    }
    result = validate_formation_separation(waypoints, min_separation_m=2)
    assert not result.valid


def test_formation_separation_rejects_mismatched_lengths():
    waypoints = {
        "drone_0": [(0.0, 0.0), (10.0, 0.0)],
        "drone_1": [(0.0, 5.0)],
    }
    result = validate_formation_separation(waypoints, min_separation_m=2)
    assert not result.valid


# --- Stage 2: validate_formation_geofence ---------------------------------------

_GEOFENCE = {"min_x": 0, "max_x": 100, "min_y": 0, "max_y": 100}


def test_formation_geofence_passes_when_all_points_inside():
    waypoints = {
        "drone_0": [(10.0, 10.0), (90.0, 90.0)],
        "drone_1": [(20.0, 20.0), (80.0, 80.0)],
    }
    result = validate_formation_geofence(waypoints, _GEOFENCE)
    assert result.valid


def test_formation_geofence_catches_offset_path_outside_the_fence():
    """This is exactly the wide-wedge scenario: the raw task's reference
    path is fine, but an offset wingman's path leaves the geofence -- stage
    1's check (on the raw task points only) would never catch this."""
    waypoints = {
        "drone_0": [(10.0, 10.0), (90.0, 90.0)],  # leader: inside
        "drone_1": [(-15.0, 10.0), (90.0, 90.0)],  # wingman: x=-15 is outside min_x=0
    }
    result = validate_formation_geofence(waypoints, _GEOFENCE)
    assert not result.valid
    assert any("drone_1" in e.field for e in result.errors)
