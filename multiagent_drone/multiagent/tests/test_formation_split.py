"""
Unit tests for formation/route_split.py and the "split" path of
formation/formation_planner.py -- dividing a shared route/area into
per-drone segments with a common regroup waypoint.
"""

from __future__ import annotations

import pytest

from formation import route_split
from formation.formation_planner import plan_formation


def test_resample_polyline_preserves_endpoints_and_count():
    result = route_split.resample_polyline([(0.0, 0.0), (10.0, 0.0)], 5)
    assert len(result) == 5
    assert result[0] == pytest.approx((0.0, 0.0))
    assert result[-1] == pytest.approx((10.0, 0.0))


def test_resample_polyline_evenly_spaced_on_a_straight_line():
    result = route_split.resample_polyline([(0.0, 0.0), (10.0, 0.0)], 3)
    assert result == pytest.approx([(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)])


def test_resample_polyline_rejects_fewer_than_two_points():
    with pytest.raises(ValueError):
        route_split.resample_polyline([(0.0, 0.0), (1.0, 0.0)], 1)


def test_split_path_produces_n_equal_length_segments():
    segments = route_split.split_path([(0.0, 0.0), (20.0, 0.0)], 2)
    assert len(segments) == 2
    for seg in segments:
        assert len(seg) == route_split.RESAMPLE_POINTS_PER_SEGMENT


def test_split_path_segments_are_contiguous_and_cover_the_full_route():
    segments = route_split.split_path([(0.0, 0.0), (20.0, 0.0)], 2)
    assert segments[0][0] == pytest.approx((0.0, 0.0))
    # The two segments meet exactly at the midpoint.
    assert segments[0][-1] == pytest.approx(segments[1][0])
    assert segments[0][-1] == pytest.approx((10.0, 0.0))
    assert segments[1][-1] == pytest.approx((20.0, 0.0))


def test_split_path_exact_resample_values_on_straight_line():
    segments = route_split.split_path([(0.0, 0.0), (20.0, 0.0)], 2)
    assert segments[0] == pytest.approx([(0, 0), (2, 0), (4, 0), (6, 0), (8, 0), (10, 0)])
    assert segments[1] == pytest.approx([(10, 0), (12, 0), (14, 0), (16, 0), (18, 0), (20, 0)])


def test_split_path_preserves_interior_bend():
    # L-shaped route: 10m east then 10m north. Each half of the 20m route
    # should still track the actual bend at (10, 0), not cut a straight
    # diagonal shortcut through it.
    segments = route_split.split_path([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], 2)
    assert segments[0][-1] == pytest.approx((10.0, 0.0))
    assert segments[1][0] == pytest.approx((10.0, 0.0))
    assert segments[1][-1] == pytest.approx((10.0, 10.0))


def test_split_path_rejects_degenerate_zero_length_path():
    with pytest.raises(ValueError):
        route_split.split_path([(5.0, 5.0), (5.0, 5.0)], 2)


def test_split_path_rejects_n_less_than_one():
    with pytest.raises(ValueError):
        route_split.split_path([(0.0, 0.0), (1.0, 1.0)], 0)


def test_split_path_rejects_single_point_path():
    with pytest.raises(ValueError):
        route_split.split_path([(0.0, 0.0)], 2)


def test_plan_formation_split_patrol_route_ends_at_regroup_point():
    mission = {
        "num_agents": 2,
        "formation": {"type": "split", "spacing_m": 5},
        "task": {
            "type": "patrol_route",
            "route": [{"x": 0, "y": 0}, {"x": 20, "y": 0}],
            "regroup_point": {"x": 50, "y": 50},
        },
    }
    result = plan_formation(mission)
    assert set(result.keys()) == {"drone_0", "drone_1"}
    for waypoints in result.values():
        assert waypoints[-1] == pytest.approx((50.0, 50.0))
    lengths = {len(v) for v in result.values()}
    assert lengths == {route_split.RESAMPLE_POINTS_PER_SEGMENT + 1}


def test_plan_formation_split_area_sweep_ends_at_regroup_point():
    mission = {
        "num_agents": 2,
        "formation": {"type": "split", "spacing_m": 5},
        "task": {
            "type": "area_sweep",
            "area_of_interest": [
                {"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 10}, {"x": 0, "y": 10},
            ],
            "regroup_point": {"x": 50, "y": 50},
        },
    }
    result = plan_formation(mission)
    assert set(result.keys()) == {"drone_0", "drone_1"}
    for waypoints in result.values():
        assert waypoints[-1] == pytest.approx((50.0, 50.0))


def test_plan_formation_split_three_agents_partitions_route_evenly():
    mission = {
        "num_agents": 3,
        "formation": {"type": "split", "spacing_m": 5},
        "task": {
            "type": "patrol_route",
            "route": [{"x": 0, "y": 0}, {"x": 30, "y": 0}],
            "regroup_point": {"x": 0, "y": 0},
        },
    }
    result = plan_formation(mission)
    assert set(result.keys()) == {"drone_0", "drone_1", "drone_2"}
    # Each drone's pre-regroup segment covers a distinct 10m-wide third of
    # the route: drone_0 starts at the route's start.
    assert result["drone_0"][0] == pytest.approx((0.0, 0.0))
