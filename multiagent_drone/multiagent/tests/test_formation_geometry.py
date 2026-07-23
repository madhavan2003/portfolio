"""
Unit tests for formation/geometry.py and the wedge/line/column path of
formation/formation_planner.py.

All test paths are dead straight so the expected offsets can be computed by
hand: for a path heading due +x, "right" (a rotate-tangent-by-90-degrees
convention -- see geometry._offset_path) is -y, so a negative lateral offset
shifts a drone to +y and a positive lateral offset shifts it to -y.
"""

from __future__ import annotations

import pytest

from formation import geometry
from formation.formation_planner import plan_formation


def test_leader_flies_reference_path_unmodified():
    path = [(0.0, 0.0), (10.0, 0.0)]
    result = geometry.apply_formation_offsets(path, "line", 5, ["a", "b"])
    assert result["a"] == pytest.approx(path)


def test_line_formation_symmetric_side_by_side_offsets():
    path = [(0.0, 0.0), (10.0, 0.0)]
    result = geometry.apply_formation_offsets(path, "line", 5, ["a", "b", "c"])
    assert result["a"] == pytest.approx([(0, 0), (10, 0)])
    assert result["b"] == pytest.approx([(0, 5), (10, 5)])
    assert result["c"] == pytest.approx([(0, -5), (10, -5)])


def test_column_formation_trails_directly_behind_leader():
    path = [(0.0, 0.0), (10.0, 0.0)]
    result = geometry.apply_formation_offsets(path, "column", 5, ["a", "b", "c"])
    assert result["a"] == pytest.approx([(0, 0), (10, 0)])
    assert result["b"] == pytest.approx([(-5, 0), (5, 0)])
    assert result["c"] == pytest.approx([(-10, 0), (0, 0)])


def test_wedge_formation_forms_a_v_behind_the_leader():
    path = [(0.0, 0.0), (10.0, 0.0)]
    result = geometry.apply_formation_offsets(path, "wedge", 5, ["a", "b", "c"])
    assert result["a"] == pytest.approx([(0, 0), (10, 0)])
    assert result["b"] == pytest.approx([(-5, 5), (5, 5)])
    assert result["c"] == pytest.approx([(-5, -5), (5, -5)])


def test_all_agents_get_equal_length_paths():
    path = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    result = geometry.apply_formation_offsets(path, "wedge", 3, ["a", "b", "c"])
    lengths = {len(v) for v in result.values()}
    assert lengths == {len(path)}


def test_unknown_formation_type_raises():
    with pytest.raises(ValueError):
        geometry.apply_formation_offsets([(0.0, 0.0), (1.0, 0.0)], "diamond", 5, ["a"])


def test_split_is_not_handled_by_apply_formation_offsets():
    # 'split' is a routing strategy handled by formation.route_split via
    # formation_planner, not a geometric offset -- apply_formation_offsets
    # should refuse it rather than silently doing something wrong.
    with pytest.raises(ValueError):
        geometry.apply_formation_offsets([(0.0, 0.0), (1.0, 0.0)], "split", 5, ["a"])


def test_sweep_path_for_area_picks_the_longer_axis():
    wide = [(0.0, 0.0), (100.0, 0.0), (100.0, 10.0), (0.0, 10.0)]
    assert geometry.sweep_path_for_area(wide) == pytest.approx([(0.0, 5.0), (100.0, 5.0)])

    tall = [(0.0, 0.0), (10.0, 0.0), (10.0, 100.0), (0.0, 100.0)]
    assert geometry.sweep_path_for_area(tall) == pytest.approx([(5.0, 0.0), (5.0, 100.0)])


def test_sweep_path_for_area_rejects_degenerate_polygon():
    with pytest.raises(ValueError):
        geometry.sweep_path_for_area([(0.0, 0.0), (1.0, 0.0)])


def test_plan_formation_wedge_patrol_route():
    mission = {
        "num_agents": 3,
        "formation": {"type": "wedge", "spacing_m": 5},
        "task": {
            "type": "patrol_route",
            "route": [{"x": 0, "y": 0}, {"x": 10, "y": 0}],
            "regroup_point": None,
        },
    }
    result = plan_formation(mission)
    assert set(result.keys()) == {"drone_0", "drone_1", "drone_2"}
    assert result["drone_0"] == pytest.approx([(0, 0), (10, 0)])
    lengths = {len(v) for v in result.values()}
    assert lengths == {2}


def test_plan_formation_line_area_sweep():
    mission = {
        "num_agents": 2,
        "formation": {"type": "line", "spacing_m": 4},
        "task": {
            "type": "area_sweep",
            "area_of_interest": [
                {"x": 0, "y": 0}, {"x": 40, "y": 0}, {"x": 40, "y": 4}, {"x": 0, "y": 4},
            ],
            "regroup_point": None,
        },
    }
    result = plan_formation(mission)
    assert set(result.keys()) == {"drone_0", "drone_1"}
    # The sweep line runs along the longer (x) axis at mid-height y=2; the
    # leader (drone_0) flies it unmodified, the wingman is offset laterally.
    assert result["drone_0"] == pytest.approx([(0, 2), (40, 2)])
