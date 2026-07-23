"""Unit tests for the deterministic dispatch loop itself (not any specific
action). Uses fake ActionHandlers so this needs no ROS/PX4/YOLO at all --
it's purely checking that MissionExecutor calls the right handler for each
step, in order, and stops on the first failure.

Run with:  python3 -m pytest tests/test_executor.py -v
"""
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor.actions.base import ActionHandler, ActionResult
from executor.executor import ExecutorContext, MissionExecutor


class RecordingAction(ActionHandler):
    calls = []

    def execute(self, step: Dict[str, Any], context: ExecutorContext) -> ActionResult:
        RecordingAction.calls.append(step["action"])
        return ActionResult(success=True, message="ok")


class FailingAction(ActionHandler):
    def execute(self, step, context) -> ActionResult:
        return ActionResult(success=False, message="deliberately failed")


class NeverCalledAction(ActionHandler):
    def execute(self, step, context) -> ActionResult:
        raise AssertionError("should never run: mission should have aborted before this step")


def setup_function():
    RecordingAction.calls = []


def test_runs_steps_in_order():
    mission = [{"action": "a"}, {"action": "b"}]
    registry = {"a": RecordingAction, "b": RecordingAction}
    context = ExecutorContext()

    results = MissionExecutor(mission, registry, context).run()

    assert RecordingAction.calls == ["a", "b"]
    assert all(r.success for r in results)


def test_unknown_action_raises():
    mission = [{"action": "does_not_exist"}]
    registry = {"a": RecordingAction}
    context = ExecutorContext()

    try:
        MissionExecutor(mission, registry, context).run()
        assert False, "expected ValueError for unknown action"
    except ValueError as e:
        assert "does_not_exist" in str(e)


def test_aborts_mission_after_first_failure():
    mission = [{"action": "fail"}, {"action": "never"}]
    registry = {"fail": FailingAction, "never": NeverCalledAction}
    context = ExecutorContext()

    results = MissionExecutor(mission, registry, context).run()

    assert len(results) == 1
    assert results[0].success is False
