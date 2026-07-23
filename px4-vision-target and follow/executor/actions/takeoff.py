"""Minimal `takeoff` action -- exists mainly to show search_and_follow is one
mode among several in the same registry, not a special-cased script.

    {"action": "takeoff", "altitude_m": 5}
"""
from __future__ import annotations

from typing import Any, Dict

from executor.actions.base import ActionHandler, ActionResult
from executor.executor import ExecutorContext
from executor.schema import require_fields


class TakeoffAction(ActionHandler):
    def execute(self, step: Dict[str, Any], context: ExecutorContext) -> ActionResult:
        require_fields(step, "takeoff", {"altitude_m": (int, float)})
        altitude_m = float(step["altitude_m"])

        if not context.vehicle.arm_and_takeoff(altitude_m):
            return ActionResult(False, f"takeoff to {altitude_m}m failed")
        return ActionResult(True, f"reached takeoff altitude {altitude_m}m")
