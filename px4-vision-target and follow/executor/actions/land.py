"""Minimal `land` action -- see takeoff.py for why this exists.

    {"action": "land"}
"""
from __future__ import annotations

from typing import Any, Dict

from executor.actions.base import ActionHandler, ActionResult
from executor.executor import ExecutorContext


class LandAction(ActionHandler):
    def execute(self, step: Dict[str, Any], context: ExecutorContext) -> ActionResult:
        if not context.vehicle.land():
            return ActionResult(False, "land command failed")
        return ActionResult(True, "landed")
