"""Action handler interface -- the plug-in point mission JSON steps dispatch to.

Adding a new mission "mode" (like `search_and_follow`) means writing one class
that implements `ActionHandler` and registering it under its `action` string
in `executor.registry.ACTION_REGISTRY`. Nothing about the executor's dispatch
loop changes. This is deliberately a small interface (one method) so it's
obvious what every action must provide and nothing more.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional


@dataclasses.dataclass
class ActionResult:
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None


class ActionHandler:
    """Base class for a mission step handler.

    `execute` receives the raw step dict (already schema-checked by the
    handler itself via executor.schema.require_fields) and an ExecutorContext
    carrying whatever the handler needs to actually act (vehicle interface,
    camera source, vision components, ROS node for logging/timing). Handlers
    are constructed fresh per step, so they may hold per-step state on self.
    """

    def execute(self, step: Dict[str, Any], context: "ExecutorContext") -> ActionResult:
        raise NotImplementedError
