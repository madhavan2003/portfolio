"""Lightweight per-step field checks.

Your broader pipeline already validates the LLM's mission JSON against a
schema before it ever reaches the executor (per the prompt -> LLM -> validated
mission JSON -> executor flow). This module is *not* that validator. It's a
narrow, defensive check of the fields a specific action needs right before it
runs, so a malformed step fails with a clear error instead of an
AttributeError three calls deep inside a control loop.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, Union


class StepValidationError(ValueError):
    pass


def require_fields(step: Dict[str, Any], action_name: str, required: Dict[str, Union[type, Tuple[type, ...]]]) -> None:
    """Raise StepValidationError if `step` is missing any required field or
    has the wrong type for it. `required` maps field name -> expected type."""
    for field, expected_type in required.items():
        if field not in step:
            raise StepValidationError(f"'{action_name}' step is missing required field '{field}': {step}")
        if not isinstance(step[field], expected_type):
            expected_name = getattr(expected_type, "__name__", None) or expected_type
            raise StepValidationError(
                f"'{action_name}' step field '{field}' must be {expected_name}, "
                f"got {type(step[field]).__name__}: {step}"
            )


SEARCH_AND_FOLLOW_REQUIRED = {
    "target_class": str,
    "return_after_seconds": (int, float),
}
