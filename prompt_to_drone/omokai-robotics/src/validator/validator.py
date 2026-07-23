"""
validator/validator.py
======================
The safety guardrail between the LLM planner and the executor.

This module has NO imports from llm_planner and NO imports from executor.
It can be run and tested in complete isolation — that isolation is intentional
and must be preserved: it proves the LLM cannot influence execution except
through this choke point.

Pipeline:
  raw dict (from LLM or file)
      → Pydantic parse (type errors caught here)
      → structural safety checks (Mission.__init__ / model_validator)
      → ValidationResult

On failure: returns a structured error list suitable for display or LLM retry.
On success: persists the mission JSON to missions/<mission_id>.json and returns
            an immutable Mission object.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

# Shared schema — the only cross-module import allowed here.
from src.schemas.mission import Mission

# Default output directory for validated mission files.
MISSIONS_DIR = Path(__file__).resolve().parents[2] / "missions"


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Always returned by validate().  Callers must check .success before
    accessing .mission — accessing .mission when success=False raises AttributeError
    by design (fail loudly rather than silently).

    .errors: human-readable list of every violation found.
             Multiple errors are collected in a single pass so the LLM (or a
             human operator) gets a complete picture and can fix all of them at once.
    """
    success: bool
    errors: List[str] = field(default_factory=list)
    mission: Optional[Mission] = None


# ── Public API ────────────────────────────────────────────────────────────────

def validate(
    raw: Dict[str, Any],
    *,
    missions_dir: Path = MISSIONS_DIR,
    persist: bool = True,
) -> ValidationResult:
    """
    Validate a raw JSON dict (from the LLM planner or a file) against the
    Mission schema.

    Args:
        raw:          The unvalidated dict — exactly what the LLM produced.
        missions_dir: Where to write the persisted JSON on success.
        persist:      Set to False in unit tests to skip disk writes.

    Returns:
        ValidationResult with success=True and a frozen Mission object, or
        success=False with a non-empty errors list.

    Design note:
        We deliberately do NOT auto-correct any field here.  If the LLM
        produces an altitude of 40 m we reject with a clear error and return.
        Auto-fixing would silently change the operator's intent and defeat the
        purpose of having a human-readable audit trail.
    """
    errors: List[str] = []

    # ── Phase 1: Pydantic parse ────────────────────────────────────────────
    # This catches: wrong types, missing required fields, out-of-range field
    # values (altitude, speed, loop count), malformed action 'type' values.
    try:
        mission = Mission(**raw)
    except ValidationError as exc:
        # Flatten Pydantic's nested error list into readable strings.
        for err in exc.errors():
            loc = " → ".join(str(x) for x in err["loc"]) if err["loc"] else "root"
            errors.append(f"[{loc}] {err['msg']}")
        return ValidationResult(success=False, errors=errors)
    except Exception as exc:  # noqa: BLE001 — broad catch for unexpected failures
        errors.append(f"Unexpected parse error: {exc}")
        return ValidationResult(success=False, errors=errors)

    # ── Phase 2: Persist validated mission ────────────────────────────────
    # Written to missions/<mission_id>.json for auditability.
    # If persist=False (unit tests), this block is skipped.
    if persist:
        _persist_mission(mission, missions_dir)

    return ValidationResult(success=True, mission=mission)


def validate_file(
    path: Path,
    *,
    missions_dir: Path = MISSIONS_DIR,
    persist: bool = True,
) -> ValidationResult:
    """
    Convenience wrapper: load JSON from a file and validate it.
    Used by the --mission-file CLI flag.
    """
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return ValidationResult(success=False, errors=[f"Failed to read mission file: {exc}"])
    return validate(raw, missions_dir=missions_dir, persist=persist)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _persist_mission(mission: Mission, missions_dir: Path) -> Path:
    """
    Write the validated mission to disk as canonical JSON.
    The file name includes the mission_id for easy cross-referencing with logs.
    """
    missions_dir.mkdir(parents=True, exist_ok=True)
    out_path = missions_dir / f"{mission.mission_id}.json"
    # model_dump_json handles datetime serialisation correctly.
    out_path.write_text(mission.model_dump_json(indent=2))
    return out_path
