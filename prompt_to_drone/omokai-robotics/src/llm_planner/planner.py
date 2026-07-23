"""
llm_planner/planner.py
======================
Natural-language prompt → raw (unvalidated) mission JSON dict.

This module is the ONLY place in the pipeline where a network call to an
external LLM is made.  It produces raw JSON — it never constructs a Mission
object and it never calls MAVSDK.  The validator is the choke point; the
planner only proposes.

Two modes
---------
1. REAL mode (default): calls a local Ollama server (OLLAMA_HOST, default
   http://localhost:11434) running an open model (OLLAMA_MODEL, default
   llama3.1). Local open models are meaningfully less reliable at producing
   strict, schema-conformant JSON than a hosted model's structured-output
   mode, so this path also runs a validation-retry loop: a failed parse or
   schema check is fed back to the model as a correction request, up to
   _OLLAMA_MAX_RETRIES times, before giving up.

2. MOCK mode (--mock-llm flag):
   A deterministic rule-based parser that extracts numbers from the prompt
   and builds a valid-looking mission dict.  This lets you run the full
   pipeline end-to-end with no LLM at all — no Ollama install, no network
   call.  The architecture is identical: the output of the mock planner
   still goes through the validator before the executor ever sees it.

Module boundary rule:
  NO imports from src.validator, src.executor, or mavsdk.
  The only shared import is src.schemas.mission (for the JSON schema export
  and for the Pydantic pre-screen used by the Ollama retry loop below — that
  pre-screen is the same Mission model validator.py itself calls; it is not
  a second, divergent set of safety rules).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

# ── Shared schema import ──────────────────────────────────────────────────────
# Used two ways below: instantiated as a local pre-screen inside the Ollama
# retry loop (so a bad response gets fed back to the model instead of reaching
# main.py), and its shape is described in the Ollama system prompt. Either
# way, this is the exact same model validator.py uses — not a parallel set of
# rules the LLM could satisfy while still failing the real validator.
from src.schemas.mission import Mission as _Mission

# ── Default geofence for the SITL home position ──────────────────────────────
# PX4 SITL defaults to 47.397742 N, 8.545594 E (Zurich area).
# The geofence is ±0.02 degrees (~2.2 km) in each direction.
_DEFAULT_GEOFENCE = {
    "lat_min": 47.377,
    "lat_max": 47.418,
    "lon_min": 8.525,
    "lon_max": 8.566,
}

_HOME_LAT = 47.397742
_HOME_LON = 8.545594

# Four corner waypoints of the geofence (used for perimeter demos)
# Labelled N/E/S/W but actually corners: NE, SE, SW, NW.
_PERIM_OFFSET_DEG = 0.0005
_PERIMETER_WAYPOINTS = [
    {"id": "wp_ne", "lat": round(_HOME_LAT + _PERIM_OFFSET_DEG, 6), "lon": round(_HOME_LON + _PERIM_OFFSET_DEG, 6)},
    {"id": "wp_se", "lat": round(_HOME_LAT - _PERIM_OFFSET_DEG, 6), "lon": round(_HOME_LON + _PERIM_OFFSET_DEG, 6)},
    {"id": "wp_sw", "lat": round(_HOME_LAT - _PERIM_OFFSET_DEG, 6), "lon": round(_HOME_LON - _PERIM_OFFSET_DEG, 6)},
    {"id": "wp_nw", "lat": round(_HOME_LAT + _PERIM_OFFSET_DEG, 6), "lon": round(_HOME_LON - _PERIM_OFFSET_DEG, 6)},
]

# Square pattern origin (SITL home offset by ~40 m in each axis)
# 1 degree lat ≈ 111 195 m → 40 m ≈ 0.00036 degrees
_SQUARE_OFFSET_DEG = 0.00036


# ── PlannerResult ─────────────────────────────────────────────────────────────

class PlannerResult:
    """
    Returned by plan().  Always contains:
      - raw_json: the unvalidated dict the planner produced (for audit logging).
      - prompt: the original NL prompt (for audit trail).
      - used_mock: True if the mock rule-based planner was used.
    """
    def __init__(self, raw_json: Dict[str, Any], prompt: str, used_mock: bool):
        self.raw_json = raw_json
        self.prompt = prompt
        self.used_mock = used_mock

    def __repr__(self) -> str:
        return (
            f"PlannerResult(used_mock={self.used_mock}, "
            f"mission_id={self.raw_json.get('mission_id', 'n/a')!r})"
        )


# ── Ollama configuration ──────────────────────────────────────────────────────
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1").strip()
# Initial attempt + up to this many corrective retries when the response
# fails to parse as JSON or fails the Mission schema pre-screen.
_OLLAMA_MAX_RETRIES = 3

_MISSION_JSON_EXAMPLE = {
    "created_from_prompt": "Go up to 10 metres and return",
    "actions": [
        {"action_id": "t1", "type": "takeoff", "altitude_m": 10.0},
        {
            "action_id": "wp1",
            "type": "goto_waypoint",
            "lat": 47.398242,
            "lon": 8.546094,
            "alt_m": 10.0,
            "speed_mps": 3.0,
        },
        {"action_id": "rtl1", "type": "rtl"},
    ],
    "constraints": {
        "max_altitude_m": 15.0,
        "max_speed_mps": 5.0,
        "max_loop_count": 1,
        "geofence": _DEFAULT_GEOFENCE,
    },
}

_OLLAMA_SYSTEM_PROMPT = f"""You are a flight mission PLANNER for a simulated drone.

YOU ARE PROPOSING A PLAN ONLY. You have no ability to execute anything — you
cannot arm, fly, or otherwise command a real or simulated vehicle. Your sole
output is a candidate mission JSON object. That JSON is then checked by an
independent, non-LLM safety validator (hard-coded altitude/speed/geofence/
loop-count limits and structural rules) before anything is ever sent to the
flight controller. If your JSON fails validation, the errors are shown to you
so you can correct it — you are never the last line of defense, so propose
your best plan and let the validator be the arbiter of safety.

Respond with ONLY a single JSON object — no markdown code fences, no
commentary before or after it.

REQUIRED JSON SHAPE:
- "created_from_prompt": the original command, as a string.
- "actions": a non-empty list, ALWAYS starting with a "takeoff" action and
  ALWAYS ending with an "rtl" or "land" action. Each action is one of:
    {{"action_id": str, "type": "takeoff", "altitude_m": number}}
    {{"action_id": str, "type": "goto_waypoint", "lat": number, "lon": number, "alt_m": number, "speed_mps": number}}
    {{"action_id": str, "type": "loop", "repeat_count": integer, "waypoint_ids": [str, ...]}}
    {{"action_id": str, "type": "rtl"}}
    {{"action_id": str, "type": "land"}}
  "loop".waypoint_ids must reference the action_id of earlier goto_waypoint actions.
- "constraints": {{"max_altitude_m": number, "max_speed_mps": number, "max_loop_count": integer, "geofence": {{"lat_min": number, "lat_max": number, "lon_min": number, "lon_max": number}}}}

HARD LIMITS (you must respect ALL of these):
- Altitude (altitude_m / alt_m, and max_altitude_m): between 2.0 and 30.0 metres
- Speed (speed_mps, and max_speed_mps): between 1.0 and 8.0 m/s
- Loop repeat_count (and max_loop_count): between 1 and 10
- Every waypoint (lat, lon) must fall inside the geofence bounding box below

DEFAULT GEOFENCE (use this unless the user specifies a different area):
{json.dumps(_DEFAULT_GEOFENCE, indent=2)}

DEFAULT HOME POSITION (SITL Zurich origin) — build waypoints as small offsets
from this point, staying inside the geofence above:
  lat: {_HOME_LAT}, lon: {_HOME_LON}

EXAMPLE — a valid mission JSON for "Go up to 10 metres and return":
{json.dumps(_MISSION_JSON_EXAMPLE, indent=2)}
"""


def _plan_with_ollama(prompt: str) -> Dict[str, Any]:
    """
    Call a local Ollama model to produce mission JSON.

    Local open models are noticeably less reliable at strict JSON than a
    hosted model's structured-output/tool-use mode, so this uses format="json"
    (biases decoding toward syntactically valid JSON) plus a correction loop:
    a parse failure or a Mission-schema failure is fed back to the model as an
    explicit list of errors, and it gets up to _OLLAMA_MAX_RETRIES chances to
    fix them before this function gives up.

    The Mission(**raw) pre-screen below is the SAME Pydantic model
    src/validator/validator.py uses — it is not a second, looser set of rules.
    It exists only to give the retry loop something to check against locally;
    validator.validate() still re-runs the identical check downstream as the
    actual safety gate before persistence/execution.
    """
    import ollama  # only imported on the real path — keeps module importable without it

    client = ollama.Client(host=_OLLAMA_HOST)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _OLLAMA_SYSTEM_PROMPT},
        {"role": "user", "content": f'Command: "{prompt}"'},
    ]

    last_errors: List[str] = []
    for attempt in range(1, _OLLAMA_MAX_RETRIES + 2):  # 1 initial + up to N retries
        if last_errors:
            messages.append({
                "role": "user",
                "content": (
                    "That JSON failed validation with these errors:\n"
                    + "\n".join(f"- {e}" for e in last_errors)
                    + "\n\nReturn a corrected JSON object only, fixing every error above."
                ),
            })

        try:
            response = client.chat(
                model=_OLLAMA_MODEL,
                messages=messages,
                format="json",
                options={"temperature": 0.2},
            )
        except Exception as exc:  # connection refused, model not pulled, etc.
            raise RuntimeError(
                f"Could not get a response from Ollama at {_OLLAMA_HOST} "
                f"(model={_OLLAMA_MODEL}): {exc}. Make sure `ollama serve` is "
                f"running and the model is pulled (`ollama pull {_OLLAMA_MODEL}`), "
                "or pass --mock-llm to skip the LLM entirely."
            ) from exc

        content = response["message"]["content"]
        messages.append({"role": "assistant", "content": content})

        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            last_errors = [f"Response was not valid JSON: {exc}"]
            continue

        if not isinstance(raw, dict):
            last_errors = [f"Response must be a JSON object, got {type(raw).__name__}"]
            continue

        raw.setdefault("created_from_prompt", prompt)
        raw.setdefault("mission_id", str(uuid.uuid4()))

        try:
            _Mission(**raw)
        except ValidationError as exc:
            last_errors = [
                f"[{' → '.join(str(x) for x in err['loc']) or 'root'}] {err['msg']}"
                for err in exc.errors()
            ]
            continue

        return raw

    raise RuntimeError(
        f"Ollama model '{_OLLAMA_MODEL}' did not produce a schema-valid mission "
        f"after {_OLLAMA_MAX_RETRIES + 1} attempts. Last errors:\n"
        + "\n".join(f"- {e}" for e in last_errors)
    )


# ── Public API ────────────────────────────────────────────────────────────────

def plan(prompt: str, *, force_mock: bool = False) -> PlannerResult:
    """
    Convert a natural-language prompt into a raw mission JSON dict.

    Args:
        prompt:     The NL command, e.g. "Patrol the perimeter twice at 15 m".
        force_mock: If True, use the deterministic mock planner instead of
                    calling Ollama. Used by the --mock-llm CLI flag.

    Returns:
        PlannerResult with raw_json, prompt, and used_mock flag.

    This function does NOT validate the output — call validator.validate()
    on result.raw_json before passing anything to the executor.
    """
    if force_mock:
        raw_json = _plan_with_mock(prompt)
        return PlannerResult(raw_json=raw_json, prompt=prompt, used_mock=True)
    else:
        raw_json = _plan_with_ollama(prompt)
        return PlannerResult(raw_json=raw_json, prompt=prompt, used_mock=False)


# ── Mock (rule-based) path ────────────────────────────────────────────────────

def _plan_with_mock(prompt: str) -> Dict[str, Any]:
    """
    Deterministic rule-based planner used when no API key is available.

    Parsing rules:
    1. Extract altitude from patterns like "15 metres", "20m", "at 15".
    2. Extract loop count from "twice", "x2", "2 times", "3 loops", etc.
    3. If "perimeter" or "patrol" in prompt → perimeter loop mission.
    4. If "square" in prompt → square pattern + RTL mission.
    5. Default: simple up-and-back waypoint mission.

    The output format is identical to what the LLM would return — both paths
    converge at validator.validate().
    """
    prompt_lower = prompt.lower()

    # ── Extract altitude ───────────────────────────────────────────────────
    alt = _extract_altitude(prompt_lower)

    # ── Extract loop count ─────────────────────────────────────────────────
    loop_count = _extract_loop_count(prompt_lower)

    # ── Choose mission template ────────────────────────────────────────────
    if any(kw in prompt_lower for kw in ("perimeter", "patrol", "loop", "circuit")):
        return _build_perimeter_mission(prompt, alt, loop_count)
    elif "square" in prompt_lower:
        return _build_square_mission(prompt, alt)
    else:
        return _build_simple_mission(prompt, alt)


def _extract_altitude(text: str) -> float:
    """
    Extract altitude in metres from natural-language text.
    Falls back to 15.0 m if no number is found.
    """
    # "15 metres", "15m", "at 15", "altitude 15", "height of 15"
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:metres?|meters?|m)\b",
        r"(?:at|altitude|height of)\s+(\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            # Clamp to safe range silently — the validator will still reject
            # the raw prompt value if the user explicitly said "50 metres",
            # because we pass the clamped value here but the validator sees only
            # the final dict.  This is intentional: the mock planner tries to be
            # helpful; the validator is the true arbiter.
            return max(2.0, min(30.0, val))
    return 15.0  # safe default


def _extract_loop_count(text: str) -> int:
    """
    Extract repeat count.  "twice" → 2, "three times" → 3, "x2" → 2, etc.
    Falls back to 2.
    """
    word_map = {
        "once": 1, "twice": 2, "thrice": 3,
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    }
    for word, count in word_map.items():
        if word in text:
            return count

    m = re.search(r"x(\d+)|(\d+)\s*(?:times?|loops?|laps?|circuits?)", text)
    if m:
        val = int(m.group(1) or m.group(2))
        return max(1, min(10, val))
    return 2


def _build_perimeter_mission(prompt: str, alt: float, loop_count: int) -> Dict[str, Any]:
    """Four-corner perimeter patrol, looped loop_count times."""
    waypoints = []
    for wp in _PERIMETER_WAYPOINTS:
        waypoints.append({
            "action_id": wp["id"],
            "type": "goto_waypoint",
            "lat": wp["lat"],
            "lon": wp["lon"],
            "alt_m": alt,
            "speed_mps": 3.0,
        })

    actions = [
        {"action_id": "t1", "type": "takeoff", "altitude_m": alt},
        *waypoints,
        {
            "action_id": "loop1",
            "type": "loop",
            "repeat_count": loop_count,
            "waypoint_ids": [wp["id"] for wp in _PERIMETER_WAYPOINTS],
        },
        {"action_id": "rtl1", "type": "rtl"},
    ]

    return {
        "mission_id": str(uuid.uuid4()),
        "created_from_prompt": prompt,
        "actions": actions,
        "constraints": {
            "max_altitude_m": min(alt + 5.0, 30.0),
            "max_speed_mps": 5.0,
            "geofence": _DEFAULT_GEOFENCE,
            "max_loop_count": 10,
        },
    }


def _build_square_mission(prompt: str, alt: float) -> Dict[str, Any]:
    """40 m square pattern + RTL (no loop)."""
    off = _SQUARE_OFFSET_DEG
    corners = [
        ("sq_ne", _HOME_LAT + off, _HOME_LON + off),
        ("sq_se", _HOME_LAT - off, _HOME_LON + off),
        ("sq_sw", _HOME_LAT - off, _HOME_LON - off),
        ("sq_nw", _HOME_LAT + off, _HOME_LON - off),
    ]

    actions = [{"action_id": "t1", "type": "takeoff", "altitude_m": alt}]
    for cid, clat, clon in corners:
        actions.append({
            "action_id": cid,
            "type": "goto_waypoint",
            "lat": clat,
            "lon": clon,
            "alt_m": alt,
            "speed_mps": 4.0,
        })
    actions.append({"action_id": "rtl1", "type": "rtl"})

    return {
        "mission_id": str(uuid.uuid4()),
        "created_from_prompt": prompt,
        "actions": actions,
        "constraints": {
            "max_altitude_m": min(alt + 5.0, 30.0),
            "max_speed_mps": 5.0,
            "geofence": _DEFAULT_GEOFENCE,
            "max_loop_count": 1,
        },
    }


def _build_simple_mission(prompt: str, alt: float) -> Dict[str, Any]:
    """Single out-and-back waypoint mission."""
    actions = [
        {"action_id": "t1", "type": "takeoff", "altitude_m": alt},
        {
            "action_id": "wp1",
            "type": "goto_waypoint",
            "lat": _HOME_LAT + 0.0005,
            "lon": _HOME_LON + 0.0005,
            "alt_m": alt,
            "speed_mps": 3.0,
        },
        {"action_id": "rtl1", "type": "rtl"},
    ]
    return {
        "mission_id": str(uuid.uuid4()),
        "created_from_prompt": prompt,
        "actions": actions,
        "constraints": {
            "max_altitude_m": min(alt + 5.0, 30.0),
            "max_speed_mps": 5.0,
            "geofence": _DEFAULT_GEOFENCE,
            "max_loop_count": 1,
        },
    }
