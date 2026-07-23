"""
llm_planner.planner
--------------------
Converts a natural-language operator command into a schema-shaped mission plan
JSON by calling a locally-running Ollama model.

This module's ONLY responsibility is: prompt in, raw JSON text out. It never
validates against safety rules, never talks to the formation layer, the
executor, or the simulator, and never runs more than once per mission
(the retry loop below is still "one mission," not repeated missions -- see
its docstring). Keeping that boundary strict is what makes "LLM never issues
commands directly" true in practice rather than just in the README.

Backend note: this originally called the Anthropic API. It was switched to a
local Ollama model because no Anthropic API credit was available for this
build -- see docs/design_decisions.md for the full tradeoff writeup and what
reverting to Anthropic would take (short version: only this file changes;
the JSON contract, validator, formation layer, and executor are backend-agnostic
and don't know or care which LLM produced the mission JSON).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List

OLLAMA_MODEL_ID = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "300"))

# How many times to re-prompt the model after it returns something that
# doesn't even parse as JSON, before giving up. Local models are more prone
# to this than Claude's structured-output enforcement -- see plan_mission()'s
# docstring for why this is scoped to JSON-parse failures only, not to
# connection/HTTP errors.
MAX_JSON_RETRIES = 2

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "mission_schema.json"


class LLMPlanningError(Exception):
    """Raised when the LLM call itself fails (unreachable server, HTTP
    error, or JSON the model never manages to produce) -- distinct from a
    schema/safety validation failure, which is validator.py's job to report."""


def _load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


# Unchanged from the Anthropic version -- reused verbatim per the decision
# to swap backends without rewriting the prompt. See docs/design_decisions.md.
SYSTEM_PROMPT = """You are the mission-planning component of a multi-drone squad control system.

Your ONLY job is to translate an operator's natural-language command into a single
JSON object describing a multi-agent drone mission. You do not fly drones, you do not
issue flight commands, and you never produce anything except that one JSON object.

Rules:
- Output must be a single JSON object matching the provided schema exactly. No prose,
  no markdown code fences, no explanation before or after the JSON.
- "num_agents" must match how many drones the operator's command implies (2 or 3 for
  this system). If the operator says "you three", num_agents=3. If they say "the two
  of you" or don't specify a count, default to 2.
- "formation.type": use "wedge" for area sweeps/coverage, "line" or "column" for
  simple coordinated movement along a route, and "split" whenever the command asks to
  divide a route/patrol between agents and regroup at the end.
- "task.type": use "area_sweep" with an "area_of_interest" polygon (3+ points) when
  the operator describes an area to cover. Use "patrol_route" with a "route" waypoint
  list (2+ points) when they describe a path or route to follow.
- "task.regroup_point" is always present in the JSON. Set it to null unless
  formation.type is "split", in which case it must be a real {"x": ..., "y": ...}
  point -- where the drones reconvene after their individual segments.
- All coordinates are meters in a local XY frame. Choose realistic, non-degenerate
  coordinates (tens to low hundreds of meters) that make geometric sense for the
  described task -- a sweep area should be a real polygon with non-zero area, a route
  should have waypoints that trace a sensible path, not a single point repeated.
- "safety" bounds must be internally consistent with the rest of the plan: they must
  accommodate the mission's own altitude_m, speed_mps, and formation.spacing_m (never
  request a 30 m/s cruise speed but cap max_speed_mps at 10). Reasonable defaults
  absent other guidance: max_altitude_m 120, max_speed_mps 15, min_separation_m 3,
  and a geofence generous enough to contain the task with margin.
- "mission_id": a short, unique, human-readable slug (e.g. "wedge-sweep-north-01").
- "operator_prompt": echo the operator's command verbatim.

If the command is ambiguous, make the most reasonable operational interpretation --
you will not get a chance to ask a follow-up question."""


def _call_ollama_chat(model: str, base_url: str, messages: List[Dict[str, str]], schema: Dict[str, Any]) -> str:
    """One request to Ollama's /api/chat, non-streaming. `format` is set to
    the full mission JSON Schema (Ollama compiles this into a decoding-time
    grammar constraint), not just the generic "json" mode -- the same
    schema-constrained-generation intent as the Anthropic version's
    output_config.format, kept for backend consistency.

    Known risk, not silently worked around: Ollama's schema-to-grammar
    compilation has historically had uneven support for `anyOf`/`$ref`,
    which mission_schema.json uses for its task discriminated union. If a
    given Ollama version rejects the schema outright, that surfaces as an
    HTTPError below with the server's own error message -- not retried (see
    plan_mission()'s docstring for why retries are scoped to JSON-parse
    failures only). If you hit this, the schema itself is the thing to
    simplify for your Ollama version, not something this function should
    silently paper over.
    """
    payload = {
        "model": model,
        "messages": messages,
        "format": schema,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print("Calling local LLM, this may take 1-3 minutes on CPU...")
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise LLMPlanningError(
            f"Ollama returned HTTP {e.code} for model={model!r} at {base_url}: {detail}\n"
            f"If this says the model wasn't found, run: ollama pull {model}"
        ) from e
    except urllib.error.URLError as e:
        raise LLMPlanningError(
            f"Could not reach Ollama at {base_url} (model={model}): {e.reason}. "
            "Is `ollama serve` running? See README.md for setup."
        ) from e

    return body["message"]["content"]


def plan_mission(
    operator_prompt: str,
    *,
    model: str = OLLAMA_MODEL_ID,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """
    Calls a local Ollama model with the operator's natural-language command
    and returns the RAW JSON text of its response as a string. Same
    signature shape as the Anthropic version it replaced
    (`plan_mission(prompt, **kwargs)`), so run.py needed no changes.

    This function does not validate, parse semantically, or act on the JSON
    beyond confirming it parses -- schema/safety validation is
    schema/validator.py's job, run as a separate, explicit step downstream.

    Retry scope: if the model's response doesn't even parse as JSON, we
    re-prompt it up to MAX_JSON_RETRIES times, feeding back its own bad
    output plus a corrective instruction -- local models are more prone to
    this than Claude's structured-output enforcement, even with `format`
    set to the schema (see _call_ollama_chat's docstring). This is
    deliberately narrow: a connection failure or an HTTP error from Ollama
    itself (server not running, model not pulled, schema rejected) is NOT
    retried -- those aren't transient JSON-quality problems, retrying them
    just delays a clear error message for no benefit.
    """
    schema = _load_schema()
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": operator_prompt},
    ]

    last_error: json.JSONDecodeError | None = None
    for attempt in range(MAX_JSON_RETRIES + 1):
        text = _call_ollama_chat(model, base_url, messages, schema)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            last_error = e
            messages = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"That was not valid JSON (parse error: {e}). Respond again with ONLY "
                        "a single valid JSON object matching the schema you were given -- no "
                        "prose, no markdown code fences, no explanation."
                    ),
                },
            ]
            continue

        # Defense in depth: never trust the model's echo of the prompt, or
        # let a missing mission_id block the pipeline, when we can just
        # supply both deterministically.
        obj["operator_prompt"] = operator_prompt
        if not obj.get("mission_id"):
            obj["mission_id"] = "mission-" + uuid.uuid4().hex[:8]
        return json.dumps(obj, indent=2)

    raise LLMPlanningError(
        f"Ollama model {model!r} did not return valid JSON after {MAX_JSON_RETRIES + 1} "
        f"attempts. Last parse error: {last_error}"
    )
