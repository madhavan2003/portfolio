#!/usr/bin/env python3
"""
run.py
------
CLI entrypoint for the multi-agent drone formation mission pipeline:

    operator prompt -> LLM (local Ollama model) -> raw JSON -> validated JSON
        -> formation layer -> deterministic executor -> gym-pybullet-drones sim

Usage:
    python run.py --prompt "You three sweep this area in a wedge formation"
    python run.py --mission-file demo/example_missions/wedge_sweep.json   # skip the LLM call
    python run.py --prompt "..." --gui                                    # visual demo (not for Docker)
    python run.py --prompt "..." --gui --chase-drone                      # close-up chase cam on drone_0
    python run.py --prompt "..." --gui --chase-drone drone_1              # chase cam on a specific drone

Every stage's output is printed so the pipeline is auditable end to end --
this is the CLI's whole purpose, not an afterthought: you should be able to
see exactly what the LLM produced, whether/why it was accepted, what each
drone was actually assigned to fly, and how the simulation went, without
reading any code.
"""

from __future__ import annotations

import argparse
import json
import sys

from executor import execute_mission
from formation.formation_planner import agent_ids_for
from llm_planner import LLMPlanningError, plan_mission
from schema import MissionValidationError, validate_mission


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-agent drone formation mission pipeline.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", type=str, help="Natural-language operator command.")
    source.add_argument(
        "--mission-file",
        type=str,
        help="Path to a pre-generated mission JSON file, skipping the LLM call entirely "
        "(useful for offline testing / demoing the executor without an API key -- see "
        "demo/example_missions/).",
    )
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet GUI (not for Docker).")
    parser.add_argument(
        "--chase-drone",
        type=str,
        nargs="?",
        const="",
        default=None,
        metavar="AGENT_ID",
        help="Chase-cam: follow one drone up close instead of framing the whole "
        "mission (only has an effect with --gui). Pass an agent id (e.g. drone_1) "
        "to pick which one; bare --chase-drone follows the first agent.",
    )
    parser.add_argument(
        "--output-folder", type=str, default="logs", help="Telemetry output directory (default: logs/)."
    )
    parser.add_argument("--model", type=str, default=None, help="Override the Ollama model tag.")
    args = parser.parse_args()

    # --- Stage 1: prompt -> LLM -> raw JSON ---------------------------------
    if args.mission_file:
        _print_header("STAGE 1: LLM PLANNING (skipped -- loading mission file)")
        with open(args.mission_file) as f:
            raw_json = f.read()
        print(f"Loaded: {args.mission_file}")
    else:
        _print_header("STAGE 1: LLM PLANNING")
        print(f"Operator prompt: {args.prompt!r}")
        try:
            kwargs = {"model": args.model} if args.model else {}
            raw_json = plan_mission(args.prompt, **kwargs)
        except LLMPlanningError as e:
            print(f"\nLLM planning failed: {e}", file=sys.stderr)
            return 1
        print("Raw LLM output:")
        print(raw_json)

    # --- Stage 2: validation (schema + safety, stage 1 of validator.py) ----
    _print_header("STAGE 2: VALIDATION (schema + safety)")
    result = validate_mission(raw_json)
    if not result.valid:
        print("REJECTED. Mission will not be executed. Errors:")
        for err in result.errors:
            print(f"  - {err}")
        return 1
    mission = result.mission
    print("VALID. Mission accepted:")
    print(json.dumps(mission, indent=2))

    chase_agent_id = None
    if args.chase_drone is not None:
        available = agent_ids_for(mission["num_agents"])
        chase_agent_id = args.chase_drone or available[0]
        if chase_agent_id not in available:
            print(
                f"\n--chase-drone {chase_agent_id!r} is not a valid agent for this "
                f"mission (available: {available})",
                file=sys.stderr,
            )
            return 1
        if not args.gui:
            print("\nNote: --chase-drone has no effect without --gui.", file=sys.stderr)

    # --- Stage 3: formation + deterministic executor + sim -----------------
    _print_header("STAGE 3: FORMATION + EXECUTION (deterministic, no LLM involved)")
    try:
        exec_result = execute_mission(
            mission, gui=args.gui, output_folder=args.output_folder, chase_agent_id=chase_agent_id
        )
    except MissionValidationError as e:
        print("REJECTED at post-formation separation check. Mission will not be simulated. Errors:")
        for err in e.errors:
            print(f"  - {err}")
        return 1

    print("Per-drone waypoints (formation layer output, meters, local XY frame):")
    for agent_id, waypoints in exec_result.formation_waypoints.items():
        pretty = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in waypoints)
        print(f"  {agent_id}: {pretty}")

    # --- Stage 4: simulation result -----------------------------------------
    _print_header("STAGE 4: SIMULATION RESULT")
    sim = exec_result.simulation
    print(f"Agents simulated: {sim.agent_ids}")
    print(f"Control steps run: {sim.num_control_steps}")
    print(
        f"Minimum observed inter-drone separation: {sim.min_separation_observed_m:.2f} m "
        f"(required: {mission['safety']['min_separation_m']} m)"
    )
    print(f"Telemetry written to: {sim.output_folder}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
