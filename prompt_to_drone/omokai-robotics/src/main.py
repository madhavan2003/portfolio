"""
main.py — CLI entrypoint for the Omokai Robotics pipeline.

Usage:
    python3 src/main.py --prompt "Patrol the perimeter loop twice at 15 metres"
    python3 src/main.py --mission-file missions/patrol_perimeter.json
    python3 src/main.py --prompt "Square pattern at 20 m" --dry-run
    python3 src/main.py --prompt "Patrol ..." --mock-llm   # no API key needed

Pipeline stages (printed in order):
  Stage 1: LLM Planner   → raw JSON
  Stage 2: Validator      → validated Mission or error list
  Stage 3: Executor       → MAVSDK commands + telemetry log

Each stage's output is printed to stdout so the full flow is visible and
auditable in the terminal.  The executor also writes a persistent log file
to logs/run_<mission_id>.log.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

# Pipeline modules — imported in the dependency order they run.
# Note the explicit absence of any import from executor inside llm_planner,
# and vice versa.  This import graph is the proof the LLM is out of the
# control loop.
from src.llm_planner.planner import plan
from src.validator.validator import validate, validate_file
from src.executor.executor import run_mission

console = Console()


# ── CLI definition ────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--prompt", "-p",
    default=None,
    help="Natural-language mission prompt (e.g. 'Patrol the perimeter twice at 15 m').",
)
@click.option(
    "--mission-file", "-f",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to a pre-validated mission JSON file (skips LLM planner).",
)
@click.option(
    "--mock-llm",
    is_flag=True,
    default=False,
    help="Use the deterministic mock planner instead of the LLM API.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run planner + validator but skip the executor (no SITL needed).",
)
@click.option(
    "--connection-url",
    default="udp://:14540",
    show_default=True,
    help="MAVLink UDP address of the SITL instance.",
)
def main(
    prompt: str | None,
    mission_file: Path | None,
    mock_llm: bool,
    dry_run: bool,
    connection_url: str,
) -> None:
    """
    Omokai Robotics — NL Prompt → Validated Mission → PX4 SITL.

    Exactly one of --prompt or --mission-file must be provided.
    """
    if not prompt and not mission_file:
        console.print("[bold red]Error:[/] Provide either --prompt or --mission-file.")
        sys.exit(1)
    if prompt and mission_file:
        console.print("[bold red]Error:[/] --prompt and --mission-file are mutually exclusive.")
        sys.exit(1)

    console.print(Panel.fit(
        "[bold cyan]Omokai Robotics Pipeline[/]\n"
        "NL Prompt → LLM → Validator → Executor → PX4 SITL",
        border_style="cyan",
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1: LLM Planner
    # ═══════════════════════════════════════════════════════════════════════
    raw_json: dict | None = None

    if prompt:
        console.rule("[bold yellow]Stage 1: LLM Planner[/]")
        console.print(f"[dim]Prompt:[/] {prompt}")
        if mock_llm:
            console.print("[dim]Mode: deterministic mock (--mock-llm)[/]")

        try:
            result = plan(prompt, force_mock=mock_llm)
        except RuntimeError as exc:
            console.print(f"[bold red]✗ Planner failed:[/] {exc}")
            sys.exit(1)
        raw_json = result.raw_json

        mode_label = "[yellow]MOCK[/]" if result.used_mock else "[green]LLM (Ollama)[/]"
        console.print(f"[dim]Planner mode:[/] {mode_label}")
        console.print()
        console.print("[bold]Raw JSON from planner:[/]")
        console.print(Syntax(json.dumps(raw_json, indent=2), "json", theme="monokai"))

    else:  # --mission-file
        console.rule("[bold yellow]Stage 1: Loading mission file[/]")
        console.print(f"[dim]File:[/] {mission_file}")
        raw_json = json.loads(mission_file.read_text())
        console.print("[bold]Mission JSON:[/]")
        console.print(Syntax(json.dumps(raw_json, indent=2), "json", theme="monokai"))

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2: Validator
    # ═══════════════════════════════════════════════════════════════════════
    console.rule("[bold yellow]Stage 2: Validator[/]")
    validation = validate(raw_json, persist=not dry_run)

    if not validation.success:
        console.print("[bold red]✗ Validation FAILED[/]")
        console.print()
        console.print("[red]Errors (would be fed back to LLM for retry in a real system):[/]")
        for err in validation.errors:
            console.print(f"  [red]• {err}[/]")
        sys.exit(2)

    mission = validation.mission
    console.print(f"[bold green]✓ Validation PASSED[/]  mission_id=[cyan]{mission.mission_id}[/]")
    if not dry_run:
        console.print(f"[dim]Mission persisted to:[/] missions/{mission.mission_id}.json")
    console.print(f"[dim]Actions:[/] {len(mission.actions)}")
    console.print(f"[dim]Constraints:[/] max_alt={mission.constraints.max_altitude_m} m, "
                  f"max_speed={mission.constraints.max_speed_mps} m/s, "
                  f"max_loops={mission.constraints.max_loop_count}")

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 3: Executor
    # ═══════════════════════════════════════════════════════════════════════
    if dry_run:
        console.rule("[bold yellow]Stage 3: Executor (DRY RUN — skipped)[/]")
        console.print("[dim]Pass a running SITL and omit --dry-run to execute the mission.[/]")
        console.print()
        console.print("[bold green]✓ Dry run complete.[/]")
        return

    console.rule("[bold yellow]Stage 3: Executor[/]")
    console.print(f"[dim]Connecting to SITL at[/] [cyan]{connection_url}[/] …")
    console.print(f"[dim]Log file:[/] logs/run_{mission.mission_id}.log")
    console.print()

    try:
        asyncio.run(run_mission(mission, connection_url=connection_url))
    except RuntimeError as exc:
        console.print(f"[bold red]✗ Executor failed:[/] {exc}")
        sys.exit(3)
    except KeyboardInterrupt:
        console.print("[bold yellow]Interrupted by user.[/]")
        sys.exit(130)

    console.print()
    console.print("[bold green]✓ Mission complete.[/]")
    console.print(f"[dim]Full command log:[/] logs/run_{mission.mission_id}.log")


if __name__ == "__main__":
    main()
