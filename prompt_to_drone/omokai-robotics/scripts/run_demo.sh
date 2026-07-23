#!/usr/bin/env bash
# scripts/run_demo.sh
# ====================
# Run the full pipeline demo natively (no Docker) against a local PX4 SITL.
#
# Prerequisites:
#   • PX4-Autopilot cloned and built at ~/PX4-Autopilot (px4_sitl_default)
#   • Gazebo Classic 11 installed (apt: gazebo11 + libgazebo11-dev)
#   • Python deps installed: pip3 install -r requirements.txt && pip3 install -e .
#
# Usage:
#   bash scripts/run_demo.sh                         # perimeter patrol (mock LLM)
#   bash scripts/run_demo.sh "Square at 20 m"        # custom prompt
#   # Real LLM via local Ollama (ollama serve && ollama pull llama3.1 first):
#   NO_MOCK=1 bash scripts/run_demo.sh "Patrol twice at 15 m"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
PROMPT="${1:-Patrol the perimeter loop twice at 15 metres}"

echo "================================================="
echo "  Omokai Robotics — Native Demo"
echo "================================================="
echo "PX4 dir : $PX4_DIR"
echo "Prompt  : $PROMPT"
echo ""

# ── Source PX4 Gazebo environment ─────────────────────────────────────────────
echo "[1/3] Setting up PX4 + Gazebo environment …"
# shellcheck disable=SC1091
source "$PX4_DIR/Tools/simulation/gazebo-classic/setup_gazebo.bash" \
       "$PX4_DIR" \
       "$PX4_DIR/build/px4_sitl_default" 2>/dev/null || true

export ROS_PACKAGE_PATH="$ROS_PACKAGE_PATH:$PX4_DIR"
export GAZEBO_PLUGIN_PATH="$GAZEBO_PLUGIN_PATH:$PX4_DIR/build/px4_sitl_default/build_gazebo-classic"
export GAZEBO_MODEL_PATH="$GAZEBO_MODEL_PATH:$PX4_DIR/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models"

# ── Launch SITL in background ─────────────────────────────────────────────────
echo "[2/3] Launching PX4 SITL + Gazebo Classic (headless) …"
HEADLESS=1 make -C "$PX4_DIR" px4_sitl_default gazebo-classic \
    FORCE_HEADLESS=1 &
SITL_PID=$!

echo "      SITL PID: $SITL_PID"
echo "      Waiting 30 s for SITL to initialise …"
sleep 30

# ── Run the pipeline ──────────────────────────────────────────────────────────
echo "[3/3] Running pipeline …"
cd "$REPO_DIR"

MOCK_FLAG="--mock-llm"
if [[ -n "${NO_MOCK:-}" ]]; then
    MOCK_FLAG=""
    echo "      NO_MOCK set — using the real LLM planner (local Ollama)."
else
    echo "      Using mock LLM (set NO_MOCK=1 to use Ollama instead)."
fi

python3 src/main.py \
    --prompt "$PROMPT" \
    $MOCK_FLAG \
    --connection-url "udp://:14540"

echo ""
echo "Demo complete. SITL still running (PID $SITL_PID)."
echo "To stop SITL: kill $SITL_PID"
