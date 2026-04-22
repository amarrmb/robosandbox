#!/usr/bin/env bash
# Validates the full pluggable eval pipeline:
#   demo (generate recording) → eval mujoco (single world) → eval newton (N worlds)
#
# Run this after any changes to the eval path to confirm nothing is broken.
# Usage: bash scripts/validate_eval_pipeline.sh [world_count]
#
# On a GPU machine: world_count defaults to 8
# On CPU/local:    world_count=1 still validates the plumbing

set -euo pipefail

WORLD_COUNT=${1:-8}
TASK="pick_cube_franka"

# Headless GPU: use EGL for off-screen MuJoCo rendering
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

# Newton venv: warp/newton are GPU-only and live in a separate venv on DGX.
# Override with NEWTON_VENV=/path/to/venv if installed elsewhere.
NEWTON_VENV=${NEWTON_VENV:-/home/amar/newton/.venv}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$NEWTON_VENV/bin/python3" ]; then
  # Ensure robosandbox is installed in the newton venv (idempotent, quiet)
  echo "[setup] Installing robosandbox into Newton venv: $NEWTON_VENV"
  "$NEWTON_VENV/bin/pip" install -q -e "$REPO_ROOT/packages/robosandbox-core"
  NEWTON_CLI="$NEWTON_VENV/bin/robo-sandbox"
else
  # Fall back to system robo-sandbox — will fail if warp not in PATH Python
  echo "[setup] Newton venv not found at $NEWTON_VENV — using system robo-sandbox"
  echo "        (Set NEWTON_VENV=/path/to/venv if warp is installed elsewhere)"
  NEWTON_CLI="robo-sandbox"
fi

echo "========================================"
echo "  RoboSandbox eval pipeline validation"
echo "========================================"
echo "task:         $TASK"
echo "world_count:  $WORLD_COUNT"
echo ""

# ---- Step 1: Generate a Franka recording via scripted agent ------------
echo "[ 1/3 ] Generating test episode (Franka scripted pick)..."
python3 scripts/generate_test_episode.py
EPISODE_DIR=$(ls -dt runs/20*/ 2>/dev/null | head -1)
if [ -z "$EPISODE_DIR" ]; then
  echo "ERROR: no episode found under runs/ after generate"
  exit 1
fi
EPISODE_DIR="${EPISODE_DIR%/}"  # strip trailing slash
echo "    episode: $EPISODE_DIR"
EVENTS="$EPISODE_DIR/events.jsonl"
LINES=$(wc -l < "$EVENTS")
echo "    events.jsonl: $LINES rows"
if [ "$LINES" -lt 10 ]; then
  echo "ERROR: events.jsonl too short ($LINES rows) — demo may have failed"
  exit 1
fi
echo "    OK"
echo ""

# ---- Step 2: MuJoCo eval (single world, validates policy loading + run_policy) ----
echo "[ 2/3 ] MuJoCo eval (single world, state replay)..."
robo-sandbox eval \
  --task "$TASK" \
  --policy "$EPISODE_DIR" \
  --sim-backend mujoco \
  --max-steps 600 && MUJOCO_EXIT=0 || MUJOCO_EXIT=$?
if [ $MUJOCO_EXIT -eq 0 ]; then
  echo "    result: success"
elif [ $MUJOCO_EXIT -eq 1 ]; then
  echo "    result: task not solved (open-loop replay may drift — plumbing is fine)"
else
  echo "ERROR: MuJoCo eval crashed (exit $MUJOCO_EXIT)"
  exit $MUJOCO_EXIT
fi
echo ""

# ---- Step 3: Newton eval (N parallel worlds, validates parallel harness) ----
echo "[ 3/3 ] Newton eval ($WORLD_COUNT parallel worlds) via: $NEWTON_CLI"
"$NEWTON_CLI" eval \
  --task "$TASK" \
  --policy "$EPISODE_DIR" \
  --sim-backend newton \
  --world-count "$WORLD_COUNT" \
  --max-steps 600 && NEWTON_EXIT=0 || NEWTON_EXIT=$?
if [ $NEWTON_EXIT -eq 0 ] || [ $NEWTON_EXIT -eq 1 ]; then
  echo "    harness ran cleanly (exit $NEWTON_EXIT)"
else
  echo "ERROR: Newton eval crashed (exit $NEWTON_EXIT)"
  exit $NEWTON_EXIT
fi
echo ""

echo "========================================"
echo "  VALIDATION COMPLETE"
echo "  Episode:  $EPISODE_DIR"
echo "  MuJoCo:   exit $MUJOCO_EXIT"
echo "  Newton:   exit $NEWTON_EXIT  (cli: $NEWTON_CLI)"
echo "========================================"
