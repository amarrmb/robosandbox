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

echo "========================================"
echo "  RoboSandbox eval pipeline validation"
echo "========================================"
echo "task:         $TASK"
echo "world_count:  $WORLD_COUNT"
echo ""

# ---- Step 1: Generate a recording via scripted demo --------------------
echo "[ 1/3 ] Generating test episode (scripted demo)..."
robo-sandbox demo
# LocalRecorder writes to runs/YYYYMMDD-HHMMSS-<id>/
EPISODE_DIR=$(ls -dt runs/20*/ 2>/dev/null | head -1)
if [ -z "$EPISODE_DIR" ]; then
  echo "ERROR: no episode found under runs/ after demo"
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
  --max-steps 600
MUJOCO_EXIT=$?
if [ $MUJOCO_EXIT -eq 0 ]; then
  echo "    result: success"
elif [ $MUJOCO_EXIT -eq 1 ]; then
  echo "    result: failure (policy ran but task not solved — open-loop replay may drift)"
else
  echo "ERROR: MuJoCo eval crashed (exit $MUJOCO_EXIT)"
  exit $MUJOCO_EXIT
fi
echo ""

# ---- Step 3: Newton eval (N parallel worlds, validates parallel harness) ----
echo "[ 3/3 ] Newton eval ($WORLD_COUNT parallel worlds)..."
robo-sandbox eval \
  --task "$TASK" \
  --policy "$EPISODE_DIR" \
  --sim-backend newton \
  --world-count "$WORLD_COUNT" \
  --max-steps 600
NEWTON_EXIT=$?
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
echo "  Newton:   exit $NEWTON_EXIT"
echo "========================================"
