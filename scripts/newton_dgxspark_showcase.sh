#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-dgxspark}"
REMOTE_ROOT="${REMOTE_ROOT:-\$HOME/newton}"
EXAMPLE="${EXAMPLE:-ik_franka}"
DEVICE="${DEVICE:-cuda:0}"
WORLD_COUNT="${WORLD_COUNT:-4}"
LOCAL_PORT="${LOCAL_PORT:-8080}"
REMOTE_PORT="${REMOTE_PORT:-8080}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
USE_WORLD_COUNT="${USE_WORLD_COUNT:-0}"

case "${1:-}" in
  -h|--help)
    cat <<EOF
Launch a Newton web viewer on ${REMOTE_HOST} and forward it locally.

Environment overrides:
  REMOTE_HOST   SSH host alias (default: dgxspark)
  REMOTE_ROOT   Remote Newton checkout (default: \$HOME/newton)
  EXAMPLE       Newton example name (default: ik_cube_stacking)
  DEVICE        Newton device (default: cuda:0)
  WORLD_COUNT   Passed to examples that accept --world-count (default: 4)
  LOCAL_PORT    Local forwarded port (default: 8080)
  REMOTE_PORT   Remote viser port (default: 8080)
  EXTRA_ARGS    Extra args appended to the Newton example command
  USE_WORLD_COUNT  Set to 1 to append --world-count (default: 0)

Usage:
  bash scripts/newton_dgxspark_showcase.sh

Then open:
  http://127.0.0.1:8000/showcase
EOF
    exit 0
    ;;
esac

echo "[showcase] forwarding localhost:${LOCAL_PORT} -> ${REMOTE_HOST}:localhost:${REMOTE_PORT}"
echo "[showcase] example=${EXAMPLE} device=${DEVICE} world_count=${WORLD_COUNT}"
echo "[showcase] open http://127.0.0.1:8000/showcase once the local viewer is running"

WORLD_COUNT_ARG=""
if [ "${USE_WORLD_COUNT}" = "1" ]; then
  WORLD_COUNT_ARG="--world-count ${WORLD_COUNT}"
fi

ssh -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "${REMOTE_HOST}" \
  "set -euo pipefail; \
  cd ${REMOTE_ROOT}; \
  pkill -f 'python -m newton.examples' >/dev/null 2>&1 || true; \
  if [ ! -d .venv ]; then python3 -m venv .venv; fi; \
  . .venv/bin/activate; \
  python - <<'PY'
from importlib.util import find_spec
mods = [\"newton\", \"viser\"]
missing = [m for m in mods if find_spec(m) is None]
if missing:
    raise SystemExit(\"Missing remote Python packages: \" + \", \".join(missing))
PY
  exec python -m newton.examples \"${EXAMPLE}\" --viewer viser --device \"${DEVICE}\" ${WORLD_COUNT_ARG} ${EXTRA_ARGS}"
