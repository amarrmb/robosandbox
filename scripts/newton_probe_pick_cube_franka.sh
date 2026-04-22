#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-dgxspark}"
REMOTE_REPO="${REMOTE_REPO:-~/robosandbox}"
REMOTE_NEWTON="${REMOTE_NEWTON:-~/newton}"
LOCAL_VIEWER="${LOCAL_VIEWER:-8000}"
REMOTE_VIEWER="${REMOTE_VIEWER:-8000}"
LOCAL_PORT="${LOCAL_PORT:-8090}"
REMOTE_PORT="${REMOTE_PORT:-8090}"
DEVICE="${DEVICE:-cuda:0}"
TASK="${TASK:-pick_cube_franka}"

case "${1:-}" in
  -h|--help)
    cat <<EOF
Forward a Newton-backed RoboSandbox viewer from dgxspark to localhost.

Usage:
  bash scripts/newton_probe_pick_cube_franka.sh [--task <name>]

Arguments:
  --task <name>   Newton task to load (default: \$TASK or pick_cube_franka)
                  Newton tasks: pick_cube_franka, cloth_fold_franka, cloth_fold_franka_x10

Environment overrides:
  REMOTE_HOST   SSH host alias (default: dgxspark)
  REMOTE_REPO   Remote robosandbox checkout (default: ~/robosandbox)
  REMOTE_NEWTON Remote newton checkout (default: ~/newton)
  LOCAL_VIEWER  Local robosandbox viewer port (default: 8000)
  REMOTE_VIEWER Remote robosandbox viewer port on dgxspark (default: 8000)
  LOCAL_PORT    Local forwarded Viser port (default: 8090)
  REMOTE_PORT   Remote Viser port (default: 8090)
  DEVICE        Newton device (default: cuda:0)
  TASK          Built-in task name (default: pick_cube_franka)

Then open:
  RoboSandbox viewer: http://127.0.0.1:\${LOCAL_VIEWER:-8000}
  Viser 3D (direct):  http://127.0.0.1:\${LOCAL_PORT}
EOF
    exit 0
    ;;
  --task)
    TASK="${2:?--task requires a value}"
    shift 2
    ;;
esac

echo "[probe] task=${TASK} device=${DEVICE}"
echo "[probe] Viser tunnel: localhost:${LOCAL_PORT} -> ${REMOTE_HOST}:localhost:${REMOTE_PORT}"
echo "[probe] open RoboSandbox viewer: http://127.0.0.1:${LOCAL_VIEWER}"
echo "[probe]   or Viser direct:       http://127.0.0.1:${LOCAL_PORT}"

ssh -o ExitOnForwardFailure=yes \
  -L "${LOCAL_VIEWER}:127.0.0.1:${REMOTE_VIEWER}" \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "${REMOTE_HOST}" \
  "pkill -f 'robo-sandbox viewer --sim-backend newton' >/dev/null 2>&1 || true; \
   cd ${REMOTE_NEWTON} && \
   . .venv/bin/activate && \
   python -m pip install -e '${REMOTE_REPO}/packages/robosandbox-core[viewer]' -q && \
   cd ${REMOTE_REPO} && \
   exec robo-sandbox viewer --sim-backend newton --viser-port ${REMOTE_PORT} --task ${TASK} --device ${DEVICE}"
