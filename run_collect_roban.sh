#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON_BIN="${ROBAN_COLLECT_PYTHON:-$PWD/.venv_roban_collect/bin/python}"
exec "$PYTHON_BIN" -u tools/collect_roban_acp.py "$@"
