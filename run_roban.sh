#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
PYTHON_BIN="${ACP_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Configure .venv using README_ENVIRONMENT.md, or set ACP_PYTHON to a compatible Python executable.\n' >&2
    exit 1
fi
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
exec "$PYTHON_BIN" -m supermimic.acp --robot roban "$@"
