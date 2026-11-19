#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export ACP_ROBOT=roban
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$ROOT/.venv_preview/bin/uv" run --no-project \
  --python "$ROOT/.venv_blink/bin/python" python -m supermimic.acp.collect_mimic "$@"
