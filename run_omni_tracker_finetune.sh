#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export ACP_ROBOT=roban
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec .venv_footstep/bin/python -m supermimic.acp.footstep.train_planned_tracker "$@"
