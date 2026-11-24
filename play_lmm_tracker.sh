#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export ACP_ROBOT=roban
export ACP_ROBAN_DESCRIPTION="${ACP_ROBAN_DESCRIPTION:-$HOME/kuavo_amp_mjlab/src/mjlab/asset_zoo/robots/biped_s17_ver}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export PYTHONUNBUFFERED=1
PYTHON="${FOOTSTEP_PYTHON:-$ROOT/.venv_footstep/bin/python}"
exec "$PYTHON" -m supermimic.acp.footstep.play_lmm_tracker \
    --lmm logs/roban_lmm_frax_s17_001/model \
    --checkpoint logs/roban_lafan1_frax_s17_mimic_002/last.pt "$@"
