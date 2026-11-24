#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${FOOTSTEP_PYTHON:-$ROOT/.venv_footstep/bin/python}"
export ACP_ROBOT=roban
export ACP_ROBAN_DESCRIPTION="${ACP_ROBAN_DESCRIPTION:-$HOME/kuavo_amp_mjlab/src/mjlab/asset_zoo/robots/biped_s17_ver}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$ROOT"
stage="${1:-}"
case "$stage" in
  stage1) shift; exec "$PYTHON" -m supermimic.acp.footstep.train_lmm_contacts "$@" ;;
  stage2) shift; exec "$PYTHON" -m supermimic.acp.footstep.finetune_lmm_contacts "$@" ;;
  *) printf '%s\n' 'Usage: bash run_lmm_contact_planner.sh {stage1|stage2} [options]' >&2; exit 2 ;;
esac
