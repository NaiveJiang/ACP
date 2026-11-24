#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${FOOTSTEP_PYTHON:-$ROOT/.venv_footstep/bin/python}"
export ACP_ROBOT=roban
export ACP_ROBAN_DESCRIPTION="${ACP_ROBAN_DESCRIPTION:-$HOME/kuavo_amp_mjlab/src/mjlab/asset_zoo/robots/biped_s17_ver}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$ROOT"
exec "$PYTHON" -m supermimic.acp.collect_mimic \
  --upstream-roban "$ROOT/logs/motion_matching_upstream_roban_frax_001" "$@"
