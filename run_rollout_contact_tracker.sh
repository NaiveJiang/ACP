#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACP_ROBAN_DESCRIPTION="${ACP_ROBAN_DESCRIPTION:-$HOME/kuavo_amp_mjlab/src/mjlab/asset_zoo/robots/biped_s17_ver}"
exec bash "$ROOT/run_contact_tracker_fast.sh" --profile wbc --controller rollout "$@"
