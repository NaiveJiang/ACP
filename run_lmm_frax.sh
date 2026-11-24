#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${FOOTSTEP_PYTHON:-$ROOT/.venv_footstep/bin/python}"
UPSTREAM="$ROOT/logs/motion_matching_upstream_roban_frax_001"
OUT="$ROOT/logs/roban_lmm_frax_s17_001"
check_only=0
STEPPER_WINDOW=62

while (( $# )); do
    case "$1" in
        --check-only) check_only=1; shift ;;
        --out)
            if (( $# < 2 )); then printf '%s\n' 'Missing directory after --out' >&2; exit 2; fi
            OUT="$2"
            shift 2
            ;;
        --upstream-roban)
            if (( $# < 2 )); then printf '%s\n' 'Missing directory after --upstream-roban' >&2; exit 2; fi
            UPSTREAM="$2"
            shift 2
            ;;
        --stepper-window)
            if (( $# < 2 )) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]] || (( $2 < 3 )); then
                printf '%s\n' 'Expected a Stepper window of at least three frames' >&2
                exit 2
            fi
            STEPPER_WINDOW="$2"
            shift 2
            ;;
        --help|-h)
            printf '%s\n' 'Usage: bash run_lmm_frax.sh [--check-only] [--upstream-roban DIRECTORY] [--out NEW_DIRECTORY] [--stepper-window FRAMES]'
            exit 0
            ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

export ACP_ROBOT=roban
export ACP_ROBAN_DESCRIPTION="${ACP_ROBAN_DESCRIPTION:-$HOME/kuavo_amp_mjlab/src/mjlab/asset_zoo/robots/biped_s17_ver}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$ROOT"

if (( check_only )); then
    exec "$PYTHON" - "$UPSTREAM" <<'PY'
import sys
from pathlib import Path

import numpy as np

from supermimic.acp.footstep.motion_codec import upstream_holdout
from supermimic.acp.footstep.motion_matching import Database, Robot, upstream_manifest
from supermimic.acp.footstep.reference_planner import MATCH_HZ, POLICY_HZ, REFERENCE_OFFSETS

source = Path(sys.argv[1])
manifest = upstream_manifest(source)
if manifest.get("retargeter") != "frax" or manifest.get("robot") != "biped_s17_ver":
    raise SystemExit(f"Expected Frax S17-ver references at {source}")
robot = Robot()
db = Database.from_upstream_preview(robot, source, MATCH_HZ)
holdout = upstream_holdout(db)
training = ~np.isin(db.clip[db.candidates], holdout)
if not np.any(db.idle[db.candidates[training]]) or not np.intersect1d(db.running_frames, db.candidates[training]).size:
    raise SystemExit("The training split must include standing and high-step running frames")
print(f"{len(db.names)} clips, {len(db.pose)} frames at {MATCH_HZ} Hz; {training.sum()} eligible training frames")
print(f"Held out: {', '.join(db.names[i] for i in holdout)}")
print(f"Tracker offsets at {POLICY_HZ} Hz: {list(REFERENCE_OFFSETS)}")
PY
fi

if [[ -e "$OUT" ]]; then
    printf 'Output already exists: %s\n' "$OUT" >&2
    exit 1
fi

"$PYTHON" -m supermimic.acp.footstep.motion_codec --upstream-roban "$UPSTREAM" --out "$OUT/codec"
"$PYTHON" -m supermimic.acp.footstep.motion_stepper --codec "$OUT/codec" --window "$STEPPER_WINDOW" --out "$OUT/stepper"
"$PYTHON" -m supermimic.acp.footstep.learned_motion --upstream-roban "$UPSTREAM" \
    --codec "$OUT/codec" --stepper "$OUT/stepper/model.pt" --out "$OUT/model"
