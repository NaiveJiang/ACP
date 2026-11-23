#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out="${1:-logs/phase_pose_rollout_ft_001}"
mkdir "$out"
python=.venv_footstep/bin/python
tracker="${2:-logs/contact_tracker_train_planner_fixed_001/last.pt}"
clock="${3:-logs/omni_mlp_clock_train_001/model.pt}"
initial_pose="${4:-logs/phase_pose_full_train_001/model.pt}"
samples=()
# Match the commands used in the original closed-loop comparison, with two
# speeds each for forward and backward walking.
while read -r name vx vy wz; do
  for dt in 0.3 0.5; do
    for init in 0 2; do
      tag="${name}_${dt}_${init}"
      sample="$out/$tag.pt"
      log="$out/$tag.log"
      bash play_contact_tracker.sh --checkpoint "$tracker" \
        --category forward --split train --clip-index "$init" \
        --velocity "$vx" "$vy" "$wz" --query-dt "$dt" \
        --recorded-plan --auto-clock --clock-model "$clock" \
        --phase-pose-samples "$sample" --seconds 4 --no-loop --headless --no-pace \
        >"$log" 2>&1 || { tail -20 "$log"; exit 1; }
      if "$python" - "$log" "$sample" <<'PY'
import json
from pathlib import Path
import sys

log, samples = map(Path, sys.argv[1:])
r = json.loads(log.read_text().splitlines()[-1])
ok = (r["steps"] == 200 and r["terminations"] == 0
      and r["invalid_transitions"] == 0 and samples.is_file())
print(f"{samples.stem}: {'accepted' if ok else 'discarded'} {r['mean_velocity']}")
raise SystemExit(not ok)
PY
      then samples+=("$sample")
      fi
    done
  done
done <<'COMMANDS'
forward_slow 0.2 0 0
forward 0.5 0 0
backward_slow -0.16 0 0
backward -0.4 0 0
left 0 0.3 0
right 0 -0.24 0
ccw 0 0 0.5
cw 0 0 -0.7
forward_ccw 0.5 0 1
left_ccw 0 0.36 0.4
COMMANDS
if (("${#samples[@]}" == 0)); then
  printf '%s\n' 'No valid closed-loop samples; inspect the rollout logs.' >&2
  exit 1
fi
"$python" -m supermimic.acp.footstep.phase_pose_full \
  --checkpoint "$tracker" --init "$initial_pose" \
  --rollout-samples "${samples[@]}" --rollout-ratio 0.5 \
  --epochs 3 --batch-size 2048 --lr 0.0003 --device cuda:0 --out "$out/model.pt"
