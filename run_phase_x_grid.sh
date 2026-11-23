#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out="${1:-logs/phase_x_grid_train_001}"
mkdir "$out"
python=.venv_footstep/bin/python
samples=()
for v in 0.2 0.5; do
  case "$v" in 0.2) clip=0021_forward_r0;; 0.5) clip=0084_forward_r0;; esac
  for dt in 0.3 0.5; do
    for init in 0 2; do
      phase=0
      if [[ "$init" == 2 ]]; then phase=0.4; fi
      sample="$out/train_${v}_${dt}_${init}.pt"
      bash play_contact_tracker.sh --checkpoint logs/contact_tracker_train_planner_fixed_001/last.pt \
        --category forward --split train --clip-index "$init" --phase "$phase" \
        --velocity "$v" 0 0 --query-dt "$dt" --recorded-plan --recorded-case "$clip" \
        --phase-x-samples "$sample" --seconds 4 --no-loop --headless --no-pace
      samples+=("$sample")
    done
  done
done
"$python" -m supermimic.acp.footstep.phase_x "${samples[@]}" --out "$out/model.pt"
