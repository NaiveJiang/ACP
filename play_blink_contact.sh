#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CHECKPOINT="${1:-logs/blink_contact_train_003/last.pt}"
if [[ $# -gt 0 ]]; then shift; fi
VARIANT="online"
EVAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ema) VARIANT="ema"; EVAL_ARGS+=("$1"); shift ;;
    --ground-friction|--box-friction)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a coefficient" >&2
        exit 2
      fi
      EVAL_ARGS+=("$1" "$2")
      shift 2
      ;;
    *) break ;;
  esac
done
UV="$ROOT/.venv_preview/bin/uv"
OUT="$(mktemp -d "$ROOT/logs/blink_play_${VARIANT}_XXXXXX")"
export ACP_ROBOT=roban
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "Recording $VARIANT policy from $CHECKPOINT into $OUT"
"$UV" run --no-project --python "$ROOT/.venv_blink/bin/python" python \
  -m supermimic.acp.evaluate --ckpt "$CHECKPOINT" "${EVAL_ARGS[@]}" \
  --record "$OUT/rollout.npz" --out "$OUT/evaluation.json"
echo "Recorded results: $OUT/evaluation.json"
exec "$UV" run --no-project --python "$ROOT/.venv_preview/bin/python" \
  tools/play_motion_matching.py --rollout "$OUT/rollout.npz" "$@"
