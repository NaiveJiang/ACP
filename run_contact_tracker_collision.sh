#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/run_contact_tracker_fast.sh" --controller collision --wm-only-iters 0 --ema 0.995 "$@"
