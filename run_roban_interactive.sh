#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
exec ./run_roban.sh interactive "$@"
