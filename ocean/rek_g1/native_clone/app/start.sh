#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 1 || { echo 'Usage: start.sh absolute-fresh-run-directory' >&2; exit 2; }
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
exec node "$app_dir/launch_logged.cjs" "$1"
