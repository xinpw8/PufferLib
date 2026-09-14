#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 || $# == 4 ]] || { printf 'Usage: %s BUILD NEW_OUTPUT KERNEL_CATALOG_JSON [TOTAL_STEPS=65536]\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$(realpath "$1")
task_catalog=$(realpath "$3")
task_steps=${4:-65536}
task_runner=${REK_TRAINING_RUNNER:-$task_source/../run_frozen_training.sh}
task_nsys=${NSYS:-/usr/local/bin/nsys}
[[ "$task_steps" =~ ^[1-9][0-9]*$ && $((task_steps%8192)) == 0 && "$task_steps" -ge 16384 ]] || { printf 'Profile steps must be a multiple of8192 and at least16384\n' >&2; exit 2; }
[[ -f "$task_runner" && -x "$task_nsys" && -x "$task_build/puffer-rek-native5" ]] || { printf 'Native trainer, runner, or Nsight missing\n' >&2; exit 2; }
command -v sqlite3 >/dev/null
command -v node >/dev/null
mkdir "$2"
task_output=$(realpath "$2")
task_profile=("$task_nsys" profile --trace=cuda,nvtx,cublas --cuda-graph-trace=node
    --sample=none --cpuctxsw=none --backtrace=none --cuda-memory-usage=false --stats=false
    --force-overwrite=false --output="$task_output/training" bash "$task_runner" "$task_build" "$task_output/run" "$task_steps")
task_export=("$task_nsys" export --type=sqlite --force-overwrite=false --output="$task_output/training.sqlite" "$task_output/training.nsys-rep")
task_analyze=(node "$task_source/profile_training.mjs" --db "$task_output/training.sqlite" --catalog "$task_catalog"
    --out "$task_output/summary.json" --agents 512 --horizon 16 --requested-steps "$task_steps")
{ printf '%q ' "${task_profile[@]}";printf '\n';printf '%q ' "${task_export[@]}";printf '\n';printf '%q ' "${task_analyze[@]}";printf '\n'; } > "$task_output/commands.txt"
{ hostname;id;date -u --iso-8601=seconds;nvidia-smi -L;
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv;
  "$task_nsys" --version;sqlite3 --version;node --version;
  sha256sum "$task_build/puffer-rek-native5" "$task_runner" "$task_catalog" "$task_source/profile_training.mjs"; } > "$task_output/provenance.txt"
set +e
/usr/bin/time -v -o "$task_output/profile-process-timing.txt" timeout --signal=TERM --kill-after=15s 480s "${task_profile[@]}" > "$task_output/profile.stdout.txt" 2> "$task_output/profile.stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/profile-exit-code.txt"
[[ "$task_status" == 0 ]] || exit "$task_status"
[[ -f "$task_output/run/exit-code.txt" && "$(<"$task_output/run/exit-code.txt")" == 0 ]] || { printf 'Training did not complete successfully\n' >&2; exit 2; }
"${task_export[@]}" > "$task_output/export.stdout.txt" 2> "$task_output/export.stderr.txt"
"${task_analyze[@]}" > "$task_output/analysis.stdout.txt" 2> "$task_output/analysis.stderr.txt"
cat "$task_output/analysis.stdout.txt"
