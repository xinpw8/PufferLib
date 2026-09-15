#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s BUILD NEW_OUTPUT\n' "$0" >&2;exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$(realpath "$1")
mkdir "$2"
task_output=$(realpath "$2")
task_profile=(/usr/local/bin/nsys profile --trace=cuda,nvtx,cublas --cuda-graph-trace=node
    --sample=none --cpuctxsw=none --backtrace=none --cuda-memory-usage=false --stats=false
    --force-overwrite=false --output="$task_output/training" bash "$task_source/run_fast_training.sh"
    "$task_build" "$task_output/run" 1048576 512 16)
task_export=(/usr/local/bin/nsys export --type=sqlite --force-overwrite=false --output="$task_output/training.sqlite" "$task_output/training.nsys-rep")
task_analysis=(node "$task_source/fast_profile.mjs" "$task_output/training.sqlite" "$task_output/summary.json")
{ printf '%q ' "${task_profile[@]}";printf '\n';printf '%q ' "${task_export[@]}";printf '\n';printf '%q ' "${task_analysis[@]}";printf '\n'; } > "$task_output/commands.sh"
{ hostname;id;date -u --iso-8601=seconds;/usr/local/bin/nsys --version;
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv;
  sha256sum "$task_source/run_fast_training.sh" "$task_source/fast_profile.sh" "$task_source/fast_profile.mjs" "$task_build/puffer-rek-native5"; } > "$task_output/provenance.txt"
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" timeout --signal=TERM --kill-after=10s 180s "${task_profile[@]}" > "$task_output/profile.stdout.txt" 2> "$task_output/profile.stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
[[ $task_status == 0 ]] || exit "$task_status"
[[ $(<"$task_output/run/exit-code.txt") == 0 ]] || exit 2
"${task_export[@]}" > "$task_output/export.stdout.txt" 2> "$task_output/export.stderr.txt"
"${task_analysis[@]}" > "$task_output/analysis.stdout.txt" 2> "$task_output/analysis.stderr.txt"
cat "$task_output/analysis.stdout.txt"
