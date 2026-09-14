#!/usr/bin/env bash
set -euo pipefail
[[ $# == 5 ]] || { printf 'Usage: %s NEW_OUTPUT PROBE_BINARY REK_MODEL_XML KERNEL_CATALOG_JSON CONDITIONAL_PTX\n' "$0" >&2; exit 2; }
task_binary=$(realpath "$2")
task_model=$(realpath "$3")
task_catalog=$(realpath "$4")
task_conditional=$(realpath "$5")
task_cuda=${CUDA_HOME:-/usr/local/cuda}
[[ -x "$task_cuda/bin/compute-sanitizer" ]] || { printf 'CUDA compute-sanitizer is unavailable\n' >&2; exit 2; }
mkdir "$1"
task_output=$(realpath "$1")
task_hash=$(sha256sum "$task_conditional" | cut -d' ' -f1)
task_command=(timeout --signal=TERM --kill-after=10s 180s "$task_cuda/bin/compute-sanitizer"
    --tool memcheck --error-exitcode 99 --log-file "$task_output/memcheck.txt"
    "$task_binary" "$task_model" "$task_catalog" "$task_conditional" "$task_hash" 10)
printf '%q ' "${task_command[@]}" > "$task_output/command.txt"
printf '\n' >> "$task_output/command.txt"
{ date -u --iso-8601=seconds; hostname; id; uname -m; nvidia-smi -L;
  "$task_cuda/bin/compute-sanitizer" --version;
  sha256sum "$task_binary" "$task_model" "$task_catalog" "$task_conditional"; } > "$task_output/provenance.txt"
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" "${task_command[@]}" > "$task_output/result.jsonl" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
cat "$task_output/memcheck.txt" "$task_output/stderr.txt"
exit "$task_status"
