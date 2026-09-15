#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 10 ]] || { printf 'Usage: %s EVALUATOR_BUILD RUNTIME_JSON CHECKPOINT SHA256 NEW_OUTPUT ARENAS ROUNDS_PER_SIDE SEED sampled|greedy bf16|fp32\n' "$0" >&2;exit 2; }
task_build=$(realpath "$1")
task_config=$(realpath "$2")
task_checkpoint=$(realpath "$3")
task_sha=$4
[[ "$task_sha" =~ ^[0-9a-f]{64}$ ]] || { printf 'Explicit checkpoint SHA256 required\n' >&2;exit 2; }
[[ "$(sha256sum "$task_checkpoint" | cut -d' ' -f1)" == "$task_sha" ]] || { printf 'Checkpoint hash mismatch\n' >&2;exit 2; }
mkdir "$5"
task_output=$(realpath "$5")
task_command=("$task_build/fast-policy-eval" "$task_config" "$task_checkpoint" "$task_sha" "$6" "$7" "$8" "$9" "${10}" "$task_output/matches.private.jsonl")
if [[ -n "${REK_EVAL_ROUND_FEATURE:-}" ]]; then
  [[ "$REK_EVAL_ROUND_FEATURE" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid diagnostic round feature\n' >&2;exit 2; }
  task_command=(env "REK_EVAL_ROUND_FEATURE=$REK_EVAL_ROUND_FEATURE" "${task_command[@]}")
fi
printf '%q ' "${task_command[@]}" > "$task_output/command.txt"
printf '\n' >> "$task_output/command.txt"
cp "$task_build/build-hashes.txt" "$task_output/build-hashes.txt"
cp "$task_build/elf-dependencies.txt" "$task_output/elf-dependencies.txt"
{ hostname;id;date -u --iso-8601=seconds;nvidia-smi -L;
  sha256sum "$task_config" "$task_checkpoint" "$task_build/fast-policy-eval"; } > "$task_output/provenance.txt"
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" timeout --signal=TERM --kill-after=10s 300s "${task_command[@]}" > "$task_output/summary.jsonl" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
cat "$task_output/summary.jsonl"
if [[ "$task_status" != 0 ]];then cat "$task_output/stderr.txt" >&2;fi
exit "$task_status"
