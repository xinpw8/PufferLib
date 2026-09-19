#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 ]] || { printf 'Usage: %s NEW_OUTPUT\n' "$0" >&2;exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir "$1"
task_out=$(realpath "$1")
task_nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
{ date -u --iso-8601=seconds;hostname;id;"$task_nvcc" --version;nvidia-smi -L;
  sha256sum "$task_source/round_reward.h" "$task_source/test_round_reward.cpp" "$task_source/test_round_reward.cu";
} > "$task_out/provenance.txt"
task_cpu=("${CXX:-g++}" -std=c++17 -O2 "$task_source/test_round_reward.cpp" -o "$task_out/test-cpu")
task_gpu=("$task_nvcc" -std=c++17 -O3 -arch=sm_121 "$task_source/test_round_reward.cu" -o "$task_out/test-cuda")
for task_kind in cpu gpu;do
    declare -n task_command="task_$task_kind"
    printf '%q ' "${task_command[@]}" > "$task_out/build-$task_kind.command.txt"
    "${task_command[@]}" > "$task_out/build-$task_kind.stdout.txt" 2> "$task_out/build-$task_kind.stderr.txt"
done
for task_kind in cpu cuda;do
    set +e
    "$task_out/test-$task_kind" > "$task_out/test-$task_kind.stdout.jsonl" 2> "$task_out/test-$task_kind.stderr.txt"
    task_status=$?
    set -e
    printf '%s\n' "$task_status" > "$task_out/test-$task_kind.exit-code.txt"
    cat "$task_out/test-$task_kind.stdout.jsonl" "$task_out/test-$task_kind.stderr.txt"
    [[ $task_status == 0 ]] || exit "$task_status"
done
sha256sum "$task_out/test-cpu" "$task_out/test-cuda" >> "$task_out/provenance.txt"
