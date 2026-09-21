#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 ]] || { printf 'Usage: %s NEW_OUTPUT\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir "$1"
task_out=$(realpath "$1")
task_cxx=${CXX:-g++}
task_nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
{
    date -u --iso-8601=seconds
    hostname
    "$task_cxx" --version
    sha256sum "$task_source/normalized_reward.h" "$task_source/test_normalized_reward.cpp" \
        "$task_source/test_normalized_reward.cu" \
        "$task_source/test_normalized_reward.sh" "$task_source/../g1_fall_state.h" \
        "$task_source/../g1_fall_state.c"
} > "$task_out/provenance.txt"
"$task_cxx" -std=c++20 -O2 -Wall -Wextra -c "$task_source/../g1_fall_state.c" \
    -o "$task_out/fall-state.o" > "$task_out/build-fall.stdout.txt" 2> "$task_out/build-fall.stderr.txt"
task_cpu=("$task_cxx" -std=c++17 -O2 -Wall -Wextra -Werror \
    -fsanitize=undefined -fno-sanitize-recover=all "$task_source/test_normalized_reward.cpp" \
    "$task_out/fall-state.o" -o "$task_out/test-cpu")
printf '%q ' "${task_cpu[@]}" > "$task_out/build-cpu.command.txt"
"${task_cpu[@]}" > "$task_out/build-cpu.stdout.txt" 2> "$task_out/build-cpu.stderr.txt"
"$task_out/test-cpu" > "$task_out/test-cpu.stdout.jsonl" 2> "$task_out/test-cpu.stderr.txt"
cat "$task_out/test-cpu.stdout.jsonl"
sha256sum "$task_out/test-cpu" >> "$task_out/provenance.txt"
if [[ ${REK_NORMALIZED_REWARD_TEST_CUDA:-0} == 1 ]]; then
    "$task_nvcc" --version >> "$task_out/provenance.txt"
    nvidia-smi -L >> "$task_out/provenance.txt"
    task_gpu=("$task_nvcc" -std=c++17 -O3 -arch="${CUDA_ARCH:-sm_121}" \
        "$task_source/test_normalized_reward.cu" "$task_out/fall-state.o" -o "$task_out/test-cuda")
    printf '%q ' "${task_gpu[@]}" > "$task_out/build-cuda.command.txt"
    "${task_gpu[@]}" > "$task_out/build-cuda.stdout.txt" 2> "$task_out/build-cuda.stderr.txt"
    "$task_out/test-cuda" > "$task_out/test-cuda.stdout.jsonl" 2> "$task_out/test-cuda.stderr.txt"
    cat "$task_out/test-cuda.stdout.jsonl"
    sha256sum "$task_out/test-cuda" >> "$task_out/provenance.txt"
fi
