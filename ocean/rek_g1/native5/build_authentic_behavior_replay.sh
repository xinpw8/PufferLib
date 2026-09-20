#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s EXACT_NATIVE_POLICY_OBJECT NEW_BUILD_DIRECTORY\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_object=$(realpath "$1")
[[ -f "$task_object" && ! -e "$2" ]] || { printf 'Missing native object or output exists\n' >&2; exit 2; }
mkdir "$2"
task_output=$(realpath "$2")
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
"${CXX:-g++}" -std=c++17 -O2 "$task_source/test_authentic_trajectory.cpp" -lcrypto -o "$task_output/test-authentic-trajectory"
"$task_output/test-authentic-trajectory" > "$task_output/cpu-test.stdout.txt"
"$task_cuda/bin/nvcc" -std=c++17 -O3 "-arch=${REK_CUDA_ARCH:-sm_121}" -I"$task_source" \
    "$task_source/replay_authentic_behavior.cu" "$task_object" \
    -L"$task_cuda/lib64" -Xlinker=-rpath -Xlinker="$task_cuda/lib64" \
    -lcublas -lcurand -lcrypto -o "$task_output/replay-authentic-behavior"
sha256sum "$task_source/"{authentic_trajectory.h,replay_authentic_behavior.cu,build_authentic_behavior_replay.sh,test_authentic_trajectory.cpp} \
    "$task_object" "$task_output/replay-authentic-behavior" > "$task_output/build-hashes.txt"
printf 'Built only; no GPU execution: %s/replay-authentic-behavior\n' "$task_output"
