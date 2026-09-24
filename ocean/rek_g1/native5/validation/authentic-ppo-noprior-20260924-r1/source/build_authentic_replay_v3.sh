#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s EXACT_NATIVE_POLICY_OBJECT NEW_BUILD_DIRECTORY\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_object=$(realpath "$1")
[[ -f "$task_object" && ! -e "$2" ]] || exit 2
mkdir "$2"
task_output=$(realpath "$2")
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_object_hash=$(sha256sum "$task_object" | awk '{print $1}')
g++ -std=c++17 -O2 "$task_source/test_authentic_trajectory.cpp" -lcrypto -o "$task_output/test-authentic-trajectory"
"$task_output/test-authentic-trajectory" > "$task_output/cpu-test.stdout.txt"
node --test "$task_source/authentic_trajectory_data.test.cjs" "$task_source/authentic_trajectory_v3.test.cjs" > "$task_output/export-tests.stdout.txt"
"$task_cuda/bin/nvcc" -std=c++17 -O3 "-arch=${REK_CUDA_ARCH:-sm_121}" -I"$task_source" \
  "-DREK_AUTHENTIC_NATIVE_OBJECT_SHA256=\"$task_object_hash\"" \
  "$task_source/replay_authentic_behavior.cu" "$task_object" \
  -L"$task_cuda/lib64" -Xlinker=-rpath -Xlinker="$task_cuda/lib64" -lcublas -lcurand -lcrypto \
  -o "$task_output/replay-authentic-behavior-v3"
sha256sum "$task_source/"{authentic_trajectory.h,authentic_trajectory_data.cjs,authentic_trajectory_v3.cjs,replay_authentic_behavior.cu,build_authentic_replay_v3.sh,authentic_trajectory_v3.test.cjs} \
  "$task_object" "$task_output/replay-authentic-behavior-v3" > "$task_output/build-hashes.txt"
printf 'Built only, no GPU execution: %s/replay-authentic-behavior-v3\n' "$task_output"
