#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s EXACT_NATIVE_POLICY_BUILD NEW_BUILD_DIRECTORY\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_objects=$(realpath "$1")
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_arch=${REK_CUDA_ARCH:-sm_121}
for task_object in native_policy.o cJSON.o; do
  [[ -f "$task_objects/$task_object" ]] || { printf 'Missing %s\n' "$task_object" >&2; exit 2; }
done
mkdir "$2"
task_output=$(realpath "$2")
task_protocol=("${CXX:-g++}" -std=c++17 -O2 -DREK_LIVE_PROTOCOL_TEST -x c++
  "$task_source/live_policy_worker.cu" -x none "$task_objects/cJSON.o"
  -o "$task_output/live-policy-protocol-test")
printf '%q ' "${task_protocol[@]}" > "$task_output/protocol-build-command.txt"
printf '\n' >> "$task_output/protocol-build-command.txt"
"${task_protocol[@]}" > "$task_output/protocol-build.stdout.txt" 2> "$task_output/protocol-build.stderr.txt"
node "$task_source/live_policy_worker.test.cjs" protocol "$task_output/live-policy-protocol-test" \
  > "$task_output/protocol-test.stdout.json" 2> "$task_output/protocol-test.stderr.txt"
REK_OBSERVATION_SCHEMA=rek.native5.observable_balance.v1 node "$task_source/live_policy_worker.test.cjs" protocol "$task_output/live-policy-protocol-test" \
  > "$task_output/balance-protocol-test.stdout.json" 2> "$task_output/balance-protocol-test.stderr.txt"
task_native=("$task_cuda/bin/nvcc" -std=c++17 -O3 "-arch=$task_arch" -I"$task_source"
  "$task_source/live_policy_worker.cu" "$task_objects/native_policy.o" "$task_objects/cJSON.o"
  -L"$task_cuda/lib64" -Xlinker=-rpath -Xlinker="$task_cuda/lib64"
  -lcublas -lcurand -lcrypto -o "$task_output/live-policy-worker")
printf '%q ' "${task_native[@]}" > "$task_output/build-command.txt"
printf '\n' >> "$task_output/build-command.txt"
"${task_native[@]}" > "$task_output/build.stdout.txt" 2> "$task_output/build.stderr.txt"
readelf -d "$task_output/live-policy-worker" > "$task_output/elf-dependencies.txt"
if rg -qi '(libpython|libtorch|libonnxruntime|libmujoco)' "$task_output/elf-dependencies.txt"; then
  printf 'Unexpected interpreter, framework, or physics dependency\n' >&2; exit 2
fi
sha256sum "$task_source/live_policy_worker.cu" "$task_source/live_policy_worker.test.cjs" \
  "$task_source/policy_feature_mask.h" "$task_source/owned_yaw_observation.h" "$task_source/observable_balance.h" "$task_source/live_policy_selection.test.cjs" \
  "$task_objects/native_policy.o" "$task_objects/cJSON.o" \
  "$task_output/live-policy-protocol-test" "$task_output/live-policy-worker" > "$task_output/build-hashes.txt"
printf 'Built native CUDA live policy worker: %s/live-policy-worker\n' "$task_output"
