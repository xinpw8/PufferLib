#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 || $# == 3 ]] || { printf 'Usage: %s EXISTING_FAST_BUILD NEW_OUTPUT [223_BYTE_MASK]\n' "$0" >&2;exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$(realpath "$1")
task_out=$2
mkdir "$task_out"
task_out=$(realpath "$task_out")
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
task_nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
task_command=("$task_nvcc" -std=c++17 -O3 -arch=sm_121
    -I"$task_source/.." -I"$task_source/../../../vendor"
    "$task_source/test_policy_feature_mask.cu" "$task_build/fast_assets.o" "$task_build/cJSON.o"
    -L"$task_mujoco" -Xlinker=-rpath -Xlinker="$task_mujoco" -l:libmujoco.so.3.7.0 -lcrypto
    -o "$task_out/test-policy-feature-mask")
printf '%q ' "${task_command[@]}" > "$task_out/build.command.txt"
{ date -u --iso-8601=seconds;hostname;"$task_nvcc" --version;nvidia-smi -L;
  sha256sum "$task_source/test_policy_feature_mask.cu" "$task_source/fast_runtime.cu" "$task_source/policy_feature_mask.h" "$task_build/fast_assets.o" "$task_build/cJSON.o";
} > "$task_out/provenance.txt"
"${task_command[@]}" > "$task_out/build.stdout.txt" 2> "$task_out/build.stderr.txt"
task_test=("$task_out/test-policy-feature-mask")
if [[ $# == 3 ]];then task_test+=("$(realpath "$3")");sha256sum "$3" >> "$task_out/provenance.txt";fi
printf '%q ' "${task_test[@]}" > "$task_out/test.command.txt"
set +e
"${task_test[@]}" > "$task_out/test.stdout.jsonl" 2> "$task_out/test.stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_out/exit-code.txt"
sha256sum "$task_out/test-policy-feature-mask" >> "$task_out/provenance.txt"
cat "$task_out/test.stdout.jsonl" "$task_out/test.stderr.txt"
exit "$task_status"
