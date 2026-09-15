#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s EXACT_FAST_TRAINING_BUILD NEW_EVALUATOR_BUILD\n' "$0" >&2;exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$(realpath "$1")
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
for task_object in fast_runtime.o fast_assets.o native_policy.o cJSON.o;do [[ -f "$task_build/$task_object" ]] || { printf 'Missing %s\n' "$task_object" >&2;exit 2; };done
mkdir "$2"
task_output=$(realpath "$2")
task_command=("$task_cuda/bin/nvcc" -std=c++17 -O3 -arch=sm_121 -I"$task_source"
  "$task_source/fast_policy_eval.cu" "$task_build/fast_runtime.o" "$task_build/fast_assets.o"
  "$task_build/native_policy.o" "$task_build/cJSON.o" -L"$task_mujoco"
  -Xlinker=-rpath -Xlinker="$task_mujoco" -l:libmujoco.so.3.7.0 -lcrypto -lcublas -lcurand
  -o "$task_output/fast-policy-eval")
printf '%q ' "${task_command[@]}" > "$task_output/build-command.txt"
printf '\n' >> "$task_output/build-command.txt"
"${task_command[@]}" > "$task_output/build.stdout.txt" 2> "$task_output/build.stderr.txt"
sha256sum "$task_source/fast_policy_eval.cu" "$task_build/fast_runtime.o" "$task_build/fast_assets.o" \
  "$task_build/native_policy.o" "$task_build/cJSON.o" "$task_output/fast-policy-eval" > "$task_output/build-hashes.txt"
readelf -d "$task_output/fast-policy-eval" > "$task_output/elf-dependencies.txt"
if rg -qi '(libpython|libtorch|libonnxruntime)' "$task_output/elf-dependencies.txt";then printf 'Unexpected inference dependency\n' >&2;exit 2;fi
printf 'Built native batched evaluator: %s/fast-policy-eval\n' "$task_output"
