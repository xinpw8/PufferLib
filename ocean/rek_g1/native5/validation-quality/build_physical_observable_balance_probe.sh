#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s NEW_BUILD RUNTIME_SOURCE\n' "$0" >&2; exit 2; }
task_source=$(cd -- "$(dirname -- "$0")" && pwd)
task_runtime=$(realpath "$2")
mkdir "$1"; task_output=$(realpath "$1")
task_corrected=/home/spark-advantage/rek-training/physical-measurement-integration-20260921-r1
task_base=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/build-native-v2
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
task_cuda=/usr/local/cuda
cp -a "$task_corrected/source-r2" "$task_output/source"
task_native=$task_output/source/ocean/rek_g1/native5
cp "$task_runtime/runtime.cu" "$task_runtime/runtime_api.h" "$task_runtime/normalized_reward.h" "$task_runtime/observable_balance.h" "$task_native/"
cp "$task_source/physical_observable_balance_probe.cu" "$task_source/build_physical_observable_balance_probe.sh" "$task_source/run_physical_observable_balance_probe.sh" "$task_output/"
task_objects=("$task_corrected/build-r2/measurement.o")
for task_object in "$task_base"/*.o; do
    case $(basename "$task_object") in pufferl.o|runtime.o|measurement.o) continue;; esac
    task_objects+=("$task_object")
done
task_flags=(-std=c++17 -O2 -arch=sm_121 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false -Xcompiler=-fPIC -Xcompiler=-ffp-contract=off -I"$task_native/.." -I"$task_native" -I"$task_mujoco/include" -I"$task_cuda/include/cccl" -DREK_NATIVE5_MUJOCO_GPU=1)
task_runtime_compile=("$task_cuda/bin/nvcc" "${task_flags[@]}" -c "$task_native/runtime.cu" -o "$task_output/runtime.o")
# The probe includes runtime.cu so CPU fixtures exercise the production adapter.
task_probe_compile=("$task_cuda/bin/nvcc" "${task_flags[@]}" -c "$task_output/physical_observable_balance_probe.cu" -o "$task_output/probe.o")
task_link=("$task_cuda/bin/nvcc" -arch=sm_121 "$task_output/probe.o" "${task_objects[@]}" -L"$task_mujoco" -L"$task_cuda/lib64" -L"$task_cuda/lib64/stubs" -Xlinker=-rpath -Xlinker="$task_mujoco" -Xlinker=-rpath -Xlinker="$task_cuda/lib64" -Xlinker=--wrap=mj_step -Xlinker=--wrap=mj_step1 -Xlinker=--wrap=mj_step2 -Xlinker=--wrap=mj_forward -Xlinker=--wrap=mj_kinematics -lcudart -lcuda -lcublas -lcrypto -l:libmujoco.so.3.7.0 -lz -lm -lpthread -o "$task_output/physical-observable-balance-probe")
{ printf '%q ' "${task_runtime_compile[@]}"; printf '\n'; printf '%q ' "${task_probe_compile[@]}"; printf '\n'; printf '%q ' "${task_link[@]}"; printf '\n'; } > "$task_output/build-command.txt"
"${task_runtime_compile[@]}" > "$task_output/build.stdout.txt" 2> "$task_output/build.stderr.txt"
"${task_probe_compile[@]}" >> "$task_output/build.stdout.txt" 2>> "$task_output/build.stderr.txt"
"${task_link[@]}" >> "$task_output/build.stdout.txt" 2>> "$task_output/build.stderr.txt"
"$task_output/physical-observable-balance-probe" --cpu-self-test > "$task_output/cpu-test.jsonl"
sha256sum "$task_output/physical-observable-balance-probe" "$task_output/physical_observable_balance_probe.cu" "$task_native/runtime.cu" "$task_native/runtime_api.h" "$task_native/observable_balance.h" "$task_output/runtime.o" "${task_objects[@]}" > "$task_output/build-provenance.txt"
cat "$task_output/cpu-test.jsonl"
