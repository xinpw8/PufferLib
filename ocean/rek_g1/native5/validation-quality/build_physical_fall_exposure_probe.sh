#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: bash build_physical_fall_exposure_probe.sh NEW_BUILD_DIRECTORY RUNTIME_SOURCE_DIRECTORY\n' >&2; exit 2; }
task_runtime_source=$(realpath "$2")
task_out=$1
mkdir "$task_out"
task_out=$(realpath "$task_out")
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_corrected=/home/spark-advantage/rek-training/physical-measurement-integration-20260921-r1
task_base=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/build-native-v2
cp -a "$task_corrected/source-r2" "$task_out/source"
task_native=$task_out/source/ocean/rek_g1/native5
cp "$task_runtime_source/runtime.cu" "$task_runtime_source/normalized_reward.h" "$task_native/"
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
task_cuda=/usr/local/cuda
cp "$task_source/physical_fall_exposure_probe.cpp" "$task_source/build_physical_fall_exposure_probe.sh" "$task_source/run_physical_fall_exposure_probe.sh" "$task_out/"
task_objects=("$task_out/runtime.o" "$task_corrected/build-r2/measurement.o")
for task_object in "$task_base"/*.o; do
    case $(basename "$task_object") in pufferl.o|runtime.o|measurement.o) continue;; esac
    task_objects+=("$task_object")
done
task_compile=("$task_cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_121 -I"$task_native" -I"$task_native/.." -I"$task_mujoco/include" -c "$task_out/physical_fall_exposure_probe.cpp" -o "$task_out/probe.o")
task_runtime_compile=("$task_cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_121 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false -Xcompiler=-fPIC -Xcompiler=-ffp-contract=off -I"$task_native/.." -I"$task_native" -I"$task_mujoco/include" -I"$task_cuda/include/cccl" -DREK_NATIVE5_MUJOCO_GPU=1 -c "$task_native/runtime.cu" -o "$task_out/runtime.o")
task_link=("$task_cuda/bin/nvcc" -arch=sm_121 "$task_out/probe.o" "${task_objects[@]}" -L"$task_mujoco" -L"$task_cuda/lib64" -L"$task_cuda/lib64/stubs" -Xlinker=-rpath -Xlinker="$task_mujoco" -Xlinker=-rpath -Xlinker="$task_cuda/lib64" -Xlinker=--wrap=mj_step -Xlinker=--wrap=mj_step1 -Xlinker=--wrap=mj_step2 -Xlinker=--wrap=mj_forward -Xlinker=--wrap=mj_kinematics -lcudart -lcuda -lcublas -lcrypto -l:libmujoco.so.3.7.0 -lz -lm -lpthread -o "$task_out/physical-fall-exposure-probe")
{ printf '%q ' "${task_runtime_compile[@]}";printf '\n';printf '%q ' "${task_compile[@]}";printf '\n';printf '%q ' "${task_link[@]}";printf '\n'; } > "$task_out/build-command.txt"
"${task_runtime_compile[@]}" > "$task_out/build.stdout.txt" 2> "$task_out/build.stderr.txt"
"${task_compile[@]}" >> "$task_out/build.stdout.txt" 2>> "$task_out/build.stderr.txt"
"${task_link[@]}" >> "$task_out/build.stdout.txt" 2>> "$task_out/build.stderr.txt"
sha256sum "$task_out/physical-fall-exposure-probe" "$task_out/physical_fall_exposure_probe.cpp" "$task_native/runtime_api.h" "$task_native/runtime.cu" "$task_native/normalized_reward.h" "$task_native/measurement.cu" "${task_objects[@]}" > "$task_out/build-provenance.txt"
