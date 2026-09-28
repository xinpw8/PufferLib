#!/usr/bin/env bash
set -euo pipefail
umask 077
task_root=/home/spark-advantage/rek-training/rek-native-clone-20260927-r1
task_g1=${REK_CLONE_SOURCE_DIR:-$task_root/source}/ocean/rek_g1
task_source=$task_g1/native5
task_base=/home/spark-advantage/rek-training/physical-bot1-integration-20260921-r1/build-r1
task_build=${REK_CLONE_BUILD_DIR:-$task_root/build}
task_cuda=/usr/local/cuda
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
test ! -e "$task_build"
sha256sum --quiet -c "$task_base/reused-object-pins.sha256"
mkdir "$task_build"
trap 'task_status=$?; printf "%s\n" "$task_status" > "$task_build/exit-code.txt"' EXIT
task_objects=()
while read -r task_hash task_object; do
    case $(basename "$task_object") in native_policy.o|robot_state.o|sonic_motion_composer_native.device.o|g1_native_combat_cuda.o) continue;; esac
    task_objects+=("$task_object")
done < "$task_base/reused-object-pins.sha256"
sha256sum "${task_objects[@]}" > "$task_build/reused-inputs.sha256"
task_flags=(-std=c++17 -O2 -arch=sm_121 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false
    -Xcompiler=-fPIC -Xcompiler=-ffp-contract=off -I"$task_g1" -I"$task_source"
    -I"$task_mujoco/include" -I"$task_cuda/include/cccl" -DREK_NATIVE5_MUJOCO_GPU=1)
for task_unit in runtime.cu native_policy.cu eval_worker.cpp motion_assets.cu robot_state.cu; do
    task_name=${task_unit%.*}
    printf 'Compiling %s\n' "$task_unit"
    "$task_cuda/bin/nvcc" "${task_flags[@]}" -c "$task_source/$task_unit" -o "$task_build/$task_name.o" \
        > "$task_build/$task_name.stdout.txt" 2> "$task_build/$task_name.stderr.txt"
    task_objects+=("$task_build/$task_name.o")
done
printf 'Compiling ordered native scheduler and composer\n'
"$task_cuda/bin/nvcc" "${task_flags[@]}" -dc "$task_g1/g1_semantic_scheduler_cuda.cu" -o "$task_build/g1_semantic_scheduler_cuda.o" \
    > "$task_build/g1_semantic_scheduler_cuda.stdout.txt" 2> "$task_build/g1_semantic_scheduler_cuda.stderr.txt"
"$task_cuda/bin/nvcc" "${task_flags[@]}" -dc '-DREK_G1_CUDA_SOURCE="sonic_motion_composer_native.c"' "$task_g1/g1_cuda_device.cu" -o "$task_build/sonic_motion_composer_native.device.o" \
    > "$task_build/sonic_motion_composer_native.stdout.txt" 2> "$task_build/sonic_motion_composer_native.stderr.txt"
"$task_cuda/bin/nvcc" "${task_flags[@]}" -dc "$task_g1/g1_native_combat_cuda.cu" -o "$task_build/g1_native_combat_cuda.o" \
    > "$task_build/g1_native_combat_cuda.stdout.txt" 2> "$task_build/g1_native_combat_cuda.stderr.txt"
task_objects+=("$task_build/g1_semantic_scheduler_cuda.o" "$task_build/sonic_motion_composer_native.device.o" "$task_build/g1_native_combat_cuda.o")
"$task_cuda/bin/nvcc" -arch=sm_121 "${task_objects[@]}" \
    -L"$task_mujoco" -L"$task_cuda/lib64" -L"$task_cuda/lib64/stubs" \
    -Xlinker=-rpath -Xlinker="$task_mujoco" -Xlinker=-rpath -Xlinker="$task_cuda/lib64" \
    -lcudart -lcuda -lcublas -lcrypto -l:libmujoco.so.3.7.0 -lEGL -lGL -lz -lm -lpthread \
    -o "$task_build/rek-native-clone" > "$task_build/link.stdout.txt" 2> "$task_build/link.stderr.txt"
sha256sum --quiet -c "$task_build/reused-inputs.sha256"
sha256sum "$task_build"/*.o "$task_build/rek-native-clone" > "$task_build/outputs.sha256"
printf 'Native clone compiled and linked\n'
