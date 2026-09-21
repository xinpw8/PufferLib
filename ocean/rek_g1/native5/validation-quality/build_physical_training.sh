#!/usr/bin/env bash
set -euo pipefail
task_stage=/home/spark-advantage/rek-training/physical-fall-exposure-20260921-r1
task_source=/home/spark-advantage/rek-training/normalized-falls-20260921-r1/source/ocean/rek_g1/native5
task_build=$task_stage/build-training-r1
task_base=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/build-native-v2
task_corrected=/home/spark-advantage/rek-training/physical-measurement-integration-20260921-r1/build-r2
task_cuda=/usr/local/cuda
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
task_nccl=/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl
task_raylib=/home/spark-advantage/pufferlib-5.0-wr64/raylib-5.5_linux_aarch64
cp -a "$task_stage/build-r2/source" "$task_stage/source-training-r1"
task_native=$task_stage/source-training-r1/ocean/rek_g1/native5
cp "$task_stage/runtime-training-r1.cu" "$task_native/runtime.cu"
[[ $(sha256sum "$task_native/runtime.cu" | cut -d' ' -f1) == b57235df70c3d10e0677d1617ba479c8d795cef27cdea8c8d98513760ed18dc6 ]]
"$task_cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_121 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false -Xcompiler=-fPIC -Xcompiler=-ffp-contract=off -I"$task_native/.." -I"$task_native" -I"$task_mujoco/include" -I"$task_cuda/include/cccl" -DREK_NATIVE5_MUJOCO_GPU=1 -c "$task_native/runtime.cu" -o "$task_stage/runtime-training-r1.o" > "$task_stage/runtime-training-r1.stdout" 2> "$task_stage/runtime-training-r1.stderr"
task_objects=("$task_stage/runtime-training-r1.o" "$task_corrected/measurement.o")
for task_object in "$task_base"/*.o;do
  case $(basename "$task_object") in runtime.o|measurement.o|pufferl.o)continue;;esac
  task_objects+=("$task_object")
done
export REK_NATIVE5_ENABLE_MUJOCO_GPU=1
unset REK_NATIVE5_COMPACT_AUTORESET
bash "$task_source/build_native.sh" "$task_build" "${task_objects[@]}" > "$task_stage/training-build.stdout" 2> "$task_stage/training-build.stderr"
g++ -std=c++17 -O2 -I"$task_mujoco/include" -c "$task_stage/cpu_physics_guard.cpp" -o "$task_build/cpu_physics_guard.o"
task_link=("$task_cuda/bin/nvcc" -arch=sm_121 -Xcompiler=-fopenmp
  "$task_build/pufferl.o" "$task_build/cpu_physics_guard.o" "${task_objects[@]}" "$task_raylib/lib/libraylib.a"
  -L"$task_cuda/lib64" -L"$task_nccl/lib" -L"$task_mujoco" -L"$task_cuda/lib64/stubs"
  -Xlinker=-rpath -Xlinker="$task_nccl/lib" -Xlinker=-rpath -Xlinker="$task_cuda/lib64" -Xlinker=-rpath -Xlinker="$task_mujoco"
  -Xlinker=--wrap=mj_step -Xlinker=--wrap=mj_step1 -Xlinker=--wrap=mj_step2 -Xlinker=--wrap=mj_forward -Xlinker=--wrap=mj_kinematics
  -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -l:libmujoco.so.3.7.0 -lcrypto -lGL -lm -lpthread -lomp5 -lcuda
  -o "$task_build/puffer-rek-physical-guarded")
printf '%q ' "${task_link[@]}" > "$task_build/guarded-link-command.txt"
printf '\n' >> "$task_build/guarded-link-command.txt"
"${task_link[@]}" > "$task_build/guarded-link.stdout" 2> "$task_build/guarded-link.stderr"
readelf -d "$task_build/puffer-rek-physical-guarded" > "$task_build/guarded-elf-dependencies.txt"
! grep -Ei 'libpython|libtorch' "$task_build/guarded-elf-dependencies.txt"
sha256sum "$task_build/puffer-rek-physical-guarded" "$task_build/pufferl.o" "$task_build/cpu_physics_guard.o" "$task_stage/cpu_physics_guard.cpp" "$task_source/puffer_env.cu" "$task_source/pufferlib5_temporal_credit.patch" "${task_objects[@]}" > "$task_build/guarded-source-manifest.txt"
