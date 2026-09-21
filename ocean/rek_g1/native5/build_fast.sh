#!/usr/bin/env bash
# Build the reduced combat runtime with the same pinned native Puffer trainer.
set -euo pipefail
[[ $# == 1 ]] || { printf 'Usage: %s NEW_BUILD_DIRECTORY\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_root=$(cd "$task_source/../../.." && pwd)
task_g1=$(cd "$task_source/.." && pwd)
task_build=$1
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_nvcc=${NVCC:-$task_cuda/bin/nvcc}
task_arch=${REK_CUDA_ARCH:-sm_121}
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
task_raylib=${PUFFER5_RAYLIB:-/home/spark-advantage/pufferlib-5.0-wr64/raylib-5.5_linux_aarch64}
task_nccl=${PUFFER5_NCCL:-/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl}
REK_NATIVE5_COMPACT_AUTORESET=1 bash "$task_source/build_native.sh" "$task_build" --compile-trainer-only
task_build=$(realpath "$task_build")
"$task_nvcc" -std=c++17 -O3 "-arch=$task_arch" -Xcompiler=-fPIC \
    -I"$task_source" -I"$task_g1" -I"$task_root/vendor" \
    -I"$task_cuda/include/cccl" -c "$task_source/fast_runtime.cu" -o "$task_build/fast_runtime.o"
"${CXX:-g++}" -std=c++17 -O3 -fPIC -I"$task_source" -I"$task_g1" \
    -I"$task_root/vendor" -I"$task_mujoco/include" -I"$task_cuda/include" \
    -c "$task_source/fast_assets.cpp" -o "$task_build/fast_assets_loader.o"
"$task_nvcc" -std=c++17 -O3 "-arch=$task_arch" -Xcompiler=-fPIC \
    -I"$task_source" -I"$task_g1" -I"$task_cuda/include/cccl" \
    -c "$task_source/native_policy.cu" -o "$task_build/native_policy.o"
"${CC:-gcc}" -std=c11 -O2 -fPIC -c "$task_root/vendor/cJSON.c" -o "$task_build/cJSON.o"
for task_unit in g1_strike_catalog native_motion_routes; do
    "${CC:-gcc}" -std=c11 -O2 -fPIC -c "$task_g1/$task_unit.c" -o "$task_build/$task_unit.o"
done
"${LD:-ld}" -r "$task_build/fast_assets_loader.o" "$task_build/g1_strike_catalog.o" "$task_build/native_motion_routes.o" -o "$task_build/fast_assets.o"
task_objects=("$task_build/fast_runtime.o" "$task_build/fast_assets.o" "$task_build/native_policy.o" "$task_build/cJSON.o")
"$task_nvcc" "-arch=$task_arch" -Xcompiler=-fopenmp "$task_build/pufferl.o" \
    "${task_objects[@]}" "$task_raylib/lib/libraylib.a" \
    -L"$task_cuda/lib64" -L"$task_nccl/lib" -L"$task_mujoco" \
    -Xlinker=-rpath -Xlinker="$task_cuda/lib64" \
    -Xlinker=-rpath -Xlinker="$task_nccl/lib" \
    -Xlinker=-rpath -Xlinker="$task_mujoco" \
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
    -l:libmujoco.so.3.7.0 -lcrypto -lGL -lm -lpthread -lomp5 \
    -o "$task_build/puffer-rek-native5"
printf '%s\n' "${task_objects[@]}" > "$task_build/runtime-objects.txt"
readelf -d "$task_build/puffer-rek-native5" > "$task_build/elf-dependencies.txt"
if rg -qi '(libpython|libtorch)' "$task_build/elf-dependencies.txt"; then
    printf 'Unexpected interpreter/framework link dependency\n' >&2; exit 2
fi
{ printf 'backend=semantic_cuda\ncontrol_hz=50\ncpu_physics=0\npython_runtime=0\n';
  sha256sum "$task_source/fast_runtime.cu" "$task_source/fast_assets.cpp" \
    "$task_source/fast_assets.h" "$task_source/primitive_contacts.cuh" \
    "$task_source/primitive_motion.cuh" "$task_source/native_contact_geometry.h" \
    "$task_source/contact_potential.h" "$task_source/contact_potential_loader.h" \
    "$task_source/round_reward.h" "$task_source/policy_feature_mask.h" "$task_source/owned_yaw_observation.h" \
    "$task_source/action_cadence.h" "$task_source/keyboard_yaw.h" "$task_source/contact_entry.h" "$task_source/contact_velocity.h" \
    "$task_build/puffer-rek-native5"; } > "$task_build/fast-build.txt"
printf 'Built reduced GPU trainer: %s/puffer-rek-native5\n' "$task_build"
