#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir=${1:?Usage: build_g1_motion_cuda.sh BUILD_DIRECTORY}
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
architecture=${REK_CUDA_ARCH:-native}
mkdir -p "$build_dir"
build_dir=$(cd "$build_dir" && pwd)
flags=(-std=c++20 -O2 "-arch=$architecture" --fmad=false --prec-div=true
    --prec-sqrt=true --ftz=false -Xcompiler=-fPIC -I"$source_dir")
objects=()
for module in puffer_action_adapter g1_semantic_action_table native_locomotion_command sonic_motion_composer_native \
        sonic_motion_composer_libm_candidate sonic_motion_entry_matcher_native; do
    "$nvcc" "${flags[@]}" -dc "-DREK_G1_CUDA_SOURCE=\"$module.c\"" \
        "$source_dir/g1_cuda_device.cu" -o "$build_dir/$module.device.o"
    objects+=("$build_dir/$module.device.o")
done
"$nvcc" "${flags[@]}" -dc "$source_dir/g1_motion_cuda.cu" \
    -o "$build_dir/g1_motion_cuda.o"
"$nvcc" "${flags[@]}" -dc "$source_dir/g1_semantic_scheduler_cuda.cu" \
    -o "$build_dir/g1_semantic_scheduler_cuda.o"
"$nvcc" "${flags[@]}" -shared "${objects[@]}" \
    "$build_dir/g1_motion_cuda.o" "$build_dir/g1_semantic_scheduler_cuda.o" \
    -o "$build_dir/librek_g1_motion_cuda.so"
printf 'CUDA motion library: %s\n' "$build_dir/librek_g1_motion_cuda.so"
