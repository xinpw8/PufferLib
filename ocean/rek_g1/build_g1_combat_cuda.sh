#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir=${1:?Usage: build_g1_combat_cuda.sh BUILD_DIRECTORY}
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
cc=${CC:-cc}
architecture=${REK_CUDA_ARCH:-native}
mkdir -p "$build_dir"
build_dir=$(cd "$build_dir" && pwd)
flags=(-std=c++20 -O2 "-arch=$architecture" --fmad=false --prec-div=true
    --prec-sqrt=true --ftz=false -Xcompiler=-fPIC -I"$source_dir")
objects=()
for module in native_motion_routes g1_strike_catalog; do
    "$cc" -std=c11 -O2 -fPIC -I"$source_dir" -c \
        "$source_dir/$module.c" -o "$build_dir/$module.host.o"
    objects+=("$build_dir/$module.host.o")
done
for module in g1_combat_tick g1_fight_state g1_fall_state g1_hit_detector; do
    "$nvcc" "${flags[@]}" -dc \
        "-DREK_G1_CUDA_SOURCE=\"$module.c\"" \
        "$source_dir/g1_cuda_device.cu" -o "$build_dir/$module.device.o"
    objects+=("$build_dir/$module.device.o")
done
"$nvcc" "${flags[@]}" -dc "$source_dir/g1_combat_cuda.cu" \
    -o "$build_dir/g1_combat_cuda.o"
"$nvcc" "${flags[@]}" -dc "$source_dir/g1_native_combat_cuda.cu" \
    -o "$build_dir/g1_native_combat_cuda.o"
"$nvcc" "${flags[@]}" -shared "${objects[@]}" \
    "$build_dir/g1_combat_cuda.o" "$build_dir/g1_native_combat_cuda.o" \
    -o "$build_dir/librek_g1_combat_cuda.so"
printf 'CUDA combat library: %s\n' "$build_dir/librek_g1_combat_cuda.so"
