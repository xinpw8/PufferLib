#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: build_decoder_row_exploration.sh PREPARED_PPO_BUILD EXACT_NATIVE_OBJECT NEW_OUTPUT' >&2;exit 2; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
prepared=$(realpath -- "$1")
native=$(realpath -- "$2")
output=$(realpath -m -- "$3")
test -f "$prepared/source/puffer5_bc_core.cuh"
test -f "$native"
test ! -e "$output"
mkdir -p "$output/source"
cp -a "$prepared/source/." "$output/source/"
cp "$here/decoder_row_exploration.cu" "$here/authentic_trajectory.h" "$here/device_storage.cuh" \
    "$here/native_policy.h" "$here/round_reward.h" "$here/build_decoder_row_exploration.sh" "$output/source/"
cuda=${CUDA_HOME:-/usr/local/cuda}
nccl=${NCCL_HOME:-/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl}
set -x
"$cuda/bin/nvcc" -std=c++17 -O2 --threads 1 -arch=sm_121 -I"$output/source" \
    -I"$nccl/include" -I"$cuda/include/cccl" "$output/source/decoder_row_exploration.cu" "$native" \
    -L"$cuda/lib64" -Xlinker=-rpath,"$cuda/lib64" -lcublas -lcurand -lcrypto -o "$output/decoder-row-exploration"
"$output/decoder-row-exploration" --cpu-self-test
sha256sum "$output/decoder-row-exploration" "$native" "$output/source/decoder_row_exploration.cu" \
    "$output/source/build_decoder_row_exploration.sh" "$output/source/algo.cu" "$output/source/puffer5_bc_core.cuh" \
    "$output/source/authentic_trajectory.h" "$output/source/native_policy.h" "$output/source/device_storage.cuh" \
    > "$output/build-hashes.sha256"
