#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: build_owned_yaw_migration.sh PREPARED_PPO_BUILD EXACT_NATIVE_OBJECT NEW_OUTPUT' >&2;exit 2; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
prepared=$(realpath -- "$1")
native=$(realpath -- "$2")
output=$(realpath -m -- "$3")
test -f "$prepared/source/puffer5_bc_core.cuh"
test -f "$native"
test ! -e "$output"
mkdir -p "$output/source"
cp -a "$prepared/source/." "$output/source/"
cp "$here/owned_yaw_migration.cu" "$here/owned_yaw_trajectory.h" "$here/owned_yaw_observation.h" \
    "$here/authentic_trajectory.h" "$here/device_storage.cuh" "$here/native_policy.h" "$here/round_reward.h" \
    "$here/build_owned_yaw_migration.sh" "$output/source/"
cuda=${CUDA_HOME:-/usr/local/cuda}
nccl=${NCCL_HOME:-/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl}
set -x
"$cuda/bin/nvcc" -std=c++17 -O2 --threads 1 -arch=sm_121 -I"$output/source" \
    -I"$nccl/include" -I"$cuda/include/cccl" "$output/source/owned_yaw_migration.cu" "$native" \
    -L"$cuda/lib64" -Xlinker=-rpath,"$cuda/lib64" -lcublas -lcurand -lcrypto -o "$output/owned-yaw-migration"
"$output/owned-yaw-migration" --cpu-self-test
sha256sum "$output/owned-yaw-migration" "$native" "$output/source/owned_yaw_migration.cu" \
    "$output/source/owned_yaw_trajectory.h" "$output/source/owned_yaw_observation.h" "$output/source/algo.cu" \
    "$output/source/puffer5_bc_core.cuh" "$output/source/authentic_trajectory.h" \
    > "$output/build-hashes.sha256"
