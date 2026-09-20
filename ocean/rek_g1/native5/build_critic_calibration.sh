#!/usr/bin/env bash
set -euo pipefail
if (( $# != 2 )); then
    echo "usage: build_critic_calibration.sh PREPARED_PPO_BUILD NEW_BUILD_DIRECTORY" >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
prepared=$(realpath -- "$1")
output=$(realpath -m -- "$2")
test -f "$prepared/source/puffer5_bc_core.cuh"
test ! -e "$output"
mkdir -p -- "$output/source"
cp -a -- "$prepared/source/." "$output/source/"
cp -- "$here/critic_calibration.cu" "$here/authentic_trajectory.h" "$here/device_storage.cuh" \
    "$here/round_reward.h" "$here/build_critic_calibration.sh" "$output/source/"
cuda=${CUDA_HOME:-/usr/local/cuda}
nccl=${NCCL_HOME:-/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl}
set -x
"$cuda/bin/nvcc" -std=c++17 -O2 --threads 1 -arch=sm_121 \
    -I"$output/source" -I"$nccl/include" -I"$cuda/include/cccl" \
    "$output/source/critic_calibration.cu" -L"$cuda/lib64" \
    -Xlinker=-rpath,"$cuda/lib64" -lcublas -lcurand -lcrypto -o "$output/critic-calibration"
# Metadata registration and byte-preservation tests only. No CUDA API calls.
"$output/critic-calibration" --cpu-self-test
sha256sum "$output/critic-calibration" "$output/source/critic_calibration.cu" \
    "$output/source/algo.cu" "$output/source/puffer5_bc_core.cuh" \
    "$output/source/authentic_trajectory.h" "$output/source/build_critic_calibration.sh" \
    > "$output/build-hashes.sha256"
readelf -d "$output/critic-calibration" | grep NEEDED
