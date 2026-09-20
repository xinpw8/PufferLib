#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s PREPARED_BC_BUILD NEW_BUILD_DIRECTORY\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_prepared=$(realpath "$1")
task_build=$2
[[ -f "$task_prepared/source/puffer5_bc_core.cuh" && ! -e "$task_build" ]] || { printf 'Missing prepared core or existing destination\n' >&2; exit 2; }
mkdir -p "$task_build/source"
task_build=$(realpath "$task_build")
cp "$task_prepared/source/"* "$task_build/source/"
cp "$task_source/owned_yaw_trajectory.h" "$task_source/owned_yaw_observation.h" "$task_build/source/"
node "$task_source/prepare_authentic_ppo_kernel.cjs" "$task_build/source/algo.cu" "$task_build/source/puffer5_ppo_fp32.cuh"
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_nccl=${PUFFER5_NCCL:-/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl}
"${NVCC:-$task_cuda/bin/nvcc}" -std=c++17 -O2 --threads 1 "-arch=${REK_CUDA_ARCH:-sm_121}" \
    -I"$task_build/source" -I"$task_nccl/include" -I"$task_cuda/include/cccl" \
    "$task_source/authentic_ppo.cu" -o "$task_build/authentic-ppo" -lcublas -lcurand -lcrypto
g++ -std=c++17 -O2 "$task_source/test_authentic_gae.cpp" -o "$task_build/test-authentic-gae"
"$task_build/test-authentic-gae"
g++ -std=c++17 -O2 "$task_source/test_authentic_parity.cpp" -o "$task_build/test-authentic-parity"
"$task_build/test-authentic-parity"
node --test "$task_source/prepare_authentic_ppo_kernel.test.cjs"
{
    printf 'backend=pufferlib5_native_cuda_authentic_trajectory_ppo\nphysics=none\nbehavior_logprobs=immutable_fp32\n'
    sha256sum "$task_build/source/"* "$task_source/"{authentic_ppo.cu,authentic_trajectory.h,owned_yaw_trajectory.h,owned_yaw_observation.h,authentic_gae.h,authentic_parity.h,test_authentic_parity.cpp,bc_train.cu,bc_dataset.h,prepare_authentic_ppo_kernel.cjs,build_authentic_ppo.sh} "$task_build/authentic-ppo"
} > "$task_build/build-provenance.txt"
readelf -d "$task_build/authentic-ppo" > "$task_build/elf-dependencies.txt"
if grep -Eqi '(libpython|libtorch)' "$task_build/elf-dependencies.txt"; then
    printf 'Unexpected interpreter/framework dependency\n' >&2; exit 2
fi
