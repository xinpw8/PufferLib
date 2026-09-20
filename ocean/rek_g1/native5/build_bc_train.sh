#!/usr/bin/env bash
# Reuse a prepared, pinned PufferLib5 source stage from build_native.sh.
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s PREPARED_PUFFER5_SRC NEW_BUILD_DIRECTORY\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_puffer=$(realpath "$1")
task_build=$2
[[ ! -e "$task_build" ]] || { printf 'Build destination exists\n' >&2; exit 2; }
for task_file in pufferl.cu algo.cu ocean.cu ini.h; do
    [[ -f "$task_puffer/$task_file" ]] || { printf 'Missing Puffer5 source: %s\n' "$task_file" >&2; exit 2; }
done
grep -q 'scan.terminals_ptr = terminals.data' "$task_puffer/algo.cu"
grep -q 'puf_dw_join(stream)' "$task_puffer/algo.cu"
mkdir -p "$task_build/source"
task_build=$(realpath "$task_build")
cp "$task_puffer/"{pufferl.cu,algo.cu,ocean.cu,ini.h} "$task_build/source/"
# Generated build adapter stops at the existing model/optimizer boundary.
# It changes no kernel bodies and includes no environment or runner.
awk '/^#include ENV_HEADER$/ {print "#define ACT_SIZES {33}\n#define NUM_ATNS 1"; env++; next}
     /^#include "protein.cu"$/ {boundary++; exit}
     {print}
     END {if (env != 1 || boundary != 1) exit 2}' \
    "$task_build/source/pufferl.cu" > "$task_build/source/puffer5_bc_core.cuh"
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_nccl=${PUFFER5_NCCL:-/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl}
task_nvcc=${NVCC:-$task_cuda/bin/nvcc}
"$task_nvcc" -std=c++17 -O2 --threads 1 "-arch=${REK_CUDA_ARCH:-sm_121}" \
    -I"$task_build/source" -I"$task_nccl/include" -I"$task_cuda/include/cccl" \
    "$task_source/bc_train.cu" -o "$task_build/bc-train" -lcublas -lcurand -lcrypto
g++ -std=c++17 -O2 "$task_source/test_bc_dataset.cpp" -o "$task_build/test-bc-dataset"
"$task_build/test-bc-dataset"
{
    printf 'backend=pufferlib5_native_cuda_bc\nphysics=none\nprecision=bf16_fp32_master\n'
    "$task_nvcc" --version
    sha256sum "$task_build/source/"* "$task_source/"{bc_train.cu,bc_dataset.h,build_bc_train.sh,test_bc_dataset.cpp} "$task_build/bc-train"
} > "$task_build/build-provenance.txt"
readelf -d "$task_build/bc-train" > "$task_build/elf-dependencies.txt"
if grep -Eqi '(libpython|libtorch)' "$task_build/elf-dependencies.txt"; then
    printf 'Unexpected interpreter/framework dependency\n' >&2; exit 2
fi
