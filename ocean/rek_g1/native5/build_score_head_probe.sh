#!/usr/bin/env bash
# Standalone CUDA only: no simulator, Python, Torch, or training-framework dependency.
set -euo pipefail
[[ $# == 1 ]] || { printf 'Usage: %s NEW_BUILD_DIRECTORY\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$1
[[ ! -e "$task_build" ]] || { printf 'Build destination already exists\n' >&2; exit 2; }
mkdir -p -- "$task_build"
task_build=$(realpath "$task_build")
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_nvcc=${NVCC:-$task_cuda/bin/nvcc}
task_arch=${REK_CUDA_ARCH:-sm_121}
"$task_nvcc" -std=c++17 -O3 --threads 1 "-arch=$task_arch" \
  "$task_source/score_head_probe.cu" -o "$task_build/score-head-probe"
{ printf 'backend=native_cuda_full_batch_sgd\nphysics=none\npolicy_training=0\n';
  "$task_nvcc" --version; sha256sum "$task_source/score_head_probe.cu" "$task_source/build_score_head_probe.sh" "$task_build/score-head-probe"; } > "$task_build/build-provenance.txt"
readelf -d "$task_build/score-head-probe" > "$task_build/elf-dependencies.txt"
if grep -Eqi '(libpython|libtorch)' "$task_build/elf-dependencies.txt"; then
  printf 'Unexpected interpreter/framework dependency\n' >&2; exit 2
fi
printf 'Built standalone CUDA probe: %s/score-head-probe\n' "$task_build"
