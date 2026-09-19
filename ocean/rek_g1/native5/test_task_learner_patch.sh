#!/usr/bin/env bash
# Compile a patched trainer in a disposable test directory. Never starts training.
set -euo pipefail
[[ $# == 3 ]] || { printf 'Usage: %s PATCHED_TRAINER_DIRECTORY REK_SOURCE_DIRECTORY NEW_OBJECT_PATH\n' "$0" >&2; exit 2; }
task_trainer=$(realpath "$1")
task_source=$(realpath "$2")
task_object=$3
[[ ! -e "$task_object" ]]
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_raylib=${PUFFER5_RAYLIB:-/home/spark-advantage/pufferlib-5.0-wr64/raylib-5.5_linux_aarch64}
task_nccl=${PUFFER5_NCCL:-/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl}
"$task_cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_121 --threads 0 \
    -I"$task_trainer" -I"$task_trainer/src" -I"$task_source" \
    -I"$task_raylib/include" -I"$task_cuda/include" -I"$task_cuda/include/cccl" \
    -I"$task_nccl/include" -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing \
    --diag-suppress=2361 -DPLATFORM_DESKTOP -DPUFFERLIB_BUILD_MAIN \
    -DPUFFER_ENV_GPU_ROLLOUT_BOOTSTRAP -DENV_NAME=rek_native5 \
    '-DPUFFER_ENV_NAME="rek_native5"' "-DENV_HEADER=\"$task_source/puffer_env.cu\"" \
    -c "$task_trainer/src/pufferl.cu" -o "$task_object"
printf 'patched_native_trainer_compile=passed\n'
