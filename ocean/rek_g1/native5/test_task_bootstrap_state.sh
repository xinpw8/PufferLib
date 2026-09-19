#!/usr/bin/env bash
set -euo pipefail
[[ $# == 4 ]] || { printf 'Usage: %s PATCHED_TRAINER REK_NATIVE_SOURCE EXISTING_RUNTIME_BUILD NEW_EXECUTABLE\n' "$0" >&2; exit 2; }
task_trainer=$(realpath "$1")
task_source=$(realpath "$2")
task_runtime=$(realpath "$3")
task_executable=$4
[[ ! -e "$task_executable" ]]
task_test=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_raylib=${PUFFER5_RAYLIB:-/home/spark-advantage/pufferlib-5.0-wr64/raylib-5.5_linux_aarch64}
task_nccl=${PUFFER5_NCCL:-/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl}
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
"$task_cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_121 --threads 0 \
    -I"$task_trainer" -I"$task_trainer/src" -I"$task_source" \
    -I"$task_raylib/include" -I"$task_cuda/include" -I"$task_cuda/include/cccl" \
    -I"$task_nccl/include" -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing \
    --diag-suppress=2361 -DPLATFORM_DESKTOP -DPUFFER_ENV_GPU_ROLLOUT_BOOTSTRAP \
    -DENV_NAME=rek_native5 '-DPUFFER_ENV_NAME="rek_native5"' \
    "-DENV_HEADER=\"$task_source/puffer_env.cu\"" \
    "-DPUFFER_TEST_TRAINER_SOURCE=\"$task_trainer/src/pufferl.cu\"" \
    "$task_test/test_task_bootstrap_state.cu" \
    "$task_runtime/fast_runtime.o" "$task_runtime/fast_assets.o" "$task_runtime/native_policy.o" "$task_runtime/cJSON.o" \
    "$task_raylib/lib/libraylib.a" -L"$task_cuda/lib64" -L"$task_nccl/lib" -L"$task_mujoco" \
    -Xlinker=-rpath -Xlinker="$task_cuda/lib64" -Xlinker=-rpath -Xlinker="$task_nccl/lib" \
    -Xlinker=-rpath -Xlinker="$task_mujoco" \
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -l:libmujoco.so.3.7.0 \
    -lcrypto -lGL -lm -lpthread -lomp5 -o "$task_executable"
printf 'native_bootstrap_state_probe_built=%s\n' "$task_executable"
