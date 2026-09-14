#!/usr/bin/env bash
# Compile native inference against the pinned learner's real CUDA implementation.
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s NEW_OUTPUT_DIRECTORY PRIVATE_CHECKPOINT\n' "$0" >&2; exit 2; }
task_output=$1
task_checkpoint=$(realpath "$2")
mkdir "$task_output"
task_output=$(realpath "$task_output")
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
task_trainer=${REK_NATIVE5_REFERENCE_SRC:-/home/spark-advantage/rek-training/native5-rek-20260913-v1/pufferlib5/src}
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_nccl=${PUFFER5_NCCL:-/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl}
task_raylib=${PUFFER5_RAYLIB:-/home/spark-advantage/pufferlib-5.0-wr64/raylib-5.5_linux_aarch64}
task_arch=${REK_CUDA_ARCH:-sm_121}
sha256sum "$task_trainer/pufferl.cu" "$task_trainer/algo.cu" \
    "$task_source/native_policy.cu" "$task_source/test_native_policy.cu" > "$task_output/source-sha256.txt"
{ hostname; uname -m; nvidia-smi -L; } > "$task_output/host.txt"
"$task_cuda/bin/nvcc" -std=c++17 -O2 "-arch=$task_arch" -I"$task_source" \
    -c "$task_source/native_policy.cu" -o "$task_output/native_policy.o"
task_flags=(-std=c++17 -O2 "-arch=$task_arch" -I"$task_source" -I"$task_trainer"
    -I"$task_nccl/include" -I"$task_raylib/include" -I"$task_cuda/include/cccl"
    -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing --diag-suppress=2361 --diag-suppress=128
    -DPLATFORM_DESKTOP -DENV_NAME=rek_native5 '-DPUFFER_ENV_NAME="rek_native5"'
    '-DENV_HEADER="native_policy_reference_env.cuh"')
for task_precision in bf16 fp32; do
    task_precision_flags=()
    if [[ $task_precision == fp32 ]]; then task_precision_flags=(-DPRECISION_FLOAT); fi
    "$task_cuda/bin/nvcc" "${task_flags[@]}" "${task_precision_flags[@]}" \
        -c "$task_source/test_native_policy.cu" -o "$task_output/test-$task_precision.o" \
        > "$task_output/compile-$task_precision.stdout" 2> "$task_output/compile-$task_precision.stderr"
    "$task_cuda/bin/nvcc" "$task_output/test-$task_precision.o" "$task_output/native_policy.o" \
        "$task_raylib/lib/libraylib.a" -L"$task_nccl/lib" -L"$task_cuda/lib64/stubs" \
        -Xlinker=-rpath -Xlinker="$task_nccl/lib" -lcublas -lcusolver -lcurand -lnccl -lnvidia-ml \
        -lcrypto -lGL -lm -lpthread -lomp5 -o "$task_output/test-$task_precision"
    set +e
    timeout 30s "$task_output/test-$task_precision" "$task_checkpoint" \
        > "$task_output/test-$task_precision.stdout" 2> "$task_output/test-$task_precision.stderr"
    task_status=$?
    set -e
    printf '%d\n' "$task_status" > "$task_output/test-$task_precision.exit-code"
    cat "$task_output/test-$task_precision.stdout" "$task_output/test-$task_precision.stderr"
    [[ $task_status == 0 ]] || exit "$task_status"
done
