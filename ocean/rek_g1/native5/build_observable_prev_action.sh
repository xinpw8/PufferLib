#!/usr/bin/env bash
# Compile and CPU tests only. No GPU initialization or inference is requested.
set -euo pipefail
if (( $# != 2 )); then
    echo "usage: build_observable_prev_action.sh PREPARED_PPO_BUILD FRESH_BUILD" >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
prepared=$(realpath -- "$1")
output=$(realpath -m -- "$2")
test ! -e "$output"
test "$(sha256sum "$prepared/source/algo.cu" | cut -d' ' -f1)" = 8a514cb8dd12d49b79cbd5afe7298875b6f0ca0491270bb19a8696bd527f4d92
test "$(sha256sum "$prepared/source/puffer5_bc_core.cuh" | cut -d' ' -f1)" = d3e07e6c5f376584543cdba457d582d6674183a0efb29c0755f236adcf284454
mkdir -p -- "$output/source"
for source in puffer5_bc_core.cuh algo.cu ocean.cu ini.h; do
    cp -- "$prepared/source/$source" "$output/source/"
done
for source in observable_balance.h observable_prev_action.h observable_prev_action_test.cpp observable_prev_action_migration.cu build_observable_prev_action.sh; do
    cp -- "$here/$source" "$output/source/"
done
cp -- "$here/../../../vendor/cJSON.c" "$here/../../../vendor/cJSON.h" "$output/source/"
cuda=${CUDA_HOME:-/usr/local/cuda}
nccl=${NCCL_HOME:-/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl}
set -x
g++ -std=c++17 -O2 -Wall -Wextra -Werror "$output/source/observable_prev_action_test.cpp" -o "$output/observable-prev-action-test"
CUDA_VISIBLE_DEVICES= "$output/observable-prev-action-test" | tee "$output/cpu-helper-tests.json"
gcc -O2 -c "$output/source/cJSON.c" -o "$output/cJSON.o"
"$cuda/bin/nvcc" -std=c++17 -O2 --threads 1 -arch=sm_121 \
    -I"$output/source" -I"$nccl/include" -I"$cuda/include/cccl" \
    "$output/source/observable_prev_action_migration.cu" "$output/cJSON.o" \
    -L"$cuda/lib64" -Xlinker=-rpath,"$cuda/lib64" -lcublas -lcurand -lcrypto \
    -o "$output/observable-prev-action-migration"
CUDA_VISIBLE_DEVICES= "$output/observable-prev-action-migration" --cpu-self-test | tee "$output/cpu-migration-tests.json"
sha256sum "$output/observable-prev-action-test" "$output/observable-prev-action-migration" \
    "$output/cJSON.o" "$output/source/"* > "$output/build-hashes.sha256"
