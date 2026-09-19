#!/usr/bin/env bash
set -euo pipefail
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=${1:?Usage: test_recovered_balance.sh NEW_BUILD_DIRECTORY [cuda]}
mkdir "$task_build"
task_build=$(cd "$task_build" && pwd)
task_modules=(g1_fall_state g1_fight_state g1_combat_tick g1_hit_detector)
task_host_objects=()
for task_module in "${task_modules[@]}"; do
    gcc -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror -pedantic \
        -c "$task_source/../$task_module.c" -o "$task_build/$task_module.host.o"
    task_host_objects+=("$task_build/$task_module.host.o")
done
g++ -std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Werror \
    "$task_source/test_recovered_balance.cpp" "${task_host_objects[@]}" \
    -o "$task_build/test_recovered_balance"
"$task_build/test_recovered_balance"
if [[ ${2:-} == cuda ]]; then
    task_nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
    task_flags=(-std=c++20 -O2 "-arch=${REK_CUDA_ARCH:-native}" --fmad=false --prec-div=true --prec-sqrt=true --ftz=false)
    task_device_objects=()
    for task_module in "${task_modules[@]}"; do
        "$task_nvcc" "${task_flags[@]}" -dc \
            "-DREK_G1_CUDA_SOURCE=\"$task_module.c\"" \
            "$task_source/../g1_cuda_device.cu" -o "$task_build/$task_module.device.o"
        task_device_objects+=("$task_build/$task_module.device.o")
    done
    "$task_nvcc" "${task_flags[@]}" -dc "$task_source/test_recovered_balance.cu" \
        -o "$task_build/test_recovered_balance.device.o"
    g++ -std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Werror \
        -DREK5_BALANCE_TEST_CUDA -c "$task_source/test_recovered_balance.cpp" \
        -o "$task_build/test_recovered_balance.host.o"
    "$task_nvcc" "${task_flags[@]}" "$task_build/test_recovered_balance.host.o" \
        "$task_build/test_recovered_balance.device.o" "${task_host_objects[@]}" \
        "${task_device_objects[@]}" -o "$task_build/test_recovered_balance_cuda"
    "$task_build/test_recovered_balance_cuda"
fi
