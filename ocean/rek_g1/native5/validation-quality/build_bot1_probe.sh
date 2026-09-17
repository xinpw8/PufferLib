#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { printf 'Usage: %s SOURCE OLD_RECOVERED_BUILD BUILD\n' "$0" >&2;exit 2; }
task_source=$(realpath "$1")
task_old=$(realpath "$2")
task_build=$(realpath "$3")
task_native=$task_source/ocean/rek_g1/native5
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
g++ -std=c++17 -O2 -Wall -Wextra -Werror "$task_native/validation-quality/native_bot1_test.cpp" -o "$task_build/native-bot1-cpu-test"
"$task_build/native-bot1-cpu-test" > "$task_build/bot1-cpu.jsonl"
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_121 -Xcompiler=-fPIC -c "$task_native/fast_runtime.cu" -o "$task_build/fast_runtime.o"
cp "$task_old/fast_assets.o" "$task_old/native_policy.o" "$task_old/cJSON.o" "$task_build/"
task_rename=()
for task_symbol in create reset step bind_action_mask bind_external_actions get_device_view encode_fighter_observations read_snapshot check_status close error;do
  task_rename+=(--redefine-sym "rek_native5_$task_symbol=legacy_$task_symbol")
done
objcopy "${task_rename[@]}" "$task_old/fast_runtime.o" "$task_build/legacy_fast_runtime.o"
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_121 "$task_native/validation-quality/native_bot1_runtime_test.cu" "$task_build/fast_runtime.o" "$task_build/legacy_fast_runtime.o" "$task_build/fast_assets.o" "$task_build/cJSON.o" -L"$task_mujoco" -Xlinker=-rpath -Xlinker="$task_mujoco" -l:libmujoco.so.3.7.0 -lcrypto -o "$task_build/native-bot1-runtime-test"
sha256sum "$task_native/fast_runtime.cu" "$task_native/native_bot1.cuh" "$task_native/rendered_pose_observation.h" "$task_native/validation-quality/native_bot1_runtime_test.cu" "$task_build/fast_runtime.o" "$task_build/fast_assets.o" "$task_build/native-bot1-runtime-test" > "$task_build/bot1-hashes.txt"
