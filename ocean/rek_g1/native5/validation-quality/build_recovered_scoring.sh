#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { printf 'Usage: %s SOURCE_ROOT OLD_V4_BUILD NEW_BUILD\n' "$0" >&2;exit 2; }
task_source=$(realpath "$1")
task_old=$(realpath "$2")
mkdir "$3"
task_build=$(realpath "$3")
task_g1=$task_source/ocean/rek_g1
task_native=$task_g1/native5
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
for task_unit in g1_strike_catalog native_motion_routes;do
  gcc -std=c11 -O2 -fPIC -c "$task_g1/$task_unit.c" -o "$task_build/$task_unit.o"
done
g++ -std=c++17 -O3 -fPIC -I/usr/local/cuda/include -I"$task_mujoco/include" -c "$task_native/fast_assets.cpp" -o "$task_build/fast_assets_loader.o"
ld -r "$task_build/fast_assets_loader.o" "$task_build/g1_strike_catalog.o" "$task_build/native_motion_routes.o" -o "$task_build/fast_assets.o"
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_121 -Xcompiler=-fPIC -c "$task_native/fast_runtime.cu" -o "$task_build/fast_runtime.o"
cp "$task_old/native_policy.o" "$task_old/cJSON.o" "$task_build/"
task_renames=()
for task_symbol in create reset step bind_action_mask bind_external_actions get_device_view encode_fighter_observations read_snapshot check_status close error;do
  task_renames+=(--redefine-sym "rek_native5_$task_symbol=legacy_$task_symbol")
done
objcopy "${task_renames[@]}" "$task_old/fast_runtime.o" "$task_build/legacy_fast_runtime.o"
# Legacy object needs the old FastAssets ABI; rename its loader and link the old
# asset object under that name. Production uses only the freshly compiled pair.
task_loader=$(nm "$task_old/fast_assets.o" | awk '$2=="T" && $3 ~ /load_fast_assets/ {print $3}')
[[ -n "$task_loader" ]]
objcopy --redefine-sym "$task_loader=legacy_load_fast_assets" "$task_build/legacy_fast_runtime.o"
objcopy --redefine-sym "$task_loader=legacy_load_fast_assets" "$task_old/fast_assets.o" "$task_build/legacy_fast_assets.o"
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_121 "$task_native/validation-quality/recovered_scoring_test.cu" "$task_build/fast_runtime.o" "$task_build/fast_assets.o" "$task_build/legacy_fast_runtime.o" "$task_build/legacy_fast_assets.o" "$task_build/native_policy.o" "$task_build/cJSON.o" -L"$task_mujoco" -Xlinker=-rpath -Xlinker="$task_mujoco" -l:libmujoco.so.3.7.0 -lcrypto -lcublas -lcurand -o "$task_build/recovered-scoring-test"
sha256sum "$task_native/fast_runtime.cu" "$task_native/fast_assets.cpp" "$task_native/fast_assets.h" "$task_native/recovered_contact_rules.cuh" "$task_build/fast_runtime.o" "$task_build/fast_assets.o" "$task_build/recovered-scoring-test" > "$task_build/hashes.txt"
"$task_build/recovered-scoring-test" --cpu-only > "$task_build/cpu-test.jsonl"
printf 'Built recovered-scoring candidate and isolated reference test: %s\n' "$task_build"
