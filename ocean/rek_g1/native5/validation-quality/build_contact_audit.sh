#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s SOURCE_ROOT EXACT_V4_BUILD\n' "$0" >&2;exit 2; }
task_source=$(realpath "$1")
task_build=$(realpath "$2")
task_output=$(dirname "$task_source")/build
mkdir "$task_output"
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
g++ -std=c++17 -O3 -fPIC -I/usr/local/cuda/include -I"$task_mujoco/include" -c "$task_source/ocean/rek_g1/native5/fast_assets.cpp" -o "$task_output/fast_assets.o"
for task_unit in g1_hit_detector g1_strike_catalog native_motion_routes;do
  gcc -std=c11 -O2 -ffp-contract=off -c "$task_source/ocean/rek_g1/$task_unit.c" -o "$task_output/$task_unit.o"
done
task_command=(/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_121
  "$task_source/ocean/rek_g1/native5/validation-quality/contact_audit.cu"
  "$task_output/fast_assets.o" "$task_build/native_policy.o" "$task_build/cJSON.o"
  "$task_output/g1_hit_detector.o" "$task_output/g1_strike_catalog.o" "$task_output/native_motion_routes.o"
  -L"$task_mujoco" -Xlinker=-rpath -Xlinker="$task_mujoco"
  -l:libmujoco.so.3.7.0 -lcrypto -lcublas -lcurand -o "$task_output/contact-audit")
printf '%q ' "${task_command[@]}" > "$task_output/command.txt"
printf '\n' >> "$task_output/command.txt"
"${task_command[@]}" > "$task_output/stdout.txt" 2> "$task_output/stderr.txt"
sha256sum "$task_source/ocean/rek_g1/native5/fast_runtime.cu" "$task_source/ocean/rek_g1/native5/validation-quality/contact_audit.cu" "$task_output/fast_assets.o" "$task_build/native_policy.o" "$task_build/cJSON.o" "$task_output/contact-audit" > "$task_output/hashes.txt"
printf 'Built %s/contact-audit\n' "$task_output"
