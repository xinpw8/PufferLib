#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build-cpu-r1
src="$PWD/source-r1/ocean/rek_g1"
flags=(-std=c++20 -O2 -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-variable -I"$src" -I"$src/test_semantic_direct_support")
for name in test_ordered_reset test_g1_semantic_direct_cpu; do
 g++ "${flags[@]}" "tests/$name.cpp" -x c++ "$src/g1_semantic_action_table.c" "$src/puffer_action_adapter.c" "$src/native_locomotion_command.c" "$src/sonic_motion_composer_native.c" -lm -o "build-cpu-r1/$name"
 "build-cpu-r1/$name" > "build-cpu-r1/$name.jsonl"
done
g++ "${flags[@]}" history_trace.cpp -lm -o build-cpu-r1/history_trace
build-cpu-r1/history_trace tests/history-input.bin build-cpu-r1/history-native.f32 build-cpu-r1/heading-native.f32
sha256sum build-cpu-r1/* > build-cpu-r1/output-hashes.sha256
cat build-cpu-r1/test_ordered_reset.jsonl build-cpu-r1/test_g1_semantic_direct_cpu.jsonl
