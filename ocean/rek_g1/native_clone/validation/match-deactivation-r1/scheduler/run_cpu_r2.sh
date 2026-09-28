#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
test ! -e cpu-run-r2
mkdir cpu-run-r2
trap 'code=$?; printf "%s\n" "$code" > cpu-run-r2/exit-code.txt' EXIT
src="$PWD/cpu-source"
flags=(-std=c++20 -O2 -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-variable -I"$src" -I"$src/test_semantic_direct_support")
for name in test_deactivation test_ordered_reset test_g1_semantic_direct_cpu; do
 g++ "${flags[@]}" "tests/$name.cpp" -x c++ "$src/g1_semantic_action_table.c" "$src/puffer_action_adapter.c" "$src/native_locomotion_command.c" "$src/sonic_motion_composer_native.c" -lm -o "cpu-run-r2/$name" > "cpu-run-r2/$name.build.stdout.txt" 2> "cpu-run-r2/$name.build.stderr.txt"
 "cpu-run-r2/$name" > "cpu-run-r2/$name.jsonl"
done
sha256sum cpu-run-r2/test_* > cpu-run-r2/hashes.sha256
cat cpu-run-r2/*.jsonl
