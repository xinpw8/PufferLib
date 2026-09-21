#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir=${1:?Usage: test_g1_semantic_direct_cpu.sh FRESH_BUILD_DIRECTORY}
test ! -e "$build_dir"
mkdir -p "$build_dir"
baseline_flags=()
if [[ $# == 2 ]]; then
    test -f "$2"
    baseline_flags+=("-DREK_G1_SEMANTIC_BASELINE_SOURCE=\"$2\"")
fi
g++ -std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Werror \
    "${baseline_flags[@]}" \
    -I"$source_dir/test_semantic_direct_support" -I"$source_dir" \
    "$source_dir/test_g1_semantic_direct_cpu.cpp" \
    -x c++ "$source_dir/g1_semantic_action_table.c" \
    "$source_dir/puffer_action_adapter.c" "$source_dir/native_locomotion_command.c" \
    "$source_dir/sonic_motion_composer_native.c" -lm \
    -o "$build_dir/test_g1_semantic_direct_cpu"
"$build_dir/test_g1_semantic_direct_cpu"
