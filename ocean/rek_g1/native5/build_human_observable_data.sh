#!/usr/bin/env bash
# CPU-only offline adapter. No CUDA or environment dependencies.
set -euo pipefail
umask 077
[[ $# == 1 ]] || { printf 'Usage: %s NEW_BUILD_DIRECTORY\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_root=$(cd "$task_source/../../.." && pwd)
mkdir "$1"
task_build=$(realpath "$1")
"${CC:-gcc}" -std=c11 -O2 -c "$task_root/vendor/cJSON.c" -o "$task_build/cJSON.o"
for task_target in human_observable_data human_observable_data_test; do
    "${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
        "$task_source/$task_target.cpp" "$task_build/cJSON.o" -lcrypto -o "$task_build/$task_target"
done
"$task_build/human_observable_data_test" | tee "$task_build/unit-tests.json"
sha256sum "$task_source/"{human_observable_data.cpp,human_observable_data_test.cpp,human_observable_data.test.cjs,build_human_observable_data.sh,observable_balance.h,bc_dataset.h} \
    "$task_root/vendor/"{cJSON.c,cJSON.h} "$task_build/"{human_observable_data,human_observable_data_test} > "$task_build/build-hashes.txt"
readelf -d "$task_build/human_observable_data" > "$task_build/elf-dependencies.txt"
if grep -Eqi 'libpython|libtorch|libmujoco|libcuda' "$task_build/elf-dependencies.txt"; then exit 2; fi
