#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 1 ]] || { printf 'Usage: %s NEW_BUILD_DIRECTORY\n' "$0" >&2;exit 2; }
task_src=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_root=$(cd "$task_src/../../../.." && pwd)
task_out=$1
mkdir "$task_out"
task_out=$(realpath "$task_out")
exec > >(tee "$task_out/build.stdout.txt") 2> >(tee "$task_out/build.stderr.txt" >&2)
set -x
"${CC:-gcc}" -std=c11 -O2 -c "$task_root/vendor/cJSON.c" -o "$task_out/cJSON.o"
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -Wno-misleading-indentation \
 "$task_src/encode_observable_balance.cpp" "$task_out/cJSON.o" -lcrypto -o "$task_out/encode-observable-balance"
readelf -d "$task_out/encode-observable-balance" > "$task_out/elf-dependencies.txt"
if rg -qi 'libpython|libtorch|libmujoco|libcuda' "$task_out/elf-dependencies.txt";then exit 2;fi
sha256sum "$task_src/encode_observable_balance.cpp" "$task_src/observable_encoder_test.cjs" \
 "$task_src/../observable_balance.h" "$task_src/../action_cadence.h" "$task_out/encode-observable-balance" > "$task_out/build-hashes.txt"
