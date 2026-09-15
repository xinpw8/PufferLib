#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 ]] || { printf 'Usage: %s NEW_BUILD_DIRECTORY\n' "$0" >&2; exit 2; }
task_src=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_root=$(cd "$task_src/../../../.." && pwd)
task_out=$1
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
mkdir -p "$task_out"
"${CC:-gcc}" -std=c11 -O2 -c "$task_root/vendor/cJSON.c" -o "$task_out/encoder-cJSON.o"
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation \
  -I"$task_mujoco/include" "$task_src/encode_live.cpp" "$task_out/encoder-cJSON.o" \
  -L"$task_mujoco" -Wl,-rpath,"$task_mujoco" -l:libmujoco.so.3.7.0 -lcrypto \
  -o "$task_out/encode-live"
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation \
  -I"$task_mujoco/include" "$task_src/encoder_test.cpp" "$task_out/encoder-cJSON.o" \
  -L"$task_mujoco" -Wl,-rpath,"$task_mujoco" -l:libmujoco.so.3.7.0 -lcrypto \
  -o "$task_out/encoder-test"
readelf -d "$task_out/encode-live" > "$task_out/encoder-elf-dependencies.txt"
if rg -qi '(libpython|libtorch)' "$task_out/encoder-elf-dependencies.txt"; then exit 2; fi
sha256sum "$task_src/encode_live.cpp" "$task_out/encode-live" > "$task_out/encoder-hashes.txt"
