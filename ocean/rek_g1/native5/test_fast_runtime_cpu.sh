#!/usr/bin/env bash
# Build and run the CPU emulation tests of fast_runtime.cu (no GPU needed).
# Optional REK_FAST_ORIGINAL=/path/to/previous/fast_runtime.cu also checks that
# disabled lite falls and move reward are bit-identical to that version.
# REK_CPU_TEST_CUDA_INCLUDE points at CUDA headers (default /usr/local/cuda/include).
set -euo pipefail
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_include=${REK_CPU_TEST_CUDA_INCLUDE:-/usr/local/cuda/include}
task_out=$(mktemp -d)
trap 'rm -rf "$task_out"' EXIT
# The device half ends where the host runtime API begins.
device_half(){ sed '/^struct RekNative5Runtime {/,$d' "$1" > "$2"; }
task_flags=(-std=c++17 -O2 -Wall -Wno-unused-function -Wno-unused-variable -Wno-misleading-indentation -Wno-ignored-attributes
    -I"$task_source" -I"$task_source/.." -I"$task_source/../../../vendor" -I"$task_include")
device_half "$task_source/fast_runtime.cu" "$task_out/modified.inc"
task_objects=()
"${CXX:-g++}" "${task_flags[@]}" -DFAST_DEVICE_SOURCE="\"$task_out/modified.inc\"" \
    -c "$task_source/test_fast_runtime_cpu_modified.cpp" -o "$task_out/modified.o"
task_objects+=("$task_out/modified.o")
task_have_original=0
if [[ -n ${REK_FAST_ORIGINAL:-} ]]; then
    device_half "$REK_FAST_ORIGINAL" "$task_out/original.inc"
    "${CXX:-g++}" "${task_flags[@]}" -DFAST_DEVICE_SOURCE="\"$task_out/original.inc\"" \
        -c "$task_source/test_fast_runtime_cpu_original.cpp" -o "$task_out/original.o"
    task_objects+=("$task_out/original.o");task_have_original=1
fi
"${CXX:-g++}" "${task_flags[@]}" -DHAVE_ORIGINAL=$task_have_original \
    "$task_source/test_fast_runtime_cpu_main.cpp" "${task_objects[@]}" \
    -x c "$task_source/../../../vendor/cJSON.c" -x none -lcrypto -o "$task_out/test_fast_runtime_cpu"
mkdir "$task_out/data"
"$task_out/test_fast_runtime_cpu" "$task_source/lite_falls_smoke_v1.json" "$task_out/data"
# Closed loop: fit the logged data and recover the known smoke model.
python3 "$task_source/fit_lite_falls.py" fit "$task_out/data/train.json" "$task_out/data/fitted.json" \
    --holdout "$task_out/data/holdout.json" --model-id recovery-test > "$task_out/data/fit.json"
python3 "$task_source/check_lite_fall_recovery.py" "$task_source/lite_falls_smoke_v1.json" \
    "$task_out/data/fitted.json" "$task_out/data/holdout.json"
