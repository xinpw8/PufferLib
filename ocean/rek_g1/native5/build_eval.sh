#!/usr/bin/env bash
# Build the native evaluator using the existing validated runtime objects.
# Recompile changed modules; preserve all earlier build and benchmark outputs.
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s BASE_BUILD NEW_BUILD\n' "$0" >&2; exit 2; }
base=$(realpath "$1")
mkdir "$2"
build=$(realpath "$2")
source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
g1=$(cd "$source/.." && pwd)
cuda=/usr/local/cuda
mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
nvcc=$cuda/bin/nvcc
runtime=${REK_EVAL_RUNTIME:-native5}
if [[ "$runtime" == semantic_cuda ]]; then
    # Copy the exact runtime objects linked into the compact trainer. The only
    # extra host work in this executable is I/O and kinematics for pictures.
    for object in fast_runtime.o fast_assets.o native_policy.o cJSON.o; do
        [[ -f "$base/$object" ]] || { printf 'Missing compact runtime object: %s\n' "$object" >&2; exit 2; }
        cp "$base/$object" "$build/"
    done
    "$nvcc" -std=c++17 -O2 -arch=sm_121 -Xcompiler=-fPIC '-DREK_EVAL_BACKEND="semantic_cuda"' \
        -I"$g1" -I"$source" -I"$mujoco/include" -I"$cuda/include/cccl" \
        -c "$source/eval_worker.cpp" -o "$build/eval_worker.o"
    "$nvcc" -arch=sm_121 "$build"/*.o -L"$mujoco" -L"$cuda/lib64" \
        -Xlinker=-rpath -Xlinker="$mujoco" -Xlinker=-rpath -Xlinker="$cuda/lib64" \
        -lcudart -lcublas -lcrypto -l:libmujoco.so.3.7.0 -lEGL -lGL -lz -lm -lpthread \
        -o "$build/rek-eval-worker"
    { printf 'backend=semantic_cuda\ncontrol_hz=50\ncpu_physics=0\npython_runtime=0\n';
      sha256sum "$source/runtime_api.h" "$source/eval_worker.cpp" "$source/eval_renderer.h" \
        "$build"/*.o "$build/rek-eval-worker"; } > "$build/eval-build-hashes.txt"
    printf 'Built reduced GPU evaluator: %s/rek-eval-worker\n' "$build"
    exit 0
fi
[[ "$runtime" == native5 ]] || { printf 'Unknown evaluator runtime: %s\n' "$runtime" >&2; exit 2; }
for object in "$base"/*.o; do
    case $(basename "$object") in pufferl.o|physics.o|runtime.o|native_policy.o) continue;; esac
    cp "$object" "$build/"
done
common=(-std=c++17 -O2 -arch=sm_121 -Xcompiler=-fPIC -I"$g1" -I"$source"
    -I"$mujoco/include" -I"$cuda/include/cccl")
"$nvcc" "${common[@]}" --fmad=false --prec-div=true --prec-sqrt=true --ftz=false \
    -Xcompiler=-ffp-contract=off -c "$source/runtime.cu" -o "$build/runtime.o"
"$nvcc" "${common[@]}" -O3 -c "$source/physics.cu" -o "$build/physics.o"
"$nvcc" "${common[@]}" -c "$source/native_policy.cu" -o "$build/native_policy.o"
"$nvcc" "${common[@]}" -c "$source/eval_worker.cpp" -o "$build/eval_worker.o"
"$nvcc" -arch=sm_121 "$build"/*.o -L"$mujoco" -L"$cuda/lib64" \
    -Xlinker=-rpath -Xlinker="$mujoco" -Xlinker=-rpath -Xlinker="$cuda/lib64" \
    -lcudart -lcublas -lcrypto -l:libmujoco.so.3.7.0 -lEGL -lGL -lz -lm -lpthread \
    -o "$build/rek-eval-worker"
sha256sum "$source"/runtime_api.h "$source"/runtime.cu "$source"/physics.cu \
    "$source"/native_policy.cu "$source"/eval_worker.cpp "$source"/eval_renderer.h \
    "$build/rek-eval-worker" > "$build/eval-build-hashes.txt"
printf 'Built %s/rek-eval-worker\n' "$build"
