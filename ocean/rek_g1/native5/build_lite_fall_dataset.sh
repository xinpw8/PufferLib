#!/usr/bin/env bash
# Build the Phase 0 lite-fall dataset logger against a runtime's objects.
#
#   build_lite_fall_dataset.sh NEW_BUILD [RUNTIME_OBJECT ...]
#
# Without objects it links the physical MuJoCo + SONIC object set used by
# validation-quality/build_physical_training.sh (2026-09-21). A compact build
# directory's fast_runtime.o fast_assets.o native_policy.o cJSON.o also work,
# for a pipeline check against the lite runtime itself.
set -euo pipefail
[[ $# -ge 1 ]] || { printf 'Usage: %s NEW_BUILD [RUNTIME_OBJECT ...]\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_root=$(cd "$task_source/../../.." && pwd)
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_arch=${REK_CUDA_ARCH:-sm_121}
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
mkdir "$1"
task_build=$(realpath "$1")
shift
task_objects=("$@")
if [[ ${#task_objects[@]} == 0 ]]; then
    task_stage=/home/spark-advantage/rek-training/physical-fall-exposure-20260921-r1
    task_base=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/build-native-v2
    task_objects=("$task_stage/runtime-training-r1.o"
        /home/spark-advantage/rek-training/physical-measurement-integration-20260921-r1/build-r2/measurement.o)
    for task_object in "$task_base"/*.o; do
        case $(basename "$task_object") in runtime.o|measurement.o|pufferl.o|eval_worker.o) continue;; esac
        task_objects+=("$task_object")
    done
fi
for task_object in "${task_objects[@]}"; do test -f "$task_object"; done
"$task_cuda/bin/nvcc" -std=c++17 -O3 "-arch=$task_arch" -I"$task_source" -I"$task_source/.." \
    -c "$task_source/lite_fall_dataset.cu" -o "$task_build/lite_fall_dataset.o"
task_link=("$task_cuda/bin/nvcc" "-arch=$task_arch" "$task_build/lite_fall_dataset.o" "${task_objects[@]}")
if ! printf '%s\n' "${task_objects[@]}" | grep -q '/cJSON.o$'; then
    "${CC:-gcc}" -std=c11 -O2 -fPIC -c "$task_root/vendor/cJSON.c" -o "$task_build/cJSON.o"
    task_link+=("$task_build/cJSON.o")
fi
task_link+=(-L"$task_cuda/lib64" -L"$task_cuda/lib64/stubs" -L"$task_mujoco"
    -Xlinker=-rpath -Xlinker="$task_cuda/lib64" -Xlinker=-rpath -Xlinker="$task_mujoco"
    -lcudart -lcuda -lcublas -lcusolver -lcurand -l:libmujoco.so.3.7.0 -lcrypto -lGL -lz -lm -lpthread
    -o "$task_build/lite-fall-dataset")
printf '%q ' "${task_link[@]}" > "$task_build/link-command.txt"; printf '\n' >> "$task_build/link-command.txt"
"${task_link[@]}"
sha256sum "$task_source/lite_fall_dataset.cu" "$task_source/lite_fall_dataset_observe.h" "$task_source/lite_falls.h" \
    "${task_objects[@]}" "$task_build/lite-fall-dataset" > "$task_build/build-hashes.txt"
printf 'Built %s/lite-fall-dataset\n' "$task_build"
