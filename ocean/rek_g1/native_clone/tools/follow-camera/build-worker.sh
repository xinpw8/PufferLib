#!/usr/bin/env bash
set -euo pipefail
task_root=/home/spark-advantage/rek-training/rek-follow-camera-20260929-r1
task_base=/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/build-r6
task_source=$task_root/checkout/source/ocean/rek_g1/native5
task_build=$task_root/build-r1
task_cuda=/usr/local/cuda
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
test ! -e "$task_build"
test "$(sha256sum "$task_base/rek-native-clone" | cut -d' ' -f1)" = 90a004a6aa89a72d13e1c44abe10f2adea70059c746075b062bedb442ed36b73
sha256sum --quiet -c "$task_base/outputs.sha256"
sha256sum --quiet -c "$task_base/reused-inputs.sha256"
mkdir "$task_build"
trap 'task_status=$?; printf "%s\n" "$task_status" > "$task_build/exit-code.txt"' EXIT
task_objects=()
while read -r task_hash task_object; do task_objects+=("$task_object"); done < "$task_base/reused-inputs.sha256"
for task_object in "$task_base"/*.o; do
    if test "$(basename "$task_object")" != eval_worker.o; then task_objects+=("$task_object"); fi
done
sha256sum "${task_objects[@]}" > "$task_build/reused-inputs.sha256"
sha256sum "$task_source"/* > "$task_build/source-inputs.sha256"
"$task_cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_121 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false \
    -Xcompiler=-fPIC -Xcompiler=-ffp-contract=off -I"$task_source/.." -I"$task_source" \
    -I"$task_mujoco/include" -I"$task_cuda/include/cccl" -DREK_NATIVE5_MUJOCO_GPU=1 \
    -c "$task_source/eval_worker.cpp" -o "$task_build/eval_worker.o" > "$task_build/compile.stdout.txt" 2> "$task_build/compile.stderr.txt"
task_objects+=("$task_build/eval_worker.o")
"$task_cuda/bin/nvcc" -arch=sm_121 "${task_objects[@]}" \
    -L"$task_mujoco" -L"$task_cuda/lib64" -L"$task_cuda/lib64/stubs" \
    -Xlinker=-rpath -Xlinker="$task_mujoco" -Xlinker=-rpath -Xlinker="$task_cuda/lib64" \
    -lcudart -lcuda -lcublas -lcrypto -l:libmujoco.so.3.7.0 -lEGL -lGL -lz -lm -lpthread \
    -o "$task_build/rek-native-clone" > "$task_build/link.stdout.txt" 2> "$task_build/link.stderr.txt"
sha256sum --quiet -c "$task_build/reused-inputs.sha256"
sha256sum --quiet -c "$task_build/source-inputs.sha256"
sha256sum "$task_build/rek-native-clone" "$task_build/eval_worker.o" > "$task_build/outputs.sha256"
