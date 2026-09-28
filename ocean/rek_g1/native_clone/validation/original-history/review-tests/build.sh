#!/usr/bin/env bash
set -euo pipefail
umask 077
task_root=/home/spark-advantage/rek-training/rek-native-clone-20260927-r1
task_here=$task_root/original-history/review-tests
task_build=$task_here/build-r1
task_cuda=/usr/local/cuda
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
test ! -e "$task_build"
sha256sum --quiet -c "$task_root/build/outputs.sha256"
sha256sum --quiet -c "$task_root/build/reused-inputs.sha256"
mkdir "$task_build"
trap 'task_status=$?; printf "%s\n" "$task_status" > "$task_build/exit-code.txt"' EXIT
task_objects=()
while read -r task_hash task_object; do task_objects+=("$task_object"); done < "$task_root/build/reused-inputs.sha256"
for task_object in "$task_root/build"/*.o; do
    if [[ $(basename "$task_object") != eval_worker.o ]]; then task_objects+=("$task_object"); fi
done
sha256sum "${task_objects[@]}" "$task_here/direct_api_test.cpp" > "$task_build/inputs.sha256"
"$task_cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_121 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false \
    -Xcompiler=-ffp-contract=off -I"$task_root/source/ocean/rek_g1/native5" -I"$task_root/source/ocean/rek_g1" -I"$task_root/source" \
    -I"$task_cuda/include/cccl" -c "$task_here/direct_api_test.cpp" -o "$task_build/direct_api_test.o" \
    > "$task_build/compile.stdout.txt" 2> "$task_build/compile.stderr.txt"
"$task_cuda/bin/nvcc" -arch=sm_121 "${task_objects[@]}" "$task_build/direct_api_test.o" \
    -L"$task_mujoco" -L"$task_cuda/lib64" -L"$task_cuda/lib64/stubs" \
    -Xlinker=-rpath -Xlinker="$task_mujoco" -Xlinker=-rpath -Xlinker="$task_cuda/lib64" \
    -lcudart -lcuda -lcublas -lcrypto -l:libmujoco.so.3.7.0 -lEGL -lGL -lz -lm -lpthread \
    -o "$task_build/direct-api-test" > "$task_build/link.stdout.txt" 2> "$task_build/link.stderr.txt"
sha256sum --quiet -c "$task_build/inputs.sha256"
sha256sum "$task_build/direct-api-test" "$task_build/direct_api_test.o" > "$task_build/outputs.sha256"
