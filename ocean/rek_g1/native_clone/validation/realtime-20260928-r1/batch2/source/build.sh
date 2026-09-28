#!/usr/bin/env bash
set -euo pipefail
task_root=/home/spark-advantage/rek-training/rek-native-runtime-speed-20260928-r1/crossbatch-r1
task_build=$task_root/build-r1
task_object=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/build-native-v2/sonic_controller.o
task_headers=/home/spark-advantage/rek-training/rek-playback-native-20260928-r1/source/ocean/rek_g1/native5
test ! -e "$task_build"
test "$(sha256sum "$task_object" | cut -d' ' -f1)" = 22f87a417bd5ed43fabae1188ed3cd444571e55f6af26e0ea34ac085bdeb4bc4
mkdir "$task_build"
sha256sum "$task_object" "$task_headers/sonic_controller.cuh" "$task_root/crossbatch.cu" > "$task_build/inputs.sha256"
/usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_121 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false \
 -Xcompiler=-ffp-contract=off -I"$task_headers" "$task_root/crossbatch.cu" "$task_object" \
 -lcudart -lcublas -lcrypto -o "$task_build/crossbatch" > "$task_build/compile.stdout.txt" 2> "$task_build/compile.stderr.txt"
sha256sum --quiet -c "$task_build/inputs.sha256"
sha256sum "$task_build/crossbatch" > "$task_build/outputs.sha256"
"$task_build/crossbatch" --self-test > "$task_build/cpu-self-test.json" 2> "$task_build/cpu-self-test.stderr.txt"
