#!/usr/bin/env bash
set -euo pipefail
[[ $# == 4 ]] || { printf 'Usage: %s NEW_OUTPUT FEATURE_BUILD MODEL.xml EXPORT.json\n' "$0" >&2;exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_output=$1
task_build=$(realpath "$2")
task_model=$(realpath "$3")
task_export=$(realpath "$4")
task_cuda=${CUDA_HOME:-/usr/local/cuda}
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
: "${REK_MUJOCO_KERNEL_CATALOG:?Required native kernel catalog}"
: "${REK_MUJOCO_CONDITIONAL_PTX:?Required conditional PTX}"
: "${REK_MUJOCO_CONDITIONAL_SHA256:?Required conditional PTX SHA256}"
mkdir "$task_output"
task_output=$(realpath "$task_output")
task_objects=("$task_build/physics.o" "$task_build"/mujoco_*.o "$task_build/cJSON.o")
for task_object in "${task_objects[@]}"; do [[ -f "$task_object" ]] || { printf 'Missing object: %s\n' "$task_object" >&2;exit 2; };done
task_compile=("$task_cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_121 -I"$task_mujoco/include"
    -c "$task_source/physics_mujoco_gpu_probe.cu" -o "$task_output/probe.o")
task_link=(g++ "$task_output/probe.o" "${task_objects[@]}" -L"$task_cuda/lib64" -L"$task_cuda/lib64/stubs"
    -L"$task_mujoco" -Wl,-rpath,"$task_mujoco" -Wl,-rpath,"$task_cuda/lib64"
    -Wl,--wrap=mj_step -Wl,--wrap=mj_forward -Wl,--wrap=mj_kinematics
    -lcudart -lcuda -lcrypto -l:libmujoco.so.3.7.0 -o "$task_output/physics-mujoco-gpu-probe")
{ printf '%q ' "${task_compile[@]}";printf '\n';printf '%q ' "${task_link[@]}";printf '\n'; } > "$task_output/build-command.txt"
"${task_compile[@]}" > "$task_output/build.stdout.txt" 2> "$task_output/build.stderr.txt"
"${task_link[@]}" >> "$task_output/build.stdout.txt" 2>> "$task_output/build.stderr.txt"
readelf -d "$task_output/physics-mujoco-gpu-probe" > "$task_output/elf-dependencies.txt"
if rg -qi '(libpython|libtorch|warp\.so)' "$task_output/elf-dependencies.txt";then printf 'Unexpected runtime dependency\n' >&2;exit 2;fi
task_run=("$task_output/physics-mujoco-gpu-probe" "$task_model" "$task_export")
printf '%q ' "${task_run[@]}" > "$task_output/command.txt"
{ date -u --iso-8601=seconds;hostname;id;uname -m;nvidia-smi -L;
  sha256sum "$task_model" "$task_export" "$REK_MUJOCO_KERNEL_CATALOG" "$REK_MUJOCO_CONDITIONAL_PTX" "$task_output/physics-mujoco-gpu-probe";
  sha256sum "$task_source/physics.cu" "$task_source/physics.cuh" "$task_source/physics_mujoco_gpu.cuh" "$task_source/physics_mujoco_gpu_probe.cu" "${task_objects[@]}"; } > "$task_output/provenance.txt"
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" "${task_run[@]}" > "$task_output/result.json" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
cat "$task_output/result.json" "$task_output/stderr.txt"
exit "$task_status"
