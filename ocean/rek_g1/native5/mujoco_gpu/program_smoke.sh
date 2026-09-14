#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { printf 'Usage: %s NEW_OUTPUT REK_MODEL_XML KERNEL_CATALOG_JSON\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_root=$(cd "$task_source/../../../.." && pwd)
task_output=$1
task_model=$(realpath "$2")
task_catalog=$(realpath "$3")
task_cuda=${CUDA_HOME:-/usr/local/cuda}
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
mkdir "$task_output"
task_output=$(realpath "$task_output")
task_cjson=(gcc -std=c11 -O2 -c "$task_root/vendor/cJSON.c" -o "$task_output/cJSON.o")
task_compile=(g++ -std=c++17 -O2 -ffp-contract=off -I"$task_cuda/include" -I"$task_mujoco/include" -I"$task_root/vendor"
    "$task_source/native_module.cpp" "$task_source/model_data.cpp" "$task_source/schedule.cpp"
    "$task_source/kernel_program.cpp" "$task_source/program_smoke.cpp" "$task_output/cJSON.o"
    -L"$task_cuda/lib64/stubs" -L"$task_mujoco" -Wl,-rpath,"$task_mujoco"
    -lcuda -lcrypto -l:libmujoco.so.3.7.0 -o "$task_output/program-smoke")
{ printf '%q ' "${task_cjson[@]}"; printf '\n'; printf '%q ' "${task_compile[@]}"; printf '\n'; } > "$task_output/build-command.txt"
"${task_cjson[@]}" > "$task_output/build.stdout.txt" 2> "$task_output/build.stderr.txt"
"${task_compile[@]}" >> "$task_output/build.stdout.txt" 2>> "$task_output/build.stderr.txt"
readelf -d "$task_output/program-smoke" > "$task_output/elf-dependencies.txt"
if rg -qi '(libpython|libtorch|warp\.so)' "$task_output/elf-dependencies.txt"; then printf 'Unexpected runtime framework dependency\n' >&2;exit 2;fi
task_command=("$task_output/program-smoke" "$task_model" "$task_catalog")
printf '%q ' "${task_command[@]}" > "$task_output/command.txt"
printf '\n' >> "$task_output/command.txt"
{ date -u --iso-8601=seconds;hostname;id;uname -m;nvidia-smi -L;
  sha256sum "$task_model" "$task_catalog" "$task_output/program-smoke" "$task_mujoco/libmujoco.so.3.7.0";
  find "$task_source" -maxdepth 1 -type f -exec sha256sum {} +; } > "$task_output/provenance.txt"
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" "${task_command[@]}" > "$task_output/result.json" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
cat "$task_output/result.json" "$task_output/stderr.txt"
exit "$task_status"
