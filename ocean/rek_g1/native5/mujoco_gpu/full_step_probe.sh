#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 || $# == 4 ]] || { printf 'Usage: %s NEW_OUTPUT REK_MODEL_XML KERNEL_CATALOG_JSON [MAX_STEPS=1000]\n' "$0" >&2; exit 2; }
task_max_steps=${4:-1000}
[[ "$task_max_steps" == 1 || "$task_max_steps" == 10 || "$task_max_steps" == 1000 ]] || { printf 'MAX_STEPS must be1,10,or1000\n' >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_root=${REK_NATIVE5_ROOT:-$(cd "$task_source/../../../.." && pwd)}
task_model=$(realpath "$2")
task_catalog=$(realpath "$3")
task_cuda=${CUDA_HOME:-/usr/local/cuda}
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
[[ -f "$task_root/vendor/cJSON.c" ]] || { printf 'Set REK_NATIVE5_ROOT to the PufferLib checkout containing vendor/cJSON.c\n' >&2; exit 2; }
mkdir "$1"
task_output=$(realpath "$1")
task_cjson=(gcc -std=c11 -O2 -c "$task_root/vendor/cJSON.c" -o "$task_output/cJSON.o")
task_conditional=("$task_cuda/bin/nvcc" -std=c++17 -arch=sm_121 -ptx "$task_source/conditional.cu" -o "$task_output/conditional.ptx")
task_compile=(g++ -std=c++17 -O2 -ffp-contract=off -I"$task_cuda/include" -I"$task_mujoco/include" -I"$task_root/vendor"
    "$task_source/native_module.cpp" "$task_source/model_data.cpp" "$task_source/schedule.cpp"
    "$task_source/collision_schedule.cpp" "$task_source/constraint_schedule.cpp" "$task_source/solver_schedule.cpp"
    "$task_source/kernel_program.cpp" "$task_source/conditional.cpp" "$task_source/native_step.cpp"
    "$task_source/full_step_probe.cpp" "$task_output/cJSON.o"
    -L"$task_cuda/lib64/stubs" -L"$task_mujoco" -Wl,-rpath,"$task_mujoco" -lcuda -lcrypto -l:libmujoco.so.3.7.0
    -o "$task_output/full-step-probe")
{ printf '%q ' "${task_cjson[@]}"; printf '\n'; printf '%q ' "${task_conditional[@]}"; printf '\n';
  printf '%q ' "${task_compile[@]}"; printf '\n'; } > "$task_output/build-commands.txt"
"${task_cjson[@]}" > "$task_output/build.stdout.txt" 2> "$task_output/build.stderr.txt"
"${task_conditional[@]}" >> "$task_output/build.stdout.txt" 2>> "$task_output/build.stderr.txt"
"${task_compile[@]}" >> "$task_output/build.stdout.txt" 2>> "$task_output/build.stderr.txt"
readelf -d "$task_output/full-step-probe" > "$task_output/elf-dependencies.txt"
if rg -qi '(libpython|libtorch|warp\.so)' "$task_output/elf-dependencies.txt"; then printf 'Unexpected runtime framework dependency\n' >&2; exit 2; fi
task_hash=$(sha256sum "$task_output/conditional.ptx" | cut -d' ' -f1)
task_command=("$task_output/full-step-probe" "$task_model" "$task_catalog" "$task_output/conditional.ptx" "$task_hash" "$task_max_steps")
printf '%q ' "${task_command[@]}" > "$task_output/command.txt"
printf '\n' >> "$task_output/command.txt"
{ date -u --iso-8601=seconds; hostname; id; uname -m; nvidia-smi -L;
  sha256sum "$task_model" "$task_catalog" "$task_output/full-step-probe" "$task_output/conditional.ptx" "$task_mujoco/libmujoco.so.3.7.0";
  find "$task_source" -maxdepth 1 -type f -exec sha256sum {} +; } > "$task_output/provenance.txt"
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" "${task_command[@]}" > "$task_output/result.jsonl" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
cat "$task_output/result.jsonl" "$task_output/stderr.txt"
exit "$task_status"
