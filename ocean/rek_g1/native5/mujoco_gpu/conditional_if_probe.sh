#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 ]] || { printf 'Usage: %s NEW_OUTPUT\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir "$1"
task_output=$(realpath "$1")
task_cuda=${CUDA_HOME:-/usr/local/cuda}
task_compile=("$task_cuda/bin/nvcc" -std=c++17 -arch=sm_121 -ptx -DREK_CONDITIONAL_TEST=1 "$task_source/conditional.cu" -o "$task_output/conditional-test.ptx")
printf '%q ' "${task_compile[@]}" > "$task_output/build-command.txt"
printf '\n' >> "$task_output/build-command.txt"
"${task_compile[@]}" > "$task_output/build.stdout.txt" 2> "$task_output/build.stderr.txt"
task_compile=(g++ -std=c++17 -O2 -I"$task_cuda/include" "$task_source/native_module.cpp"
    "$task_source/conditional_if.cpp" "$task_source/conditional_if_probe.cpp"
    -L"$task_cuda/lib64/stubs" -lcuda -lcrypto -o "$task_output/conditional-if-probe")
printf '%q ' "${task_compile[@]}" >> "$task_output/build-command.txt"
printf '\n' >> "$task_output/build-command.txt"
"${task_compile[@]}" >> "$task_output/build.stdout.txt" 2>> "$task_output/build.stderr.txt"
task_hash=$(sha256sum "$task_output/conditional-test.ptx" | cut -d' ' -f1)
task_command=("$task_output/conditional-if-probe" "$task_output/conditional-test.ptx" "$task_hash")
printf '%q ' "${task_command[@]}" > "$task_output/command.txt"
printf '\n' >> "$task_output/command.txt"
{ hostname;uname -m;nvidia-smi -L;sha256sum "$task_source/conditional.cu" "$task_source/conditional_if.cpp" "$task_source/conditional_if.h" "$task_source/conditional_if_probe.cpp" "$task_output/conditional-test.ptx" "$task_output/conditional-if-probe"; } > "$task_output/provenance.txt"
readelf -d "$task_output/conditional-if-probe" > "$task_output/elf-dependencies.txt"
set +e
"${task_command[@]}" > "$task_output/result.json" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
cat "$task_output/result.json" "$task_output/stderr.txt"
exit "$task_status"
