#!/usr/bin/env bash
set -euo pipefail
[[ $# == 4 ]] || { printf 'Usage: %s NEW_OUTPUT CACHED_PTX WARP_NATIVE_HEADERS NEXT_VELOCITY_SYMBOL\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_output=$1
task_ptx=$(realpath "$2")
task_headers=$(realpath "$3")
task_symbol=$4
mkdir "$task_output"
task_output=$(realpath "$task_output")
task_cuda=${CUDA_HOME:-/usr/local/cuda}
task_compile=(g++ -std=c++17 -O2 -ffp-contract=off -I"$task_cuda/include" -I"$task_headers"
    "$task_source/native_module.cpp" "$task_source/kernel_smoke.cpp"
    -L"$task_cuda/lib64/stubs" -lcuda -lcrypto -o "$task_output/kernel-smoke")
printf '%q ' "${task_compile[@]}" > "$task_output/build-command.txt"
printf '\n' >> "$task_output/build-command.txt"
"${task_compile[@]}" > "$task_output/build.stdout.txt" 2> "$task_output/build.stderr.txt"
readelf -d "$task_output/kernel-smoke" > "$task_output/elf-dependencies.txt"
if rg -qi '(libpython|libtorch|warp\.so)' "$task_output/elf-dependencies.txt"; then
    printf 'Unexpected interpreter/framework runtime dependency\n' >&2; exit 2
fi
task_hash=$(sha256sum "$task_ptx" | cut -d' ' -f1)
task_command=("$task_output/kernel-smoke" "$task_ptx" "$task_hash" "$task_symbol")
printf '%q ' "${task_command[@]}" > "$task_output/command.txt"
printf '\n' >> "$task_output/command.txt"
{ date -u --iso-8601=seconds; hostname; id; uname -m; nvidia-smi -L;
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv;
  sha256sum "$task_source/native_module.h" "$task_source/native_module.cpp" "$task_source/warp_abi.h" \
      "$task_source/kernel_smoke.cpp" "$task_source/kernel_smoke.sh" "$task_headers/array.h" "$task_headers/builtin.h" \
      "$task_ptx" "$task_output/kernel-smoke"; } > "$task_output/provenance.txt"
task_cached_source=${task_ptx%.sm*.ptx}.cu
task_cached_meta=${task_ptx%.sm*.ptx}.meta
if [[ -f "$task_cached_source" ]]; then
    sha256sum "$task_cached_source" >> "$task_output/provenance.txt"
    rg -n '^#define WP_TILE_BLOCK_DIM' "$task_cached_source" > "$task_output/cached-block-dimension.txt"
fi
if [[ -f "$task_cached_meta" ]]; then sha256sum "$task_cached_meta" >> "$task_output/provenance.txt"; fi
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" "${task_command[@]}" > "$task_output/result.json" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
cat "$task_output/result.json" "$task_output/stderr.txt"
exit "$task_status"
