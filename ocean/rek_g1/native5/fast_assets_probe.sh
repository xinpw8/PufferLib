#!/usr/bin/env bash
set -euo pipefail
[[ $# == 4 ]] || { printf 'Usage: %s NEW_OUTPUT MODEL_XML ASSETS_DIRECTORY FEATURES_DIRECTORY\n' "$0" >&2;exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_root=${REK_NATIVE5_ROOT:-$(cd "$task_source/../../.." && pwd)}
task_cuda=${CUDA_HOME:-/usr/local/cuda}
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
mkdir "$1"
task_output=$(realpath "$1")
task_cjson=(gcc -std=c11 -O2 -c "$task_root/vendor/cJSON.c" -o "$task_output/cJSON.o")
for task_unit in g1_strike_catalog native_motion_routes;do
  gcc -std=c11 -O2 -c "$task_source/../$task_unit.c" -o "$task_output/$task_unit.o"
done
task_compile=(g++ -std=c++17 -O2 -Wall -Wextra -I"$task_cuda/include" -I"$task_mujoco/include"
  "$task_source/fast_assets.cpp" "$task_source/fast_assets_probe.cpp" "$task_output/cJSON.o" "$task_output/g1_strike_catalog.o" "$task_output/native_motion_routes.o"
  -L"$task_mujoco" -Wl,-rpath,"$task_mujoco" -l:libmujoco.so.3.7.0 -lcrypto
  -Wl,--wrap=mj_step,--wrap=mj_forward,--wrap=mj_step1,--wrap=mj_step2 -o "$task_output/fast-assets-probe")
task_command=("$task_output/fast-assets-probe" "$(realpath "$2")" "$(realpath "$3")" "$(realpath "$4")")
{ printf '%q ' "${task_cjson[@]}";printf '\n';printf '%q ' "${task_compile[@]}";printf '\n';printf '%q ' "${task_command[@]}";printf '\n'; } > "$task_output/commands.txt"
"${task_cjson[@]}" > "$task_output/build.stdout.txt" 2> "$task_output/build.stderr.txt"
"${task_compile[@]}" >> "$task_output/build.stdout.txt" 2>> "$task_output/build.stderr.txt"
{ hostname;id;date -u --iso-8601=seconds;sha256sum "$task_source/fast_assets.h" "$task_source/fast_assets.cpp" "$task_source/fast_assets_probe.cpp" "$task_output/fast-assets-probe"; } > "$task_output/provenance.txt"
readelf -d "$task_output/fast-assets-probe" > "$task_output/dependencies.txt"
if rg -qi '(libpython|libtorch|libcuda|libcudart)' "$task_output/dependencies.txt";then printf 'Unexpected execution dependency\n' >&2;exit 2;fi
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" "${task_command[@]}" > "$task_output/result.jsonl" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
cat "$task_output/result.jsonl" "$task_output/stderr.txt"
exit "$task_status"
