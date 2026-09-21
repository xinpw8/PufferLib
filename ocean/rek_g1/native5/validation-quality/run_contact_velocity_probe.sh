#!/usr/bin/env bash
# CPU-only, fresh output; based on the existing fast_assets_probe.sh workflow.
set -euo pipefail
[[ $# == 4 ]] || { printf 'Usage: %s NEW_OUTPUT MODEL_XML ASSETS_DIRECTORY FEATURES_DIRECTORY\n' "$0" >&2; exit 2; }
task_probe=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_source=$(cd "$task_probe/.." && pwd)
task_root=$(cd "$task_source/../../.." && pwd)
task_cuda=${CUDA_HOME:-/usr/local/cuda}
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
mkdir "$1"
task_output=$(realpath "$1")
exec 3> "$task_output/commands.txt"
run_build() { printf '%q ' "$@" >&3; printf '\n' >&3; "$@" >> "$task_output/build.stdout.txt" 2>> "$task_output/build.stderr.txt"; }
run_build gcc -std=c11 -O2 -c "$task_root/vendor/cJSON.c" -o "$task_output/cJSON.o"
for task_unit in g1_strike_catalog native_motion_routes; do
    run_build gcc -std=c11 -O2 -c "$task_source/../$task_unit.c" -o "$task_output/$task_unit.o"
done
run_build g++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -I"$task_cuda/include" -I"$task_mujoco/include" \
    "$task_source/fast_assets.cpp" "$task_probe/contact_velocity_probe.cpp" "$task_output/cJSON.o" \
    "$task_output/g1_strike_catalog.o" "$task_output/native_motion_routes.o" \
    -L"$task_mujoco" -Wl,-rpath,"$task_mujoco" -l:libmujoco.so.3.7.0 -lcrypto \
    -Wl,--wrap=mj_step,--wrap=mj_forward,--wrap=mj_step1,--wrap=mj_step2 -o "$task_output/contact-velocity-probe"
task_command=("$task_output/contact-velocity-probe" "$(realpath "$2")" "$(realpath "$3")" "$(realpath "$4")")
printf '%q ' "${task_command[@]}" >&3; printf '\n' >&3
{ hostname; date -u --iso-8601=ns; g++ --version; sha256sum "$task_source/fast_assets.h" "$task_source/fast_assets.cpp" \
    "$task_source/runtime_api.h" "$task_source/primitive_contacts.cuh" "$task_source/native_contact_geometry.h" \
    "$task_source/recovered_contact_rules.cuh" "$task_source/../g1_strike_catalog.c" "$task_source/../g1_strike_catalog.h" \
    "$task_source/../native_motion_routes.c" "$task_source/../native_motion_routes.h" "$task_source/../g1_hit_detector.c" \
    "$task_source/../g1_hit_detector.h" "$task_source/../g1_combat_types.h" "$task_source/../g1_cuda_qualifiers.h" \
    "$task_root/vendor/cJSON.c" "$task_root/vendor/cJSON.h" "$task_probe/contact_velocity_probe.cpp" "$0" \
    "$task_mujoco/libmujoco.so.3.7.0" "$task_output/contact-velocity-probe" "$2" \
    "$3/semantic_duel_assets_manifest.json" "$4/foot_features_manifest.json"; } > "$task_output/provenance.txt"
readelf -d "$task_output/contact-velocity-probe" > "$task_output/dependencies.txt"
if rg -qi '(libpython|libtorch|libcuda|libcudart)' "$task_output/dependencies.txt"; then printf 'Unexpected execution dependency\n' >&2; exit 2; fi
date -u --iso-8601=ns > "$task_output/start.txt"
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" timeout 120 "${task_command[@]}" > "$task_output/result.jsonl" 2> "$task_output/stderr.txt"
task_status=$?
set -e
date -u --iso-8601=ns > "$task_output/end.txt"
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
sha256sum "$task_output/result.jsonl" "$task_output/stderr.txt" "$task_output/commands.txt" "$task_output/provenance.txt" > "$task_output/result-hashes.txt"
cat "$task_output/result.jsonl" "$task_output/stderr.txt"
exit "$task_status"
