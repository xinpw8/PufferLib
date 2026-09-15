#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s BUILD NEW_OUTPUT\n' "$0" >&2;exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$(realpath "$1")
mkdir "$2"
task_output=$(realpath "$2")
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
task_compile=(/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_121 -I"$task_source"
    "$task_source/fast_diversity_probe.cu" "$task_build/fast_runtime.o" "$task_build/fast_assets.o" "$task_build/cJSON.o"
    -L"$task_mujoco" -Xlinker=-rpath -Xlinker="$task_mujoco" -l:libmujoco.so.3.7.0 -lcrypto -o "$task_output/fast-diversity-probe")
task_run=("$task_output/fast-diversity-probe" "$task_assets/model.two_fighter_arena.xml"
    /home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
    "$task_assets" /home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features)
{ printf '%q ' "${task_compile[@]}";printf '\n';printf '%q ' "${task_run[@]}";printf '\n'; } > "$task_output/commands.sh"
"${task_compile[@]}" > "$task_output/build.stdout.txt" 2> "$task_output/build.stderr.txt"
{ hostname;id;date -u --iso-8601=seconds;sha256sum "$task_source/fast_diversity_probe.cu" "$task_source/fast_diversity_probe.sh" "$task_build/fast_runtime.o" "$task_build/fast_assets.o" "$task_output/fast-diversity-probe"; } > "$task_output/provenance.txt"
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" timeout --signal=TERM --kill-after=10s 60s "${task_run[@]}" > "$task_output/stdout.txt" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
cat "$task_output/stdout.txt"
if (( task_status )); then tail -8 "$task_output/stderr.txt";fi
exit "$task_status"
