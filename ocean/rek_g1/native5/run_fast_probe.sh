#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s FAST_BUILD NEW_OUTPUT\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$(realpath "$1")
mkdir "$2"
task_output=$(realpath "$2")
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
task_export=/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
task_features=/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
task_compile=("$task_cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_121 -I"$task_source"
    "$task_source/fast_probe.cu" "$task_build/fast_runtime.o" "$task_build/fast_assets.o" "$task_build/cJSON.o"
    -L"$task_mujoco" -Xlinker=-rpath -Xlinker="$task_mujoco" -l:libmujoco.so.3.7.0 -lcrypto
    -o "$task_output/fast-probe")
printf '%q ' "${task_compile[@]}" > "$task_output/build-command.txt"
"${task_compile[@]}" > "$task_output/build.stdout.txt" 2> "$task_output/build.stderr.txt"
task_run=("$task_output/fast-probe" "$task_assets/model.two_fighter_arena.xml" "$task_export" "$task_assets" "$task_features")
printf '%q ' "${task_run[@]}" > "$task_output/command.txt"
{ hostname; id; date -u --iso-8601=seconds; nvidia-smi -L;
  sha256sum "$task_output/fast-probe" "$task_build/fast_runtime.o" "$task_build/fast_assets.o";
} > "$task_output/provenance.txt"
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" timeout --signal=TERM --kill-after=5s 90s \
    "${task_run[@]}" > "$task_output/result.json" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
cat "$task_output/result.json" "$task_output/stderr.txt"
exit "$task_status"
