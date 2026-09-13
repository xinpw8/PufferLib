#!/usr/bin/env bash
# Headless native training probe using existing, private Spark assets.
set -euo pipefail
[[ $# == 3 ]] || { printf 'Usage: %s BUILD_DIRECTORY NEW_OUTPUT_DIRECTORY TOTAL_STEPS\n' "$0" >&2; exit 2; }
task_build=$(realpath "$1")
task_output=$2
task_steps=$3
mkdir "$task_output"
task_output=$(realpath "$task_output")
task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
task_models=/home/spark-advantage/rek-training/gpu-runtime-20260910/controller-batch1024-20260910
task_arenas=${REK_NATIVE5_ARENAS:-512}
task_horizon=${REK_NATIVE5_HORIZON:-16}
task_fighters=$((task_arenas * 2))
if [[ $task_arenas == 4096 ]]; then
    task_models=/home/spark-advantage/rek-training/gpu-runtime-20260910/controller-batch8192-e06109ec-20260910
elif [[ $task_arenas != 512 ]]; then
    printf 'Probe supports existing 512- and 4096-arena model exports only\n' >&2
    exit 2
fi
task_executable=${REK_NATIVE5_EXECUTABLE:-$task_build/puffer-rek-native5}
task_command=("$task_executable" train --headless
    --vec.total_agents="$task_arenas" --train.horizon="$task_horizon"
    --train.minibatch_size="$((task_arenas * task_horizon))"
    --train.total_timesteps="$task_steps" --sweep.metric=perf/train --sweep.downsample=5
    --base.run_id=native5-probe --base.log_dir="$task_output/logs"
    --base.checkpoint_dir="$task_output/checkpoints"
    --env.model_path="$task_assets/model.two_fighter_arena.xml"
    --env.physics_export_path=/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
    --env.assets_path="$task_assets"
    --env.motion_features_path=/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
    --env.controller_encoder_path="$task_models/model_encoder.batch$task_fighters.onnx"
    --env.controller_decoder_path="$task_models/model_decoder.batch$task_fighters.onnx")
printf '%q ' "${task_command[@]}" > "$task_output/command.txt"
printf '\n' >> "$task_output/command.txt"
{ hostname; id; uname -m; nvidia-smi -L; sha256sum "$task_executable"; } > "$task_output/host.txt"
readelf -d "$task_executable" > "$task_output/elf-dependencies.txt"
cd "$task_build"
if [[ ${REK_NATIVE5_PROFILE:-0} == 1 ]]; then
    task_command=(nsys profile --trace=cuda --sample=none --cpuctxsw=none
        --inherit-environment=false --cuda-graph-trace=node
        --output="$task_output/native-profile" "${task_command[@]}")
    printf '%q ' "${task_command[@]}" > "$task_output/profile-command.txt"
    printf '\n' >> "$task_output/profile-command.txt"
fi
ulimit -c 0
set +e
timeout --signal=TERM --kill-after=10s 90s "${task_command[@]}" > "$task_output/stdout.txt" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
cat "$task_output/stdout.txt" "$task_output/stderr.txt"
printf 'native_training_exit_code=%s\noutput=%s\n' "$task_status" "$task_output"
exit "$task_status"
