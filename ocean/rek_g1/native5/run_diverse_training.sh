#!/usr/bin/env bash
# Native CUDA curriculum stages. Checkpoints, logs, and assets remain private.
set -euo pipefail
[[ $# == 6 || $# == 7 ]] || { printf 'Usage: %s BUILD NEW_OUTPUT STEPS ARENAS HORIZON ROUND_SECONDS [INITIAL_CHECKPOINT]\n' "$0" >&2;exit 2; }
task_build=$(realpath "$1")
task_output=$2
task_steps=$3
task_arenas=$4
task_horizon=$5
task_seconds=$6
for task_number in "$task_steps" "$task_arenas" "$task_horizon" "$task_seconds";do
    [[ "$task_number" =~ ^[1-9][0-9]*$ ]] || exit 2
done
(( task_steps%(task_arenas*task_horizon)==0 )) || exit 2
task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
task_features=/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
task_export=/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
mkdir "$task_output"
task_output=$(realpath "$task_output")
task_run=$(basename "$task_output")
task_environment=(REK_PHYSICS_BACKEND=semantic_cuda REK_ALLOW_CPU_EVALUATION=0)
# Only these task-specific values are recorded. Never dump the shell environment.
for task_key in REK_FAST_SCORING REK_FAST_OPPONENT_MODE REK_FAST_RANDOM_RESETS REK_FAST_RESET_GAP_MIN REK_FAST_RESET_GAP_MAX REK_FAST_RESET_HEADING_SPREAD_RAD REK_FAST_SHAPING_WEIGHT REK_FAST_SHAPING_GAMMA REK_FAST_SHAPING_TARGET REK_FAST_SHAPING_BEARING_WEIGHT REK_FROZEN_OPPONENT_FRACTION;do
    if [[ -v "$task_key" ]];then task_environment+=("$task_key=${!task_key}");fi
done
task_command=(env "${task_environment[@]}" "$task_build/puffer-rek-native5" train --headless
    --vec.total_agents="$task_arenas" --train.horizon="$task_horizon"
    --train.minibatch_size="${REK_TRAIN_MINIBATCH:-8192}" --train.total_timesteps="$task_steps"
    --train.gamma="${REK_TRAIN_GAMMA:-0.999}" --train.gae_lambda="${REK_TRAIN_GAE_LAMBDA:-0.995}"
    --train.learning_rate="${REK_TRAIN_LEARNING_RATE:-0.0003}" --train.ent_coef="${REK_TRAIN_ENTROPY:-0.01}"
    --sweep.metric=perf/train --sweep.downsample=64 --base.run_id="$task_run"
    --base.checkpoint_interval=64 --base.log_dir="$task_output/logs" --base.checkpoint_dir="$task_output/checkpoints"
    --env.round_seconds="$task_seconds" --env.seed="${REK_TRAIN_SEED:-73}"
    --env.model_path="$task_assets/model.two_fighter_arena.xml" --env.physics_export_path="$task_export"
    --env.assets_path="$task_assets" --env.motion_features_path="$task_features")
if [[ $# == 7 ]];then
    task_initial=$(realpath "$7");test -f "$task_initial"
    task_command+=(--base.load_model_path="$task_initial")
    sha256sum "$task_initial" > "$task_output/initial-checkpoint.sha256"
fi
if [[ -n ${REK_TRAIN_OPPONENT_CHECKPOINT:-} ]];then
    task_opponent=$(realpath "$REK_TRAIN_OPPONENT_CHECKPOINT");test -f "$task_opponent"
    task_opponent_sha=$(sha256sum "$task_opponent" | cut -d' ' -f1)
    task_command+=(--env.opponent_checkpoint="$task_opponent" --env.opponent_sha256="$task_opponent_sha"
        --env.opponent_observation_encoding=scaled_polar_xy --env.opponent_precision=0 --env.opponent_deterministic=0)
else task_command+=(--env.opponent_checkpoint=None);fi
printf '%q ' "${task_command[@]}" > "$task_output/command.txt"
printf '\n' >> "$task_output/command.txt"
{ date -u --iso-8601=seconds;hostname;id;uname -m;nvidia-smi -L;
  sha256sum "$task_build/puffer-rek-native5" "$task_build/pufferl.o" "$task_build/fast_runtime.o";
} > "$task_output/provenance.txt"
cp "$task_build/config/default.ini" "$task_build/config/rek_native5.ini" "$task_output/"
cd "$task_build"
ulimit -c 0
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" timeout --signal=TERM --kill-after=10s "${REK_TRAIN_TIMEOUT:-900}" \
    "${task_command[@]}" > "$task_output/stdout.txt" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
sed -n 's/^native5_round_summary=//p' "$task_output/stdout.txt" > "$task_output/round-summary.json"
if [[ -d "$task_output/checkpoints" ]];then
    find "$task_output/checkpoints" -type f -name '*.bin' -exec sha256sum {} + > "$task_output/checkpoint-hashes.txt"
fi
if [[ $# == 7 && "$task_status" == 0 ]];then
    task_readback=$task_output/checkpoints/rek_native5/$task_run/0000000000000000.bin
    cmp "$task_initial" "$task_readback"
    sha256sum "$task_initial" "$task_readback" > "$task_output/verified-warm-start.txt"
fi
tail -20 "$task_output/stdout.txt"
cat "$task_output/stderr.txt"
printf 'exit_code=%s\noutput=%s\n' "$task_status" "$task_output"
exit "$task_status"
