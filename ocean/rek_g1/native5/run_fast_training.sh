#!/usr/bin/env bash
# Headless reduced-state training. One SPS unit is one 20 ms learner transition.
set -euo pipefail
[[ $# == 4 || $# == 5 ]] || { printf 'Usage: %s BUILD NEW_OUTPUT TOTAL_STEPS ARENAS [HORIZON=16]\n' "$0" >&2; exit 2; }
task_build=$(realpath "$1")
task_output=$2
task_steps=$3
task_arenas=$4
task_horizon=${5:-16}
for task_number in "$task_steps" "$task_arenas" "$task_horizon"; do
    [[ "$task_number" =~ ^[1-9][0-9]*$ ]] || { printf 'Positive integer sizes required\n' >&2; exit 2; }
done
task_batch=$((task_arenas*task_horizon))
(( task_steps%task_batch==0 )) || { printf 'TOTAL_STEPS must divide into complete arena*horizon batches\n' >&2; exit 2; }
task_minibatch=${REK_FAST_MINIBATCH:-$task_batch}
task_limit=${REK_FAST_TIMEOUT_SECONDS:-300}
task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
task_features=/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
task_export=/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
task_executable="$task_build/puffer-rek-native5"
mkdir "$task_output"
task_output=$(realpath "$task_output")
task_epochs=$((task_steps/task_batch))
task_downsample=$task_epochs
if (( task_downsample>64 )); then task_downsample=64; fi
task_command=(env REK_PHYSICS_BACKEND=semantic_cuda REK_ALLOW_CPU_EVALUATION=0
    "$task_executable" train --headless --vec.total_agents="$task_arenas"
    --train.horizon="$task_horizon" --train.minibatch_size="$task_minibatch"
    --train.total_timesteps="$task_steps" --sweep.metric=perf/train --sweep.downsample="$task_downsample"
    --base.run_id="semantic-cuda-$task_arenas-$task_horizon" --base.checkpoint_interval=64
    --base.log_dir="$task_output/logs" --base.checkpoint_dir="$task_output/checkpoints"
    --env.round_seconds=20 --env.seed=73 --env.opponent_checkpoint=None
    --env.model_path="$task_assets/model.two_fighter_arena.xml"
    --env.physics_export_path="$task_export" --env.assets_path="$task_assets"
    --env.motion_features_path="$task_features")
printf '%q ' "${task_command[@]}" > "$task_output/command.txt"
printf '\n' >> "$task_output/command.txt"
{ date -u --iso-8601=seconds; hostname; id; uname -m; nvidia-smi -L;
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv;
  printf 'backend=semantic_cuda\ncontrol_hz=50\narenas=%s\nhorizon=%s\nminibatch=%s\n' "$task_arenas" "$task_horizon" "$task_minibatch";
  sha256sum "$task_executable" "$task_assets/model.two_fighter_arena.xml";
} > "$task_output/provenance.txt"
cp "$task_build/config/default.ini" "$task_build/config/rek_native5.ini" "$task_output/"
cp "$task_build/fast-build.txt" "$task_build/elf-dependencies.txt" "$task_output/"
cd "$task_build"
ulimit -c 0
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" \
    timeout --signal=TERM --kill-after=10s "$task_limit" "${task_command[@]}" \
    > "$task_output/stdout.txt" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
sed -n 's/^native5_round_summary=//p' "$task_output/stdout.txt" > "$task_output/round-summary.json"
if [[ -d "$task_output/checkpoints" ]]; then
    find "$task_output/checkpoints" -type f -name '*.bin' -exec sha256sum {} + > "$task_output/checkpoint-hashes.txt"
fi
tail -20 "$task_output/stdout.txt"
cat "$task_output/stderr.txt"
printf 'training_exit_code=%s\noutput=%s\n' "$task_status" "$task_output"
exit "$task_status"
