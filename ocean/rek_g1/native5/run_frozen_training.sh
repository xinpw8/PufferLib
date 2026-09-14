#!/usr/bin/env bash
# Native CUDA training against a frozen native5 policy or the scripted baseline.
set -euo pipefail
[[ $# == 3 ]] || { printf 'Usage: %s BUILD NEW_OUTPUT TOTAL_STEPS\n' "$0" >&2; exit 2; }
task_build=$(realpath "$1")
task_output=$2
task_steps=$3
task_backend=${REK_TRAINING_BACKEND:-puffysics}
task_opponent_mode=${REK_TRAINING_OPPONENT:-frozen}
[[ "$task_backend" == puffysics || "$task_backend" == mujoco_cuda ]] || { printf 'Unsupported GPU training backend\n' >&2; exit 2; }
[[ "$task_opponent_mode" == frozen || "$task_opponent_mode" == scripted ]] || { printf 'Opponent must be frozen or scripted\n' >&2; exit 2; }
if [[ "$task_backend" == mujoco_cuda ]]; then
    : "${REK_MUJOCO_KERNEL_CATALOG:?Native MuJoCo kernel catalog is required}"
    : "${REK_MUJOCO_CONDITIONAL_PTX:?Native conditional PTX is required}"
    : "${REK_MUJOCO_CONDITIONAL_SHA256:?Native conditional PTX hash is required}"
fi
[[ "$task_steps" =~ ^[1-9][0-9]*$ && $((task_steps % 8192)) == 0 ]] || { printf 'TOTAL_STEPS must be a positive multiple of 8192\n' >&2; exit 2; }
task_downsample=$((task_steps / 8192))
if (( task_downsample > 64 )); then task_downsample=64; fi
mkdir "$task_output"
task_output=$(realpath "$task_output")
task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
task_models=/home/spark-advantage/rek-training/gpu-runtime-20260910/controller-batch1024-20260910
task_opponent=/home/spark-advantage/rek-training/native5-rek-20260913-v1/final-4096/checkpoints/rek_native5/native5-probe/0000000000196608.bin
task_opponent_hash=0b6acd72fd1a26117e60d5e717e316342f73d490b94d60eecc3f49a9544fa4f3
task_physics_export=/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
task_features=/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
task_executable="$task_build/puffer-rek-native5"
task_opponent_flags=(--env.opponent_checkpoint=None)
if [[ "$task_opponent_mode" == frozen ]]; then
    task_opponent_flags=(--env.opponent_checkpoint="$task_opponent" --env.opponent_sha256="$task_opponent_hash"
        --env.opponent_observation_encoding=scaled_polar_xy
        --env.opponent_hidden_size=256 --env.opponent_num_layers=2
        --env.opponent_precision=0 --env.opponent_legacy_fast_hidden=0 --env.opponent_deterministic=0)
fi
task_runid=native5-frozen20s
if [[ "$task_backend" != puffysics || "$task_opponent_mode" != frozen ]]; then task_runid="native5-$task_backend-$task_opponent_mode-20s"; fi
task_backend_environment=()
if [[ "$task_backend" == mujoco_cuda ]]; then
    task_backend_environment=("REK_MUJOCO_KERNEL_CATALOG=$REK_MUJOCO_KERNEL_CATALOG"
        "REK_MUJOCO_CONDITIONAL_PTX=$REK_MUJOCO_CONDITIONAL_PTX"
        "REK_MUJOCO_CONDITIONAL_SHA256=$REK_MUJOCO_CONDITIONAL_SHA256")
fi
task_command=(env REK_PHYSICS_BACKEND="$task_backend" REK_PUFFYSICS_STABILIZATION=joint_cold_start REK_ALLOW_CPU_EVALUATION=0
    "${task_backend_environment[@]}"
    "$task_executable" train --headless
    --vec.total_agents=512 --train.horizon=16 --train.minibatch_size=8192
    --train.total_timesteps="$task_steps" --sweep.metric=perf/train --sweep.downsample="$task_downsample"
    --base.run_id="$task_runid" --base.checkpoint_interval=8
    --base.log_dir="$task_output/logs" --base.checkpoint_dir="$task_output/checkpoints"
    --env.round_seconds=20 --env.seed=73
    --env.model_path="$task_assets/model.two_fighter_arena.xml"
    --env.physics_export_path="$task_physics_export" --env.assets_path="$task_assets"
    --env.motion_features_path="$task_features"
    --env.controller_encoder_path="$task_models/model_encoder.batch1024.onnx"
    --env.controller_decoder_path="$task_models/model_decoder.batch1024.onnx"
    "${task_opponent_flags[@]}")
printf '%q ' "${task_command[@]}" > "$task_output/command.txt"
printf '\n' >> "$task_output/command.txt"
{
    date -u --iso-8601=seconds
    hostname
    id
    uname -m
    nvidia-smi -L
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
    printf 'physics_backend=%s\nopponent_mode=%s\n' "$task_backend" "$task_opponent_mode"
    if [[ "$task_backend" == mujoco_cuda ]]; then
        sha256sum "$REK_MUJOCO_KERNEL_CATALOG" "$REK_MUJOCO_CONDITIONAL_PTX"
        printf 'conditional_expected_sha256=%s\n' "$REK_MUJOCO_CONDITIONAL_SHA256"
    fi
    if [[ "$task_opponent_mode" == frozen ]]; then sha256sum "$task_opponent"; fi
    sha256sum "$task_executable" "$task_assets/model.two_fighter_arena.xml" \
        "$task_physics_export" "$task_models/model_encoder.batch1024.onnx" "$task_models/model_decoder.batch1024.onnx"
} > "$task_output/host-and-models.txt"
find "$task_assets" "$task_features" -type f -exec sha256sum {} + > "$task_output/asset-hashes.txt"
readelf -d "$task_executable" > "$task_output/elf-dependencies.txt"
if rg -qi '(libpython|libtorch)' "$task_output/elf-dependencies.txt"; then
    printf 'Interpreter/framework dependency detected\n' >&2
    exit 2
fi
cp "$task_build/config/default.ini" "$task_output/default.ini"
cp "$task_build/config/rek_native5.ini" "$task_output/rek_native5.ini"
cd "$task_build"
ulimit -c 0
set +e
/usr/bin/time -v -o "$task_output/process-timing.txt" \
    timeout --signal=TERM --kill-after=10s 360s "${task_command[@]}" \
    > "$task_output/stdout.txt" 2> "$task_output/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
if [[ -d "$task_output/checkpoints" ]]; then
    find "$task_output/checkpoints" -type f -name '*.bin' -exec sha256sum {} + > "$task_output/checkpoint-hashes.txt"
fi
sed -n 's/^native5_round_summary=//p' "$task_output/stdout.txt" > "$task_output/round-summary.json"
tail -20 "$task_output/stdout.txt"
cat "$task_output/stderr.txt"
printf 'native_training_exit_code=%s\noutput=%s\n' "$task_status" "$task_output"
exit "$task_status"
