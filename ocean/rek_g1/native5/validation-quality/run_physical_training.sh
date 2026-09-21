#!/usr/bin/env bash
set -euo pipefail
task_stage=/home/spark-advantage/rek-training/physical-fall-exposure-20260921-r1
task_build=$task_stage/build-training-r1
task_out=$task_stage/train-physical-normalized-r1
task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
task_models=/home/spark-advantage/rek-training/gpu-runtime-20260910/controller-batch1024-20260910
task_export=/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
task_features=/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
task_initial=/home/spark-advantage/rek-training/owned-yaw-migration-20260920-r1/matched-gae-r1/train-control-v1/ppo.bin
[[ $(sha256sum "$task_initial" | cut -d' ' -f1) == f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4 ]]
[[ $(stat -c%s "$task_initial") == 1836032 ]]
mkdir "$task_out"
export REK_PHYSICS_BACKEND=mujoco_cuda REK_NATIVE5_REWARD=normalized_points_falls_v1 REK_ALLOW_CPU_EVALUATION=0
export REK_OBSERVATION_SCHEMA=rek.native5.scaled_polar_xy.v1
export REK_MUJOCO_KERNEL_CATALOG=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/kernel-catalog-v3.json
export REK_MUJOCO_CONDITIONAL_PTX=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/build-native-v2/mujoco-conditional.ptx
export REK_MUJOCO_CONDITIONAL_SHA256=2300284dfc4f4560234ef6ffe8a3fab8c918a1abc8ffcdcf270e180042837b21
unset REK_PUFFYSICS_STABILIZATION REK_POLICY_FEATURE_MASK REK_POLICY_ACTION_STRIDE REK_FROZEN_OPPONENT_FRACTION
task_run=("$task_build/puffer-rek-physical-guarded" train --headless
 --vec.total_agents=512 --train.horizon=512 --train.minibatch_size=8192 --train.total_timesteps=4194304
 --train.gamma=.9998844821426083 --train.gae_lambda=.9978673240629938 --train.learning_rate=.0001 --train.ent_coef=.01
 --policy.hidden_size=256 --policy.num_layers=2 --base.seed=419 --env.seed=419
 --sweep.metric=perf/train --sweep.downsample=16 --base.run_id=physical-normalized-r1 --base.checkpoint_interval=1
 --base.log_dir="$task_out/logs" --base.checkpoint_dir="$task_out/checkpoints" --base.load_model_path="$task_initial"
 --env.round_seconds=120 --env.opponent_checkpoint=None
 --env.model_path="$task_assets/model.two_fighter_arena.xml" --env.physics_export_path="$task_export" --env.assets_path="$task_assets"
 --env.motion_features_path="$task_features" --env.controller_encoder_path="$task_models/model_encoder.batch1024.onnx" --env.controller_decoder_path="$task_models/model_decoder.batch1024.onnx")
{ printf '%s\n' "REK_PHYSICS_BACKEND=$REK_PHYSICS_BACKEND" "REK_NATIVE5_REWARD=$REK_NATIVE5_REWARD" "REK_ALLOW_CPU_EVALUATION=$REK_ALLOW_CPU_EVALUATION" "REK_OBSERVATION_SCHEMA=$REK_OBSERVATION_SCHEMA";printf '%q ' "${task_run[@]}";printf '\n'; } > "$task_out/command.txt"
sha256sum "$task_build/puffer-rek-physical-guarded" "$task_initial" "$task_assets/model.two_fighter_arena.xml" "$task_export" "$task_models"/*.onnx "$REK_MUJOCO_KERNEL_CATALOG" "$REK_MUJOCO_CONDITIONAL_PTX" > "$task_out/provenance.txt"
cp "$task_build/config/default.ini" "$task_build/config/rek_native5.ini" "$task_out/"
cd "$task_build"
ulimit -c 0
date -u --iso-8601=ns > "$task_out/start-utc.txt"
set +e
/usr/bin/time -v -o "$task_out/process-timing.txt" timeout --signal=TERM --kill-after=10s 900 "${task_run[@]}" > "$task_out/stdout.txt" 2> "$task_out/stderr.txt"
task_status=$?
set -e
date -u --iso-8601=ns > "$task_out/end-utc.txt"
printf '%s\n' "$task_status" > "$task_out/exit-code.txt"
find "$task_out/checkpoints" -type f -name '*.bin' -exec sha256sum {} + > "$task_out/checkpoint-hashes.txt"
sed -n 's/^native5_round_summary=//p' "$task_out/stdout.txt" > "$task_out/round-summary.json"
task_readback=$task_out/checkpoints/rek_native5/physical-normalized-r1/0000000000000000.bin
if [[ -f "$task_readback" ]];then cmp "$task_initial" "$task_readback";sha256sum "$task_initial" "$task_readback" > "$task_out/verified-warm-start.txt";fi
sha256sum "$task_out"/*.txt "$task_out/round-summary.json" > "$task_out/artifact-sha256.txt"
exit "$task_status"
