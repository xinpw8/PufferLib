#!/usr/bin/env bash
set -euo pipefail
[[ $# -le 1 ]] || { printf 'Usage: %s [--check]\n' "$0" >&2; exit 2; }
task_variant=zero_joints
task_coherent=1
task_name=F7-joint-mask-no-prior-16777216
task_check=${1:-}
[[ -z "$task_check" || "$task_check" == --check ]] || exit 2
task_root=/home/spark-advantage/rek-training/joint-mask-transfer-20260924-r1
task_baseline=/home/spark-advantage/rek-training/attack-gate-sweep-20260921-r1/final/F7-interrupt-fromF6
task_run="$task_root/runs/$task_name"
task_exe=/home/spark-advantage/rek-training/f7-action-id-fix-20260924-r1/build/puffer-rek-native5
task_mask="$task_root/zero-joints.bin"
task_mask_hash=d26c5f2f7d2a7faf14f89a7290189fd8fb45223ff4a548abe7be51061ec4bf85
task_checkpoint="$task_baseline/checkpoints/rek_native5/F7-interrupt-fromF6/0000000016777216.bin"
task_checkpoint_hash=6a5082750aee85183bdd23b42470775374c89740d3c8c00445ead2d6d2e39263
task_exe_hash=1754a66278059fdaa207e02d7231bf2a8d7dcc235880b2d055c4d628a344072a
task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
task_steps=16777216
[[ -x "$task_exe" && -x /usr/bin/time && ! -e "$task_run" ]]
[[ $(sha256sum "$task_checkpoint" | awk '{print $1}') == "$task_checkpoint_hash" ]]
[[ $(sha256sum "$task_exe" | awk '{print $1}') == "$task_exe_hash" ]]
[[ -f "$task_baseline/config/default.ini" && -f "$task_baseline/config/rek_native5.ini" ]]

# Restore exactly the recorded REK environment before selecting the one factor.
while IFS= read -r task_key; do unset "$task_key"; done < <(compgen -v REK_)
while IFS='=' read -r task_key task_value; do
  [[ "$task_key" == REK_* && "$task_key" =~ ^[A-Z0-9_]+$ ]] || exit 2
  export "$task_key=$task_value"
done < "$task_baseline/environment.txt"
export REK_FAST_CALIBRATION_COHERENT="$task_coherent"
export REK_FAST_CALIBRATION_INDEPENDENT_RNG=0
export REK_FAST_CALIBRATION_ACTION_IDS=1
export REK_FAST_KICK_FALL_P=0
export REK_POLICY_FEATURE_MASK="$task_mask"
[[ -f "$task_mask" && $(stat -c %s "$task_mask") == 223 ]]
[[ $(sha256sum "$task_mask" | awk '{print $1}') == "$task_mask_hash" ]]

task_cmd=("$task_exe" train --headless
  --vec.total_agents=512 --train.minibatch_size=8192 --train.total_timesteps="$task_steps"
  --train.gamma=.9998844821426083 --train.gae_lambda=.9978673240629938
  --train.learning_rate=5.5e-05 --train.ent_coef=1.7e-04 --train.clip_coef=0.13
  --train.vf_coef=1.02 --train.horizon=128
  --base.run_id="$task_name" --base.seed=73 --base.checkpoint_interval=64
  --base.log_dir="$task_run/logs" --base.checkpoint_dir="$task_run/checkpoints"
  --env.round_seconds=120 --env.seed=419
  --env.model_path="$task_assets/model.two_fighter_arena.xml"
  --env.physics_export_path=/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
  --env.assets_path="$task_assets"
  --env.motion_features_path=/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
  --base.load_model_path="$task_checkpoint" --env.opponent_checkpoint=None)
if [[ "$task_check" == --check ]]; then
  printf 'Validated variant=%s coherent=%s independent_rng=0 action_ids=1 kick_fall_p=0 checkpoint_sha256=%s\n' "$task_variant" "$task_coherent" "$task_checkpoint_hash"
  sha256sum "$task_mask"
  printf '%q ' "${task_cmd[@]}"; printf '\n'
  exit 0
fi

# Existing run directories are never reused or overwritten.
mkdir -p "$task_root/runs"
mkdir "$task_run"
mkdir "$task_run/config"
cp "$task_baseline/config/default.ini" "$task_baseline/config/rek_native5.ini" "$task_run/config/"
cp "$task_baseline/command.txt" "$task_run/F7-source-command.txt"
cp "$task_baseline/environment.txt" "$task_run/F7-source-environment.txt"
cd "$task_run"
printf '%q ' "${task_cmd[@]}" > command.txt; printf '\n' >> command.txt
env | LC_ALL=C sort | awk '/^REK_/' > environment.txt
sha256sum "$task_checkpoint" > input-checkpoint.sha256
sha256sum "$task_exe" > executable.sha256
sha256sum "$task_mask" > feature-mask.sha256
sha256sum config/default.ini config/rek_native5.ini > config.sha256
date -u +%FT%TZ > started.utc
task_started_ns=$(date +%s%N)
if /usr/bin/time -f 'native_process_wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nprocess_exit_code=%x' -o process-time.txt \
    "${task_cmd[@]}" > stdout.txt 2> stderr.txt; then
  task_exit=0
else
  task_exit=$?
fi
task_finished_ns=$(date +%s%N)
date -u +%FT%TZ > finished.utc
printf '%s\n' "$task_exit" > exit-code.txt
task_wall=$(awk -v a="$task_started_ns" -v b="$task_finished_ns" 'BEGIN {printf "%.6f", (b-a)/1000000000}')
task_log="$task_run/logs/rek_native5/$task_name.ini"
if [[ -f "$task_log" ]]; then
  awk '/^\[metrics\]$/ {found=1; next} /^\[/ {found=0} found' "$task_log" > native-metrics.txt
else
  printf 'native_metrics_missing=1\n' > native-metrics.txt
fi
awk '/native5_round_summary=/' stdout.txt > native-round-summary.txt
awk '/SPS/ {last=$0} END {if (last!="") print last}' stdout.txt > native-final-dashboard-sps.txt
task_final="$task_run/checkpoints/rek_native5/$task_name/0000000016777216.bin"
task_complete=0
if [[ -f "$task_final" ]]; then
  sha256sum "$task_final" > final-checkpoint.sha256
  [[ "$task_exit" == 0 ]] && task_complete=1
fi
{
  printf 'variant=%s\nplanned_agent_steps=%s\ntraining_exit_code=%s\ncompleted_expected_checkpoint=%s\n' "$task_variant" "$task_steps" "$task_exit" "$task_complete"
  printf 'full_native_process_wall_seconds=%s\n' "$task_wall"
  if [[ "$task_complete" == 1 ]]; then
    awk -v steps="$task_steps" -v wall="$task_wall" 'BEGIN {printf "full_native_process_sps=%.6f\n",steps/wall}'
  else
    printf 'full_native_process_sps=unavailable_incomplete_run\n'
  fi
  printf 'timing_scope=entire_headless_native_process_including_startup_rollouts_PPO_checkpoint_writes_and_shutdown\n'
  printf 'python_runtime=0\nnative_ini_metrics_scope=last_persisted_native_snapshot_may_precede_final_step\n'
  awk '/^(SPS|agent_steps|uptime|epoch|env\/score|env\/episode_return|env\/hits|env\/wins|env\/losses|env\/draws|env\/falls|env\/n|perf\/train|perf\/rollout) =/' native-metrics.txt
  cat native-round-summary.txt native-final-dashboard-sps.txt
  [[ ! -f final-checkpoint.sha256 ]] || cat final-checkpoint.sha256
} > training-summary.txt
cat training-summary.txt
if [[ "$task_exit" != 0 ]]; then exit "$task_exit"; fi
[[ "$task_complete" == 1 ]] || { printf 'Expected final checkpoint missing\n' >&2; exit 3; }
