#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 4 ]] || { printf 'Usage: %s NATIVE_SOURCE EVAL_BUILD SWEEP NEW_OUTPUT\n' "$0" >&2; exit 2; }
task_source=$(realpath "$1"); task_build=$(realpath "$2"); task_sweep=$(realpath "$3")
mkdir "$4"; task_output=$(realpath "$4")
export REK_OBSERVATION_SCHEMA=rek.native5.scaled_polar_xy.v1 REK_POLICY_ACTION_STRIDE=1
export REK_FAST_YAW_COMMAND=keyboard_reset_v1 REK_FAST_CONTACT_ENTRY=geom_pair_v1 REK_FAST_CONTACT_VELOCITY=body_cvel_v1
export REK_FAST_REWARD=normalized_points_falls_v1 REK_FAST_OPPONENT_MODE=scripted
export REK_FAST_RANDOM_RESETS=1 REK_FAST_RESET_GAP_MIN=.55 REK_FAST_RESET_GAP_MAX=2.5 REK_FAST_RESET_HEADING_SPREAD_RAD=3.14159265
export REK_FAST_SHAPING_WEIGHT=0
unset REK_FAST_CONTACT_POTENTIAL REK_POLICY_FEATURE_MASK REK_EVAL_ROUND_FEATURE REK_FAST_REWARD_GAMMA
while IFS=$'\t' read -r task_arm task_mode task_scale task_entropy task_lr task_steps task_initial; do
    [[ "$task_arm" == arm ]] && continue
    task_checkpoint=$task_sweep/$task_arm/checkpoints/rek_native5/$task_arm/0000000008388608.bin
    task_sha=$(sha256sum "$task_checkpoint" | cut -d' ' -f1)
    printf 'EVAL %s %s\n' "$task_arm" "$task_sha"
    bash "$task_source/run_fast_policy_eval.sh" "$task_build" "$task_sweep/../scripts/sweep-runtime.json" \
        "$task_checkpoint" "$task_sha" "$task_output/$task_arm" 256 1 10001 sampled bf16 \
        > "$task_output/$task_arm.runner.stdout.txt" 2> "$task_output/$task_arm.runner.stderr.txt"
done < "$task_sweep/arms.tsv"
