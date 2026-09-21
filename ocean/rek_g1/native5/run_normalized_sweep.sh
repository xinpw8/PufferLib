#!/usr/bin/env bash
# Bounded native-CUDA screen. Frozen and authentic evaluations are separate.
set -euo pipefail
umask 077
[[ $# == 5 ]] || { printf 'Usage: %s BUILD NEW_OUTPUT ORIGINAL_OUTCOME_CHECKPOINT SCALED_001_CHECKPOINT ZERO_CRITIC_CHECKPOINT\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$(realpath "$1")
mkdir "$2"
task_output=$(realpath "$2")
task_original=$(realpath "$3")
task_scaled=$(realpath "$4")
task_zero=$(realpath "$5")
sha256sum "$task_original" "$task_scaled" "$task_zero" > "$task_output/initial-checkpoints.sha256"
export REK_OBSERVATION_SCHEMA=rek.native5.scaled_polar_xy.v1 REK_POLICY_ACTION_STRIDE=1
export REK_FAST_YAW_COMMAND=keyboard_reset_v1 REK_FAST_CONTACT_ENTRY=geom_pair_v1 REK_FAST_CONTACT_VELOCITY=body_cvel_v1
export REK_FAST_SCORING=recovered_hit_rules_v2 REK_FAST_GEOMETRY=primitive_samples_v1 REK_FAST_CONTACT_SUBSTEPS=8
export REK_FAST_OPPONENT=recovered_bot1_v1 REK_FAST_OBSERVATION=rendered_pose_v1 REK_FAST_OPPONENT_MODE=scripted
export REK_FAST_RANDOM_RESETS=1 REK_FAST_RESET_GAP_MIN=.55 REK_FAST_RESET_GAP_MAX=2.5 REK_FAST_RESET_HEADING_SPREAD_RAD=3.14159265
export REK_FAST_SHAPING_WEIGHT=0 REK_TRAIN_SEED=419 REK_TRAIN_BASE_SEED=73
export REK_TRAIN_GAMMA=.9998844821426083 REK_TRAIN_GAE_LAMBDA=.9978673240629938
export REK_TRAIN_METRIC=wins REK_TRAIN_DOWNSAMPLE=1 REK_TRAIN_MINIBATCH=8192
unset REK_FAST_CONTACT_POTENTIAL REK_FAST_REWARD_GAMMA REK_TRAIN_OPPONENT_CHECKPOINT REK_POLICY_FEATURE_MASK REK_EVAL_ROUND_FEATURE
printf 'arm\treward\tcritic_scale\tentropy\tlearning_rate\tsteps\tcheckpoint\n' > "$task_output/arms.tsv"
task_index=0
for task_mode in normalized_points_falls_v1 round_outcome_v1; do
    if [[ "$task_mode" == normalized_points_falls_v1 ]]; then task_initial=$task_zero; task_scale=0; else task_initial=$task_original; task_scale=1; fi
    for task_entropy in .0001 .001 .01; do
        for task_lr in .00003 .0001; do
            task_index=$((task_index+1))
            printf -v task_arm 'arm-%02d' "$task_index"
            printf '%s\t%s\t%s\t%s\t%s\t8388608\t%s\n' "$task_arm" "$task_mode" "$task_scale" "$task_entropy" "$task_lr" "$task_initial" >> "$task_output/arms.tsv"
            printf 'START %s reward=%s entropy=%s lr=%s\n' "$task_arm" "$task_mode" "$task_entropy" "$task_lr"
            REK_FAST_REWARD=$task_mode REK_TRAIN_ENTROPY=$task_entropy REK_TRAIN_LEARNING_RATE=$task_lr \
                bash "$task_source/run_diverse_training.sh" "$task_build" "$task_output/$task_arm" 8388608 512 512 120 "$task_initial" \
                > "$task_output/$task_arm.runner.stdout.txt" 2> "$task_output/$task_arm.runner.stderr.txt"
            printf 'DONE %s\n' "$task_arm"
        done
    done
done
# Reproduce the previous run's objective-incompatible critic, on this budget.
printf 'arm-13\tnormalized_points_falls_v1\t1\t.01\t.0001\t8388608\t%s\n' "$task_original" >> "$task_output/arms.tsv"
REK_FAST_REWARD=normalized_points_falls_v1 REK_TRAIN_ENTROPY=.01 REK_TRAIN_LEARNING_RATE=.0001 \
    bash "$task_source/run_diverse_training.sh" "$task_build" "$task_output/arm-13" 8388608 512 512 120 "$task_original" \
    > "$task_output/arm-13.runner.stdout.txt" 2> "$task_output/arm-13.runner.stderr.txt"
printf 'DONE arm-13\n'
# Scaling an outcome critic is a heuristic control, not a points-unit conversion.
printf 'arm-14\tnormalized_points_falls_v1\t.01\t.01\t.0001\t8388608\t%s\n' "$task_scaled" >> "$task_output/arms.tsv"
REK_FAST_REWARD=normalized_points_falls_v1 REK_TRAIN_ENTROPY=.01 REK_TRAIN_LEARNING_RATE=.0001 \
    bash "$task_source/run_diverse_training.sh" "$task_build" "$task_output/arm-14" 8388608 512 512 120 "$task_scaled" \
    > "$task_output/arm-14.runner.stdout.txt" 2> "$task_output/arm-14.runner.stderr.txt"
printf 'DONE arm-14\n'
