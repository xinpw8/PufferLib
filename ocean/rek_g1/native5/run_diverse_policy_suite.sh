#!/usr/bin/env bash
set -euo pipefail
[[ $# == 7 ]] || { printf 'Usage: %s EVALUATOR_BUILD RUNTIME_JSON CHECKPOINT SHA256 NEW_OUTPUT OLD_CHECKPOINT OLD_SHA256\n' "$0" >&2;exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$(realpath "$1")
task_config=$(realpath "$2")
task_checkpoint=$(realpath "$3")
task_sha=$4
task_old=$(realpath "$6")
task_old_sha=$7
mkdir "$5"
task_output=$(realpath "$5")
for task_fixture in fixed heldout;do
  for task_opponent in neutral scripted retreat strafe checkpoint;do
    task_args=("$task_build" "$task_config" "$task_checkpoint" "$task_sha" "$task_output/$task_fixture-$task_opponent-20" 128 4 10001 sampled bf16 "$task_opponent" "$task_fixture" 20)
    if [[ "$task_opponent" == checkpoint ]];then task_args+=("$task_old" "$task_old_sha");fi
    bash "$task_source/run_diverse_policy_eval.sh" "${task_args[@]}"
  done
done
# Longer human-round checks are smaller, explicitly separate samples. They
# change duration and timer observations together, not only one input feature.
for task_opponent in neutral scripted;do
  bash "$task_source/run_diverse_policy_eval.sh" "$task_build" "$task_config" "$task_checkpoint" "$task_sha" \
    "$task_output/fixed-$task_opponent-300" 32 1 10001 sampled bf16 "$task_opponent" fixed 300
done
node "$task_source/summarize_diverse_policy.cjs" "$task_output"
