#!/usr/bin/env bash
set -euo pipefail
[[ $# == 4 || $# == 6 ]] || { printf 'Usage: %s PRIVATE_BASE TRAIN_RUN TRANSITIONS NEW_OUTPUT_PREFIX [OLD_CHECKPOINT OLD_SHA256_FOR_FULL_SUITE]\n' "$0" >&2;exit 2; }
task_base=$(realpath "$1")
task_name=$2
[[ "$task_name" =~ ^[a-z0-9-]+$ && "$3" =~ ^[0-9]+$ && "$4" =~ ^[a-z0-9-]+$ ]] || exit 2
task_run=$task_base/$task_name
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_ready=0
for ((task_attempt=0;task_attempt<72;task_attempt++));do
  if [[ -f "$task_run/exit-code.txt" ]];then
    [[ "$(cat "$task_run/exit-code.txt")" == 0 ]] || { printf 'Training failed: %s\n' "$task_name" >&2;exit 2; }
    if [[ -s "$task_run/verified-warm-start.txt" ]];then
      task_ready=1
      break
    fi
  fi
  sleep 5
done
[[ "$task_ready" == 1 ]] || { printf 'Timed out waiting for completed training\n' >&2;exit 2; }
printf -v task_file '%016d.bin' "$3"
task_checkpoint=$task_run/checkpoints/rek_native5/$task_name/$task_file
task_sha=$(sha256sum "$task_checkpoint" | cut -d' ' -f1)
printf 'checkpoint=%s\nsha256=%s\n' "$task_checkpoint" "$task_sha"
if [[ $# == 6 ]];then
  bash "$task_source/run_diverse_policy_suite.sh" "$task_base/diverse-eval-build-v4" \
    "$task_base/eval-run-v3/semantic_cuda-worker.json" "$task_checkpoint" "$task_sha" "$task_base/$4" "$5" "$6"
  exit
fi
for task_spec in fixed-neutral fixed-scripted heldout-neutral;do
  task_fixture=${task_spec%-*}
  task_opponent=${task_spec#*-}
  bash "$task_source/run_diverse_policy_eval.sh" "$task_base/diverse-eval-build-v4" \
    "$task_base/eval-run-v3/semantic_cuda-worker.json" "$task_checkpoint" "$task_sha" \
    "$task_base/$4-$task_spec-20" 64 2 10001 sampled bf16 "$task_opponent" "$task_fixture" 20
done
