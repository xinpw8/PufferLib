#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && ( $2 == --check || $2 == --run ) ]] || { printf 'Usage: %s replay|ppo --check|--run\n' "$0" >&2; exit 2; }
task_mode=$1
task_root=/home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1
task_data=$task_root/export/authentic-trajectories-v3.bin
task_identity=$task_root/export/behavior-identity.json
task_worker=/home/spark-advantage/rek-training/semantic-fast-20260914-v1/live-policy-20260915/build-r1/live-policy-worker
task_checkpoint=/home/spark-advantage/rek-training/f7-action-id-fix-20260924-r1/runs/F7-no-unmeasured-kick-prior-16777216/checkpoints/rek_native5/F7-no-unmeasured-kick-prior-16777216/0000000016777216.bin
task_cp_sha=7561859790d0ed55a55471d96442fe954a593c52d8617b51c143687ba9d98f96
task_replay=$task_root/behavior-replay-v3.bin
task_pin() { [[ $(sha256sum "$1" | awk '{print $1}') == "$2" ]] || { printf 'Hash mismatch: %s\n' "$1" >&2; exit 2; }; }
task_pin "$task_data" 8ced592947fc1167f771d9480a0a56da3bab025d5bae292dc90308552f0bf83b
task_pin "$task_identity" 674009772cbdbb719ff5ec7e15b4022c123cd3e61370a578a6de99aed97eb537
task_pin "$task_checkpoint" "$task_cp_sha"
case "$task_mode" in
  replay)
    task_exe=$task_root/replay-build/replay-authentic-behavior-v3
    task_pin "$task_exe" f275cee3098e35cac9ebad68f125c8d0444c1a9661bc69b4d37ce00877e85fed
    task_pin "$task_worker" 3a12e88268bca78ccec4813adaeacaec74bd68652594b46007d24ab711c0a029
    task_cmd=("$task_exe" "$task_data" "$task_checkpoint" "$task_cp_sha" "$task_replay" "$task_identity" "$task_worker")
    ;;
  ppo)
    task_exe=$task_root/ppo-build/authentic-ppo
    task_pin "$task_exe" 2772caae58c6072c2805303b7daf41d8d5ee210304d45403330149b9551e238a
    task_cmd=("$task_exe" "$task_data" "$task_replay" "$task_checkpoint" "$task_cp_sha"
      "$task_root/authentic-mc-epoch1.bin" 1 0.00001 128 0.2 0.2 0 0.001
      --allow-bounded-bf16-batch --targets=complete-mc-zero-baseline)
    ;;
  *) printf 'Unknown mode\n' >&2; exit 2 ;;
esac
printf '%q ' "${task_cmd[@]}"; printf '\n'
[[ $2 == --run ]] || exit 0
[[ "$task_mode" != ppo || -f "$task_replay" ]] || { printf 'Exact behavior replay must pass first\n' >&2; exit 2; }
task_logs=$task_root/$task_mode-execution
[[ ! -e "$task_logs" ]] || { printf 'Execution directory exists\n' >&2; exit 2; }
mkdir "$task_logs"
printf '%q ' "${task_cmd[@]}" > "$task_logs/command.txt"; printf '\n' >> "$task_logs/command.txt"
date -u +%FT%TZ > "$task_logs/started.utc"
task_exit=0
/usr/bin/time -f 'full_native_process_wall_seconds=%e\nexit_code=%x' -o "$task_logs/time.txt" \
  "${task_cmd[@]}" > "$task_logs/stdout.jsonl" 2> "$task_logs/stderr.txt" || task_exit=$?
date -u +%FT%TZ > "$task_logs/finished.utc"
printf '%s\n' "$task_exit" > "$task_logs/exit-code.txt"
cat "$task_logs/time.txt" "$task_logs/stdout.jsonl" "$task_logs/stderr.txt"
exit "$task_exit"
