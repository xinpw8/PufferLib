#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && ( $1 == --check || $1 == --diagnose || $1 == --train ) ]] || exit 2
task_root=/home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1
task_candidate=$task_root/distributional-candidate
task_pin() { [[ $(sha256sum "$1" | awk '{print $1}') == "$2" ]] || { printf 'Hash mismatch: %s\n' "$1" >&2; exit 2; }; }
task_pin "$task_candidate/ppo-build/authentic-ppo" 1ddff37719ad26670c8ebf8a19abddd83872882c893d4c120b6c863335bde680
task_pin "$task_root/export/authentic-trajectories-v3.bin" 8ced592947fc1167f771d9480a0a56da3bab025d5bae292dc90308552f0bf83b
task_pin "$task_root/behavior-replay-v3.bin" ff391c9898357023e2f4d9185a69d2aec3ae5a68feb5c3fe690ff8116b2ba0f8
task_epochs=1
[[ $1 != --diagnose ]] || task_epochs=0
task_output=$task_candidate/epoch-$task_epochs
task_cmd=("$task_candidate/ppo-build/authentic-ppo" "$task_root/export/authentic-trajectories-v3.bin"
  "$task_root/behavior-replay-v3.bin"
  /home/spark-advantage/rek-training/f7-action-id-fix-20260924-r1/runs/F7-no-unmeasured-kick-prior-16777216/checkpoints/rek_native5/F7-no-unmeasured-kick-prior-16777216/0000000016777216.bin
  7561859790d0ed55a55471d96442fe954a593c52d8617b51c143687ba9d98f96
  "$task_output/policy.bin" "$task_epochs" 0.00001 128 0.2 0.2 0 0.001
  --allow-distributional-bf16-batch --targets=complete-mc-zero-baseline)
printf '%q ' "${task_cmd[@]}"; printf '\n'
[[ $1 != --check ]] || exit 0
[[ ! -e "$task_output" ]] || exit 2
mkdir "$task_output"
printf '%q ' "${task_cmd[@]}" > "$task_output/command.txt"; printf '\n' >> "$task_output/command.txt"
task_code=0
/usr/bin/time -f 'wall_seconds=%e\nexit_code=%x' -o "$task_output/time.txt" \
  "${task_cmd[@]}" > "$task_output/stdout.jsonl" 2> "$task_output/stderr.txt" || task_code=$?
cat "$task_output/stdout.jsonl" "$task_output/stderr.txt" "$task_output/time.txt"
exit "$task_code"
