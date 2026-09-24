#!/usr/bin/env bash
set -euo pipefail
stage=/home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1/ppo-variant
data=/home/spark-advantage/rek-training/timing500-ppo-refresh-20260924-r1/score-delta-5s
baseline=/home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1/fit-fixed-r1/row-baselines.bin
protocol=/home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1/PROTOCOL.md
initial=/home/spark-advantage/rek-training/timing500-onpolicy-20260924-r2/train-score-delta/policy.bin
run=$stage/train-fixed-r1
[[ ! -e "$run" ]] || { printf 'Fixed output already exists; preserving it.\n' >&2; exit 2; }
mkdir "$run"
printf '%s  %s\n' \
  4087ab38dd972673370af39261a2bcb5a36713f608b9db54566e5c61de6c5344 "$stage/build/authentic-ppo-state-baseline" \
  66e34920cf3707724512fa67715a414d2b6c48b94b62f91a0d157da39cbe8c50 "$stage/authentic_ppo.cu" \
  ab4519399e6274d9e607091c5c4da3dbb87927e18710f184e51bb5a2929af5fb "$stage/crossfit_baseline.h" \
  eb6b1ae210b3b3911597f93a513ae32081f15e4871b29c1cb29e75829f718655 "$data/authentic-score-delta-v5.bin" \
  6aa4bdb5b8c28b41524d4902f667e183fb0adea37584b2af6402a0c02bb8a702 "$data/behavior-replay-v5.bin" \
  c2c4987b268996cd912fe35e6ba5f9b20a94d69fd93e4ef15b6c5fc71ee5e533 "$initial" \
  8d4a24282f834a687a7740c14729974b87757a3b80779cb7fe004ba0936b268a "$baseline" \
  32ee5bb1dd2cdb427cfeff7ae01f663a350858b3d35d80655d86987f64293dd9 "$protocol" \
  | tee "$run/inputs.sha256" | sha256sum --check --strict
sha256sum "$stage/run-plan.json" "$stage/run_once.sh" > "$run/launch-plan.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/started-utc.txt"
"$stage/build/authentic-ppo-state-baseline" \
  "$data/authentic-score-delta-v5.bin" "$data/behavior-replay-v5.bin" "$initial" \
  c2c4987b268996cd912fe35e6ba5f9b20a94d69fd93e4ef15b6c5fc71ee5e533 \
  "$run/policy.bin" 1 .00003 128 .2 .2 0 .001 \
  --allow-distributional-bf16-batch \
  --targets=complete-mc-cross-fitted-state-baseline \
  --observation-schema=rek.native5.scaled_polar_xy.balance8_v1 \
  "--state-baseline=$baseline" \
  --state-baseline-sha256=8d4a24282f834a687a7740c14729974b87757a3b80779cb7fe004ba0936b268a \
  --state-baseline-protocol-sha256=32ee5bb1dd2cdb427cfeff7ae01f663a350858b3d35d80655d86987f64293dd9 \
  2> "$run/stderr.txt" | tee "$run/stdout.jsonl"
sha256sum --check --strict "$run/inputs.sha256"
cmp "$run/policy.bin" "$run/policy.bin.epoch-1.bin"
sha256sum "$run/policy.bin" "$run/policy.bin.epoch-1.bin" > "$run/checkpoint-hashes.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/completed-utc.txt"
