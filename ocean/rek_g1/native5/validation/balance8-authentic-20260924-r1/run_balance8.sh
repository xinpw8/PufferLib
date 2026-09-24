#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && ( $2 == --check || $2 == --run ) ]] || exit 2
stage=/home/spark-advantage/rek-training/balance8-authentic-20260924-r1
original=/home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1
teacher=$stage/migrated-base7561.bin
teacher_sha=4906e39d0afa5e92334c5303b8c15cbadf48455537944c5a3d4c1f161d8b5263
data=$stage/export/authentic-balance8-v4.bin
replay=$stage/behavior-replay-v4.bin
pin(){ [[ $(sha256sum "$1" | awk '{print $1}') == "$2" ]] || { printf 'Hash mismatch %s\n' "$1" >&2; exit 2; }; }
pin "$data" fab40a24a588bebb8dc0a55fa8c0534e2a6bcc2e666cfb557e0efc1513d57dde
pin "$stage/export/derived-identity.json" f0486538aff2af4ddc99d24a5866b4344167054b542ce72f757adfc78d4bc7e1
pin "$teacher" "$teacher_sha"
case "$1" in
 replay)
  pin "$stage/build/replay-balance8" eafc6445887723527f9979b6c763020cb5917d5b9350fba85e3d3396199d1814
  command=("$stage/build/replay-balance8" "$data" "$teacher" "$teacher_sha" "$replay" "$stage/export/derived-identity.json"
   /home/spark-advantage/rek-training/semantic-fast-20260914-v1/live-policy-20260915/build-r1/live-policy-worker
   "$original/export/authentic-trajectories-v3.bin" "$original/behavior-replay-v3.bin")
  ;;
 diagnose|train)
  pin "$stage/build/authentic-ppo-balance8" 43aa664a0e666c17f3f4a5466bbc4962fcf1cc3a7f4cff62ea09e503e4fcac7d
  epochs=0;[[ $1 != train ]] || epochs=1
  command=("$stage/build/authentic-ppo-balance8" "$data" "$replay" "$teacher" "$teacher_sha" "$stage/$1-execution/policy.bin" "$epochs" .00001 128 .2 .2 0 .001
   --allow-distributional-bf16-batch --targets=complete-mc-zero-baseline --observation-schema=rek.native5.scaled_polar_xy.balance8_v1)
  ;;
 *) exit 2;;
esac
printf '%q ' "${command[@]}";printf '\n'
[[ $2 != --check ]] || exit 0
if [[ $1 != replay ]];then
 node -e 'const fs=require("fs"),x=JSON.parse(fs.readFileSync(process.argv[1]));if(!x.verification_passed||x.matching_sampled_actions!==28715||!x.original_logprobs_values_logits_copied_bitwise||x.derived_teacher_was_recorded_worker!==false)process.exit(2)' "$replay.json"
fi
logs=$stage/$1-execution
[[ ! -e "$logs" ]] || exit 2
mkdir "$logs"
printf '%q ' "${command[@]}" > "$logs/command.txt";printf '\n' >> "$logs/command.txt"
date -u +%FT%TZ > "$logs/started.utc"
status=0
/usr/bin/time -f 'full_native_process_wall_seconds=%e\nexit_code=%x' -o "$logs/time.txt" "${command[@]}" > "$logs/stdout.jsonl" 2> "$logs/stderr.txt" || status=$?
printf '%s\n' "$status" > "$logs/exit-code.txt"
date -u +%FT%TZ > "$logs/finished.utc"
cat "$logs/stdout.jsonl" "$logs/stderr.txt" "$logs/time.txt"
exit "$status"
