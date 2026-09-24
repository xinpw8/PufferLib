#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && ( $2 == --check || $2 == --run ) ]] || exit 2
stage=/home/spark-advantage/rek-training/balance8-onpolicy-20260924-r2
prior=/home/spark-advantage/rek-training/balance8-authentic-20260924-r1
data=$stage/score-delta-5s/authentic-score-delta-v5.bin
identity=$stage/score-delta-5s/behavior-identity.json
behavior=$prior/train-execution/policy.bin
behavior_sha=9a875c347b512bb46dde25481be75d276ea1b4c0799886b30aaf03be6ae1c5ce
worker=$prior/build/live-policy-worker-balance8
replay=$stage/score-delta-5s/behavior-replay-v5.bin
pin(){ [[ $(sha256sum "$1" | awk '{print $1}') == "$2" ]] || { printf 'Hash mismatch %s\n' "$1" >&2; exit 2; }; }
pin "$data" a28c938abe35794497ef9e36577574af28064fb669a1dc2f6d560aa0878567b1
pin "$identity" 03b3c21a698c850a1471f086d5116e7507a52beef7787ad128e24f6ddc4510ac
pin "$behavior" "$behavior_sha"
pin "$worker" 52741aed67037073bff5ecb21af63864550cba93a316dfe59a41b8b50126d7b2
case "$1" in
 replay)
  pin "$stage/build-score-delta/replay-authentic-behavior" fa622e8a238d5b44a1fbb839c41b4750eb80d328795eeb8d992da5771b6c9927
  command=("$stage/build-score-delta/replay-authentic-behavior" "$data" "$behavior" "$behavior_sha" "$replay" "$identity" "$worker" --observation-schema=rek.native5.scaled_polar_xy.balance8_v1)
  ;;
 diagnose|train)
  pin "$stage/build-score-delta/authentic-ppo" 741035dad9303be7636c77be97ec44584011c95987170e86008d5515020b30e8
  epochs=0; [[ $1 != train ]] || epochs=1
  command=("$stage/build-score-delta/authentic-ppo" "$data" "$replay" "$behavior" "$behavior_sha" "$stage/$1-score-delta/policy.bin" "$epochs" .00001 128 .2 .2 0 .001 --allow-distributional-bf16-batch --targets=complete-mc-zero-baseline --observation-schema=rek.native5.scaled_polar_xy.balance8_v1)
  ;;
 *) exit 2;;
esac
printf '%q ' "${command[@]}";printf '\n'
[[ $2 != --check ]] || exit 0
if [[ $1 != replay ]];then
 node -e 'const fs=require("fs"),x=JSON.parse(fs.readFileSync(process.argv[1]));if(!x.verification_passed||x.matching_sampled_actions!==15078||x.mismatches!==0||x.teacher_was_recorded_worker!==true||x.checkpoint_sha256!==process.argv[2])process.exit(2)' "$replay.json" "$behavior_sha"
fi
logs=$stage/$1-score-delta
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
