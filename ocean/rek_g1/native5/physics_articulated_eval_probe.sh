#!/usr/bin/env bash
# Bounded CPU-viewer comparison with GPU controller/combat; never PPO training.
set -euo pipefail
[[ $# == 3 ]] || { printf 'Usage: %s WORKER CONFIG NEW_OUTPUT_DIRECTORY\n' "$0" >&2; exit 2; }
worker=$(realpath "$1")
config=$(realpath "$2")
mkdir "$3"
out=$(realpath "$3")
export REK_PHYSICS_BACKEND=puffysics_cpu_eval
export REK_ALLOW_CPU_EVALUATION=1
export REK_PUFFYSICS_STABILIZATION=joint_cold_start
sha256sum "$worker" "$config" > "$out/inputs.sha256"
hostname > "$out/host.txt"
set +e
{
    printf '%s\n' '{"id":1,"op":"policy","side":0,"checkpoint":"","scripted":true}'
    for ((i=0;i<8;i++)); do
        printf '{"id":%d,"op":"step","steps":128,"stopAtRound":true}\n' "$((i+2))"
    done
} | /usr/bin/time -f '%e' -o "$out/wall-seconds.txt" timeout 120s "$worker" --config "$config" > "$out/raw.ndjson" 2> "$out/stderr.txt"
result=${PIPESTATUS[1]}
printf '%s\n' "$result" > "$out/worker-exit-code.txt"
jq -Rc 'fromjson? | select(.state != null) | .state | {tick,timeRemaining,score,falls,terminal,completedRounds,failureBits,failure,root_height:[.raw[2],.raw[88]],joint_speed_max:[([.raw[42:71][]|fabs]|max),([.raw[128:157][]|fabs]|max)]}' "$out/raw.ndjson"
if [[ $result == 0 ]]; then
    jq -Rse 'split("\n") | map(fromjson?) | all(.ok != false and .event != "fatal") and ([.[] | select(.state != null) | .state] | last | .terminal == 1 and .completedRounds == 1 and .failureBits == 0)' "$out/raw.ndjson" > "$out/round-passed.txt" || result=3
fi
printf '%s\n' "$result" > "$out/exit-code.txt"
exit "$result"
