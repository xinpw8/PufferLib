#!/usr/bin/env bash
set -euo pipefail
task_base=/home/spark-advantage/rek-training/semantic-fast-20260914-v1
task_stage=$task_base/human-viewer-v4-20260915
task_mode=${1:?Expected test or live}
[[ "$task_mode" == test || "$task_mode" == live ]]
task_run=$task_stage/$task_mode-run
task_port=18770
[[ "$task_mode" != live ]] || task_port=18769
test -f "$task_run/server.json"
if [[ "$task_mode" == live ]];then
    task_expected="node $task_base/human-viewer-v3-20260915/league/server.cjs $task_base/human-viewer-v3-20260915/live-run/server.json"
    [[ "$(ps -p 145654 -o args=)" == "$task_expected" ]] || { printf 'Prior process identity changed; not stopping it\n' >&2;exit 1; }
    curl -fsS http://127.0.0.1:18769/api/state > "$task_run/prior-state.private.json"
    node -e 'const s=JSON.parse(require("fs").readFileSync(process.argv[1])); if(!s.paused) throw new Error("Prior viewer is actively running; cutover deferred");' "$task_run/prior-state.private.json"
    kill -TERM 145654
    for task_attempt in {1..50};do
        [[ -z $(ss -ltnH "sport = :18769") ]] && break
        sleep .1
    done
fi
[[ -z $(ss -ltnH "sport = :$task_port") ]]
nohup node "$task_stage/league/server.cjs" "$task_run/server.json" > "$task_run/server.stdout.txt" 2> "$task_run/server.stderr.txt" < /dev/null &
task_pid=$!
printf '%s\n' "$task_pid" > "$task_run/server.pid"
for task_attempt in {1..100};do
    if curl -fsS "http://127.0.0.1:$task_port/health" > "$task_run/health.json" 2>/dev/null;then
        printf '\nready pid=%s port=%s\n' "$task_pid" "$task_port"
        ps -p "$task_pid" -o pid=,args=
        exit 0
    fi
    kill -0 "$task_pid"
    sleep .1
done
printf 'New evaluator health did not become ready\n' >&2
exit 1
