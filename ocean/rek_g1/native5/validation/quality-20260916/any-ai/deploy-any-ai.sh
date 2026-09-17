#!/usr/bin/env bash
set -euo pipefail
old_sha=${1:?old bridge SHA required}
new_sha=${2:?new bridge SHA required}
relay_sha=${3:?relay SHA required}
[[ "$old_sha" =~ ^[a-f0-9]{64}$ && "$new_sha" =~ ^[a-f0-9]{64}$ && "$relay_sha" =~ ^[a-f0-9]{64}$ ]]
run=/home/spark-advantage/codexrook-runtime/live-transfer-20260915
game=/home/spark-advantage/codexrook-runtime/rek-policy-transfer-20260915
installed=$game/BepInEx/plugins/RekUiBridgeAgent.dll
staged=$run/RekUiBridgeAgent-any-ai-20260917.dll
relay=$run/RekUiPipeClient-any-ai-20260917.exe
backup=$run/RekUiBridgeAgent-pre-any-ai-20260917.dll
evidence=/home/spark-advantage/rek-training/policy-quality-20260916-r1/any-ai-deployment
mkdir "$evidence"
exec > >(tee "$evidence/stdout.txt") 2> >(tee "$evidence/stderr.txt" >&2)
set -x
date -u +%FT%TZ
hostname
id
printf '%s  %s\n' "$old_sha" "$installed" | sha256sum -c -
printf '%s  %s\n' "$new_sha" "$staged" | sha256sum -c -
printf '%s  %s\n' "$relay_sha" "$relay" | sha256sum -c -
test ! -e "$backup"
cp -p -- "$installed" "$backup"
printf '%s  %s\n' "$old_sha" "$backup" | sha256sum -c -
owned_pid=$(cat "$run/game.pid")
[[ "$owned_pid" =~ ^[1-9][0-9]*$ ]]
docker exec -i codexrook-xserver bash -s -- "$owned_pid" <<'VERIFY_AND_STOP'
set -euo pipefail
owned_pid=$1
test -d "/proc/$owned_pid"
mapfile -d '' -t args < "/proc/$owned_pid/cmdline"
matched=false
for arg in "${args[@]}"; do
  if [[ "$arg" == '/opt/codexrook/rek-policy-transfer-20260915/REK.exe' ]]; then matched=true; fi
done
[[ "$matched" == true ]]
display_ok=false
isolation_ok=false
while IFS= read -r -d '' entry; do
  [[ "$entry" != 'DISPLAY=:98' ]] || display_ok=true
  [[ "$entry" != 'REK_EVIDENCE_ISOLATED_SESSION=spark-x98' ]] || isolation_ok=true
done < "/proc/$owned_pid/environ"
[[ "$display_ok" == true && "$isolation_ok" == true ]]
printf 'verified_owned_pid=%s display=:98 isolated_session=spark-x98\n' "$owned_pid"
kill -TERM "$owned_pid"
for attempt in {1..10}; do
  if [[ ! -d "/proc/$owned_pid" ]]; then exit 0; fi
  sleep 1
done
printf 'Owned process did not exit; installation withheld\n' >&2
exit 1
VERIFY_AND_STOP
install -m 644 -- "$staged" "$installed"
printf '%s  %s\n' "$new_sha" "$installed" | sha256sum -c -
bash "$run/live_transfer_runtime.sh" start
date -u +%FT%TZ
