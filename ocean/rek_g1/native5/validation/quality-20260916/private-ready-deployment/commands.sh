set -euo pipefail
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
run=/home/spark-advantage/codexrook-runtime/live-transfer-20260915
target=/home/spark-advantage/codexrook-runtime/rek-policy-transfer-20260915/BepInEx/plugins/RekUiBridgeAgent.dll
stage=$run/RekUiBridgeAgent-private-ready.dll
backup=$run/RekUiBridgeAgent-pre-private-ready-20260916.dll
mkdir "$b/private-ready-deployment"
exec > >(tee "$b/private-ready-deployment/stdout.txt") 2> >(tee "$b/private-ready-deployment/stderr.txt" >&2)
date -u
hostname
test "$(sha256sum "$stage" | cut -d' ' -f1)" = 2edf75c65e6693db28eb7e87c45d88522668f549c7b457cdf0aa108582667952
test "$(sha256sum "$run/RekUiPipeClient-private-ready-20260916.exe" | cut -d' ' -f1)" = 0438528bbda9ad736b10bd68a1d2dbfee8f1f555f74e1e6b73b4a61799d8a674
test "$(sha256sum "$target" | cut -d' ' -f1)" = 40109bdcb2b84fabb2d66b8c995855fee1253a2c7ca943382aff8dbd7dc4eb99
test ! -e "$backup"
cp -p "$target" "$backup"
sha256sum "$stage" "$target" "$backup"
pid=$(cat "$run/game.pid")
test "$pid" = 3412
docker exec codexrook-xserver bash -c '
set -eu
pid=$1
args=$(ps -p "$pid" -o args=)
printf "verified_process=%s\n" "$args"
case "$args" in "REK.exe "*" /opt/codexrook/rek-policy-transfer-20260915/REK.exe "*) ;; *) exit 11;; esac
tr "\0" "\n" < /proc/$pid/environ | grep -xE "DISPLAY=:98|REK_EVIDENCE_ISOLATED_SESSION=spark-x98"
test "$(tr "\0" "\n" < /proc/$pid/environ | grep -xEc "DISPLAY=:98|REK_EVIDENCE_ISOLATED_SESSION=spark-x98")" = 2
kill -TERM "$pid"
for n in 1 2 3 4 5 6 7 8 9 10; do
  if ! kill -0 "$pid" 2>/dev/null; then printf "owned_game_stopped=true\n"; exit 0; fi
  sleep 1
done
printf "Owned process remains alive; refusing DLL replacement\n" >&2
exit 12
' bash "$pid"
install -m 644 "$stage" "$target"
test "$(sha256sum "$target" | cut -d' ' -f1)" = 2edf75c65e6693db28eb7e87c45d88522668f549c7b457cdf0aa108582667952
sha256sum "$target"
bash "$run/live_transfer_runtime.sh" start
