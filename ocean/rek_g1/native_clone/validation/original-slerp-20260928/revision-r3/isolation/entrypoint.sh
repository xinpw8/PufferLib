#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$DISPLAY" = :99
test "$WINEPREFIX" = /session/wineprefix
test "$HOME" = /session/home
test ! -e /session/run.started
test -f /session/STAGED.json
test -f /session/PLUGIN.json
test -f /session/INPUT.json
test -f /session/input/run-config.json
test -f /session/game/REK_SLERP_ORACLE_ISOLATED.json
test ! -e /session/out/oracle
test ! -e /tmp/.X11-unix/X98
test ! -e /dev/nvidia0
test ! -e /dev/nvidiactl
test ! -e /dev/nvidia-uvm
test ! -e /dev/dri
test "$(find /session/game/BepInEx/plugins -maxdepth 1 -type f -name '*.dll' | wc -l)" -eq 1
test ! -e /session/game/BepInEx/plugins/RekUiBridgeAgent.dll
test ! -e /session/game/BepInEx/plugins/RekEvidenceRecorder.dll
python3 - <<'PY'
import hashlib,json
from pathlib import Path
r=Path('/session')
receipt=json.loads((r/'STAGED.json').read_text())
plugin=json.loads((r/'PLUGIN.json').read_text())
inputs=json.loads((r/'INPUT.json').read_text())
config_bytes=(r/'input/run-config.json').read_bytes()
config=json.loads(config_bytes)
marker=json.loads((r/'game/REK_SLERP_ORACLE_ISOLATED.json').read_text())
if marker['run_id']!=config['run_id'] or marker['config_sha256']!=hashlib.sha256(config_bytes).hexdigest():
    raise SystemExit('Isolated oracle marker/config mismatch')
if config['output_directory'].replace('\\','/')!='Z:/session/out/oracle':
    raise SystemExit('Oracle output must be the fresh private child directory')
for item in receipt['files']+[plugin]+inputs['files']:
    p=r/item['path']
    if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
        raise SystemExit('Staged input hash mismatch: '+item['path'])
if not all(x['path'].startswith('game/') for x in receipt['files']):
    raise SystemExit('Unexpected staged source path')
PY
test "$(sha256sum /opt/codexrook/box64/bin/box64 | cut -d' ' -f1)" = 12a50a0f629f1ddeb08524c8f7399829e0b7101f79f4e20c094376c73a1af7ae
date -u +%FT%TZ > /session/run.started
cp /rek-harness-packages.txt /session/out/image-packages.txt
Xvfb :99 -screen 0 1280x720x24 -nolisten tcp -noreset -ac > /session/out/xvfb.stdout.log 2> /session/out/xvfb.stderr.log &
rek_xvfb_pid=$!
trap 'kill "$rek_xvfb_pid" 2>/dev/null || true' EXIT
for rek_i in $(seq 1 50); do
    if xdpyinfo -display :99 > /session/out/x-display.txt 2>/dev/null; then break; fi
    sleep 0.1
done
xdpyinfo -display :99 > /session/out/x-display.txt
export WINEARCH=win64 WINEDEBUG=-all,err+all,warn+seh
export BOX64_NOBANNER=1 BOX64_NOPULSE=1 BOX64_NOVULKAN=1 BOX64_UNITY=1
export BOX64_DYNAREC_STRONGMEM=2 BOX64_DYNAREC_WEAKBARRIER=0
export BOX64_LOG=1 BOX64_SHOWSEGV=1 BOX64_SHOWBT=1
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
export WINEDLLOVERRIDES='winhttp=n,b;winealsa.drv=d;winepulse.drv=d;mscoree=d;mshtml=d;winemenubuilder.exe=d'
export BOX64_LD_LIBRARY_PATH=/opt/codexrook/x64root-ubuntu-24.04/lib/x86_64-linux-gnu:/opt/codexrook/x64root-ubuntu-24.04/usr/lib/x86_64-linux-gnu:/opt/codexrook/wine-11.13/lib:/opt/codexrook/wine-11.13/lib/wine/x86_64-unix
export REK_ORIGINAL_FUNCTIONS_OUTPUT=Z:/session/out
export REK_ORIGINAL_FUNCTIONS_INPUT=Z:/session/input/fixtures.json
export REK_SLERP_ORACLE_ENABLE=detached-v1
export REK_SLERP_ORACLE_CONFIG=Z:/session/input/run-config.json
cd /session/game
set +e
timeout --signal=TERM --kill-after=10s 180s /opt/codexrook/box64/bin/box64 /opt/codexrook/wine-11.13/bin/wine /session/game/REK.exe \
    -batchmode -nographics -job-worker-count 2 --rek-slerp-oracle -screen-fullscreen 0 -screen-width 1280 -screen-height 720 -disable-audio -nosound \
    -logFile Z:/session/out/unity.log > /session/out/game.stdout.log 2> /session/out/game.stderr.log
rek_exit=$?
set -e
printf '%s\n' "$rek_exit" > /session/out/game.exit-code.txt
date -u +%FT%TZ > /session/out/game.finished.utc
exit "$rek_exit"
