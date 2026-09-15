#!/usr/bin/env bash
# Authentic REK isolated-runtime launcher. No policy or controller starts here.
set -euo pipefail
task_base=/home/spark-advantage/codexrook-runtime
task_game=rek-policy-transfer-20260915
task_run=$task_base/live-transfer-20260915
task_container=codexrook-xserver
task_mode=${1:?Expected prepare, start, or status}
if [[ "$task_mode" == prepare ]];then
    test ! -e "$task_base/$task_game"
    test ! -e "$task_run"
    mkdir "$task_run"
    cp -a --reflink=auto "$task_base/rek-motion-parity-current-001" "$task_base/$task_game"
    sha256sum "$task_base/$task_game/GameAssembly.dll" \
      "$task_base/$task_game/REK_Data/il2cpp_data/Metadata/global-metadata.dat" > "$task_run/game.sha256"
    exit
fi
if [[ "$task_mode" == status ]];then
    docker top "$task_container" -eo pid,comm
    test ! -f "$task_run/game.pid" || cat "$task_run/game.pid"
    exit
fi
[[ "$task_mode" == start ]]
test -d "$task_base/$task_game"
test -d "$task_run"
if pgrep -f '[/]opt/codexrook/rek[^ ]*/REK[.]exe' >/dev/null;then
    printf 'An authentic REK process is already running; no duplicate launch\n' >&2
    exit 1
fi
task_stamp=$(date -u +%Y%m%dT%H%M%SZ)
DISPLAY=:98 xrandr --fb 1280x720
docker exec -d --user 1000:1000 \
    -e DISPLAY=:98 -e HOME=/opt/codexrook/home \
    -e WINEPREFIX=/opt/codexrook/wineprefix -e WINEARCH=win64 -e WINEDEBUG=-all \
    -e REK_EVIDENCE_ISOLATED_SESSION=spark-x98 \
    -e 'WINEDLLOVERRIDES=winhttp=n,b;winealsa.drv=d;winepulse.drv=d;mscoree=d;mshtml=d;winemenubuilder.exe=d' \
    -e BOX64_NOBANNER=1 -e BOX64_NOPULSE=1 -e BOX64_NOVULKAN=1 -e BOX64_UNITY=1 \
    -e BOX64_DYNAREC_STRONGMEM=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -e BOX64_LD_LIBRARY_PATH=/opt/codexrook/x64root-ubuntu-24.04/lib/x86_64-linux-gnu:/opt/codexrook/x64root-ubuntu-24.04/usr/lib/x86_64-linux-gnu:/opt/codexrook/wine-11.13/lib:/opt/codexrook/wine-11.13/lib/wine/x86_64-unix \
    "$task_container" bash -c '
      set -euo pipefail
      cd "/opt/codexrook/$1"
      printf "%s\n" "$$" > /opt/codexrook/live-transfer-20260915/game.pid
      exec /opt/codexrook/box64/bin/box64 /opt/codexrook/wine-11.13/bin/wine "/opt/codexrook/$1/REK.exe" \
        -screen-fullscreen 0 -screen-width 1280 -screen-height 720 -popupwindow -force-d3d11 -disable-audio -nosound \
        -logFile "C:\\rekagent\\live-transfer-$2-unity.log" \
        > "/opt/codexrook/live-transfer-20260915/game-$2.stdout.txt" \
        2> "/opt/codexrook/live-transfer-20260915/game-$2.stderr.txt"
    ' bash "$task_game" "$task_stamp"
printf 'launch_requested_utc=%s\ncontroller_started=false\n' "$task_stamp"
