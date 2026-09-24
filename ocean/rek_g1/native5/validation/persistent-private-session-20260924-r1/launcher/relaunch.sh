#!/usr/bin/env bash
set -euo pipefail
umask 077
# Derived from launch.sh (authentic-policy-ab-20260919-r1). Changes: parametrized
# output suffix, and the pair-capture config may already be the disabled copy.
rek_suffix=${1:?usage: relaunch.sh SUFFIX}
rek_stage=/home/spark-advantage/rek-training/authentic-policy-ab-20260919-r1
rek_game=/home/spark-advantage/codexrook-runtime/rek-core-referee-20260924-r1
rek_out=/home/spark-advantage/codexrook-runtime/live-transfer-20260915/live-attack-gate-20260921-$rek_suffix
rek_guest=/opt/codexrook/live-transfer-20260915/live-attack-gate-20260921-$rek_suffix
rek_pair=$rek_game/BepInEx/config/rek-consented-pair-capture.json
test ! -e "$rek_out"
test "$(docker inspect --format='{{.State.Running}}' codexrook-xserver)" = true
test "$(sha256sum "$rek_game/REK.exe" | cut -d' ' -f1)" = 5fe6a5c3da371cb7b75a1795eb073b2e69ec718197cf68826bdcd461dcd986c1
test "$(sha256sum "$rek_game/GameAssembly.dll" | cut -d' ' -f1)" = 6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412
test "$(sha256sum "$rek_game/BepInEx/plugins/RekUiBridgeAgent.dll" | cut -d' ' -f1)" = 5a2edac6c586f1ea401d92e0086ebfc468dbc591e2bb11115aef056682280e7a
test "$(sha256sum "$rek_game/BepInEx/plugins/RekEvidenceRecorder.dll" | cut -d' ' -f1)" = 525bc57feb05e623ea152292c824f03c5f068f7a71b999adba53a150d637b071
rek_pair_sha=$(sha256sum "$rek_pair" | cut -d' ' -f1)
test "$rek_pair_sha" = c8e80ab5ef6240b417c68d319d287a449c8456a321b3a6d3b86f3749abbbe0e4 -o "$rek_pair_sha" = 57892634aea35f7af7ec4c96963a3e2ddaa2511856e7a899495be5fa5c17f8a4
test "$(sha256sum "$rek_stage/pair-disabled.json" | cut -d' ' -f1)" = 57892634aea35f7af7ec4c96963a3e2ddaa2511856e7a899495be5fa5c17f8a4
rg -q '^PreloadIL2CPPInteropAssemblies = false' "$rek_game/BepInEx/config/BepInEx.cfg"
rek_available=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
test "$rek_available" -ge 16777216
for rek_proc in /proc/[0-9]*/cmdline; do
    rek_args=()
    mapfile -d '' -t rek_args < "$rek_proc" 2>/dev/null || continue
    for rek_arg in "${rek_args[@]}"; do
        if [[ "$rek_arg" == REK.exe || "$rek_arg" == /opt/codexrook/rek*/REK.exe ]]; then
            printf 'Existing REK process; launch cancelled\n' >&2; exit 2
        fi
    done
    if grep -azFxq 'WINEPREFIX=/opt/codexrook/wineprefix' "${rek_proc%/cmdline}/environ" 2>/dev/null; then
        printf 'Existing prefix user; launch cancelled\n' >&2; exit 2
    fi
done
rek_glx=$(docker exec --user 1000:1000 -e DISPLAY=:98 -e __GLX_VENDOR_LIBRARY_NAME=nvidia codexrook-xserver glxinfo -B)
mkdir -m 700 "$rek_out"
cp --no-clobber "$0" "$rek_out/launch.sh"
cp --no-clobber "$rek_pair" "$rek_out/pair-config.before.json"
if test -f "$rek_game/BepInEx/LogOutput.log"; then
    cp --no-clobber "$rek_game/BepInEx/LogOutput.log" "$rek_out/prior-BepInEx-LogOutput.log"
fi
sha256sum "$rek_game/BepInEx/config/BepInEx.cfg" "$rek_game/BepInEx/plugins/"*.dll > "$rek_out/unchanged-inputs.sha256"
# Exact task-scoped configuration replacement: only Enabled changed to false.
cp "$rek_stage/pair-disabled.json" "$rek_pair"
sha256sum "$rek_pair" > "$rek_out/pair-config.after.sha256"
printf '%s\n' "$rek_glx" > "$rek_out/glx-before.txt"
free -m > "$rek_out/memory-before.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$rek_out/gpu-processes-before.txt"
date -u +%FT%TZ > "$rek_out/launch-requested.utc.txt"
docker exec -d --user 1000:1000 \
    -e DISPLAY=:98 -e HOME=/opt/codexrook/home \
    -e WINEPREFIX=/opt/codexrook/wineprefix -e WINEARCH=win64 \
    -e WINEDEBUG=-all,err+all,warn+seh -e BOX64_LOG=1 \
    -e BOX64_SHOWSEGV=1 -e BOX64_SHOWBT=1 \
    -e REK_EVIDENCE_ISOLATED_SESSION=spark-x98 \
    -e 'WINEDLLOVERRIDES=winhttp=n,b;winealsa.drv=d;winepulse.drv=d;mscoree=d;mshtml=d;winemenubuilder.exe=d' \
    -e BOX64_NOBANNER=1 -e BOX64_NOPULSE=1 -e BOX64_NOVULKAN=1 -e BOX64_UNITY=1 \
    -e BOX64_DYNAREC_STRONGMEM=2 -e BOX64_DYNAREC_WEAKBARRIER=0 -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -e BOX64_LD_LIBRARY_PATH=/opt/codexrook/x64root-ubuntu-24.04/lib/x86_64-linux-gnu:/opt/codexrook/x64root-ubuntu-24.04/usr/lib/x86_64-linux-gnu:/opt/codexrook/wine-11.13/lib:/opt/codexrook/wine-11.13/lib/wine/x86_64-unix \
    codexrook-xserver bash -c '
        set -uo pipefail
        cd /opt/codexrook/rek-core-referee-20260924-r1
        /opt/codexrook/box64/bin/box64 /opt/codexrook/wine-11.13/bin/wine /opt/codexrook/rek-core-referee-20260924-r1/REK.exe \
            -screen-fullscreen 0 -screen-width 1280 -screen-height 720 -popupwindow -force-d3d11 -disable-audio -nosound \
            -logFile "Z:$1/unity.log" > "$1/stdout.txt" 2> "$1/stderr.txt" &
        rek_child=$!
        printf "%s\n" "$rek_child" > "$1/game.pid"
        cat "/proc/$rek_child/stat" > "$1/game.initial-stat.txt"
        wait "$rek_child"
        rek_result=$?
        printf "%s\n" "$rek_result" > "$1/game.exit-code.txt"
        date -u +%FT%TZ > "$1/game.exited.utc.txt"
        exit "$rek_result"
    ' bash "$rek_guest"
printf 'isolated_launch_submitted=true; renderer=nvidia; arena_entry=false; policy_input=false; pair_capture_enabled=false\n'
