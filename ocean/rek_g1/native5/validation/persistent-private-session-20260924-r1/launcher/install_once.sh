#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=/home/spark-advantage/rek-training/persistent-private-session-20260924-r1
old=/home/spark-advantage/rek-training/timing500-ppo-refresh-live-20260924-r1
plugin=/home/spark-advantage/codexrook-runtime/rek-core-referee-20260924-r1/BepInEx/plugins/RekUiBridgeAgent.dll
test ! -e "$stage/maintenance-install"
printf '%s  %s\n' \
  11fcfa2bea541f2829553c3833c301717c0182a59eebb0020f5f7b8cb066ca3d "$plugin" \
  5a2edac6c586f1ea401d92e0086ebfc468dbc591e2bb11115aef056682280e7a "$stage/RekUiBridgeAgent.dll" \
  41d685616f90c1f78e70a744ee7e7059537ce33d2085cc4da87b96f8163483d7 "$stage/relaunch.sh" | sha256sum --check --strict
test ! -d /proc/3051691
mkdir -m 700 "$stage/maintenance-install"
date -u +%FT%TZ > "$stage/maintenance-install/started.utc"
cp --no-clobber "$plugin" "$stage/maintenance-install/RekUiBridgeAgent.previous.dll"
sha256sum "$plugin" "$stage/RekUiBridgeAgent.dll" > "$stage/maintenance-install/before.sha256"
# One explicitly announced maintenance restart, not round-based teardown.
# Existing helper verifies one exact X98 game and no policy worker before stopping its prefix.
bash "$old/root-campaign/recycle_owned_client.sh"
bash "$old/root-campaign/clear_dead_prefix.sh"
cp "$stage/RekUiBridgeAgent.dll" "$plugin"
printf '%s  %s\n' 5a2edac6c586f1ea401d92e0086ebfc468dbc591e2bb11115aef056682280e7a "$plugin" | sha256sum --check --strict
sha256sum "$plugin" "$stage/maintenance-install/RekUiBridgeAgent.previous.dll" > "$stage/maintenance-install/after.sha256"
date -u +%FT%TZ > "$stage/maintenance-install/completed.utc"
bash "$stage/relaunch.sh" persistent-diagnostic-r1
