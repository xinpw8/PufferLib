#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
test ! -e cpu-run-r1
mkdir cpu-run-r1
trap 'status=$?; printf "%s\n" "$status" > cpu-run-r1/exit-code.txt' EXIT
g++ -std=c++20 -O2 -ffp-contract=off -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -DREK_G1_CUDA_DEVICE=1 -D__device__= -D__constant__= -Istubs -Isource -Idependencies -I. \
  test_match.cpp dependencies/g1_fight_state.c dependencies/g1_fall_state.c \
  dependencies/g1_combat_tick.c dependencies/g1_hit_detector.c -lm -o cpu-run-r1/test-match \
  >cpu-run-r1/build.stdout.txt 2>cpu-run-r1/build.stderr.txt
./cpu-run-r1/test-match | tee cpu-run-r1/test.stdout.txt
gcc -std=c11 -O2 -ffp-contract=off -Idependencies dependencies/test_g1_fight_state.c dependencies/g1_fight_state.c -lm -o cpu-run-r1/test-original-fight
./cpu-run-r1/test-original-fight | tee cpu-run-r1/original-fight.stdout.txt
gcc -std=c11 -O2 -ffp-contract=off -Idependencies dependencies/test_g1_combat_tick.c dependencies/g1_combat_tick.c dependencies/g1_fight_state.c dependencies/g1_fall_state.c dependencies/g1_hit_detector.c -lm -o cpu-run-r1/test-original-combat
./cpu-run-r1/test-original-combat | tee cpu-run-r1/original-combat.stdout.txt
sha256sum source/* cpu-run-r1/test-* > cpu-run-r1/artifact-hashes.sha256
