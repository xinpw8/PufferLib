#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/native"
task_build=${1:-build-r1}
[[ "$task_build" =~ ^build-r[0-9]+$ ]]
test ! -e "$task_build"
mkdir "$task_build"
trap 'code=$?; printf "%s\n" "$code" > "$task_build/exit-code.txt"' EXIT
sha256sum source/* assets/* fixture* native_trace.c slerp_boundary.h > "$task_build/inputs.sha256"
cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror -Isource -I. \
 source/sonic_motion_composer_native.c source/sonic_motion_composer_libm_candidate.c \
 source/sonic_motion_entry_matcher_native.c native_trace.c -lm -o "$task_build/native-trace" \
 >"$task_build/build.stdout.txt" 2>"$task_build/build.stderr.txt"
sha256sum --quiet -c "$task_build/inputs.sha256"
sha256sum "$task_build/native-trace" > "$task_build/binary.sha256"
# Compile only. Capturing or replaying the fixture is a separate explicit step.
