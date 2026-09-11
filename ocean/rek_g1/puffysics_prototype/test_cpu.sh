#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
task_output=$(mktemp -d /tmp/rek-puffysics-tests.XXXXXX)
printf 'Test artifacts: %s\n' "$task_output"
g++ -std=c++17 -O2 test_cylinder_geometry.cpp -o "$task_output/cylinders"
"$task_output/cylinders"
g++ -std=c++17 -O2 test_exact_touch.cpp -o "$task_output/exact_touch"
"$task_output/exact_touch"
g++ -std=c++17 -O2 cpu-review/test_world_com_regression.cpp -o "$task_output/dynamics"
"$task_output/dynamics"
g++ -std=c++17 -O2 cpu-review/test_armature.cpp -o "$task_output/armature"
"$task_output/armature"
g++ -std=c++17 -O2 -DTEST_ORIGINAL=1 -DB3_ART_FIXED_POSE_CACHE=0 test_art_cache.cpp -o "$task_output/original"
g++ -std=c++17 -O2 -DTEST_ORIGINAL=0 -DB3_ART_FIXED_POSE_CACHE=1 test_art_cache.cpp -o "$task_output/cached"
"$task_output/original" "$task_output/original.bin"
"$task_output/cached" "$task_output/cached.bin"
cmp "$task_output/original.bin" "$task_output/cached.bin"
printf 'PASS: geometry, exact touch, world-COM dynamics, armature and bitwise cache comparison\n'
