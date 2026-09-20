#!/usr/bin/env bash
# Build and run only CPU schedule tests. Never starts the CUDA probe.
set -euo pipefail
[[ $# == 1 ]] || { printf 'Usage: bash build_physical_schedule_probe.sh NEW_OUTPUT_DIRECTORY\n' >&2; exit 2; }
task_out=$1
mkdir "$task_out"
task_out=$(realpath "$task_out")
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_stage=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914
task_base=$task_stage/build-native-v2
task_native=$task_stage/native-source/ocean/rek_g1/native5
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
task_cuda=/usr/local/cuda
cp "$task_source/physical_schedule_probe.cpp" "$task_source/build_physical_schedule_probe.sh" "$task_source/run_physical_schedule_probe.sh" "$task_out/"
grep -F '6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16,' "$task_native/../g1_semantic_action_table.c" > "$task_out/actual-registry-order.txt"
g++ -std=c++17 -O2 -DREK_SCHEDULE_CPU_ONLY "$task_out/physical_schedule_probe.cpp" -o "$task_out/schedule-cpu"
"$task_out/schedule-cpu" --self-test > "$task_out/cpu-tests.jsonl"
"$task_out/schedule-cpu" --schedule-only > "$task_out/schedule.jsonl"
task_objects=()
for task_object in "$task_base"/*.o; do
    case $(basename "$task_object") in pufferl.o) continue;; esac
    task_objects+=("$task_object")
done
task_compile=("$task_cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_121 -I"$task_native" -I"$task_native/.." -I"$task_mujoco/include" -c "$task_out/physical_schedule_probe.cpp" -o "$task_out/probe.o")
task_link=("$task_cuda/bin/nvcc" -arch=sm_121 "$task_out/probe.o" "${task_objects[@]}" -L"$task_mujoco" -L"$task_cuda/lib64" -L"$task_cuda/lib64/stubs" -Xlinker=-rpath -Xlinker="$task_mujoco" -Xlinker=-rpath -Xlinker="$task_cuda/lib64" -Xlinker=--wrap=mj_step -Xlinker=--wrap=mj_forward -Xlinker=--wrap=mj_kinematics -lcudart -lcuda -lcublas -lcrypto -l:libmujoco.so.3.7.0 -lz -lm -lpthread -o "$task_out/physical-schedule-probe")
{ printf '%q ' "${task_compile[@]}";printf '\n';printf '%q ' "${task_link[@]}";printf '\n'; } > "$task_out/build-command.txt"
"${task_compile[@]}" > "$task_out/build.stdout.txt" 2> "$task_out/build.stderr.txt"
"${task_link[@]}" >> "$task_out/build.stdout.txt" 2>> "$task_out/build.stderr.txt"
{ date -u --iso-8601=ns; sha256sum "$task_out/physical-schedule-probe" "$task_out/physical_schedule_probe.cpp" "$task_out/build_physical_schedule_probe.sh" "$task_out/run_physical_schedule_probe.sh" "$task_native/runtime_api.h" "$task_native/../g1_semantic_action_table.c" "$task_native/../native_motion_routes.c" "${task_objects[@]}"; } > "$task_out/build-provenance.txt"
printf 'Build and CPU schedule tests completed; CUDA probe was not executed.\n'
