#!/usr/bin/env bash
# Private, offline replay of two pinned received-pose captures. No physics steps.
# Usage: bash run-pose-contact-replay.sh PRIVATE_STAGE NEW_RUN_LABEL
set -euo pipefail
umask 077
[[ $# == 2 ]] || exit 2
task_stage=$(realpath "$1")
task_label=$2
[[ $task_stage == /home/spark-advantage/rek-training/human-pose-replay-20260917-r1 ]] || exit 2
[[ $task_label =~ ^run-[a-z0-9-]+$ ]] || exit 2
task_run=$task_stage/$task_label
[[ ! -e $task_run ]] || exit 2
mkdir -m 700 "$task_run"
exec > "$task_run/stdout.txt" 2> "$task_run/stderr-and-commands.txt"
trap 'task_status=$?; printf "%s\n" "$task_status" > "$task_run/exit-code.txt"' EXIT
set -x
task_source=$task_stage/source/ocean/rek_g1/native5
task_vendor=$task_stage/source/vendor
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
task_model=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact/model.two_fighter_arena.xml
task_model_sha=6cec7d81b69187bfdf2429d71b6288ebb999b5359ecaab7990721323b21722aa
task_round1=$task_stage/captures/round1.private.jsonl
task_round2=$task_stage/captures/round2.private.jsonl
task_round1_sha=547cec42f700e97b9594f8d2c6df88f966052c0e0e5ce7395c4862b619fb6d7f
task_round2_sha=ec55c32a6e2272a8e7656d73260ca35876be2cd6c8d8d6fe8c9dfe77cd25a309
hostname
date -u --iso-8601=seconds
/usr/local/cuda/bin/nvcc --version
printf '%s  %s\n' "$task_model_sha" "$task_model" "$task_round1_sha" "$task_round1" "$task_round2_sha" "$task_round2" | sha256sum --check --strict
sha256sum "$task_source/pose_contact_replay.cu" "$task_source/primitive_contacts.cuh" "$task_vendor/cJSON.c" "$task_vendor/cJSON.h" "$task_source/validation/human-contact-20260917/run-pose-contact-replay.sh" "$task_model" "$task_round1" "$task_round2" "$task_mujoco/libmujoco.so.3.7.0" > "$task_run/input-hashes.txt"
gcc -std=c11 -O2 -c "$task_vendor/cJSON.c" -o "$task_run/cJSON.o"
/usr/local/cuda/bin/nvcc -std=c++17 -O2 -lineinfo -arch=sm_121 -I"$task_mujoco/include" "$task_source/pose_contact_replay.cu" "$task_run/cJSON.o" -Xlinker "$task_mujoco/libmujoco.so.3.7.0" -lcrypto -Xlinker -rpath -Xlinker "$task_mujoco" -o "$task_run/pose-contact-replay"
task_args=("$task_model" "$task_model_sha" "$task_round1" "$task_round1_sha" "$task_round2" "$task_round2_sha")
timeout 120 "$task_run/pose-contact-replay" "${task_args[@]}" "$task_run/pose-results.private.jsonl" > "$task_run/replay.stdout.txt" 2> "$task_run/replay.stderr.txt"
timeout 180 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 --log-file "$task_run/memcheck.txt" "$task_run/pose-contact-replay" "${task_args[@]}" "$task_run/memcheck-results.private.jsonl" > "$task_run/memcheck.stdout.txt" 2> "$task_run/memcheck.stderr.txt"
cmp "$task_run/pose-results.private.jsonl" "$task_run/memcheck-results.private.jsonl"
cmp "$task_run/replay.stdout.txt" "$task_run/memcheck.stdout.txt"
sha256sum "$task_run/pose-contact-replay" "$task_run/pose-results.private.jsonl" "$task_run/replay.stdout.txt" "$task_run/replay.stderr.txt" "$task_run/memcheck-results.private.jsonl" "$task_run/memcheck.stdout.txt" "$task_run/memcheck.stderr.txt" "$task_run/memcheck.txt" > "$task_run/output-hashes.txt"
printf 'Replay and memcheck completed; outputs are byte-identical.\n'
