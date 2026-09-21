#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: %s BUILD NEW_OUTPUT\n' "$0" >&2; exit 2; }
task_build=$(realpath "$1"); mkdir "$2"; task_output=$(realpath "$2")
task_stage=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914
task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
task_controller=/home/spark-advantage/codexrook-runtime/generated/gear-sonic-g1batch8-mode0-20260909T0203Z
task_export=/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
task_features=/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
export REK_PHYSICS_BACKEND=mujoco_cuda REK_ALLOW_CPU_EVALUATION=0 REK_NATIVE5_REWARD=normalized_points_falls_v1
export REK_MUJOCO_KERNEL_CATALOG=$task_stage/kernel-catalog-v3.json
export REK_MUJOCO_CONDITIONAL_PTX=$task_stage/build-native-v2/mujoco-conditional.ptx
export REK_MUJOCO_CONDITIONAL_SHA256=2300284dfc4f4560234ef6ffe8a3fab8c918a1abc8ffcdcf270e180042837b21
unset REK_PUFFYSICS_STABILIZATION REK_FAST_SCORING REK_FAST_OPPONENT REK_FAST_OBSERVATION
for task_mode in legacy observable; do
    mkdir "$task_output/$task_mode"
    if [[ "$task_mode" == legacy ]]; then unset REK_OBSERVATION_SCHEMA; else export REK_OBSERVATION_SCHEMA=rek.native5.observable_balance.v1; fi
    task_command=(timeout 120s "$task_build/physical-observable-balance-probe" "$task_assets/model.two_fighter_arena.xml" "$task_export" "$task_assets" "$task_features" "$task_controller/model_encoder.batch8.onnx" "$task_controller/model_decoder.batch8.onnx")
    { printf '%s\n' "REK_PHYSICS_BACKEND=$REK_PHYSICS_BACKEND" "REK_ALLOW_CPU_EVALUATION=$REK_ALLOW_CPU_EVALUATION" "REK_NATIVE5_REWARD=$REK_NATIVE5_REWARD" "REK_OBSERVATION_SCHEMA=${REK_OBSERVATION_SCHEMA:-unset}" "REK_MUJOCO_KERNEL_CATALOG=$REK_MUJOCO_KERNEL_CATALOG" "REK_MUJOCO_CONDITIONAL_PTX=$REK_MUJOCO_CONDITIONAL_PTX" "REK_MUJOCO_CONDITIONAL_SHA256=$REK_MUJOCO_CONDITIONAL_SHA256"; printf '%q ' "${task_command[@]}"; printf '\n'; } > "$task_output/$task_mode/command.txt"
    set +e
    "${task_command[@]}" > "$task_output/$task_mode/stdout.jsonl" 2> "$task_output/$task_mode/stderr.txt"
    task_status=$?
    set -e
    printf '%s\n' "$task_status" > "$task_output/$task_mode/exit-code.txt"
    cat "$task_output/$task_mode/stdout.jsonl"
    if [[ "$task_status" != 0 ]]; then cat "$task_output/$task_mode/stderr.txt" >&2; exit "$task_status"; fi
done
sha256sum "$task_build/physical-observable-balance-probe" "$task_assets/model.two_fighter_arena.xml" "$task_export" "$task_controller/model_encoder.batch8.onnx" "$task_controller/model_decoder.batch8.onnx" "$REK_MUJOCO_KERNEL_CATALOG" "$REK_MUJOCO_CONDITIONAL_PTX" > "$task_output/provenance.txt"
