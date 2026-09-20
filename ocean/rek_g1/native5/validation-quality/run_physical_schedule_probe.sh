#!/usr/bin/env bash
# Explicit future GPU invocation, only after the parent releases its live cohort.
set -euo pipefail
[[ $# == 2 ]] || { printf 'Usage: bash run_physical_schedule_probe.sh BUILT_DIRECTORY NEW_RUN_DIRECTORY\n' >&2; exit 2; }
task_build=$(realpath "$1")
task_out=$2
mkdir "$task_out"
task_out=$(realpath "$task_out")
task_stage=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914
task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
task_controller=/home/spark-advantage/codexrook-runtime/generated/gear-sonic-g1batch8-mode0-20260909T0203Z
task_export=/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
task_features=/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
export REK_PHYSICS_BACKEND=mujoco_cuda REK_ALLOW_CPU_EVALUATION=0
unset REK_PUFFYSICS_STABILIZATION REK_FAST_SCORING REK_FAST_OPPONENT REK_FAST_OBSERVATION
export REK_MUJOCO_KERNEL_CATALOG=$task_stage/kernel-catalog-v3.json
export REK_MUJOCO_CONDITIONAL_PTX=$task_stage/build-native-v2/mujoco-conditional.ptx
export REK_MUJOCO_CONDITIONAL_SHA256=2300284dfc4f4560234ef6ffe8a3fab8c918a1abc8ffcdcf270e180042837b21
task_run=(timeout 120s "$task_build/physical-schedule-probe" --run "$task_assets/model.two_fighter_arena.xml" "$task_export" "$task_assets" "$task_features" "$task_controller/model_encoder.batch8.onnx" "$task_controller/model_decoder.batch8.onnx")
{ printf '%s\n' "REK_PHYSICS_BACKEND=$REK_PHYSICS_BACKEND" "REK_ALLOW_CPU_EVALUATION=$REK_ALLOW_CPU_EVALUATION" "REK_MUJOCO_KERNEL_CATALOG=$REK_MUJOCO_KERNEL_CATALOG" "REK_MUJOCO_CONDITIONAL_PTX=$REK_MUJOCO_CONDITIONAL_PTX" "REK_MUJOCO_CONDITIONAL_SHA256=$REK_MUJOCO_CONDITIONAL_SHA256"; printf '%q ' "${task_run[@]}";printf '\n'; } > "$task_out/run-command.txt"
{ date -u --iso-8601=ns;sha256sum "$task_build/physical-schedule-probe" "$task_build/physical_schedule_probe.cpp" "$task_assets/model.two_fighter_arena.xml" "$task_export" "$task_assets"/*.json "$task_features"/*.json "$task_controller"/*.onnx "$REK_MUJOCO_KERNEL_CATALOG" "$REK_MUJOCO_CONDITIONAL_PTX"; } > "$task_out/provenance.txt"
date -u --iso-8601=ns > "$task_out/start-utc.txt"
set +e
/usr/bin/time -v -o "$task_out/process-timing.txt" "${task_run[@]}" > "$task_out/stdout.jsonl" 2> "$task_out/stderr.txt"
task_status=$?
set -e
date -u --iso-8601=ns > "$task_out/end-utc.txt"
printf '%s\n' "$task_status" > "$task_out/exit-code.txt"
sha256sum "$task_out"/*.txt "$task_out/stdout.jsonl" > "$task_out/artifact-sha256.txt"
exit "$task_status"
