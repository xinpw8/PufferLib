#!/usr/bin/env bash
# Build one new diagnostic against preserved, feature-enabled runtime objects.
set -euo pipefail
[[ $# == 1 ]] || { printf 'Usage: %s NEW_OUTPUT_DIRECTORY\n' "$0" >&2; exit 2; }
task_out=$1
mkdir "$task_out"
task_out=$(realpath "$task_out")
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_stage=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914
task_base=$task_stage/build-native-v2
task_native=$task_stage/native-source/ocean/rek_g1/native5
task_cuda=/usr/local/cuda
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
task_controller=/home/spark-advantage/codexrook-runtime/generated/gear-sonic-g1batch8-mode0-20260909T0203Z
cp "$task_source/physical_quality_probe.cpp" "$task_source/physical_quality_probe.sh" "$task_out/"
task_objects=()
for task_object in "$task_base"/*.o; do
    case $(basename "$task_object") in pufferl.o) continue;; esac
    task_objects+=("$task_object")
done
task_compile=("$task_cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_121
    -I"$task_native" -I"$task_native/.." -I"$task_mujoco/include"
    -c "$task_out/physical_quality_probe.cpp" -o "$task_out/probe.o")
task_link=("$task_cuda/bin/nvcc" -arch=sm_121 "$task_out/probe.o" "${task_objects[@]}"
    -L"$task_mujoco" -L"$task_cuda/lib64" -L"$task_cuda/lib64/stubs"
    -Xlinker=-rpath -Xlinker="$task_mujoco" -Xlinker=-rpath -Xlinker="$task_cuda/lib64"
    -Xlinker=--wrap=mj_step -Xlinker=--wrap=mj_forward -Xlinker=--wrap=mj_kinematics
    -lcudart -lcuda -lcublas -lcrypto -l:libmujoco.so.3.7.0 -lz -lm -lpthread
    -o "$task_out/physical-quality-probe")
{ printf '%q ' "${task_compile[@]}";printf '\n';printf '%q ' "${task_link[@]}";printf '\n'; } > "$task_out/build-command.txt"
"${task_compile[@]}" > "$task_out/build.stdout.txt" 2> "$task_out/build.stderr.txt"
"${task_link[@]}" >> "$task_out/build.stdout.txt" 2>> "$task_out/build.stderr.txt"
export REK_PHYSICS_BACKEND=mujoco_cuda REK_ALLOW_CPU_EVALUATION=0
unset REK_PUFFYSICS_STABILIZATION REK_FAST_SCORING REK_FAST_OPPONENT REK_FAST_OBSERVATION
export REK_MUJOCO_KERNEL_CATALOG=$task_stage/kernel-catalog-v3.json
export REK_MUJOCO_CONDITIONAL_PTX=$task_base/mujoco-conditional.ptx
export REK_MUJOCO_CONDITIONAL_SHA256=2300284dfc4f4560234ef6ffe8a3fab8c918a1abc8ffcdcf270e180042837b21
task_export=/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
task_features=/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
task_run=(timeout 120s "$task_out/physical-quality-probe" "$task_assets/model.two_fighter_arena.xml"
    "$task_export" "$task_assets" "$task_features"
    "$task_controller/model_encoder.batch8.onnx" "$task_controller/model_decoder.batch8.onnx")
{ printf '%s\n' "REK_PHYSICS_BACKEND=$REK_PHYSICS_BACKEND" "REK_ALLOW_CPU_EVALUATION=$REK_ALLOW_CPU_EVALUATION" \
    "REK_MUJOCO_KERNEL_CATALOG=$REK_MUJOCO_KERNEL_CATALOG" "REK_MUJOCO_CONDITIONAL_PTX=$REK_MUJOCO_CONDITIONAL_PTX" \
    "REK_MUJOCO_CONDITIONAL_SHA256=$REK_MUJOCO_CONDITIONAL_SHA256";
    printf '%q ' "${task_run[@]}";printf '\n'; } > "$task_out/run-command.txt"
{ date -u --iso-8601=seconds;hostname;uname -m;
    sha256sum "$task_out/physical-quality-probe" "$task_out/physical_quality_probe.cpp" "$task_out/physical_quality_probe.sh"
    sha256sum "$task_native/runtime_api.h" "$task_assets/model.two_fighter_arena.xml" "$task_export"
    sha256sum "$task_controller"/*.onnx "$REK_MUJOCO_KERNEL_CATALOG" "$REK_MUJOCO_CONDITIONAL_PTX"
    sha256sum "${task_objects[@]}"; } > "$task_out/provenance.txt"
readelf -d "$task_out/physical-quality-probe" > "$task_out/elf-dependencies.txt"
set +e
/usr/bin/time -v -o "$task_out/process-timing.txt" "${task_run[@]}" > "$task_out/stdout.jsonl" 2> "$task_out/stderr.txt"
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_out/exit-code.txt"
tail -n 6 "$task_out/stdout.jsonl"
tail -n 8 "$task_out/stderr.txt"
exit "$task_status"
