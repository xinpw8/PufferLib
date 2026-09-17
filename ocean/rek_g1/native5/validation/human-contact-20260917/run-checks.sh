#!/usr/bin/env bash
# Bounded paired geometry experiment, using a frozen existing policy.
set -euo pipefail
[[ $# == 1 ]] || exit 2
task_stage=$(realpath "$1")
task_source=$task_stage/source/ocean/rek_g1/native5
task_config=$task_source/validation/human-contact-20260917
task_old=/home/spark-advantage/rek-training/policy-quality-20260916-r1
task_checkpoint=$task_old/train-bot1-rendered-r1/checkpoints/rek_native5/train-bot1-rendered-r1/0000000536870912.bin
task_sha=f87dae69a777e4ac28782bdee89b30d7434208d773be75bf97f56fba4a52b07e
mkdir "$task_stage/checks"
exec > "$task_stage/checks/stdout.txt" 2> "$task_stage/checks/stderr-and-commands.txt"
set -x
hostname
id
date -u --iso-8601=seconds
g++ -std=c++17 -O2 -Wall -Wextra -Werror "$task_source/test_primitive_contacts.cpp" -o "$task_stage/checks/primitive-test"
"$task_stage/checks/primitive-test"
bash "$task_source/fast_assets_probe.sh" "$task_stage/assets-probe" /home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact/model.two_fighter_arena.xml /home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact /home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
bash "$task_source/build_diverse_policy_eval.sh" "$task_stage/build" "$task_stage/eval"
for task_mode in baseline primitive-1 primitive-4 primitive-8; do
    case "$task_mode" in
        baseline) task_runtime=$task_source/validation/quality-20260916/recovered-bot1-rendered-runtime.json ;;
        primitive-4) task_runtime=$task_config/primitive-runtime.json ;;
        primitive-1) task_runtime=$task_config/primitive-runtime-1.json ;;
        primitive-8) task_runtime=$task_config/primitive-runtime-8.json ;;
    esac
    bash "$task_source/run_diverse_policy_eval.sh" "$task_stage/eval" "$task_runtime" "$task_checkpoint" "$task_sha"       "$task_stage/eval-$task_mode" 4 2 9171058 sampled bf16 scripted heldout 120
done
sha256sum "$task_source/primitive_contacts.cuh" "$task_source/primitive_motion.cuh"  "$task_source/test_primitive_contacts.cpp" "$task_source/fast_assets.h" "$task_source/fast_assets.cpp"  "$task_source/fast_runtime.cu" "$task_source/fast_mode_config.h" "$task_stage/build/puffer-rek-native5"  > "$task_stage/checks/hashes.txt"
