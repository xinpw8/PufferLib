#!/usr/bin/env bash
# Reproduce the existing optimized REK candidate's real native PPO workload.
set -euo pipefail
task_out=${1:?usage: run_current_training_baseline.sh NEW_OUTPUT_DIRECTORY}
mkdir "$task_out"
task_runtime=/home/spark-advantage/rek-training/gpu-runtime-20260910
task_previous=/home/spark-advantage/rek-training/deferred-observe-20260911-v1
export PYTHONPATH="$task_runtime/native-external-bootstrap-v2:$task_previous/source/ocean/rek_g1:$task_runtime/controller-deps:$task_runtime/deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cp "$0" "$task_out/command.sh"
hostname > "$task_out/hostname.txt"
nvidia-smi -q > "$task_out/nvidia-smi-before.txt"
for task_repeat in 1 2 3; do
    task_args=(/home/spark-advantage/.venv/bin/python -u -B
        "$task_previous/source/ocean/rek_g1/train_gpu_duel.py"
        --gpu-duel-config "$task_previous/training-benchmark-v1/deferred.json"
        --default-config /home/spark-advantage/rek-training/training-opt-20260911/baseline/config/default.ini
        --native-config "$task_previous/training-benchmark-v1/training/deferred-r1.inputs/resolved-native.ini"
        --run-dir "$task_out/training-r$task_repeat"
        --output "$task_out/report-r$task_repeat.json"
        --total-timesteps 262144 --total-agents 1024 --horizon 256
        --minibatch-size 4096 --log-every 1 --checkpoint-every 0
        --opponent candidate-dummy --policy-observation-encoder scaled_polar_xy_v1
        --load-checkpoint /home/spark-advantage/rek-training/winrate-20260911-v1/training/encoded-long-r1/0000000003276800.bin)
    printf '%q ' "${task_args[@]}" > "$task_out/command-r$task_repeat.txt"
    printf '\n' >> "$task_out/command-r$task_repeat.txt"
    task_status=0
    "${task_args[@]}" > "$task_out/stdout-r$task_repeat.log" 2> "$task_out/stderr-r$task_repeat.log" || task_status=$?
    printf '%s\n' "$task_status" > "$task_out/exit-r$task_repeat.txt"
    printf 'repeat=%s exit=%s report=%s\n' "$task_repeat" "$task_status" "$task_out/report-r$task_repeat.json"
    if (( task_status != 0 )); then
        tail -40 "$task_out/stderr-r$task_repeat.log"
        exit "$task_status"
    fi
done
nvidia-smi -q > "$task_out/nvidia-smi-after.txt"
