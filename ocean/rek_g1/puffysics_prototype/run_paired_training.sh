#!/usr/bin/env bash
# Each invocation runs one real PPO arm; run arms sequentially on one GPU.
set -euo pipefail
task_backend=${1:?backend: mujoco or puffysics}
task_label=${2:?unique output label}
task_horizon=${3:-16}
task_epochs=${4:-2}
task_profile=${5:-plain}
task_root=${REK_BENCH_ROOT:-/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z}
task_runtime=${REK_GPU_RUNTIME:-/home/spark-advantage/rek-training/gpu-runtime-20260910}
task_config=${REK_DUEL_CONFIG:-/home/spark-advantage/rek-training/deferred-observe-20260911-v1/training-benchmark-v1/deferred.json}
task_native=${REK_NATIVE_CONFIG:-/home/spark-advantage/rek-training/deferred-observe-20260911-v1/training-benchmark-v1/training/deferred-r1.inputs/resolved-native.ini}
task_checkpoint=${REK_CHECKPOINT:-/home/spark-advantage/rek-training/winrate-20260911-v1/training/encoded-long-r1/0000000003276800.bin}
task_source="$task_root/source/ocean/rek_g1"
task_prototype="$task_source/puffysics_prototype"
task_python=${REK_PYTHON:-/home/spark-advantage/.venv/bin/python}
task_extension=${REK_NATIVE_EXTENSION:-$task_runtime/native-external-bootstrap-v2}
export PYTHONPATH="$task_extension:$task_source:$task_runtime/controller-deps:$task_runtime/deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ ! "$task_label" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    printf 'Output label must be a single path component\n' >&2
    exit 2
fi
if [[ -e "$task_root/$task_label" || -e "$task_root/$task_label.stdout.log" ]]; then
    printf 'Output already exists: %s\n' "$task_root/$task_label" >&2
    exit 2
fi
task_args=("$task_python" -u -B "$task_prototype/profile_training.py"
    --backend "$task_backend" --gpu-duel-config "$task_config"
    --default-config "$task_root/source/config/default.ini" --native-config "$task_native"
    --run-dir "$task_root/$task_label" --horizon "$task_horizon" --epochs "$task_epochs"
    --warmup-updates "${REK_BENCH_WARMUP_UPDATES:-1}" --minibatch-size 4096
    --policy-observation-encoder scaled_polar_xy_v1 --load-checkpoint "$task_checkpoint"
    --no-conditional-reset-forward)
if [[ "$task_backend" == puffysics ]]; then
    task_args+=(--puffysics-library "${REK_PUFFYSICS_LIBRARY:-$task_prototype/librek_puffysics_semantic_v1.so}"
        --solver-mode "${REK_PUFFYSICS_SOLVER_MODE:-1}"
        --model-export "${REK_PUFFYSICS_MODEL_EXPORT:-/home/spark-advantage/rek-training/rek-puffysics-prototype-20260911/model-export.json}")
fi
if [[ "$task_profile" == nsys ]]; then
    task_args+=(--cuda-profiler-range)
    task_args=(nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
        --capture-range=cudaProfilerApi --capture-range-end=stop
        --output="$task_root/$task_label" "${task_args[@]}")
elif [[ "$task_profile" == events ]]; then
    task_args+=(--step-timing)
elif [[ "$task_profile" != plain ]]; then
    printf 'Unknown profiling mode: %s\n' "$task_profile" >&2
    exit 2
fi
printf '%q ' "${task_args[@]}" > "$task_root/$task_label.command.txt"
printf '\n' >> "$task_root/$task_label.command.txt"
nvidia-smi --query-gpu=name,utilization.gpu,power.draw,temperature.gpu --format=csv > "$task_root/$task_label.gpu-before.txt"
task_status=0
"${task_args[@]}" > "$task_root/$task_label.stdout.log" 2> "$task_root/$task_label.stderr.log" || task_status=$?
printf '%s\n' "$task_status" > "$task_root/$task_label.exit.txt"
nvidia-smi --query-gpu=name,utilization.gpu,power.draw,temperature.gpu --format=csv > "$task_root/$task_label.gpu-after.txt"
tail -4 "$task_root/$task_label.stdout.log"
if (( task_status != 0 )); then
    tail -30 "$task_root/$task_label.stderr.log"
fi
exit "$task_status"
