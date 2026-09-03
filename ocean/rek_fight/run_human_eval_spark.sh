#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"

export MUJOCO_HOME=${MUJOCO_HOME:-/home/spark-advantage/rek-evidence/plant-7cc6ce1/venv-mujoco-3.7.0/lib/python3.12/site-packages/mujoco}
export MUJOCO_LIB=${MUJOCO_LIB:-$MUJOCO_HOME/libmujoco.so.3.7.0}
export REK_MJCF_PATH=${REK_MJCF_PATH:-/home/spark-advantage/rek-evidence/plant-7cc6ce1/evidence_out/t800_t800_factory_arena.diagnostic.xml}
export REK_HUMAN_EVAL_PYTHON=${REK_HUMAN_EVAL_PYTHON:-/home/spark-advantage/rek-evidence/plant-7cc6ce1/venv-mujoco-3.7.0/bin/python}
export MUJOCO_GL=${MUJOCO_GL:-egl}

expected_mjcf_sha=01caa6ed90277a90fc71b4c16d7959fb70f7702fa15bea7c68d3eaa9d5f27b2c
actual_mjcf_sha=$(sha256sum "$REK_MJCF_PATH" | cut -d' ' -f1)
if [[ "$actual_mjcf_sha" != "$expected_mjcf_sha" ]]; then
    echo "unexpected REK_MJCF_PATH SHA-256: $actual_mjcf_sha" >&2
    exit 1
fi

output_dir=${REK_HUMAN_EVAL_OUTPUT_DIR:-/tmp/rek-human-eval}
mkdir -p "$output_dir"
library_path="$output_dir/librek_human_eval.so"

cc -std=c11 -Wall -Wextra -O2 -fPIC -shared \
    -I"$MUJOCO_HOME/include" \
    ocean/rek_fight/human_eval_bridge.c \
    -o "$library_path" \
    "$MUJOCO_LIB" -Wl,-rpath,"$(dirname "$MUJOCO_LIB")" -lm

exec "$REK_HUMAN_EVAL_PYTHON" ocean/rek_fight/human_eval_server.py \
    --host "${REK_HUMAN_EVAL_HOST:-127.0.0.1}" \
    --port "${REK_HUMAN_EVAL_PORT:-18766}" \
    --library "$library_path" \
    --model "$REK_MJCF_PATH" \
    --log "${REK_HUMAN_EVAL_LOG:-$output_dir/actions.jsonl}"
