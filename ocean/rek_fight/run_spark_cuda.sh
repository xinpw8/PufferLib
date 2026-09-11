#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"

PUFFER_VENV=${PUFFER_VENV:-/home/spark-advantage/pufferlib-4.0/.venv}
set +u
source "$PUFFER_VENV/bin/activate"
set -u
export CUDA_HOME=/usr/local/cuda
export MACHINE=aarch64
export NVCC_ARCH=sm_121
export MUJOCO_HOME=${MUJOCO_HOME:-/home/spark-advantage/rek-evidence/plant-7cc6ce1/venv-mujoco-3.7.0/lib/python3.12/site-packages/mujoco}
export MUJOCO_LIB=${MUJOCO_LIB:-$MUJOCO_HOME/libmujoco.so.3.7.0}
export REK_MJCF_PATH=${REK_MJCF_PATH:-/home/spark-advantage/rek-evidence/plant-7cc6ce1/evidence_out/t800_t800_factory_arena.diagnostic.xml}

expected_mjcf_sha=01caa6ed90277a90fc71b4c16d7959fb70f7702fa15bea7c68d3eaa9d5f27b2c
actual_mjcf_sha=$(sha256sum "$REK_MJCF_PATH" | cut -d' ' -f1)
if [[ "$actual_mjcf_sha" != "$expected_mjcf_sha" ]]; then
    echo "unexpected REK_MJCF_PATH SHA-256: $actual_mjcf_sha" >&2
    exit 1
fi

./build.sh rek_fight --float
python ocean/rek_fight/vector_smoke.py \
    --mode both \
    --steps "${REK_FIGHT_VECTOR_STEPS:-1000}" \
    --warmup "${REK_FIGHT_VECTOR_WARMUP:-20}" \
    --total-agents "${REK_FIGHT_TOTAL_AGENTS:-64}" \
    --num-threads "${REK_FIGHT_NUM_THREADS:-16}"
