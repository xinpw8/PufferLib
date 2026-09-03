#!/bin/bash
# Run from the checked-out repository on the DGX Spark.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"

train_steps=${REK_FIGHT_TRAIN_STEPS:-8192}
if [[ ! "$train_steps" =~ ^[1-9][0-9]*$ ]]; then
    echo "REK_FIGHT_TRAIN_STEPS must be a positive integer" >&2
    exit 2
fi
if (( train_steps > 8192 )) &&
   [[ ${REK_FIGHT_ALLOW_PROVISIONAL_LONG_RUN:-0} != 1 ]]; then
    echo "rek_fight has not passed the held-out REK parity gate" >&2
    echo "set REK_FIGHT_ALLOW_PROVISIONAL_LONG_RUN=1 only for an explicitly provisional run" >&2
    exit 2
fi

PUFFER_VENV=${PUFFER_VENV:-/home/spark-advantage/pufferlib-4.0/.venv}
set +u
source "$PUFFER_VENV/bin/activate"
set -u
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MACHINE=${MACHINE:-aarch64}
export NVCC_ARCH=${NVCC_ARCH:-sm_121}
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
artifact_root=${REK_FIGHT_ARTIFACT_ROOT:-$repo_root/train_artifacts/rek_fight}
mkdir -p "$artifact_root/checkpoints" "$artifact_root/logs"
python -m pufferlib.pufferl train rek_fight \
    --train.total-timesteps "$train_steps" \
    --checkpoint-dir "$artifact_root/checkpoints" \
    --log-dir "$artifact_root/logs" \
    --tag "${REK_FIGHT_TAG:-rek-fight-spark}"
