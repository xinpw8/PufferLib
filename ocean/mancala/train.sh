#!/bin/bash
# train.sh — build (if needed) and train pufferlib-4.0 mancala on DGX Spark.
# Lives in ocean/mancala/ in the local clone; the deployed copy on spark is
# /home/spark-advantage/pufferlib-test/PufferLib/train_mancala.sh.
#
# Usage:
#   ./train_mancala.sh                                      # 10M steps, no wandb
#   ./train_mancala.sh --train.total-timesteps 200000000    # 200M steps
#   ./train_mancala.sh --train.total-timesteps 200000000 --wandb \
#       --wandb-project mancala --wandb-group myrun         # with wandb
#
# Environment overrides (optional):
#   PUFFERLIB_DIR=/path/to/PufferLib    # override default spark location
#   SKIP_BUILD=1                        # skip the auto-rebuild check

set -euo pipefail

PUFFERLIB_DIR="${PUFFERLIB_DIR:-/home/spark-advantage/pufferlib-test/PufferLib}"
VENV_DIR="${VENV_DIR:-/home/spark-advantage/pufferlib-test/venv}"
cd "$PUFFERLIB_DIR"

# venv with pufferlib + cuda/nccl/cudnn wheels
source "$VENV_DIR/bin/activate"

# nvcc/nccl/cudnn paths the build.sh and runtime need
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
SP="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
export CPATH="$SP/nccl/include:${CPATH:-}"
export LIBRARY_PATH="$SP/nccl/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$SP/nccl/lib:$SP/cudnn/lib:${LD_LIBRARY_PATH:-}"

# Rebuild only if env source is newer than the compiled _C.so
if [[ -z "${SKIP_BUILD:-}" ]]; then
    SO=$(ls pufferlib/_C*.so 2>/dev/null | head -1 || true)
    if [[ -z "$SO" ]] || find ocean/mancala config/mancala.ini -newer "$SO" 2>/dev/null | grep -q .; then
        echo "[train.sh] rebuilding (env source newer than $SO)"
        ./build.sh mancala
    else
        echo "[train.sh] $SO is up to date — skipping rebuild"
    fi
fi

# Default total-timesteps if user didn't pass one
ARGS=("$@")
HAS_TS=0
for a in "${ARGS[@]}"; do
    case "$a" in --train.total-timesteps|--train.total-timesteps=*) HAS_TS=1;; esac
done
if [[ "$HAS_TS" -eq 0 ]]; then
    ARGS=(--train.total-timesteps 10000000 "${ARGS[@]}")
fi

echo "[train.sh] python -m pufferlib.pufferl train mancala ${ARGS[*]}"
exec python -m pufferlib.pufferl train mancala "${ARGS[@]}"
