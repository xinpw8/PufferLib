#!/bin/bash
set -euo pipefail
cd /home/spark-advantage/pufferlib-rek
set +u
source /home/spark-advantage/pufferlib-4.0/.venv/bin/activate
set -u
export CUDA_HOME=/usr/local/cuda
export MACHINE=aarch64
export NVCC_ARCH=sm_121
export MUJOCO_HOME=/home/spark-advantage/rek-evidence/plant-7cc6ce1/venv-mujoco-3.7.0/lib/python3.12/site-packages/mujoco
export MUJOCO_LIB=$MUJOCO_HOME/libmujoco.so.3.7.0
export REK_MJCF_PATH=/home/spark-advantage/rek-evidence/plant-7cc6ce1/evidence_out/t800_t800_factory_arena.diagnostic.xml
./build.sh rek_fight --cpu
./build.sh rek_fight
python - <<'PY'
from pufferlib import _C
print('compiled env', getattr(_C, 'env_name', None))
assert _C.env_name == 'rek_fight'
PY
