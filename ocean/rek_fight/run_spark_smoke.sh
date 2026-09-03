#!/bin/bash
set -euo pipefail
cd /home/spark-advantage/pufferlib-rek
export MUJOCO_HOME=/home/spark-advantage/rek-evidence/plant-7cc6ce1/venv-mujoco-3.7.0/lib/python3.12/site-packages/mujoco
export MUJOCO_LIB=$MUJOCO_HOME/libmujoco.so.3.7.0
export REK_MJCF_PATH=/home/spark-advantage/rek-evidence/plant-7cc6ce1/evidence_out/t800_t800_factory_arena.diagnostic.xml
cc -std=c11 -Wall -Wextra -O2 -I"$MUJOCO_HOME/include" \
  ocean/rek_fight/test_rek_fight.c -o /tmp/test_rek_fight \
  "$MUJOCO_LIB" -Wl,-rpath,"$(dirname "$MUJOCO_LIB")" -lm
/tmp/test_rek_fight
