#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"

export MUJOCO_HOME=${MUJOCO_HOME:-/home/spark-advantage/rek-evidence/plant-7cc6ce1/venv-mujoco-3.7.0/lib/python3.12/site-packages/mujoco}
export MUJOCO_LIB=${MUJOCO_LIB:-$MUJOCO_HOME/libmujoco.so.3.7.0}
export REK_MJCF_PATH=${REK_MJCF_PATH:-/home/spark-advantage/rek-evidence/plant-7cc6ce1/evidence_out/t800_t800_factory_arena.diagnostic.xml}

expected_mjcf_sha=01caa6ed90277a90fc71b4c16d7959fb70f7702fa15bea7c68d3eaa9d5f27b2c
actual_mjcf_sha=$(sha256sum "$REK_MJCF_PATH" | cut -d' ' -f1)
if [[ "$actual_mjcf_sha" != "$expected_mjcf_sha" ]]; then
    echo "unexpected REK_MJCF_PATH SHA-256: $actual_mjcf_sha" >&2
    exit 1
fi

cc -std=c11 -Wall -Wextra -O2 -I"$MUJOCO_HOME/include" \
  ocean/rek_fight/test_rek_fight.c -o /tmp/test_rek_fight \
  "$MUJOCO_LIB" -Wl,-rpath,"$(dirname "$MUJOCO_LIB")" -lm
/tmp/test_rek_fight
/tmp/test_rek_fight --benchmark "${REK_FIGHT_BENCH_STEPS:-20000}"
