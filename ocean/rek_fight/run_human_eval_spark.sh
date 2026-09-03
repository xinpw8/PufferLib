#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"

export MUJOCO_HOME=${MUJOCO_HOME:-/home/spark-advantage/rek-evidence/plant-7cc6ce1/venv-mujoco-3.7.0/lib/python3.12/site-packages/mujoco}
export MUJOCO_LIB=${MUJOCO_LIB:-$MUJOCO_HOME/libmujoco.so.3.7.0}
export REK_MJCF_PATH=${REK_MJCF_PATH:-/home/spark-advantage/rek-evidence/plant-7cc6ce1/evidence_out/t800_t800_factory_arena.diagnostic.xml}
export REK_HUMAN_EVAL_PYTHON=${REK_HUMAN_EVAL_PYTHON:-/home/spark-advantage/rek-evidence/work/mnn-venv/bin/python}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export ENGINEAI_SDK_ROOT=${ENGINEAI_SDK_ROOT:-/home/spark-advantage/rek-evidence/upstream/engineai_robotics_native_sdk}
export REK_WALKING_POLICY=${REK_WALKING_POLICY:-$ENGINEAI_SDK_ROOT/assets/config/t800/rl_walking_example/policy/t800_260618_165257_30000.mnn}
export REK_RECOVERY_POLICY=${REK_RECOVERY_POLICY:-$ENGINEAI_SDK_ROOT/assets/config/t800/rl_supine_to_stance/policy/T800_supine_to_stance.mnn}
export REK_RECOVERY_TRAJECTORY=${REK_RECOVERY_TRAJECTORY:-$ENGINEAI_SDK_ROOT/assets/config/t800/rl_supine_to_stance/trajectory/T800_supine_to_stance.npy}

expected_mjcf_sha=01caa6ed90277a90fc71b4c16d7959fb70f7702fa15bea7c68d3eaa9d5f27b2c
actual_mjcf_sha=$(sha256sum "$REK_MJCF_PATH" | cut -d' ' -f1)
if [[ "$actual_mjcf_sha" != "$expected_mjcf_sha" ]]; then
    echo "unexpected REK_MJCF_PATH SHA-256: $actual_mjcf_sha" >&2
    exit 1
fi
expected_engineai_commit=335c60e88772c26c7852d0abd6b3c7439037dd8f
actual_engineai_commit=$(git -C "$ENGINEAI_SDK_ROOT" rev-parse HEAD)
if [[ "$actual_engineai_commit" != "$expected_engineai_commit" ]]; then
    echo "unexpected EngineAI SDK commit: $actual_engineai_commit" >&2
    exit 1
fi
expected_policy_sha=cbcb90f86dbb2fde39bdc5a25c8d0530d5c79c7a8f84b1f90863d8c9065b6427
actual_policy_sha=$(sha256sum "$REK_WALKING_POLICY" | cut -d' ' -f1)
if [[ "$actual_policy_sha" != "$expected_policy_sha" ]]; then
    echo "unexpected EngineAI walking policy SHA-256: $actual_policy_sha" >&2
    exit 1
fi
expected_recovery_policy_sha=deb9974b1f4f4a7e77801f8c9c6e77f599caab0ca4dd7709fe0bae55870e0e86
actual_recovery_policy_sha=$(sha256sum "$REK_RECOVERY_POLICY" | cut -d' ' -f1)
if [[ "$actual_recovery_policy_sha" != "$expected_recovery_policy_sha" ]]; then
    echo "unexpected EngineAI recovery policy SHA-256: $actual_recovery_policy_sha" >&2
    exit 1
fi
expected_recovery_trajectory_sha=c2f19c164093701311634024eb27999fed4631a00d38d507f8aa306ee138c161
actual_recovery_trajectory_sha=$(sha256sum "$REK_RECOVERY_TRAJECTORY" | cut -d' ' -f1)
if [[ "$actual_recovery_trajectory_sha" != "$expected_recovery_trajectory_sha" ]]; then
    echo "unexpected EngineAI recovery trajectory SHA-256: $actual_recovery_trajectory_sha" >&2
    exit 1
fi

output_dir=${REK_HUMAN_EVAL_OUTPUT_DIR:-/tmp/rek-human-eval}
mkdir -p "$output_dir"

exec "$REK_HUMAN_EVAL_PYTHON" ocean/rek_fight/human_eval_server.py \
    --host "${REK_HUMAN_EVAL_HOST:-127.0.0.1}" \
    --port "${REK_HUMAN_EVAL_PORT:-18766}" \
    --model "$REK_MJCF_PATH" \
    --walking-policy "$REK_WALKING_POLICY" \
    --recovery-policy "$REK_RECOVERY_POLICY" \
    --recovery-trajectory "$REK_RECOVERY_TRAJECTORY" \
    --log "${REK_HUMAN_EVAL_LOG:-$output_dir/actions.jsonl}"
