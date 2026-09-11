#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /absolute/cache/directory" >&2
    exit 64
fi

destination=$1
repository=https://github.com/engineai-robotics/engineai_robotics_native_sdk.git
commit=335c60e88772c26c7852d0abd6b3c7439037dd8f
walking_policy=assets/config/t800/rl_walking_example/policy/t800_260618_165257_30000.mnn
walking_sha256=cbcb90f86dbb2fde39bdc5a25c8d0530d5c79c7a8f84b1f90863d8c9065b6427
recovery_policy=assets/config/t800/rl_supine_to_stance/policy/T800_supine_to_stance.mnn
recovery_policy_sha256=deb9974b1f4f4a7e77801f8c9c6e77f599caab0ca4dd7709fe0bae55870e0e86
recovery_trajectory=assets/config/t800/rl_supine_to_stance/trajectory/T800_supine_to_stance.npy
recovery_trajectory_sha256=c2f19c164093701311634024eb27999fed4631a00d38d507f8aa306ee138c161

if [[ ! -e "$destination" ]]; then
    git clone --filter=blob:none "$repository" "$destination"
elif [[ ! -d "$destination/.git" ]]; then
    echo "destination exists and is not a git checkout: $destination" >&2
    exit 1
fi

git -C "$destination" fetch --tags origin
git -C "$destination" checkout --detach "$commit"

actual_commit=$(git -C "$destination" rev-parse HEAD)
actual_policy_sha256=$(sha256sum "$destination/$walking_policy" | cut -d' ' -f1)
actual_recovery_policy_sha256=$(sha256sum "$destination/$recovery_policy" | cut -d' ' -f1)
actual_recovery_trajectory_sha256=$(sha256sum "$destination/$recovery_trajectory" | cut -d' ' -f1)
if [[ "$actual_commit" != "$commit" ]]; then
    echo "EngineAI SDK commit mismatch: $actual_commit" >&2
    exit 1
fi
if [[ "$actual_policy_sha256" != "$walking_sha256" ]]; then
    echo "EngineAI walking policy SHA-256 mismatch: $actual_policy_sha256" >&2
    exit 1
fi
if [[ "$actual_recovery_policy_sha256" != "$recovery_policy_sha256" ]]; then
    echo "EngineAI recovery policy SHA-256 mismatch: $actual_recovery_policy_sha256" >&2
    exit 1
fi
if [[ "$actual_recovery_trajectory_sha256" != "$recovery_trajectory_sha256" ]]; then
    echo "EngineAI recovery trajectory SHA-256 mismatch: $actual_recovery_trajectory_sha256" >&2
    exit 1
fi

printf '{"repository":"%s","commit":"%s","walking_policy":"%s","walking_policy_sha256":"%s","recovery_policy":"%s","recovery_policy_sha256":"%s","recovery_trajectory":"%s","recovery_trajectory_sha256":"%s"}\n' \
    "$repository" "$actual_commit" "$destination/$walking_policy" "$actual_policy_sha256" \
    "$destination/$recovery_policy" "$actual_recovery_policy_sha256" \
    "$destination/$recovery_trajectory" "$actual_recovery_trajectory_sha256"
