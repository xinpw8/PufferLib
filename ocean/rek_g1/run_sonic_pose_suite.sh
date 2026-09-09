#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "usage: $0 PYTHON REPO_ROOT ONNX YAML ASSETS_DIR RESULTS_DIR" >&2
    exit 64
fi

PYTHON=$1
REPO_ROOT=$2
ONNX=$3
YAML=$4
ASSETS_DIR=$5
RESULTS_DIR=$6

for path in "$PYTHON" "$ONNX" "$YAML"; do
    if [ ! -f "$path" ]; then
        echo "required file is missing: $path" >&2
        exit 66
    fi
done
for path in "$REPO_ROOT" "$ASSETS_DIR" "$RESULTS_DIR"; do
    if [ ! -d "$path" ]; then
        echo "required directory is missing: $path" >&2
        exit 66
    fi
done

CANDIDATE="$REPO_ROOT/ocean/rek_g1/sonic_candidate.py"
ARENA="$REPO_ROOT/ocean/rek/evidence/evidence_out/g1_arena_physics_contract.v1.json"
ROLES=(
    idle
    walk
    strafe_left
    turn_right
    kick_left_side
    kick_left_front
    kick_right_side
    kick_right_knee
)

for role in "${ROLES[@]}"; do
    prefix="$RESULTS_DIR/$role"
    "$PYTHON" "$CANDIDATE" \
        --onnx "$ONNX" \
        --yaml "$YAML" \
        --assets-dir "$ASSETS_DIR" \
        --motion-role "$role" \
        --arena-contract "$ARENA" \
        --metrics-out "$prefix.metrics.json" \
        --trace "$prefix.trace.jsonl" \
        > "$prefix.stdout.json"
done

sha256sum "$RESULTS_DIR"/*.metrics.json \
    "$RESULTS_DIR"/*.stdout.json \
    "$RESULTS_DIR"/*.trace.jsonl
