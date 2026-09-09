#!/usr/bin/env bash
set -euo pipefail

required=(
  REK_G1_PUFFER_EXTENSION
  REK_G1_PUFFER_EXTENSION_SHA256
  REK_G1_SEMANTIC_ASSETS_DIR
  REK_G1_ASSET_MANIFEST_SHA256
  REK_G1_MODEL_SHA256
  REK_G1_ENCODER_ONNX
  REK_G1_ENCODER_SHA256
  REK_G1_DECODER_ONNX
  REK_G1_DECODER_SHA256
  REK_G1_HUMAN_EVAL_TRACE
)

for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf 'required environment variable is absent: %s\n' "$name" >&2
    exit 64
  fi
done

python_bin="${REK_G1_HUMAN_EVAL_PYTHON:-python3}"
port="${REK_G1_HUMAN_EVAL_PORT:-18766}"
physics_workers="${REK_G1_HUMAN_EVAL_PHYSICS_WORKERS:-4}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

exec "$python_bin" "$script_dir/human_eval_server.py" \
  --extension "$REK_G1_PUFFER_EXTENSION" \
  --extension-sha256 "$REK_G1_PUFFER_EXTENSION_SHA256" \
  --semantic-assets "$REK_G1_SEMANTIC_ASSETS_DIR" \
  --asset-manifest-sha256 "$REK_G1_ASSET_MANIFEST_SHA256" \
  --model-sha256 "$REK_G1_MODEL_SHA256" \
  --encoder "$REK_G1_ENCODER_ONNX" \
  --encoder-sha256 "$REK_G1_ENCODER_SHA256" \
  --decoder "$REK_G1_DECODER_ONNX" \
  --decoder-sha256 "$REK_G1_DECODER_SHA256" \
  --trace-out "$REK_G1_HUMAN_EVAL_TRACE" \
  --port "$port" \
  --physics-workers "$physics_workers"
