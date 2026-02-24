#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SELFPLAY_STEPS="${SELFPLAY_STEPS:-500000000}"
FINETUNE_STEPS="${FINETUNE_STEPS:-100000000}"
GATE_GAMES="${GATE_GAMES:-200}"
STOCKFISH_ELO="${STOCKFISH_ELO:-2200}"
STOCKFISH_MOVETIME_MS="${STOCKFISH_MOVETIME_MS:-30}"
TOTAL_AGENTS_SELFPLAY="${TOTAL_AGENTS_SELFPLAY:-8192}"
TOTAL_AGENTS_FINETUNE="${TOTAL_AGENTS_FINETUNE:-128}"
NUM_BUFFERS_SELFPLAY="${NUM_BUFFERS_SELFPLAY:-4}"
NUM_BUFFERS_FINETUNE="${NUM_BUFFERS_FINETUNE:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-puffer4}"
WANDB_GROUP="${WANDB_GROUP:-chess-stockfish}"

STOCKFISH_PATH="${PUFFER_STOCKFISH_PATH:-}"
if [[ -z "$STOCKFISH_PATH" ]]; then
  if [[ -x /usr/games/stockfish ]]; then
    STOCKFISH_PATH="/usr/games/stockfish"
  elif command -v stockfish >/dev/null 2>&1; then
    STOCKFISH_PATH="$(command -v stockfish)"
  else
    echo "ERROR: stockfish not found. Install it first (e.g. sudo apt install stockfish)"
    exit 1
  fi
fi
export PUFFER_STOCKFISH_PATH="$STOCKFISH_PATH"

echo "[1/5] Install/editable build with --no-build-isolation"
uv pip install --no-build-isolation -e .

echo "[2/5] Build chess backend"
python setup.py build_chess

find_latest_ckpt() {
  python - <<'PY'
import glob
import os
paths = glob.glob('experiments/puffer_chess/*/model_*.pt')
if not paths:
    raise SystemExit(1)
paths.sort(key=os.path.getmtime)
print(paths[-1])
PY
}

echo "[3/5] Self-play pretraining (${SELFPLAY_STEPS} steps) with wandb"
puffer train puffer_chess \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "$WANDB_GROUP" \
  --tag chess-selfplay-pretrain \
  --train.total-timesteps "$SELFPLAY_STEPS" \
  --vec.total-agents "$TOTAL_AGENTS_SELFPLAY" \
  --vec.num-buffers "$NUM_BUFFERS_SELFPLAY" \
  --train.cudagraphs -1 \
  --env.selfplay 1 \
  --env.random-bot 0 \
  --env.stockfish-bot 0

CKPT="$(find_latest_ckpt)"
echo "Latest checkpoint: $CKPT"

echo "[4/5] Stockfish fine-tuning (${FINETUNE_STEPS} steps) with wandb"
puffer train puffer_chess \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "$WANDB_GROUP" \
  --tag chess-stockfish-finetune \
  --load-model-path "$CKPT" \
  --train.total-timesteps "$FINETUNE_STEPS" \
  --vec.total-agents "$TOTAL_AGENTS_FINETUNE" \
  --vec.num-buffers "$NUM_BUFFERS_FINETUNE" \
  --train.cudagraphs -1 \
  --env.selfplay 1 \
  --env.random-bot 0 \
  --env.stockfish-bot 1 \
  --env.stockfish-limit-strength 1 \
  --env.stockfish-elo "$STOCKFISH_ELO" \
  --env.stockfish-movetime-ms "$STOCKFISH_MOVETIME_MS"

CKPT="$(find_latest_ckpt)"
echo "Latest checkpoint after fine-tune: $CKPT"

echo "[5/5] Stockfish gate eval (>=70% wins over ${GATE_GAMES} games)"
python tools/chess_stockfish_eval.py \
  --model-path "$CKPT" \
  --games "$GATE_GAMES" \
  --stockfish-elo "$STOCKFISH_ELO" \
  --stockfish-movetime-ms "$STOCKFISH_MOVETIME_MS" \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "$WANDB_GROUP" \
  --wandb-tag chess-stockfish-gate \
  --json-out "experiments/chess_stockfish_gate_summary.json"

echo "Pipeline complete"
