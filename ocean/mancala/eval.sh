#!/bin/bash
# eval.sh — find the latest mancala checkpoint and play against it
# interactively. Lives in ocean/mancala/ in the local clone; deployed copy
# on the training host is eval_mancala.sh.
#
# Usage:
#   ./eval.sh                 # auto-find newest checkpoint, play
#   ./eval.sh -c PATH         # use explicit checkpoint .bin
#
# Environment overrides:
#   PUFFERLIB_DIR=/path       # default /home/spark-advantage/pufferlib-test/PufferLib
#   SKIP_BUILD=1              # skip the auto-rebuild check

set -euo pipefail

PUFFERLIB_DIR="${PUFFERLIB_DIR:-/home/spark-advantage/pufferlib-test/PufferLib}"
cd "$PUFFERLIB_DIR"

CKPT=""
if [[ "${1:-}" == "-c" && -n "${2:-}" ]]; then
    CKPT="$2"
fi
if [[ -z "$CKPT" ]]; then
    CKPT=$(ls -t checkpoints/mancala/*/*.bin 2>/dev/null | head -1 || true)
    if [[ -z "$CKPT" ]]; then
        echo "[eval.sh] no checkpoint under checkpoints/mancala/. Train first."
        exit 1
    fi
fi
echo "[eval.sh] checkpoint: $CKPT"

if [[ -z "${SKIP_BUILD:-}" ]]; then
    if [[ ! -x ./mancala ]] || find ocean/mancala -newer ./mancala 2>/dev/null | grep -q .; then
        echo "[eval.sh] rebuilding (source newer than ./mancala)"
        ./build.sh mancala --fast >/dev/null
    fi
fi

exec ./mancala human "$CKPT"
