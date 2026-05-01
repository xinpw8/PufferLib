#!/bin/bash
# eval.sh — eval the trained pufferlib-4.0 mancala policy on the latest
# checkpoint with a single command. Lives in ocean/mancala/ in the local
# clone; deployed copy on spark is /home/spark-advantage/pufferlib-test/PufferLib/eval_mancala.sh.
#
# Usage:
#   ./eval_mancala.sh                  # 1000-episode silent eval, print stats
#   ./eval_mancala.sh watch            # render 1 verbose episode (every move)
#   ./eval_mancala.sh watch 3          # render 3 verbose episodes
#   ./eval_mancala.sh 100              # N-episode silent eval, print stats
#   ./eval_mancala.sh chain            # hand-coded 17-extra-turn demo
#   ./eval_mancala.sh chainplay        # trained policy from [6,5,4,3,2,1]
#   ./eval_mancala.sh -c PATH ...      # use an explicit checkpoint .bin
#
# Environment overrides (optional):
#   PUFFERLIB_DIR=/path/to/PufferLib
#   SKIP_BUILD=1                       # skip rebuild check

set -euo pipefail

PUFFERLIB_DIR="${PUFFERLIB_DIR:-/home/spark-advantage/pufferlib-test/PufferLib}"
cd "$PUFFERLIB_DIR"

# Optional explicit checkpoint via -c PATH; otherwise pick newest .bin under
# checkpoints/mancala/.
CKPT=""
if [[ "${1:-}" == "-c" && -n "${2:-}" ]]; then
    CKPT="$2"; shift 2
fi
if [[ -z "$CKPT" ]]; then
    CKPT=$(ls -t checkpoints/mancala/*/*.bin 2>/dev/null | head -1 || true)
    if [[ -z "$CKPT" ]]; then
        echo "[eval.sh] no checkpoint found under checkpoints/mancala/. Train first."
        exit 1
    fi
fi
echo "[eval.sh] checkpoint: $CKPT"

# Rebuild standalone if env source is newer (matches train.sh logic).
if [[ -z "${SKIP_BUILD:-}" ]]; then
    BIN=./mancala
    if [[ ! -x "$BIN" ]] || find ocean/mancala -newer "$BIN" 2>/dev/null | grep -q .; then
        echo "[eval.sh] rebuilding standalone (source newer than $BIN)"
        ./build.sh mancala --fast >/dev/null 2>&1 || ./build.sh mancala --fast
    fi
fi

MODE="${1:-stats}"
case "$MODE" in
    watch)
        N="${2:-1}"
        exec ./mancala play "$CKPT" "$N" verbose
        ;;
    chain)
        exec ./mancala chain
        ;;
    chainplay)
        exec ./mancala chainplay "$CKPT"
        ;;
    ''|stats|''|[0-9]*)
        N="${MODE:-1000}"
        [[ "$MODE" == "stats" || -z "$MODE" ]] && N=1000
        exec ./mancala play "$CKPT" "$N" silent
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "See top of $0 for usage."
        exit 1
        ;;
esac
