#!/bin/bash
# Drive the rek env on the DGX Spark from WSL, over an `ssh spark` alias.
#
#   ./ocean/rek/tools/spark_train.sh --dry-run   # preflight + sync + build + verify
#   ./ocean/rek/tools/spark_train.sh             # ...then start training
#
# The Spark's state is not assumed. Preflight checks each prerequisite and says
# what is missing rather than failing several steps later inside nvcc, because a
# half-configured CUDA install produces errors that look nothing like their
# cause.
set -uo pipefail

HOST="${SPARK_HOST:-spark}"
BRANCH="${REK_BRANCH:-rek}"
REPO="${REK_REPO:-https://github.com/xinpw8/PufferLib}"
REMOTE_DIR="${REK_REMOTE_DIR:-~/PufferLib}"
DRY_RUN=0
SKIP_BUILD=0
TRAIN_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)    DRY_RUN=1 ;;
        --skip-build) SKIP_BUILD=1 ;;
        --host)       HOST="$2"; shift ;;
        --branch)     BRANCH="$2"; shift ;;
        -h|--help)
            sed -n '2,10p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)            TRAIN_ARGS+=("$1") ;;
    esac
    shift
done

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mxx %s\033[0m\n' "$*"; exit 1; }

# Run a command on the Spark. Quoted heredoc so nothing expands locally.
on_spark() { ssh "$HOST" "bash -lc $(printf '%q' "$1")"; }

command -v ssh >/dev/null || die "no ssh client on this machine"

say "Reaching $HOST"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" true 2>/dev/null \
    || die "cannot ssh to '$HOST' non-interactively.
   Check: ssh $HOST true
   If it needs a passphrase, start an agent first: eval \$(ssh-agent) && ssh-add"
echo "ok"

# ---------------------------------------------------------------- preflight
say "Preflight"
PREFLIGHT=$(on_spark '
    printf "arch\t%s\n" "$(uname -m)"
    printf "kernel\t%s\n" "$(uname -r)"
    printf "cores\t%s\n" "$(nproc)"
    if command -v nvcc >/dev/null; then
        printf "nvcc\t%s\n" "$(nvcc --version | sed -n "s/.*release \([0-9.]*\).*/\1/p")"
    else printf "nvcc\tMISSING\n"; fi
    if command -v nvidia-smi >/dev/null; then
        printf "gpu\t%s\n" "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
        printf "cc\t%s\n" "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
    else printf "gpu\tMISSING\n"; printf "cc\t?\n"; fi
    printf "python\t%s\n" "$(command -v python3 || echo MISSING)"
    printf "torch\t%s\n" "$(python3 -c "import torch;print(torch.__version__)" 2>/dev/null || echo MISSING)"
    printf "cuda_ok\t%s\n" "$(python3 -c "import torch;print(torch.cuda.is_available())" 2>/dev/null || echo MISSING)"
    printf "repo\t%s\n" "$([ -d '"$REMOTE_DIR"'/.git ] && echo present || echo absent)"
    printf "raylib_deps\t%s\n" "$([ -f /usr/include/GL/gl.h ] && echo ok || echo MISSING)"
    printf "openmp\t%s\n" "$(ls /usr/lib/*/libomp.so /usr/lib/llvm-*/lib/libomp.so 2>/dev/null | head -1 || echo MISSING)"
')
[ -n "$PREFLIGHT" ] || die "preflight produced no output — is bash available on $HOST?"
echo "$PREFLIGHT" | while IFS=$'\t' read -r k v; do printf '  %-12s %s\n' "$k" "$v"; done

get() { echo "$PREFLIGHT" | awk -F'\t' -v k="$1" '$1==k{print $2}'; }

[ "$(get arch)" = "aarch64" ] || warn "arch is $(get arch), not aarch64 — prepare_arm64.sh will no-op and build.sh runs as upstream wrote it"
[ "$(get nvcc)" = "MISSING" ] && die "nvcc not found on $HOST. Install the CUDA toolkit, or add it to PATH in ~/.bashrc (login shell is what this script uses)."
[ "$(get torch)" = "MISSING" ] && die "torch not importable on $HOST. PufferLib needs torch>=2.9 built for aarch64+CUDA."
[ "$(get cuda_ok)" != "True" ] && warn "torch.cuda.is_available() is $(get cuda_ok) — training needs a working CUDA runtime"
# Exact set that took ./build.sh rek --fast from failing to building; libomp is
# easy to miss because the failure is a bare "cannot find -lomp" at link time,
# long after the compile looked fine.
[ "$(get raylib_deps)" = "MISSING" ] && warn "GL/X11 headers missing; the raylib build will need:
    sudo apt install libgl1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev"
[ "$(get openmp)" = "MISSING" ] && warn "libomp missing; the link step will fail with 'cannot find -lomp':
    sudo apt install libomp-dev"

# ---------------------------------------------------------------- sync
say "Syncing $BRANCH"
on_spark "
    set -e
    if [ ! -d $REMOTE_DIR/.git ]; then
        echo 'cloning...'
        git clone $REPO $REMOTE_DIR
    fi
    cd $REMOTE_DIR
    git remote set-url origin $REPO 2>/dev/null || git remote add origin $REPO
    git fetch origin $BRANCH
    git checkout -B $BRANCH origin/$BRANCH
    git log --oneline -1
" || die "sync failed"

# ---------------------------------------------------------------- build
if [ "$SKIP_BUILD" = "0" ]; then
    say "Building (first real exercise of the aarch64 path)"
    # PufferLib's build.sh is left exactly as upstream ships it. Everything
    # aarch64 needs is staged around it from ocean/rek/tools/arm64: raylib built
    # from source under the directory name build.sh looks for, and a CC shim
    # that drops the x86-only -mavx2/-mfma it compiles with unconditionally.
    on_spark "
        set -e
        cd $REMOTE_DIR
        echo 'NVCC_ARCH resolves to:'
        nvcc --version | tail -1
        ./ocean/rek/tools/arm64/prepare_arm64.sh
        eval \"\$(./ocean/rek/tools/arm64/prepare_arm64.sh --export)\"
        echo \"building with CC=\$CC \${REK_CC_EXTRA:-}\"
        ./build.sh rek
    " || die "build failed.
   If it died in raylib, install the GL/X11 headers listed above.
   If it died in nvcc over the GPU arch, pin it: NVCC_ARCH=sm_121 ./build.sh rek"
fi

# ---------------------------------------------------------------- verify
say "Verifying rules and step rate (headless, no GPU or display needed)"
on_spark "
    set -e
    cd $REMOTE_DIR
    cc -O3 -Wall -I./src ocean/rek/test_rek.c -lm -o /tmp/test_rek
    /tmp/test_rek --bench
    echo
    echo \"cores available: \$(nproc)\"
" || die "verification failed — the rules did not pass on aarch64"

if [ "$DRY_RUN" = "1" ]; then
    say "Dry run complete — stopping before training"
    echo "Run without --dry-run to start:  puffer train rek"
    exit 0
fi

# ---------------------------------------------------------------- train
say "Training"
echo "Ctrl-C detaches this shell; the run keeps going under its own tmux session."
ssh -t "$HOST" "bash -lc $(printf '%q' "
    cd $REMOTE_DIR
    if command -v tmux >/dev/null; then
        tmux new-session -A -s rek \"puffer train rek ${TRAIN_ARGS[*]:-}\"
    else
        echo 'tmux not installed — running in the foreground, closing this shell kills it'
        puffer train rek ${TRAIN_ARGS[*]:-}
    fi
")"
