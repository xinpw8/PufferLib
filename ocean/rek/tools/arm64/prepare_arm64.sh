#!/usr/bin/env bash
# Stage everything build.sh needs on aarch64, without editing build.sh.
#
# Run once from the repo root, then build normally:
#
#     ./ocean/rek/tools/arm64/prepare_arm64.sh
#     eval "$(./ocean/rek/tools/arm64/prepare_arm64.sh --export)"
#     ./build.sh rek
#
# Two things stand in the way on an aarch64 host, and both are fixed from here
# rather than in the library:
#
#   1. raylib publishes no arm64 Linux binary for 5.5, and build.sh downloads
#      the amd64 tarball by name. Its download() returns early when the target
#      directory already exists, so building raylib from source into that name
#      first means the download never happens and every downstream -I/-L path
#      keeps working untouched. The real build lands in an honestly named
#      directory and the name build.sh wants is a symlink to it.
#   2. -mavx2 -mfma are hard errors here. tools/arm64/puffer-cc strips them and
#      build.sh picks it up through ${CC}. See that script for why dropping them
#      is safe for this env.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
SHIM="$HERE/puffer-cc"

MACHINE="$(uname -m)"
REAL_NAME='raylib-5.5_linux_arm64'
# The name build.sh computes for Linux. Matching it is the whole trick.
EXPECTED_NAME='raylib-5.5_linux_amd64'
RAYLIB_VER=5.5

export_only=0
[ "${1:-}" = "--export" ] && export_only=1

# The one flag worth adding back. Probed rather than assumed: clang only knows
# -mcpu=native for a CPU it recognises, and an unknown one is another hard error.
probe_mcpu() {
    if echo 'int main(void){return 0;}' \
        | "${REK_REAL_CC:-clang}" -mcpu=native -x c - -o /dev/null 2>/dev/null; then
        echo '-mcpu=native'
    fi
}

if [ "$export_only" = 1 ]; then
    echo "export CC='$SHIM'"
    extra="$(probe_mcpu)"
    [ -n "$extra" ] && echo "export REK_CC_EXTRA='$extra'"
    exit 0
fi

if [ "$MACHINE" != "aarch64" ] && [ "$MACHINE" != "arm64" ]; then
    echo "This host is $MACHINE, not aarch64 — build.sh needs no help here."
    exit 0
fi

cd "$ROOT"

if [ -e "$EXPECTED_NAME" ]; then
    echo "raylib already staged at $EXPECTED_NAME — nothing to do."
else
    if [ ! -d "$REAL_NAME" ]; then
        echo "Building raylib $RAYLIB_VER from source for $MACHINE (no upstream arm64 binary)..."
        src="raylib-$RAYLIB_VER-src"
        if [ ! -d "$src" ]; then
            curl -sL "https://github.com/raysan5/raylib/archive/refs/tags/$RAYLIB_VER.tar.gz" \
                -o raylib-src.tar.gz || { echo "Error: failed to download raylib source"; exit 1; }
            tar xf raylib-src.tar.gz && mv "raylib-$RAYLIB_VER" "$src" && rm raylib-src.tar.gz
        fi
        make -C "$src/src" PLATFORM=PLATFORM_DESKTOP -j"$(nproc)" || {
            echo "Error: raylib build failed. Install the GL/X11 headers it needs:"
            echo "  sudo apt install libgl1-mesa-dev libx11-dev libxrandr-dev \\"
            echo "      libxinerama-dev libxcursor-dev libxi-dev"
            exit 1
        }
        mkdir -p "$REAL_NAME/lib" "$REAL_NAME/include"
        cp "$src/src/libraylib.a" "$REAL_NAME/lib/"
        cp "$src/src/raylib.h" "$src/src/raymath.h" "$src/src/rlgl.h" "$REAL_NAME/include/"
    fi
    # build.sh looks for the amd64 name and skips its download if it is there.
    # The contents are arm64; the name is what makes the untouched script work.
    ln -s "$REAL_NAME" "$EXPECTED_NAME"
    echo "Staged $REAL_NAME as $EXPECTED_NAME (build.sh will skip its download)."
fi

extra="$(probe_mcpu)"
echo
echo "Ready. Build with:"
echo "  export CC='$SHIM'"
[ -n "$extra" ] && echo "  export REK_CC_EXTRA='$extra'"
echo "  ./build.sh rek"
echo
echo "Or in one line:  eval \"\$($0 --export)\" && ./build.sh rek"
