#!/usr/bin/env bash
# build_chess.sh — compile Abseil, console demo, and Raylib GUI for the PufferLib Chess demo.
# Usage:  ./build_chess.sh
# Run from the repository root (/puffertank/release_test_pufferlib).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

#------------------------------------------------------------------------------
# 1. Build Abseil (static, one-time). Skipped if libs already exist.
#------------------------------------------------------------------------------
if [ ! -d pufferlib/abseil-cpp/build ] || ! ls pufferlib/abseil-cpp/build/libabsl_*.a >/dev/null 2>&1; then
  echo "[build] Configuring & building Abseil …"
  pushd pufferlib/abseil-cpp >/dev/null
  mkdir -p build && cd build
  cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        ..
  make -j$(nproc)
  popd >/dev/null
else
  echo "[skip] Abseil static libs already built."
fi

ABSL_LIBS="$(find pufferlib/abseil-cpp/build -name 'libabsl_*.a' | tr '\n' ' ')"

#------------------------------------------------------------------------------
# 2. Build console demo (demo_chess)
#------------------------------------------------------------------------------
echo "[build] Compiling demo_chess …"

g++ -std=c++17 \
    -I pufferlib/abseil-cpp \
    demo_chess.cc \
    ${ABSL_LIBS} \
    -lpthread -o demo_chess

echo "[ok] Created ./demo_chess"

#------------------------------------------------------------------------------
# 3. Build Raylib GUI demo (raylib_chess) — requires bundled Raylib binaries
#------------------------------------------------------------------------------
if [ -d raylib-5.5_linux_amd64 ]; then
  echo "[build] Compiling raylib_chess …"
  g++ -std=c++17 \
      -I pufferlib/abseil-cpp \
      -I raylib-5.5_linux_amd64/include \
      raylib_chess.cc \
      ${ABSL_LIBS} \
      -L raylib-5.5_linux_amd64/lib -l:libraylib.a \
      -lm -lpthread -ldl -lrt -lX11 \
      -o raylib_chess
  echo "[ok] Created ./raylib_chess"
else
  echo "[skip] raylib-5.5_linux_amd64 not found — skipping GUI build."
fi

echo "Done.  Remember to set LD_LIBRARY_PATH=. before running the binaries if libchess.so is in the current directory." 