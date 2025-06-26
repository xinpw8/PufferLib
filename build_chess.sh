#!/usr/bin/env bash
# build_chess.sh — compile Abseil, console demo, and Raylib GUI for the PufferLib Chess demo.
# Usage:  ./build_chess.sh
# Run from the repository root (/puffertank/release_test_pufferlib).

set -euo pipefail

# Absolute path to the script itself, resolving symlinks
SCRIPT_FULL_PATH="$(readlink -f "$0")"

# Absolute path to the pufferlib directory (where this script resides)
PUFFERLIB_DIR="$(dirname "$SCRIPT_FULL_PATH")"

# Absolute path to the workspace root (one level up from pufferlib_dir)
WORKSPACE_ROOT="$(dirname "$PUFFERLIB_DIR")"

cd "$PUFFERLIB_DIR" # Change to the pufferlib directory

#------------------------------------------------------------------------------
# 1. Build Abseil (static, one-time). Skipped if libs already exist.
#------------------------------------------------------------------------------
if [ ! -d "${WORKSPACE_ROOT}/open_spiel_src/open_spiel/abseil-cpp/build" ] || ! ls "${WORKSPACE_ROOT}/open_spiel_src/open_spiel/abseil-cpp/build/libabsl_*.a" >/dev/null 2>&1; then
  echo "[build] Configuring & building Abseil …"
  pushd "${WORKSPACE_ROOT}/open_spiel_src/open_spiel/abseil-cpp" >/dev/null
  mkdir -p build && cd build
  cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        ..
  make -j$(nproc)
  echo "Abseil build directory: $(pwd)"
  popd >/dev/null
else
  echo "[skip] Abseil static libs already built."
fi

ABSL_LIBS="$(find "${WORKSPACE_ROOT}/open_spiel_src/open_spiel/abseil-cpp/build" -name 'libabsl_*.a' | tr '\n' ' ' )"

#------------------------------------------------------------------------------
# 2. Build console demo (demo_chess)
#------------------------------------------------------------------------------
echo "[build] Compiling demo_chess …"
pushd pufferlib/ocean/chess >/dev/null

clang++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I abseil-cpp \
    -I ${WORKSPACE_ROOT}/open_spiel_src \
    -I ${WORKSPACE_ROOT}/open_spiel_src/open_spiel \
    -I ${WORKSPACE_ROOT}/open_spiel_src/open_spiel/games/chess/ \
    demo_chess.cc \
    ${WORKSPACE_ROOT}/open_spiel_src/open_spiel/game_parameters.cc \
    ${WORKSPACE_ROOT}/open_spiel_src/open_spiel/spiel.cc \
    ${WORKSPACE_ROOT}/open_spiel_src/open_spiel/games/chess/chess.cc \
    ${ABSL_LIBS} \
    -lpthread -o ${WORKSPACE_ROOT}/demo_chess

popd >/dev/null
echo "[ok] Created ./demo_chess"

#------------------------------------------------------------------------------
# 3. Build Raylib GUI demo (raylib_chess)
#    Requires the pre-built Raylib bundle in raylib-5.5_linux_amd64/
#------------------------------------------------------------------------------
if [ -d "${WORKSPACE_ROOT}/raylib-5.5_linux_amd64" ]; then
  echo "[build] Compiling raylib_chess …"
  pushd pufferlib/ocean/chess >/dev/null
  clang++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
      -I abseil-cpp \
      -I ${WORKSPACE_ROOT}/open_spiel_src \
      -I ${WORKSPACE_ROOT}/open_spiel_src/open_spiel \
      -I ${WORKSPACE_ROOT}/raylib-5.5_linux_amd64/include \
      raylib_chess.cc \
      ${WORKSPACE_ROOT}/open_spiel_src/open_spiel/game_parameters.cc \
      ${WORKSPACE_ROOT}/open_spiel_src/open_spiel/spiel.cc \
      ${WORKSPACE_ROOT}/open_spiel_src/open_spiel/games/chess/chess.cc \
      ${ABSL_LIBS} \
      -L ${WORKSPACE_ROOT}/raylib-5.5_linux_amd64/lib -l:libraylib.a \
      -lm -lpthread -ldl -lrt -lX11 \
      -o ${WORKSPACE_ROOT}/raylib_chess
  popd >/dev/null
  echo "[ok] Created ./raylib_chess"
else
  echo "[skip] raylib-5.5_linux_amd64 not found — skipping GUI build."
fi

#------------------------------------------------------------------------------
# 4. Build SAN-replay validator (replay_chess)
#------------------------------------------------------------------------------
echo "[build] Compiling replay_chess …"
pushd pufferlib/ocean/chess >/dev/null
clang++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I abseil-cpp \
    -I ${WORKSPACE_ROOT}/open_spiel_src \
    -I ${WORKSPACE_ROOT}/open_spiel_src/open_spiel \
    -I ${WORKSPACE_ROOT}/open_spiel_src/open_spiel/games/chess/ \
    replay_chess.cc \
    ${WORKSPACE_ROOT}/open_spiel_src/open_spiel/game_parameters.cc \
    ${WORKSPACE_ROOT}/open_spiel_src/open_spiel/spiel.cc \
    ${WORKSPACE_ROOT}/open_spiel_src/open_spiel/games/chess/chess.cc \
    ${ABSL_LIBS} \
    -lpthread -o ${WORKSPACE_ROOT}/replay_chess
popd >/dev/null
echo "[ok] Created ./replay_chess"

echo "Done. Remember to set LD_LIBRARY_PATH=. before running the binaries if libchess.so is in the current directory." 