#!/usr/bin/env bash
# build_chess.sh — build the native flat-Chess demos & tools
# This script lives in pufferlib/scripts/, but it always executes from the
# repository root so all relative paths stay consistent.
#
# Usage (from anywhere):
#   ./pufferlib/scripts/build_chess.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repository root is one level above pufferlib/ -> scripts/ is two levels deep
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

###############################################################################
# 1. Build Abseil (static) — dependency for the OpenSpiel-derived engine
###############################################################################
if [ ! -d pufferlib/abseil-cpp/build ] || ! ls pufferlib/abseil-cpp/build/libabsl_*.a >/dev/null 2>&1; then
  echo "[build] Configuring & building Abseil …"
  pushd pufferlib/abseil-cpp >/dev/null
  mkdir -p build && cd build
  cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF ..
  make -j"$(nproc)"
  popd >/dev/null
else
  echo "[skip] Abseil already built."
fi

ABSL_LIBS="$(find pufferlib/abseil-cpp/build -name 'libabsl_*.a' | tr '\n' ' ')"

###############################################################################
# 2. Console demo (demo_chess)
###############################################################################

# Common flags
INC="-I pufferlib/abseil-cpp"
LIBS="${ABSL_LIBS} -lpthread"

echo "[build] Compiling demo_chess …"
g++ -std=c++17 -pthread \
    ${INC} \
    pufferlib/pufferlib/ocean/chess/demo_chess.cc \
    ${LIBS} -o ${REPO_ROOT}/pufferlib/demo_chess

echo "[ok] ./demo_chess"

###############################################################################
# 3. Raylib GUI demo (optional)
###############################################################################
if [ -d raylib-5.5_linux_amd64 ]; then
  echo "[build] Compiling raylib_chess …"
  g++ -std=c++17 -pthread \
      ${INC} \
      -I raylib-5.5_linux_amd64/include \
      pufferlib/pufferlib/ocean/chess/raylib_chess.cc \
      ${ABSL_LIBS} \
      -L raylib-5.5_linux_amd64/lib -l:libraylib.a \
      -lm -pthread -ldl -lrt -lX11 \
      -o ${REPO_ROOT}/pufferlib/raylib_chess
  echo "[ok] ./raylib_chess"
else
  echo "[skip] raylib bundle not found — skipping GUI build"
fi

###############################################################################
# 4. SAN-replay validator (replay_chess)
###############################################################################

echo "[build] Compiling replay_chess …"
g++ -std=c++17 -pthread \
    ${INC} \
    pufferlib/pufferlib/ocean/chess/replay_chess.cc \
    ${LIBS} -o ${REPO_ROOT}/pufferlib/replay_chess

echo "[ok] ./replay_chess"

echo "Done.  Remember to export LD_LIBRARY_PATH=. if you need to locate shared libs at runtime." 