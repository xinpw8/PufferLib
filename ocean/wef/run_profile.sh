#!/usr/bin/env bash
# Build and run wef env profile.  --gprof builds -pg and dumps a flat profile.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SRC=ocean/wef/profile.c
OUT=./wef_profile
OUT_GPROF=./wef_profile_gprof
REPORT=wef_gprof_report.txt

INCS=(-I./raylib-5.5_linux_amd64/include -I./src -I./vendor -I./ocean/wef)
# pufferenv.h needs AVX2; raylib for link even though profile is headless.
LIBS=(raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt)
CC_FAST="${CC:-clang}"
CC_PROF="${CC_PROF:-gcc}"
SIMD=(-mavx2 -mfma)

GPROF=0
ARGS=()
for a in "$@"; do
    [[ "$a" == "--gprof" ]] && GPROF=1 || ARGS+=("$a")
done

if [[ $GPROF -eq 1 ]]; then
    echo "Building $OUT_GPROF (-pg, -O1)..."
    "$CC_PROF" -pg -O1 -g -DNDEBUG "${SIMD[@]}" \
        "${INCS[@]}" "$SRC" -o "$OUT_GPROF" "${LIBS[@]}" -DPLATFORM_DESKTOP

    rm -f gmon.out
    # 8192 fish, outer-t / inner-env loop (see profile.c).
    "$OUT_GPROF" \
        --total-fish "${PROFILE_TOTAL_FISH:-8192}" \
        --num-fish "${PROFILE_NUM_FISH:-4}" \
        --steps "${PROFILE_STEPS:-32}" \
        --warmup "${PROFILE_WARMUP:-1}" \
        "${ARGS[@]}"

    [[ -f gmon.out ]] || { echo "error: gmon.out missing" >&2; exit 1; }
    gprof -b -p "$OUT_GPROF" gmon.out > "$REPORT"
    echo
    echo "=== gprof flat profile (top) ==="
    head -n 60 "$REPORT"
    echo
    echo "Full flat profile: $REPORT"
else
    echo "Building $OUT (-O3)..."
    "$CC_FAST" -O3 -DNDEBUG "${SIMD[@]}" \
        "${INCS[@]}" "$SRC" -o "$OUT" "${LIBS[@]}" -DPLATFORM_DESKTOP
    echo
    "$OUT" \
        --total-fish "${PROFILE_TOTAL_FISH:-8192}" \
        --num-fish "${PROFILE_NUM_FISH:-4}" \
        --steps "${PROFILE_STEPS:-64}" \
        --warmup "${PROFILE_WARMUP:-2}" \
        "${ARGS[@]}"
fi
