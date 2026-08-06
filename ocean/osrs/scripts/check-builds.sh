#!/usr/bin/env bash
# Neither frontend alone reproduces nvcc: g++ takes _Static_assert as an extension,
# clang++ takes out-of-order designated initializers. Run both.
# --cuda runs the real nvcc build for every osrs env, and only works on the box.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

if [ "${1:-}" = "--cuda" ]; then
    status=0
    for dir in ocean/osrs_*/; do
        env="$(basename "${dir}")"
        [ -f "${dir}/${env}.h" ] || { echo "SKIP ${env} (no ${env}.h wrapper)"; continue; }
        log="$(mktemp)"
        if ./build.sh "${env}" >"${log}" 2>&1; then
            echo "OK   ${env}"
        else
            echo "FAIL ${env}"
            grep -m5 -E 'error:|error #' "${log}" || tail -5 "${log}"
            status=1
        fi
        rm -f "${log}"
    done
    exit ${status}
fi

GXX="${GXX:-g++-16}"
CXX_CLANG="${CXX_CLANG:-clang++}"
TU="$(mktemp -d)"
trap 'rm -rf "${TU}"' EXIT
status=0

for enc in colosseum inferno zulrah nh_pvp; do
    src="${TU}/${enc}.cc"
    echo "#include \"ocean/osrs/encounters/encounter_${enc}.h\"" >"${src}"
    out="$("${GXX}" -fsyntax-only -std=c++17 -I. "${src}" 2>&1)"; rc=$?
    out="${out}$("${CXX_CLANG}" -fsyntax-only -std=c++17 -Werror=c11-extensions -ferror-limit=0 -I. "${src}" 2>&1)"
    rc=$((rc + $?))
    if [ ${rc} -eq 0 ]; then
        echo "OK   encounter_${enc}.h"
    else
        echo "FAIL encounter_${enc}.h"
        printf '%s\n' "${out}"
        status=1
    fi
done

exit ${status}
