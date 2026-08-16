#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

mkdir -p build/bat-tests
cc -std=c99 -O2 -Wall -Wextra -ffunction-sections -fdata-sections \
    -I. -Isrc -Iocean/bat -Ivendor -Iraylib-5.5_linux_amd64/include \
    ocean/bat/tests/test_bat_core.c \
    -Wl,--gc-sections -lm \
    -o build/bat-tests/test_bat_core

build/bat-tests/test_bat_core
