#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 ]] || { printf 'Usage: %s FRESH_BUILD_DIRECTORY\n' "$0" >&2; exit 2; }
task_clone=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_output=$(realpath -m "$1")
exec env REK_CLONE_SOURCE_DIR="$task_clone/source" REK_CLONE_BUILD_DIR="$task_output" \
    bash "$task_clone/tools/build-native.sh"
