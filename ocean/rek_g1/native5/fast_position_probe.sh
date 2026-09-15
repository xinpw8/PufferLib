#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 || $# == 4 ]] || { printf 'Usage: %s EVAL_BUILD BASE_CONFIG NEW_OUTPUT [all|paired_i]\n' "$0" >&2;exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$(realpath "$1")
task_config=$(realpath "$2")
task_output=$3
[[ ! -e "$task_output" ]] || { printf 'Output already exists\n' >&2;exit 2; }
task_command=(node "$task_source/fast_position_probe.cjs" "$task_build/rek-eval-worker" "$task_config" "$task_output" "${4:-all}")
task_log=$(mktemp)
task_error=$(mktemp)
set +e
timeout --signal=TERM --kill-after=5s 90s "${task_command[@]}" > "$task_log" 2> "$task_error"
task_status=$?
set -e
if [[ -d "$task_output" ]]; then
  cp "$task_log" "$task_output/stdout.txt"
  cp "$task_error" "$task_output/stderr.txt"
  { if [[ -v REK_EXPECT_NO_HIT_RESETS ]]; then printf 'REK_EXPECT_NO_HIT_RESETS=%q ' "$REK_EXPECT_NO_HIT_RESETS";fi
    printf '%q ' "${task_command[@]}";printf '\n';hostname;id;date -u --iso-8601=seconds;
    sha256sum "$task_source/fast_position_probe.cjs" "$task_source/fast_position_probe.sh" "$task_config" "$task_build/rek-eval-worker" "$task_build/fast_runtime.o" "$task_build/fast_assets.o";
    printf 'exit_code=%s\n' "$task_status"; } > "$task_output/provenance.txt"
fi
cat "$task_log" "$task_error"
rm -- "$task_log" "$task_error"
exit "$task_status"
