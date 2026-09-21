#!/usr/bin/env bash
# Bounded offline conversion and checks. No GPU, training, or game connection.
set -euo pipefail
umask 077
[[ $# == 5 ]] || { printf 'Usage: %s BUILD ORIGINAL_DATASET TRAIN_RAW HELDOUT_RAW NEW_RUN_DIRECTORY\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$(realpath "$1")
task_original=$(realpath "$2")
task_train=$(realpath "$3")
task_heldout=$(realpath "$4")
mkdir "$5"
task_run=$(realpath "$5")
task_output=$task_run/dataset
task_command=("$task_build/human_observable_data" "$task_original" "$task_train" "$task_heldout" "$task_output")
printf '%q ' "${task_command[@]}" > "$task_run/command.txt"
printf '\n' >> "$task_run/command.txt"
cp "$task_build/build-hashes.txt" "$task_run/build-hashes.txt"
cp "$0" "$task_run/run-script.sh"
timeout 60s "${task_command[@]}" > "$task_run/adapter.stdout.json" 2> "$task_run/adapter.stderr.txt"
node "$task_source/human_observable_data.test.cjs" "$task_original" "$task_output" "$task_train" "$task_heldout" \
    > "$task_run/integration-tests.json" 2> "$task_run/integration-tests.stderr.txt"
# Failure cases cannot publish or modify a dataset.
task_before=$(sha256sum "$task_output/human-observable.bin")
if "$task_build/human_observable_data" "$task_original" "$task_train" "$task_heldout" "$task_output" \
    > "$task_run/reject-existing.stdout.txt" 2> "$task_run/reject-existing.stderr.txt"; then exit 2; fi
grep -q 'output directory already exists' "$task_run/reject-existing.stderr.txt"
[[ $(sha256sum "$task_output/human-observable.bin") == "$task_before" ]]
if "$task_build/human_observable_data" "$task_original" "$task_heldout" "$task_train" "$task_run/rejected-swapped-inputs" \
    > "$task_run/reject-hash.stdout.txt" 2> "$task_run/reject-hash.stderr.txt"; then exit 2; fi
grep -q 'pinned input SHA256 mismatch' "$task_run/reject-hash.stderr.txt"
[[ ! -e "$task_run/rejected-swapped-inputs" ]]
printf '{"existing_output_refused_unchanged":true,"swapped_raw_hashes_refused_before_output":true}\n' > "$task_run/cli-tests.json"
sha256sum "$task_output/"* "$task_run/"{adapter.stdout.json,integration-tests.json,cli-tests.json,run-script.sh} > "$task_run/output-hashes.txt"
printf '0\n' > "$task_run/exit-code.txt"
printf 'Offline conversion and tests passed: %s\n' "$task_run"
