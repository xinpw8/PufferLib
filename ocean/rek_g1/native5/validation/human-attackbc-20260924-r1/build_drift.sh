#!/usr/bin/env bash
set -euo pipefail
stage=/home/spark-advantage/rek-training/scorecredit-human-attackbc-20260924-r1
prior=/home/spark-advantage/rek-training/balance8-authentic-20260924-r1/source
obj=/home/spark-advantage/rek-training/semantic-fast-20260914-v1/build-v4/native_policy.o
[[ ! -e "$stage/build" ]] || exit 2
[[ $(sha256sum "$obj" | awk '{print $1}') == 4ada3de760b5a00f7bb3d6592cd2da4ca48a4a196e774d40e6d220a57d1f574c ]] || exit 2
[[ $(sha256sum "$prior/ocean/rek_g1/native5/live_policy_worker.cu" | awk '{print $1}') == 18e7d633ce0f595f873bbdc5542a69c0461f1d8a8b361733fce852f5d97e926c ]] || exit 2
mkdir "$stage/build"
/usr/local/cuda/bin/nvcc -std=c++17 -O2 --threads 1 -arch=sm_121 -I"$prior/ocean/rek_g1/native5" \
 "$stage/diagnose_live_drift.cu" "$prior/vendor/cJSON.c" "$obj" -lcublas -lcurand -lcrypto -o "$stage/build/diagnose-live-drift"
sha256sum "$stage/build/diagnose-live-drift" "$stage/diagnose_live_drift.cu" "$prior/ocean/rek_g1/native5/live_policy_worker.cu" "$obj"
printf 'Compile only, no GPU execution.\n'
