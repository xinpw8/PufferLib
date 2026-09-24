#!/usr/bin/env bash
set -euo pipefail
stage=/home/spark-advantage/rek-training/balance8-onpolicy-20260924-r2
src=$stage/source
out=$stage/build-score-delta
prior=/home/spark-advantage/rek-training/balance8-authentic-20260924-r1/source/ocean/rek_g1/native5
prepared=/home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1/distributional-candidate/ppo-build/source
cuda=/usr/local/cuda
nccl=/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl
obj=/home/spark-advantage/rek-training/semantic-fast-20260914-v1/build-v4/native_policy.o
[[ ! -e "$out" ]] || exit 2
mkdir "$out"
node --test "$src/authentic_trajectory_data.test.cjs" "$src/authentic_trajectory_v3.test.cjs" > "$out/js-tests.txt"
node --test "$stage/test_score_delta.cjs" > "$out/score-delta-js-tests.txt"
g++ -std=c++17 -O2 "$src/test_authentic_v3.cpp" -lcrypto -o "$out/test-authentic-v3"
"$out/test-authentic-v3" "$stage/export/authentic-trajectories-v3.bin" > "$out/data-cpu-test.txt"
g++ -std=c++17 -O2 "$src/test_score_delta.cpp" -lcrypto -o "$out/test-score-delta"
"$out/test-score-delta" "$stage/export/authentic-trajectories-v3.bin" "$stage/score-delta-5s/authentic-score-delta-v5.bin" > "$out/score-delta-cpu-test.json"
object_hash=$(sha256sum "$obj" | awk '{print $1}')
[[ $object_hash == 4ada3de760b5a00f7bb3d6592cd2da4ca48a4a196e774d40e6d220a57d1f574c ]] || exit 2
"$cuda/bin/nvcc" -std=c++17 -O3 --threads 1 -arch=sm_121 -I"$src" -I"$prior" \
 "-DREK_AUTHENTIC_NATIVE_OBJECT_SHA256=\"$object_hash\"" "$src/replay_authentic_behavior.cu" "$obj" \
 -lcublas -lcurand -lcrypto -o "$out/replay-authentic-behavior"
"$cuda/bin/nvcc" -std=c++17 -O2 --threads 1 -arch=sm_121 -I"$prepared" -I"$nccl/include" -I"$cuda/include/cccl" \
 "$src/authentic_ppo.cu" -lcublas -lcurand -lcrypto -o "$out/authentic-ppo"
sha256sum "$out/replay-authentic-behavior" "$out/authentic-ppo" "$obj" > "$out/binary-hashes.sha256"
printf 'Compilation and CPU tests only. No GPU execution.\n'
