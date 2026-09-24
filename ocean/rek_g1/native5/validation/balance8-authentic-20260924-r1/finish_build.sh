#!/usr/bin/env bash
set -euo pipefail
stage=/home/spark-advantage/rek-training/balance8-authentic-20260924-r1
src=$stage/source/ocean/rek_g1/native5
out=$stage/build
obj=/home/spark-advantage/rek-training/semantic-fast-20260914-v1/build-v4/native_policy.o
g++ -std=c++17 -O2 -DREK_LIVE_PROTOCOL_TEST -x c++ "$src/live_policy_worker.cu" -x none "$out/cJSON.o" -o "$out/live-policy-protocol-test"
CUDA_VISIBLE_DEVICES= REK_OBSERVATION_SCHEMA=rek.native5.scaled_polar_xy.balance8_v1 node "$src/live_policy_worker.test.cjs" protocol "$out/live-policy-protocol-test" > "$out/worker-protocol-tests.json"
/usr/local/cuda/bin/nvcc -std=c++17 -O3 --threads 1 -arch=sm_121 -I"$src" "$src/live_policy_worker.cu" "$obj" "$out/cJSON.o" -lcublas -lcurand -lcrypto -o "$out/live-policy-worker-balance8"
g++ -std=c++17 -O2 "$src/test_balance8_data.cpp" -lcrypto -o "$out/test-balance8-data"
"$out/test-balance8-data" "$stage/export/authentic-balance8-v4.bin" /home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1/export/authentic-trajectories-v3.bin | tee "$out/test-data.json"
"$out/test-authentic-v3" /home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1/export/authentic-trajectories-v3.bin
CUDA_VISIBLE_DEVICES= node "$stage/test_encoder.cjs" | tee "$out/encoder-edge-tests.json"
sha256sum "$out/encode-balance8" "$out/balance8-migration" "$out/replay-balance8" "$out/authentic-ppo-balance8" "$out/live-policy-worker-balance8" "$obj" > "$out/binary-hashes.sha256"
