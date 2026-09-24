#!/usr/bin/env bash
set -euo pipefail
stage=/home/spark-advantage/rek-training/balance8-authentic-20260924-r1
src=$stage/source/ocean/rek_g1/native5
out=$stage/build
prepared=/home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1/distributional-candidate/ppo-build/source
mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
cuda=/usr/local/cuda
nccl=/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl
obj=/home/spark-advantage/rek-training/semantic-fast-20260914-v1/build-v4/native_policy.o
mkdir -p "$out"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation "$src/test_balance8.cpp" -o "$out/test-balance8"
CUDA_VISIBLE_DEVICES= "$out/test-balance8" | tee "$out/test-balance8.json"
gcc -O2 -c "$stage/source/vendor/cJSON.c" -o "$out/cJSON.o"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation -Wno-unused-function -I"$mujoco/include" \
 "$src/live_transfer/encode_balance8.cpp" "$src/live_transfer/balance8_project_api.cpp" "$out/cJSON.o" \
 -L"$mujoco" -Wl,-rpath,"$mujoco" -l:libmujoco.so.3.7.0 -lcrypto -o "$out/encode-balance8"
"$cuda/bin/nvcc" -std=c++17 -O2 --threads 1 -arch=sm_121 -I"$prepared" -I"$nccl/include" -I"$cuda/include/cccl" -I"$stage/source/vendor" \
 "$src/balance8_migration.cu" "$out/cJSON.o" -lcublas -lcurand -lcrypto -o "$out/balance8-migration"
CUDA_VISIBLE_DEVICES= "$out/balance8-migration" --cpu-self-test | tee "$out/test-migration.json"
object_hash=$(sha256sum "$obj" | awk '{print $1}')
"$cuda/bin/nvcc" -std=c++17 -O3 --threads 1 -arch=sm_121 -I"$src" \
 "-DREK_AUTHENTIC_NATIVE_OBJECT_SHA256=\"$object_hash\"" "$src/replay_balance8.cu" "$obj" \
 -lcublas -lcurand -lcrypto -o "$out/replay-balance8"
"$cuda/bin/nvcc" -std=c++17 -O2 --threads 1 -arch=sm_121 -I"$prepared" -I"$nccl/include" -I"$cuda/include/cccl" \
 "$src/authentic_ppo.cu" -lcublas -lcurand -lcrypto -o "$out/authentic-ppo-balance8"
g++ -std=c++17 -O2 "$src/test_authentic_v3.cpp" -lcrypto -o "$out/test-authentic-v3"
"$out/test-authentic-v3" /home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1/export/authentic-trajectories-v3.bin
sha256sum "$out/encode-balance8" "$out/balance8-migration" "$out/replay-balance8" "$out/authentic-ppo-balance8" "$obj" > "$out/binary-hashes.sha256"
printf 'Compile and CPU checks completed. No GPU inference or training executed.\n'
