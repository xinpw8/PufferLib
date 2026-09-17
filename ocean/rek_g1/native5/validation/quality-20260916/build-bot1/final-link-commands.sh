set -eu
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
cp "$b/build-bot1/linked-object-hashes.txt" "$b/build-bot1/unexecuted-pre-final-link-hashes.txt"
cp "$b/build-bot1/puffer-rek-native5" "$b/build-bot1/unexecuted-pre-final-trainer"
set -euo pipefail
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
m=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
cuda=/usr/local/cuda
nccl=/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl
ray=/home/spark-advantage/pufferlib-5.0-wr64/raylib-5.5_linux_aarch64
objects=("$b/build-bot1/fast_runtime.o" "$b/build-bot1/fast_assets.o" "$b/build-bot1/native_policy.o" "$b/build-bot1/cJSON.o")
for obj in "${objects[@]}";do test -f "$obj";done
sha256sum "${objects[@]}" > "$b/build-bot1/linked-object-hashes.txt"
"$cuda/bin/nvcc" -arch=sm_121 -Xcompiler=-fopenmp "$b/build-bot1/pufferl.o" "${objects[@]}" "$ray/lib/libraylib.a" -L"$cuda/lib64" -L"$nccl/lib" -L"$m" -Xlinker=-rpath -Xlinker="$cuda/lib64" -Xlinker=-rpath -Xlinker="$nccl/lib" -Xlinker=-rpath -Xlinker="$m" -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -l:libmujoco.so.3.7.0 -lcrypto -lGL -lm -lpthread -lomp5 -o "$b/build-bot1/puffer-rek-native5"
for pair in 'eval-bot1 diverse-policy-eval' 'fast-eval-bot1 fast-policy-eval'; do
  set -- $pair
  "$cuda/bin/nvcc" -arch=sm_121 "$b/$1/evaluator.o" "${objects[@]}" -L"$m" -Xlinker=-rpath -Xlinker="$m" -l:libmujoco.so.3.7.0 -lcrypto -lcublas -lcurand -o "$b/$1/$2"
  sha256sum "$b/$1/evaluator.o" "${objects[@]}" "$b/$1/$2" > "$b/$1/build-hashes.txt"
  readelf -d "$b/$1/$2" > "$b/$1/elf-dependencies.txt"
done
"$cuda/bin/nvcc" -arch=sm_121 "$b/viewer-bot1/eval_worker.o" "${objects[@]}" -L"$m" -L"$cuda/lib64" -Xlinker=-rpath -Xlinker="$m" -Xlinker=-rpath -Xlinker="$cuda/lib64" -lcudart -lcublas -lcrypto -l:libmujoco.so.3.7.0 -lEGL -lGL -lz -lm -lpthread -o "$b/viewer-bot1/rek-eval-worker"
sha256sum "$b/build-bot1/puffer-rek-native5" "$b/eval-bot1/diverse-policy-eval" "$b/viewer-bot1/rek-eval-worker"
