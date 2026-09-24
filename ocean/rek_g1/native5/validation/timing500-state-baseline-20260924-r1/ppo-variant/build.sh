#!/usr/bin/env bash
set -euo pipefail
stage=/home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1/ppo-variant
src=/home/spark-advantage/rek-training/balance8-onpolicy-20260924-r2/source
prepared=/home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1/distributional-candidate/ppo-build/source
nccl=/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl
data=/home/spark-advantage/rek-training/timing500-ppo-refresh-20260924-r1/score-delta-5s
baseline=/home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1/fit-fixed-r1/row-baselines.bin
cd "$stage"
mkdir -p build
printf '%s  %s\n' 173bfaf5c8e511ab84f55efbfee956bd54fb1bc780542667269d5c9c742dde11 "$src/authentic_ppo.cu" 785b03ddc49a881045d64bea26faa9fd00c115bd465ecc9dcc4a4684b0cf331e "$src/authentic_trajectory.h" | sha256sum --check --strict
node test_unchanged_zero.cjs "$src/authentic_ppo.cu" authentic_ppo.cu
g++ -std=c++17 -O2 -I"$src" test_crossfit_baseline.cpp -lcrypto -o build/test-crossfit
build/test-crossfit "$data/authentic-score-delta-v5.bin" eb6b1ae210b3b3911597f93a513ae32081f15e4871b29c1cb29e75829f718655 "$baseline" 8d4a24282f834a687a7740c14729974b87757a3b80779cb7fe004ba0936b268a 32ee5bb1dd2cdb427cfeff7ae01f663a350858b3d35d80655d86987f64293dd9
/usr/local/cuda/bin/nvcc -std=c++17 -O2 --threads 1 -arch=sm_121 -I"$src" -I"$prepared" -I"$nccl/include" -I/usr/local/cuda/include/cccl authentic_ppo.cu -lcublas -lcurand -lcrypto -o build/authentic-ppo-state-baseline
sha256sum authentic_ppo.cu crossfit_baseline.h test_crossfit_baseline.cpp test_unchanged_zero.cjs build.sh build/test-crossfit build/authentic-ppo-state-baseline
printf 'CPU tests and CUDA compilation only. No GPU execution.\n'
