#!/usr/bin/env bash
set -euo pipefail
cd /home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1
mkdir -p vendor build
reader=/home/spark-advantage/rek-training/balance8-onpolicy-20260924-r2/source
printf '%s  %s\n' 785b03ddc49a881045d64bea26faa9fd00c115bd465ecc9dcc4a4684b0cf331e "$reader/authentic_trajectory.h" b49311dae9506b3c64fb50fc791e31fba143f8445af40494864358d1a52dfab1 "$reader/round_reward.h" | sha256sum --check --strict
cp -n "$reader/authentic_trajectory.h" "$reader/round_reward.h" vendor/
printf '%s  %s\n' 785b03ddc49a881045d64bea26faa9fd00c115bd465ecc9dcc4a4684b0cf331e vendor/authentic_trajectory.h b49311dae9506b3c64fb50fc791e31fba143f8445af40494864358d1a52dfab1 vendor/round_reward.h | sha256sum --check --strict
g++ -x c++ -DREK_BASELINE_CPU_ONLY -std=c++17 -O2 -Ivendor state_baseline.cu -lcrypto -o build/cpu-check
./build/cpu-check --cpu-test
./build/cpu-check --inspect /home/spark-advantage/rek-training/timing500-ppo-refresh-20260924-r1/score-delta-5s/authentic-score-delta-v5.bin eb6b1ae210b3b3911597f93a513ae32081f15e4871b29c1cb29e75829f718655
/usr/local/cuda/bin/nvcc -std=c++17 -O2 --threads 1 -arch=sm_121 -Ivendor state_baseline.cu -lcublas -lcusolver -lcrypto -o build/state-baseline
sha256sum state_baseline.cu PROTOCOL.md build.sh vendor/authentic_trajectory.h vendor/round_reward.h build/cpu-check build/state-baseline
