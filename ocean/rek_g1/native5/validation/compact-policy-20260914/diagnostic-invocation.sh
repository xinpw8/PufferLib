#!/usr/bin/env bash
# Executed on Spark through WSL SSH. Preserved invocation, not a new run.
set -eu
stage=/home/spark-advantage/rek-training/semantic-fast-20260914-v1
for feature in 1 64; do
REK_EVAL_ROUND_FEATURE="$feature" bash "$stage/source/ocean/rek_g1/native5/run_fast_policy_eval.sh" "$stage/policy-eval-build-v2" "$stage/runtime-config.json" "$stage/bench-512-r1/checkpoints/rek_native5/semantic-cuda-512-16/0000000033554432.bin" fcc5bcac7a7dabb36ebb79fcf13ca5dae640377287e4409a9ac69b4e24b9444b "$stage/policy-eval-round-feature${feature}-diagnostic-r1" 128 2 10001 sampled bf16
done
