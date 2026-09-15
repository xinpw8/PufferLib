#!/usr/bin/env bash
# Executed on Spark through WSL SSH; completed60/60, forfeits0, invalid0.
set -eu
stage=/home/spark-advantage/rek-training/semantic-fast-20260914-v1
node "$stage/source-v2/ocean/rek_g1/league/tournament.cjs" --config "$stage/eval-run-v2/server.json" --backend semantic_cuda --policies compact-v2-33m-bf16-sampled,compact-v2-1m-bf16-sampled,scripted --seeds 20001,20002,20003,20004,20005,20006,20007,20008,20009,20010 --out "$stage/eval-run-v2/tournament-sampled-r1" --timeout-ms 120000 --chunk-steps 64
