#!/bin/bash
set -euo pipefail
HOST=spark
REMOTE=/home/spark-advantage/pufferlib-rek
ROOT=/mnt/c/Users/Daniel/codex-wr64-puffer-scenario-7cc6ce1
cd "$ROOT"
scp -q ocean/rek_fight/run_spark_cuda.sh "$HOST:$REMOTE/ocean/rek_fight/run_spark_cuda.sh"
ssh -o BatchMode=yes "$HOST" "sed -i 's/\r$//' $REMOTE/ocean/rek_fight/run_spark_cuda.sh && chmod +x $REMOTE/ocean/rek_fight/run_spark_cuda.sh && $REMOTE/ocean/rek_fight/run_spark_cuda.sh"
