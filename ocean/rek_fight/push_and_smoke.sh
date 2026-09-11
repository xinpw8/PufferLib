#!/bin/bash
set -euo pipefail
HOST=spark
REMOTE=/home/spark-advantage/pufferlib-rek
ROOT=/mnt/c/Users/Daniel/codex-wr64-puffer-scenario-7cc6ce1
ssh -o BatchMode=yes "$HOST" "mkdir -p $REMOTE/ocean/rek_fight $REMOTE/ocean/rek_strategy $REMOTE/ocean/rek_match $REMOTE/ocean/rek $REMOTE/config"
cd "$ROOT"
scp -q ocean/rek_fight/rek_fight.h ocean/rek_fight/binding.c ocean/rek_fight/test_rek_fight.c ocean/rek_fight/README.md ocean/rek_fight/train_spark.sh ocean/rek_fight/run_spark_smoke.sh "$HOST:$REMOTE/ocean/rek_fight/"
scp -q ocean/rek_strategy/strategy_router.h "$HOST:$REMOTE/ocean/rek_strategy/"
scp -q ocean/rek_match/rek_match.h "$HOST:$REMOTE/ocean/rek_match/"
scp -q ocean/rek/rek_sha256.h "$HOST:$REMOTE/ocean/rek/"
scp -q config/rek_fight.ini "$HOST:$REMOTE/config/"
scp -q build.sh "$HOST:$REMOTE/build.sh"
ssh -o BatchMode=yes "$HOST" 'chmod +x /home/spark-advantage/pufferlib-rek/ocean/rek_fight/run_spark_smoke.sh; sed -i "s/\r$//" /home/spark-advantage/pufferlib-rek/ocean/rek_fight/run_spark_smoke.sh; /home/spark-advantage/pufferlib-rek/ocean/rek_fight/run_spark_smoke.sh'
