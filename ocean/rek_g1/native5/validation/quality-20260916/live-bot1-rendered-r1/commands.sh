set -euo pipefail
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
node "$b/source/ocean/rek_g1/native5/run_live_mask_trial.cjs" "$b/live-cadence-r1/trial.config.json" "$b/encoder-after/encode-live" "$b/source/ocean/rek_g1/native5/live_transfer_run.cjs" "$b/live-bot1-rendered-r1" "$b/train-bot1-rendered-r1/checkpoints/rek_native5/train-bot1-rendered-r1/0000000536870912.bin" 40109bdcb2b84fabb2d66b8c995855fee1253a2c7ca943382aff8dbd7dc4eb99
