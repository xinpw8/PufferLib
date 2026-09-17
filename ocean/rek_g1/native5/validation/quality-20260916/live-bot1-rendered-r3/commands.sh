set -euo pipefail
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
node "$b/source/ocean/rek_g1/native5/run_live_mask_trial.cjs" "$b/live-ready-base.json" "$b/encoder-after/encode-live" "$b/live_transfer_ready_run.cjs" "$b/live-bot1-rendered-r3" "$b/train-bot1-rendered-r1/checkpoints/rek_native5/train-bot1-rendered-r1/0000000536870912.bin" 2edf75c65e6693db28eb7e87c45d88522668f549c7b457cdf0aa108582667952
