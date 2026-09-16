set -eu
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
node "$b/source/ocean/rek_g1/native5/run_live_mask_trial.cjs" "$b/live-recovered-r13/trial.config.json" "$b/encoder-after/encode-live" "$b/source/ocean/rek_g1/native5/live_transfer_run.cjs" "$b/live-cadence-r1" "$b/train-recovered-r2/checkpoints/rek_native5/train-recovered-r2/0000000268435456.bin" 40109bdcb2b84fabb2d66b8c995855fee1253a2c7ca943382aff8dbd7dc4eb99

