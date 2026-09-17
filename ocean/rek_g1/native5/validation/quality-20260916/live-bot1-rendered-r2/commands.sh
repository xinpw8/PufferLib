set -euo pipefail
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
relay=/home/spark-advantage/codexrook-runtime/live-transfer-20260915/RekUiPipeClient-unexpected-ai-20260916.exe
test "$(sha256sum "$relay" | cut -d' ' -f1)" = a47dfebd5a0b3c7b2feec526469f3947d61de4297aace7848f1d304cbe9535cc
sha256sum "$relay" "$b/live_transfer_run.cjs" "$b/live-retry-base.json"
node "$b/source/ocean/rek_g1/native5/run_live_mask_trial.cjs" "$b/live-retry-base.json" "$b/encoder-after/encode-live" "$b/live_transfer_run.cjs" "$b/live-bot1-rendered-r2" "$b/train-bot1-rendered-r1/checkpoints/rek_native5/train-bot1-rendered-r1/0000000536870912.bin" 40109bdcb2b84fabb2d66b8c995855fee1253a2c7ca943382aff8dbd7dc4eb99
