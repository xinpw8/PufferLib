set -euo pipefail
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
s=$b/bot1-source/ocean/rek_g1/native5
checkpoint=$b/train-recovered-r2/checkpoints/rek_native5/train-recovered-r2/0000000268435456.bin
sha256sum -c "$b/build-bot1/linked-object-hashes.txt"
bash "$s/run_diverse_policy_eval.sh" "$b/eval-bot1" "$s/validation/quality-20260916/recovered-bot1-rendered-runtime.json" "$checkpoint" af84c4ec92e953e72c6bf4cde9ccb8cf80258a91dbf01947312e622dd492876e "$b/eval-pre-bot1-r1" 256 1 200009 sampled bf16 scripted heldout 120
