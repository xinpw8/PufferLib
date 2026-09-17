set -euo pipefail
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
s=$b/bot1-source/ocean/rek_g1/native5
checkpoint=$b/train-bot1-rendered-r1/checkpoints/rek_native5/train-bot1-rendered-r1/0000000536870912.bin
bash "$s/run_diverse_policy_eval.sh" "$b/eval-bot1" "$s/validation/quality-20260916/recovered-bot1-rendered-runtime.json" "$checkpoint" f87dae69a777e4ac28782bdee89b30d7434208d773be75bf97f56fba4a52b07e "$b/eval-post-bot1-r1-fixed" 256 1 200013 sampled bf16 scripted fixed 120
