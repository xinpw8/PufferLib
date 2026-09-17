set -euo pipefail
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
s=$b/bot1-source/ocean/rek_g1/native5
checkpoint=$b/train-bot1-rendered-r1/checkpoints/rek_native5/train-bot1-rendered-r1/0000000536870912.bin
sha256sum "$checkpoint"
cat "$b/train-bot1-rendered-r1/exit-code.txt" "$b/train-bot1-rendered-r1/verified-warm-start.txt"
set +e
timeout 10 "$b/build-bot1/scoring-v2-test" --gpu > "$b/build-bot1/scoring-v2-gpu.stdout.txt" 2> "$b/build-bot1/scoring-v2-gpu.stderr.txt"
rc=$?
set -e
printf '%s\n' "$rc" > "$b/build-bot1/scoring-v2-gpu.exit-code.txt"
cat "$b/build-bot1/scoring-v2-gpu.stdout.txt" "$b/build-bot1/scoring-v2-gpu.stderr.txt"
test "$rc" = 0
sha=$(sha256sum "$checkpoint" | cut -d' ' -f1)
bash "$s/run_diverse_policy_eval.sh" "$b/eval-bot1" "$s/validation/quality-20260916/recovered-bot1-rendered-runtime.json" "$checkpoint" "$sha" "$b/eval-post-bot1-r1" 256 1 200009 sampled bf16 scripted heldout 120
bash "$s/run_diverse_policy_eval.sh" "$b/eval-bot1" "$s/validation/quality-20260916/recovered-bot1-rendered-runtime.json" "$checkpoint" "$sha" "$b/eval-post-bot1-r1-fresh" 256 1 200011 sampled bf16 scripted heldout 120
