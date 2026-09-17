set -euo pipefail
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
s=$b/bot1-source/ocean/rek_g1/native5
sha256sum -c "$b/build-bot1/linked-object-hashes.txt"
unset REK_TRAIN_OPPONENT_CHECKPOINT REK_FROZEN_OPPONENT_FRACTION
export REK_FAST_SCORING=recovered_hit_rules_v2
export REK_FAST_OPPONENT=recovered_bot1_v1
export REK_FAST_OBSERVATION=rendered_pose_v1
export REK_FAST_OPPONENT_MODE=scripted
export REK_FAST_RANDOM_RESETS=1
export REK_FAST_RESET_GAP_MIN=.55
export REK_FAST_RESET_GAP_MAX=2.5
export REK_FAST_RESET_HEADING_SPREAD_RAD=3.14159265
export REK_FAST_SHAPING_WEIGHT=0
export REK_TRAIN_LEARNING_RATE=.0001
export REK_TRAIN_SEED=197
bash "$s/run_diverse_training.sh" "$b/build-bot1" "$b/train-bot1-rendered-r1" 536870912 512 128 120 "$b/train-recovered-r2/checkpoints/rek_native5/train-recovered-r2/0000000268435456.bin"
