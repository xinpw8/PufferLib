#!/usr/bin/env bash
# End-to-end sanity probe for the lite training path.
#
# Reward: move_start_v1, 0.01 for each accepted start of native move 7 (the HH
# left front kick, action category 17) and nothing else: no points, falls,
# knockouts or terminal reward. A working env + learner + evaluator must
# produce a policy that, in greedy evaluation, starts HH at nearly every
# opportunity and almost never starts anything else. Anything else means a
# broken reward, mask, observation, checkpoint or evaluation path.
#
# Usage: run_lite_kick_probe.sh BUILD NEW_OUTPUT [TOTAL_STEPS] [ARENAS] [HORIZON]
#   BUILD    a build_fast.sh output containing puffer-rek-native5 and objects
#   LITE     env: smoke (default), none, or a rek.lite_falls.v1 model path
set -euo pipefail
umask 077
[[ $# -ge 2 && $# -le 5 ]] || { printf 'Usage: %s BUILD NEW_OUTPUT [TOTAL_STEPS] [ARENAS] [HORIZON]\n' "$0" >&2; exit 2; }
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$(realpath "$1")
task_arenas=${4:-4096}
task_horizon=${5:-64}
task_steps=${3:-$((task_arenas*task_horizon*512))}
task_seconds=20
mkdir "$2"
task_output=$(realpath "$2")
case ${LITE:-smoke} in
    none) task_lite= ;;
    smoke) task_lite=$task_source/lite_falls_smoke_v1.json ;;
    *) task_lite=$(realpath "$LITE"); test -f "$task_lite" ;;
esac
# Current compact configuration (as in run_normalized_sweep.sh), HH-only reward.
export REK_OBSERVATION_SCHEMA=rek.native5.scaled_polar_xy.v1 REK_POLICY_ACTION_STRIDE=1
export REK_FAST_YAW_COMMAND=keyboard_reset_v1 REK_FAST_CONTACT_ENTRY=geom_pair_v1 REK_FAST_CONTACT_VELOCITY=body_cvel_v1
export REK_FAST_SCORING=recovered_hit_rules_v2 REK_FAST_GEOMETRY=primitive_samples_v1 REK_FAST_CONTACT_SUBSTEPS=8
export REK_FAST_OPPONENT=recovered_bot1_v1 REK_FAST_OBSERVATION=rendered_pose_v1 REK_FAST_OPPONENT_MODE=scripted
export REK_FAST_RANDOM_RESETS=1 REK_FAST_RESET_GAP_MIN=.55 REK_FAST_RESET_GAP_MAX=2.5 REK_FAST_RESET_HEADING_SPREAD_RAD=3.14159265
export REK_FAST_SHAPING_WEIGHT=0 REK_FAST_REWARD=move_start_v1 REK_FAST_REWARD_MOVE=7 REK_FAST_REWARD_MOVE_VALUE=0.01
export REK_TRAIN_GAMMA=${REK_TRAIN_GAMMA:-0.99} REK_TRAIN_GAE_LAMBDA=${REK_TRAIN_GAE_LAMBDA:-0.95}
# The pinned trainer uses raw advantages: keep entropy well below the 0.01 reward.
export REK_TRAIN_ENTROPY=${REK_TRAIN_ENTROPY:-0.001} REK_TRAIN_LEARNING_RATE=${REK_TRAIN_LEARNING_RATE:-0.0003}
export REK_TRAIN_MINIBATCH=${REK_TRAIN_MINIBATCH:-32768} REK_TRAIN_DOWNSAMPLE=${REK_TRAIN_DOWNSAMPLE:-16}
export REK_TRAIN_TIMEOUT=${REK_TRAIN_TIMEOUT:-1800}
unset REK_FAST_CONTACT_POTENTIAL REK_FAST_REWARD_GAMMA REK_TRAIN_OPPONENT_CHECKPOINT REK_POLICY_FEATURE_MASK REK_EVAL_ROUND_FEATURE
if [[ -n "$task_lite" ]]; then export REK_LITE_FALLS=$task_lite; else unset REK_LITE_FALLS; fi
printf 'lite_falls=%s\nsteps=%s\narenas=%s\nhorizon=%s\n' "${task_lite:-none}" "$task_steps" "$task_arenas" "$task_horizon" > "$task_output/probe.txt"

# 1. Train a fresh policy.
bash "$task_source/run_diverse_training.sh" "$task_build" "$task_output/train" "$task_steps" "$task_arenas" \
    "$task_horizon" "$task_seconds" > "$task_output/train.runner.txt" 2>&1 \
    || { tail -40 "$task_output/train.runner.txt"; exit 1; }
task_checkpoint=$(find "$task_output/train/checkpoints" -type f -name '*.bin' | sort | tail -1)
[[ -n "$task_checkpoint" ]] || { printf 'No checkpoint written\n' >&2; exit 1; }
task_sha=$(sha256sum "$task_checkpoint" | cut -d' ' -f1)

# 2. Evaluator built from the exact training objects.
bash "$task_source/build_fast_policy_eval.sh" "$task_build" "$task_output/eval-build" > "$task_output/eval-build.txt" 2>&1

# 3. Greedy BF16 evaluation on both sides with the same compact modes.
task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
task_features=/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
python3 - "$task_output/eval-runtime.json" "$task_assets" "$task_features" "$task_seconds" "$task_lite" <<'EOF'
import json, sys
path, assets, features, seconds, lite = sys.argv[1:6]
fast = {"opponent_controller": "recovered_bot1_v1", "observation_mode": "rendered_pose_v1",
        "geometry_mode": "primitive_samples_v1", "contact_substeps": 8, "scoring_mode": "recovered_hit_rules_v2"}
if lite:
    fast["lite_falls_model"] = lite
with open(path, "x") as handle:
    json.dump({"backend": "semantic_cuda", "model_path": assets + "/model.two_fighter_arena.xml",
               "assets_path": assets, "motion_features_path": features, "round_seconds": int(seconds),
               "fast": fast}, handle, indent=1)
EOF
REK_EVAL_WATCH_ACTION=17 bash "$task_source/run_fast_policy_eval.sh" "$task_output/eval-build" "$task_output/eval-runtime.json" \
    "$task_checkpoint" "$task_sha" "$task_output/eval" 256 4 911 greedy bf16 > "$task_output/eval.txt" 2>&1 \
    || { tail -40 "$task_output/eval.txt"; exit 1; }

# 4. Verdict.
python3 - "$task_output" <<'EOF'
import glob, json, os, re, sys
out = sys.argv[1]
sides = [json.loads(line) for line in open(os.path.join(out, "eval", "summary.jsonl"))
         if '"side_result"' in line]
sps = None
for ini in sorted(glob.glob(os.path.join(out, "train", "logs", "**", "*.ini"), recursive=True)):
    for line in open(ini):
        if line.startswith("SPS = "):
            values = [float(v) for v in line.split("=", 1)[1].split(",") if v.strip()]
            sps = values[-1] if values else sps
ok = len(sides) == 2
for side in sides:
    starts = side["learner_move_starts_by_category"]
    side["verdict"] = {"share_ok": side["watched_share_of_starts"] >= 0.95,
                       "rate_ok": side["watched_opportunity_use"] >= 0.90,
                       "other_starts": sum(starts) - starts[17]}
    ok = ok and side["verdict"]["share_ok"] and side["verdict"]["rate_ok"]
verdict = {"probe": "lite_hh_kick_v1", "pass": ok, "final_training_sps": sps,
           "sides": [{k: s[k] for k in ("policy_side", "watched_starts", "watched_share_of_starts",
                                         "watched_opportunity_ticks", "watched_opportunity_use",
                                         "falls", "opponent_falls", "verdict")} for s in sides]}
with open(os.path.join(out, "verdict.json"), "x") as handle:
    json.dump(verdict, handle, indent=1)
print(json.dumps(verdict, indent=1))
sys.exit(0 if ok else 1)
EOF
