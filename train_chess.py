#!/usr/bin/env python3
"""Chess selfplay training with proper configuration.
Launch with: python train_chess.py [--resume RUN_ID]
"""
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import sys
import os

args = [
    "pufferl.py", "train", "puffer_chess",
    "--vec.backend", "Multiprocessing",
    "--vec.num-envs", "4",
    "--vec.num-workers", "4",
    "--env.num-envs", "4096",
    "--env.reward-capture-scale", "1.0",
    "--env.reward-draw", "-0.05",
    "--env.reward-repetition", "-0.01",
    "--env.reward-invalid-piece", "0.0",
    "--env.reward-invalid-move", "0.0",
    "--env.fen-curric-pct", "0.25",
    "--policy.channels", "64",
    "--policy.num-blocks", "3",
    "--policy.hidden-size", "256",
    "--policy.meta-hidden", "128",
    "--policy.embed-dim", "16",
    "--policy.group-norm-groups", "8",
    "--train.total-timesteps", "50000000000",
    "--train.minibatch-size", "32768",
    "--train.max-minibatch-size", "32768",
    "--train.bptt-horizon", "64",
    "--train.ent-coef", "0.01",
    "--train.learning-rate", "0.0003",
    "--train.gamma", "0.999",
    "--train.gae-lambda", "0.95",
    "--train.clip-coef", "0.2",
    "--train.vf-coef", "1.0",
    "--train.update-epochs", "2",
    "--train.max-grad-norm", "0.5",
    "--train.checkpoint-interval", "25",
    "--train.device", "cuda",
    "--wandb",
    "--wandb-project", "pufferlib",
    "--wandb-group", "chess-selfplay-v1",
    "--tag", "resnet64x3-material-rewards",
]

# Handle --resume flag
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--resume" and i + 1 < len(sys.argv):
        args.extend(["--load-id", sys.argv[i + 1]])
    elif arg.startswith("--"):
        args.append(arg)
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            args.append(sys.argv[i + 1])

sys.argv = args

from pufferlib.pufferl import main
main()
