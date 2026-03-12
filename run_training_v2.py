#!/usr/bin/env python3
"""Chess selfplay v2: ChessSeven+LSTM with material reward shaping."""
import multiprocessing

def main():
    import sys
    sys.argv = ["pufferl.py", "train", "puffer_chess",
        "--wandb",
        "--wandb-project", "pufferlib",
        "--wandb-group", "chess-selfplay-v2",
        "--tag", "chessseven-lstm-material-rewards",
    ]
    from pufferlib.pufferl import main as puffer_main
    puffer_main()

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)
    main()
