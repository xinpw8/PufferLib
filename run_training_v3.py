"""Chess selfplay v3: ChessLight-8x2 (no LSTM) with material rewards.
Key changes from v2:
- proj_dim=8, num_spatial=2 (faster, ~935K forward obs/s)
- No LSTM (chess is fully observable; saves bptt overhead)
- ent_coef=0.05 (prevent entropy collapse)
- checkpoint_interval=10 (frequent evaluation)
- bptt_horizon=16 (no LSTM needed, smaller chunks)
"""
import multiprocessing

def main():
    import sys
    sys.argv = ['pufferl.py', 'train', 'puffer_chess',
        '--wandb',
        '--wandb-project', 'pufferlib',
        '--wandb-group', 'chess-selfplay-v3',
        '--tag', 'chesslight-8x2-no-lstm-material-rewards',
    ]
    from pufferlib.pufferl import main as puffer_main
    puffer_main()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
