'''Unified eval for PokeBattle: run all bot evals or play vs policy.

Usage:
    python -m pufferlib.ocean.poke_battle.eval                  # all 3 bot evals
    python -m pufferlib.ocean.poke_battle.eval --human          # GUI play vs policy
    python -m pufferlib.ocean.poke_battle.eval --model-path X   # specify checkpoint
    python -m pufferlib.ocean.poke_battle.eval --episodes 200   # more episodes
'''

import argparse
import glob
import os
import time

import numpy as np
import torch

import pufferlib.pytorch
from pufferlib.ocean.poke_battle.poke_battle import PokeBattle


OBS_SIZE = 140


def _latest_checkpoint():
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    # Check both patterns: directory checkpoints and top-level symlinks
    patterns = [
        str(root / 'experiments' / 'puffer_poke_battle_*' / 'model_puffer_poke_battle_*.pt'),
        str(root / 'experiments' / 'puffer_poke_battle_*.pt'),
    ]
    matches = []
    for p in patterns:
        matches.extend(glob.glob(p))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _load_policy(model_path, env, device='cuda'):
    from pufferlib.ocean.torch import PokeBattle as PokeBattlePolicy, PokeBattleLSTM
    policy = PokeBattlePolicy(env, hidden_size=256)
    policy = PokeBattleLSTM(env, policy, input_size=256, hidden_size=256)
    policy = policy.to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def eval_vs_bot(model_path, bot_mode, num_episodes, device='cuda',
                mcts_iterations=128, mcts_depth=5):
    '''Run policy vs bot eval. Returns (wins, losses, draws, elapsed).'''
    env = PokeBattle(
        num_envs=1,
        selfplay=0,
        bot_mode=bot_mode,
        mcts_iterations=mcts_iterations,
        mcts_depth=mcts_depth,
        seed=42,
        auto_reset=0,
    )
    policy = _load_policy(model_path, env, device)
    lstm_h = torch.zeros(1, policy.hidden_size, device=device)
    lstm_c = torch.zeros(1, policy.hidden_size, device=device)

    wins, losses, draws = 0, 0, 0
    start = time.time()

    try:
        for ep in range(1, num_episodes + 1):
            env.reset(seed=42 + ep)
            lstm_h.zero_()
            lstm_c.zero_()
            while True:
                obs_t = torch.as_tensor(env.observations, device=device)
                state = dict(lstm_h=lstm_h, lstm_c=lstm_c)
                with torch.no_grad():
                    logits, _ = policy.forward_eval(obs_t, state)
                    action, _, _ = pufferlib.pytorch.sample_logits(logits)
                lstm_h = state.get('lstm_h', lstm_h)
                lstm_c = state.get('lstm_c', lstm_c)

                env.step(np.array([int(action.item())], dtype=np.int32))
                if bool(env.terminals[0]):
                    r = float(env.rewards[0])
                    if r > 0:
                        wins += 1
                    elif r < 0:
                        losses += 1
                    else:
                        draws += 1
                    break
    finally:
        env.close()

    elapsed = time.time() - start
    return wins, losses, draws, elapsed


def eval_all_bots(model_path, num_episodes, device='cuda'):
    '''Run policy vs all 3 bot types and print summary table.'''
    bot_names = {0: 'Random', 1: 'Heuristic', 2: 'MCTS'}
    results = {}

    for bot_mode in [0, 1, 2]:
        name = bot_names[bot_mode]
        print(f'  Evaluating vs {name}...', end='', flush=True)
        w, l, d, elapsed = eval_vs_bot(model_path, bot_mode, num_episodes, device)
        total = w + l + d
        results[name] = (w, l, d, total, elapsed)
        wr = 100.0 * w / total if total > 0 else 0
        print(f' {w}/{total} wins ({wr:.1f}%) in {elapsed:.1f}s')

    # Summary table
    print()
    print(f'  {"Opponent":<12} {"Wins":>6} {"Losses":>6} {"Draws":>6} {"Win%":>7} {"Time":>7}')
    print(f'  {"-"*12} {"-"*6} {"-"*6} {"-"*6} {"-"*7} {"-"*7}')
    for name, (w, l, d, total, elapsed) in results.items():
        wr = 100.0 * w / total if total > 0 else 0
        print(f'  {name:<12} {w:>6} {l:>6} {d:>6} {wr:>6.1f}% {elapsed:>6.1f}s')
    print()

    return results


def eval_human(model_path, device='cuda', episodes=10):
    '''Launch GUI for human vs policy.'''
    base_seed = time.time_ns() & 0x7FFFFFFF

    env = PokeBattle(
        num_envs=1,
        selfplay=1,
        bot_mode=0,
        seed=base_seed,
        auto_reset=0,
    )
    policy = _load_policy(model_path, env, device)
    lstm_h = torch.zeros(1, policy.hidden_size, device=device)
    lstm_c = torch.zeros(1, policy.hidden_size, device=device)

    wins, losses, draws = 0, 0, 0
    closed = False

    try:
        for ep in range(1, episodes + 1):
            env.reset(seed=base_seed + ep)
            lstm_h.zero_()
            lstm_c.zero_()

            while True:
                human_action = env.render_get_action(0)
                if human_action == -1:
                    closed = True
                    break
                if human_action == -2:
                    break

                p2_obs = env.observations[0, OBS_SIZE:OBS_SIZE * 2]
                obs_t = torch.as_tensor(p2_obs, device=device).unsqueeze(0)
                state = dict(lstm_h=lstm_h, lstm_c=lstm_c)
                with torch.no_grad():
                    logits, _ = policy.forward_eval(obs_t, state)
                    policy_action, _, _ = pufferlib.pytorch.sample_logits(logits)
                lstm_h = state.get('lstm_h', lstm_h)
                lstm_c = state.get('lstm_c', lstm_c)

                env.step(np.array([human_action, int(policy_action.item())], dtype=np.int32))

                if bool(env.terminals[0]):
                    r = float(env.rewards[0])
                    if r > 0:
                        wins += 1
                        result = 'Won'
                    elif r < 0:
                        losses += 1
                        result = 'Lost'
                    else:
                        draws += 1
                        result = 'Draw'
                    print(f'  Ep {ep}: {result}  (W={wins} L={losses} D={draws})')
                    lstm_h.zero_()
                    lstm_c.zero_()

            if closed:
                break

    finally:
        env.close()
        total = wins + losses + draws
        if total > 0:
            wr = 100.0 * wins / total
            print(f'\n  Human vs Policy: {total} games played')
            print(f'  {"You":<12} {"Wins":>6} {"Losses":>6} {"Draws":>6} {"Win%":>7}')
            print(f'  {"-"*12} {"-"*6} {"-"*6} {"-"*6} {"-"*7}')
            print(f'  {"Human":<12} {wins:>6} {losses:>6} {draws:>6} {wr:>6.1f}%')
            print(f'  {"Policy":<12} {losses:>6} {wins:>6} {draws:>6} {100-wr:>6.1f}%')


def main():
    parser = argparse.ArgumentParser(
        description='PokeBattle eval: run bot evals or play vs policy')
    parser.add_argument('--human', action='store_true',
                        help='Launch GUI to play against the policy')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to checkpoint (auto-detects latest if omitted)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Episodes per bot eval (default: 100)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model_path = args.model_path or _latest_checkpoint()
    if not model_path:
        print('Error: no checkpoint found. Provide --model-path or train first.')
        return

    print(f'PokeBattle Eval')
    print(f'  Checkpoint: {model_path}')
    print()

    if args.human:
        print('  Mode: Human vs Policy (GUI)')
        print('  You are P1 (bottom). Policy is P2 (top).')
        print()
        eval_human(model_path, args.device, episodes=args.episodes)
    else:
        print(f'  Mode: Policy vs Bots ({args.episodes} episodes each)')
        print()
        eval_all_bots(model_path, args.episodes, args.device)


if __name__ == '__main__':
    main()
