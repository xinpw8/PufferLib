'''Evaluate a trained PokeBattle model against bot opponents.'''

import sys
import time
import numpy as np
import torch
import pufferlib.pytorch

from pufferlib.ocean.poke_battle.poke_battle import PokeBattle

# Default model path (most recent training run)
MODEL_PATH = None
NUM_ENVS = 256
NUM_EPISODES = 2000
BOT_MODE = 0  # 0=random, 1=heuristic, 2=mcts

def load_policy(model_path, env, device='cuda'):
    from pufferlib.ocean.torch import PokeBattle as PokeBattlePolicy, PokeBattleLSTM
    policy = PokeBattlePolicy(env, hidden_size=256)
    policy = PokeBattleLSTM(env, policy, input_size=256, hidden_size=256)
    policy = policy.to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate PokeBattle model vs bot')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--bot-mode', type=int, default=0,
                        choices=[0, 1, 2], help='0=random, 1=heuristic, 2=mcts')
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--num-episodes', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mcts-iterations', type=int, default=128)
    parser.add_argument('--mcts-depth', type=int, default=5)
    args = parser.parse_args()

    bot_names = {0: 'Random', 1: 'Heuristic (1-ply)', 2: f'MCTS (iter={args.mcts_iterations})'}
    print(f'Evaluating model vs {bot_names[args.bot_mode]} bot...')
    print(f'  Model: {args.model_path}')
    print(f'  Envs: {args.num_envs}, Target episodes: {args.num_episodes}')

    # Create env with selfplay=0 (bot opponent)
    env = PokeBattle(
        num_envs=args.num_envs,
        selfplay=0,
        bot_mode=args.bot_mode,
        mcts_iterations=args.mcts_iterations,
        mcts_depth=args.mcts_depth,
    )

    # Load trained policy
    policy = load_policy(args.model_path, env, args.device)

    # Run evaluation
    env.reset()
    device = args.device

    wins = 0
    losses = 0
    draws = 0
    total_episodes = 0
    total_reward = 0.0
    total_steps = 0

    lstm_h = torch.zeros(args.num_envs, policy.hidden_size, device=device)
    lstm_c = torch.zeros(args.num_envs, policy.hidden_size, device=device)

    start = time.time()
    while total_episodes < args.num_episodes:
        obs = torch.as_tensor(env.observations).to(device)
        state = dict(
            lstm_h=lstm_h,
            lstm_c=lstm_c,
        )

        with torch.no_grad():
            logits, value = policy.forward_eval(obs, state)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
            lstm_h = state.get('lstm_h', lstm_h)
            lstm_c = state.get('lstm_c', lstm_c)

        actions = action.cpu().numpy()
        env.step(actions)
        total_steps += args.num_envs

        # Count completed episodes
        for i in range(args.num_envs):
            if env.terminals[i]:
                total_episodes += 1
                r = env.rewards[i]
                total_reward += r
                if r > 0:
                    wins += 1
                elif r < 0:
                    losses += 1
                else:
                    draws += 1

                # Reset LSTM state for completed episodes
                lstm_h[i] = 0
                lstm_c[i] = 0

    elapsed = time.time() - start
    win_rate = wins / total_episodes * 100
    loss_rate = losses / total_episodes * 100
    draw_rate = draws / total_episodes * 100
    avg_reward = total_reward / total_episodes

    print(f'\nResults ({total_episodes} episodes, {elapsed:.1f}s):')
    print(f'  Win rate:  {wins:>5d} / {total_episodes} ({win_rate:.1f}%)')
    print(f'  Loss rate: {losses:>5d} / {total_episodes} ({loss_rate:.1f}%)')
    print(f'  Draw rate: {draws:>5d} / {total_episodes} ({draw_rate:.1f}%)')
    print(f'  Avg reward: {avg_reward:.4f}')
    print(f'  SPS: {total_steps / elapsed:.0f}')

    env.close()

if __name__ == '__main__':
    main()
