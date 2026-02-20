'''Interactive play against bot or trained policy using Showdown-style Raylib UI.

Usage:
    python -m pufferlib.ocean.poke_battle.play_showdown --bot-mode 0  # vs random bot
    python -m pufferlib.ocean.poke_battle.play_showdown --bot-mode 1  # vs heuristic bot
    python -m pufferlib.ocean.poke_battle.play_showdown --bot-mode 2  # vs MCTS bot
    python -m pufferlib.ocean.poke_battle.play_showdown --vs-policy   # vs trained policy
    python -m pufferlib.ocean.poke_battle.play_showdown --vs-policy --model-path path/to/model.pt
'''

import argparse
import json
import os
import time

import numpy as np
import torch

import pufferlib.pytorch
from pufferlib.ocean.poke_battle.poke_battle import PokeBattle

OBS_SIZE = 140


def snapshot_state(env):
    '''Capture compact battle state for replay logging.'''
    state = env.get_state(0)
    out = {
        'turn': state['turn'],
        'mode': state['mode'],
        'p1_action': state['last_p1_action'],
        'p2_action': state['last_p2_action'],
        'p1': [],
        'p2': [],
    }
    for side_key in ('p1', 'p2'):
        player = state[side_key]
        for i, mon in enumerate(player['team']):
            out[side_key].append({
                'species': mon['species'],
                'hp': mon['hp'],
                'max_hp': mon['max_hp'],
                'status': mon['status_name'] or None,
                'alive': bool(mon['is_alive']),
                'active': (i == player['active_idx']),
            })
    return out


def format_action(action, state, side_key):
    '''Human-readable action description.'''
    player = state[side_key]
    active_idx = next(i for i, m in enumerate(player) if m['active'])
    active = player[active_idx]
    if action < 4:
        # Move — get move name from env state
        return f"{active['species']} move {action}"
    elif action < 10:
        target = player[action - 4]
        return f"switch to {target['species']}"
    return f"action {action}"


def _latest_checkpoint():
    import glob
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    pattern = str(root / 'experiments' / 'puffer_poke_battle_*' / 'model_puffer_poke_battle_*.pt')
    matches = glob.glob(pattern)
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


def run_vs_policy(args):
    '''Human (p1, GUI) vs trained policy (p2) using selfplay env.'''
    base_seed = args.seed if args.seed is not None else (time.time_ns() & 0x7FFFFFFF)
    model_path = args.model_path or _latest_checkpoint()
    if not model_path:
        raise RuntimeError('No checkpoint found. Provide --model-path.')
    device = args.device

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f'session_vs_policy_{int(time.time())}.jsonl')
    log_file = open(log_path, 'w')
    print(f'Recording battle logs to {log_path}')
    print(f'Model: {model_path}')

    # selfplay=1: obs is [p1_obs(140) | p2_obs(140)], actions buffer has 2 entries
    env = PokeBattle(
        num_envs=1,
        selfplay=1,
        bot_mode=0,
        seed=base_seed,
        auto_reset=0,
    )
    policy = _load_policy(model_path, env, device)

    wins, losses, draws = 0, 0, 0
    lstm_h = torch.zeros(1, policy.hidden_size, device=device)
    lstm_c = torch.zeros(1, policy.hidden_size, device=device)

    try:
        for ep in range(1, args.episodes + 1):
            ep_seed = base_seed + ep
            env.reset(seed=ep_seed)
            lstm_h.zero_()
            lstm_c.zero_()

            init_state = snapshot_state(env)
            game_record = {
                'episode': ep,
                'seed': ep_seed,
                'opponent': 'policy',
                'model': model_path,
                'p1_team': [m['species'] for m in init_state['p1']],
                'p2_team': [m['species'] for m in init_state['p2']],
                'turns': [],
                'result': None,
            }

            while True:
                # Human picks p1 action via GUI (blocks until click)
                human_action = env.render_get_action(0)

                if human_action == -1:
                    print(f'\nFinal: W={wins} L={losses} D={draws} / {wins+losses+draws} games')
                    return
                if human_action == -2:
                    break  # restart signal

                # Policy picks p2 action from p2's observation
                p2_obs = env.observations[0, OBS_SIZE:OBS_SIZE * 2]
                obs_t = torch.as_tensor(p2_obs, device=device).unsqueeze(0)
                net_state = dict(lstm_h=lstm_h, lstm_c=lstm_c)
                with torch.no_grad():
                    logits, _ = policy.forward_eval(obs_t, net_state)
                    policy_action, _, _ = pufferlib.pytorch.sample_logits(logits)
                lstm_h = net_state.get('lstm_h', lstm_h)
                lstm_c = net_state.get('lstm_c', lstm_c)

                pre_state = snapshot_state(env)
                env.step(np.array([human_action, int(policy_action.item())], dtype=np.int32))
                post_state = snapshot_state(env)

                turn_record = {
                    'turn': post_state['turn'],
                    'p1_action': post_state['p1_action'],
                    'p2_action': post_state['p2_action'],
                    'p1_hp': [(m['hp'], m['max_hp']) for m in post_state['p1']],
                    'p2_hp': [(m['hp'], m['max_hp']) for m in post_state['p2']],
                }
                for side in ('p1', 'p2'):
                    for i in range(len(post_state[side])):
                        if pre_state[side][i]['alive'] and not post_state[side][i]['alive']:
                            turn_record.setdefault('faints', []).append(
                                f'{side} {post_state[side][i]["species"]}')
                game_record['turns'].append(turn_record)

                if bool(env.terminals[0]):
                    reward = float(env.rewards[0])
                    # reward is from learner (p1) perspective; human IS p1
                    if reward > 0:
                        wins += 1
                        game_record['result'] = 'win'
                    elif reward < 0:
                        losses += 1
                        game_record['result'] = 'loss'
                    else:
                        draws += 1
                        game_record['result'] = 'draw'

                    game_record['total_turns'] = post_state['turn']
                    log_file.write(json.dumps(game_record) + '\n')
                    log_file.flush()

                    print(f'Ep {ep}: {"Won" if reward > 0 else "Lost" if reward < 0 else "Draw"} '
                          f'in {post_state["turn"]} turns (vs Policy) '
                          f'W={wins} L={losses} D={draws}')
                    lstm_h.zero_()
                    lstm_c.zero_()

            print(f'Running: W={wins} L={losses} D={draws}')
        print(f'\nFinal: W={wins} L={losses} D={draws} / {wins+losses+draws} games')
    finally:
        log_file.close()
        env.close()
        print(f'Battle logs saved to {log_path}')


def main():
    parser = argparse.ArgumentParser(description='Play PokeBattle with Showdown UI')
    parser.add_argument('--vs-policy', action='store_true',
                        help='Play against trained policy instead of bot')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to policy checkpoint (auto-detects latest if omitted)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--bot-mode', type=int, default=1, choices=[0, 1, 2],
                        help='Bot: 0=random, 1=heuristic, 2=MCTS')
    parser.add_argument('--mcts-iterations', type=int, default=128)
    parser.add_argument('--mcts-depth', type=int, default=5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--log-dir', type=str, default='experiments/battle_logs',
                        help='Directory to save battle logs')
    args = parser.parse_args()

    if args.vs_policy:
        run_vs_policy(args)
        return

    base_seed = args.seed if args.seed is not None else (time.time_ns() & 0x7FFFFFFF)
    bot_names = {0: 'Random', 1: 'Heuristic', 2: 'MCTS'}

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f'session_{int(time.time())}.jsonl')
    log_file = open(log_path, 'w')
    print(f'Recording battle logs to {log_path}')

    wins, losses, draws = 0, 0, 0

    env = PokeBattle(
        num_envs=1,
        selfplay=0,
        bot_mode=args.bot_mode,
        mcts_iterations=args.mcts_iterations,
        mcts_depth=args.mcts_depth,
        seed=base_seed,
        auto_reset=0,
    )

    try:
        for ep in range(1, args.episodes + 1):
            ep_seed = base_seed + ep
            env.reset(seed=ep_seed)

            # Record initial state
            init_state = snapshot_state(env)
            game_record = {
                'episode': ep,
                'seed': ep_seed,
                'bot_mode': bot_names[args.bot_mode],
                'p1_team': [m['species'] for m in init_state['p1']],
                'p2_team': [m['species'] for m in init_state['p2']],
                'turns': [],
                'result': None,
            }

            game_over = False
            while not game_over:
                action = env.render_get_action(0)

                if action == -1:
                    # Window closed — finally block handles env.close()
                    print(f'\nFinal: W={wins} L={losses} D={draws} / {wins+losses+draws} games')
                    return

                if action == -2:
                    # Restart signal from result overlay click
                    break

                # Take state before step for logging
                pre_state = snapshot_state(env)
                env.step(np.array([action], dtype=np.int32))
                post_state = snapshot_state(env)

                # Record turn
                turn_record = {
                    'turn': post_state['turn'],
                    'p1_action': post_state['p1_action'],
                    'p2_action': post_state['p2_action'],
                    'p1_hp': [(m['hp'], m['max_hp']) for m in post_state['p1']],
                    'p2_hp': [(m['hp'], m['max_hp']) for m in post_state['p2']],
                }

                # Note any faints
                for side in ('p1', 'p2'):
                    for i in range(len(post_state[side])):
                        if pre_state[side][i]['alive'] and not post_state[side][i]['alive']:
                            turn_record.setdefault('faints', []).append(
                                f"{side} {post_state[side][i]['species']}")

                game_record['turns'].append(turn_record)

                if bool(env.terminals[0]):
                    reward = float(env.rewards[0])
                    if reward > 0:
                        wins += 1
                        game_record['result'] = 'win'
                    elif reward < 0:
                        losses += 1
                        game_record['result'] = 'loss'
                    else:
                        draws += 1
                        game_record['result'] = 'draw'

                    game_record['total_turns'] = post_state['turn']

                    # Write game record
                    log_file.write(json.dumps(game_record) + '\n')
                    log_file.flush()

                    print(f'Ep {ep}: {"Won" if reward > 0 else "Lost" if reward < 0 else "Draw"} '
                          f'in {post_state["turn"]} turns '
                          f'(vs {bot_names[args.bot_mode]}) '
                          f'W={wins} L={losses} D={draws}')

                    # Keep looping so render shows result overlay
                    game_over = False

        print(f'\nFinal: W={wins} L={losses} D={draws} / {wins+losses+draws} games')
    finally:
        log_file.close()
        env.close()
        print(f'Battle logs saved to {log_path}')


if __name__ == '__main__':
    main()
