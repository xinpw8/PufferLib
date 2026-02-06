"""
Usage example:
    python maia_eval.py --load-model-path model.pt
"""

import os
import shutil
import argparse
import glob
import os
import re
import matplotlib.pyplot as plt
import sys
import argparse
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

try:
    import chess
    import chess.engine
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False
    print("Warning: python-chess not installed. Run: pip install python-chess")

import pufferlib
import pufferlib.pytorch


# Constants (same as original)
PASS_ACTION = 96
OBS_SIZE = 1082

# Observation offsets
O_BOARD = 0
O_SIDE = 768
O_CASTLE = 770
O_EP = 786
O_PICK_PHASE = 851

_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def obs_to_board_fast(obs: np.ndarray, viewer_is_white: bool) -> chess.Board:
    """Convert observation to chess.Board from viewer's perspective."""
    board = chess.Board.empty()
    flip = 0 if viewer_is_white else 56

    viewer_color = chess.WHITE if viewer_is_white else chess.BLACK
    opp_color = not viewer_color

    # Pieces (planes 0-11)
    planes = obs[O_BOARD:O_BOARD + 12*64].reshape(12, 64)

    for i, pt in enumerate(_PIECE_TYPES):
        occ = np.flatnonzero(planes[i])
        for sq in occ:
            board.set_piece_at(int(sq) ^ flip, chess.Piece(pt, viewer_color))

    for i, pt in enumerate(_PIECE_TYPES, start=6):
        occ = np.flatnonzero(planes[i])
        for sq in occ:
            board.set_piece_at(int(sq) ^ flip, chess.Piece(pt, opp_color))

    # Turn
    board.turn = viewer_color if obs[O_SIDE] == 1 else opp_color

    castle_onehot = obs[O_CASTLE:O_CASTLE + 16]
    castle_idx = int(np.argmax(castle_onehot))
    
    # C code flips castling for Black viewer, flip it back to absolute
    if not viewer_is_white:
        flipped = 0
        if castle_idx & 1:  
            flipped |= 4
        if castle_idx & 2: 
            flipped |= 8
        if castle_idx & 4:
            flipped |= 1
        if castle_idx & 8:
            flipped |= 2
        castle_idx = flipped
    
    board.castling_rights = chess.BB_EMPTY
    if castle_idx & 1:  # WHITE_OO
        board.castling_rights |= chess.BB_H1
    if castle_idx & 2:  # WHITE_OOO
        board.castling_rights |= chess.BB_A1
    if castle_idx & 4:  # BLACK_OO
        board.castling_rights |= chess.BB_H8
    if castle_idx & 8:  # BLACK_OOO
        board.castling_rights |= chess.BB_A8

    # En passant (65 one-hot: 64 squares + 1 for none)
    ep_onehot = obs[O_EP:O_EP + 65]
    ep_idx = int(np.argmax(ep_onehot))
    
    if ep_idx < 64:
        # Flip back to absolute square if viewer is Black
        ep_sq_absolute = ep_idx ^ flip
        board.ep_square = ep_sq_absolute
    else:
        board.ep_square = None

    return board


@dataclass
class MaiaConfig:
    path: str = "./lc0"                     # Path to lc0 executable
    weights_path: str = "lc0/model_files/maia-1100.pb.gz"
    backend: Optional[str] = "cudnn"           # e.g., "cuda-auto", "blas", etc.
    time_limit: float = 0.1
    nodes_limit: Optional[int] = None       # If set, overrides time_limit
    threads: int = 2


class MaiaPlayer:
    def __init__(self, config: MaiaConfig):
        if not CHESS_AVAILABLE:
            raise ImportError("python-chess required: pip install python-chess")

        self.config = config

        if not shutil.which(config.path):  # Better check if lc0 is executable
            raise FileNotFoundError(f"LC0 not found at {config.path}. Check --lc0-path")

        cmd = [config.path]
        if config.weights_path:
            if not os.path.exists(config.weights_path):
                raise FileNotFoundError(f"Weights file not found: {config.weights_path}")
            cmd.append(f"--weights={config.weights_path}")
        if config.backend:
            cmd.append(f"--backend={config.backend}")

        print(f"Launching LC0 with command: {' '.join(cmd)}")

        self.engine = chess.engine.SimpleEngine.popen_uci(cmd)

        self.engine.configure({
            "Threads": config.threads,
        })

    def get_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over():
            return None
            
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        try:
            if self.config.nodes_limit is not None:
                limit = chess.engine.Limit(nodes=self.config.nodes_limit)
            else:
                limit = chess.engine.Limit(time=self.config.time_limit)
            result = self.engine.play(board, limit)
            
            # Validate the move
            if result.move and result.move in board.legal_moves:
                return result.move
            else:
                print(f"Maia returned invalid move, using random legal move")
                return legal_moves[0]
                
        except Exception as e:
            print(f"Maia error: {e}, using fallback")
            print(f"FEN: {board.fen()}")
            return legal_moves[0]
    '''def get_move(self, board: chess.Board) -> chess.Move:
        if self.config.nodes_limit is not None:
            limit = chess.engine.Limit(nodes=self.config.nodes_limit)
        else:
            limit = chess.engine.Limit(time=self.config.time_limit)
        result = self.engine.play(board, limit)
        return result.move
'''
    def close(self):
        if self.engine:
            self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# obs_to_board, get_pick_phase, is_my_turn, chess_move_to_actions unchanged from original
def obs_to_board(obs: np.ndarray, viewer_is_white: bool) -> chess.Board:
    board = chess.Board(fen=None)
    flip = 0 if viewer_is_white else 56
    viewer_color = chess.WHITE if viewer_is_white else chess.BLACK
    opponent_color = chess.BLACK if viewer_is_white else chess.WHITE
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for plane_idx, pt in enumerate(piece_types):
        for sq in range(64):
            if obs[O_BOARD + plane_idx * 64 + sq]:
                board.set_piece_at(sq ^ flip, chess.Piece(pt, viewer_color))
    for plane_idx, pt in enumerate(piece_types):
        for sq in range(64):
            if obs[O_BOARD + (6 + plane_idx) * 64 + sq]:
                board.set_piece_at(sq ^ flip, chess.Piece(pt, opponent_color))
    board.turn = viewer_color if obs[O_SIDE] == 1 else opponent_color
    return board


def get_pick_phase(obs: np.ndarray) -> int:
    return 1 if obs[O_PICK_PHASE + 1] == 1 else 0


def is_my_turn(obs: np.ndarray) -> bool:
    return obs[O_SIDE] == 1


def chess_move_to_actions(move: chess.Move, player_is_white: bool) -> Tuple[int, int]:
    flip = 0 if player_is_white else 56
    from_action = move.from_square ^ flip
    if move.promotion:
        promo_map = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
        to_action = 64 + promo_map.get(move.promotion, 0) * 8 + chess.square_file(move.to_square)
    else:
        to_action = move.to_square ^ flip
    return from_action, to_action


class VectorizedMaiaEval:
    def __init__(self, num_envs: int, maia: MaiaPlayer):
        self.num_envs = num_envs
        self.maia = maia
        self.pending_moves: List[Optional[chess.Move]] = [None] * num_envs

        self.boards: List[chess.Board] = [chess.Board.empty() for _ in range(num_envs)]

        self.learner_is_white_per_env: List[bool] = [False] * num_envs
        self.colors_detected: List[bool] = [False] * num_envs
        self.games_completed = 0
        self.policy_wins = 0
        self.maia_wins = 0
        self.draws = 0
        self.wins_as_white = 0
        self.wins_as_black = 0
        self.games_as_white = 0
        self.games_as_black = 0

    def get_actions(self, obs: np.ndarray, policy, device: str, state: dict) -> np.ndarray:
        num_envs = obs.shape[0]
        actions = np.full(num_envs * 2, PASS_ACTION, dtype=np.int32)

        learner_obs = obs[:, :OBS_SIZE]
        opponent_obs = obs[:, OBS_SIZE:]

        obs_tensor = torch.as_tensor(learner_obs).float().to(device)
        with torch.no_grad():
            logits, _ = policy.forward_eval(obs_tensor, state)
            policy_actions, _, _ = pufferlib.pytorch.sample_logits(logits)
            policy_actions = policy_actions.cpu().numpy()

        for env_idx in range(num_envs):
            if not self.colors_detected[env_idx]:
                # If learner is to move on first observation → learner is white
                self.learner_is_white_per_env[env_idx] = is_my_turn(learner_obs[env_idx])
                self.colors_detected[env_idx] = True

            maia_is_white = not self.learner_is_white_per_env[env_idx]

            learner_turn = is_my_turn(learner_obs[env_idx])
            opponent_turn = is_my_turn(opponent_obs[env_idx])

            learner_action_idx = env_idx * 2
            opponent_action_idx = env_idx * 2 + 1

            if learner_turn:
                actions[learner_action_idx] = policy_actions[env_idx]
                actions[opponent_action_idx] = PASS_ACTION

            elif opponent_turn:
                actions[learner_action_idx] = PASS_ACTION
                opp_phase = get_pick_phase(opponent_obs[env_idx])

                if opp_phase == 0:
                    board = obs_to_board_fast(opponent_obs[env_idx], maia_is_white)
                    try:
                        maia_move = self.maia.get_move(board)
                        self.pending_moves[env_idx] = maia_move
                        from_action, to_action = chess_move_to_actions(maia_move, maia_is_white)
                        actions[opponent_action_idx] = from_action
                    except Exception as e:
                        print(f"Maia error env {env_idx}: {e}")
                        print(f"FEN: {board.fen()}")
                        actions[opponent_action_idx] = PASS_ACTION

                else:
                    if self.pending_moves[env_idx] is not None:
                        maia_move = self.pending_moves[env_idx]
                        from_action, to_action = chess_move_to_actions(maia_move, maia_is_white)
                        actions[opponent_action_idx] = to_action
                        self.pending_moves[env_idx] = None
                    else:
                        actions[opponent_action_idx] = PASS_ACTION
            else:
                actions[learner_action_idx] = PASS_ACTION
                actions[opponent_action_idx] = PASS_ACTION

        return actions

    def process_results(self, rewards: np.ndarray, terminals: np.ndarray):
        for env_idx in range(len(terminals)):
            if terminals[env_idx]:
                was_white = self.learner_is_white_per_env[env_idx]
                if was_white:
                    self.games_as_white +=1
                else: 
                    self.games_as_black +=1
                self.pending_moves[env_idx] = None
                self.colors_detected[env_idx] = False
                self.games_completed += 1
                reward = rewards[env_idx]
                if reward >= 1.0:
                    self.policy_wins += 1
                    if was_white:
                        self.wins_as_white+=1
                    else:
                        self.wins_as_black+=1
                elif reward <= -1.0:
                    self.maia_wins += 1
                else:
                    self.draws += 1
                


    def get_stats(self) -> Dict:
        total = self.games_completed
        if total == 0:
            return {'games': 0, 'win_rate': 0, 'wins': 0, 'draws': 0, 'losses': 0}
        return {
            'games': total,
            'win_rate': (self.policy_wins + 0.5 * self.draws) / total,
            'wins': self.policy_wins,
            'draws': self.draws,
            'losses': self.maia_wins,
        }


def evaluate_vs_maia(
    policy,
    vecenv,
    maia_config: MaiaConfig,
    num_games: int = 4096,
    device: str = 'cuda',
    use_rnn: bool = False,
    verbose: bool = False,
    render: bool = False,
    render_delay: float = 0.5,
) -> Dict:
    num_envs = vecenv.num_envs
    print(f"Running evaluation with {num_envs} parallel environments")

    obs, _ = vecenv.reset()

    state = {}
    if use_rnn:
        state = {
            'lstm_h': torch.zeros(num_envs, policy.hidden_size, device=device),
            'lstm_c': torch.zeros(num_envs, policy.hidden_size, device=device),
        }

    with MaiaPlayer(maia_config) as maia:
        evaluator = VectorizedMaiaEval(
            num_envs=num_envs,
            maia=maia,
        )

        step = 0
        max_steps = (num_games // num_envs + 1) * 2000

        while evaluator.games_completed < num_games and step < max_steps:
            if render:
                vecenv.driver_env.render()
            actions = evaluator.get_actions(obs, policy, device, state)
            obs, rewards, terminals, truncations, infos = vecenv.step(actions)
            done_mask = terminals | truncations
            evaluator.process_results(rewards, done_mask)
            if use_rnn and done_mask.any():
                for env_idx in range(num_envs):
                    if done_mask[env_idx]:
                        state['lstm_h'][env_idx] = 0
                        state['lstm_c'][env_idx] = 0
            
            step += 1

            if step % 100 == 0 or evaluator.games_completed >= num_games:
                stats = evaluator.get_stats()
                if stats['games'] > 0:
                    nodes_str = f"nodes {maia_config.nodes_limit}" if maia_config.nodes_limit is not None else f"time {maia_config.time_limit}s"
                    print(f"Step {step}: {stats['games']}/{num_games} games ({nodes_str}), "
                          f"Win rate: {stats['win_rate']:.2%} "
                          f"(W:{stats['wins']}/D:{stats['draws']}/L:{stats['losses']})")

    return evaluator.get_stats()


def extract_epoch(filename):
    """Extract epoch number from filename like '00000.pt' or 'model_00123.pt'"""
    basename = os.path.basename(filename)
    match = re.search(r'(\d+)\.pt$', basename)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate chess policy against Maia (LC0) for multiple checkpoints')
    parser.add_argument('env_name', type=str, nargs='?', default='puffer_chess')
    parser.add_argument('--models-folder', type=str, required=True, help='Folder containing .pt model files')
    parser.add_argument('--lc0-path', type=str, default='./lc0')
    parser.add_argument('--weights-path', type=str, default='lc0/model_files/maia-1100.pb.gz')
    parser.add_argument('--backend', type=str, default=None, help='LC0 backend, e.g. cuda-auto')
    parser.add_argument('--nodes-limit', type=int, default=1, help='Fixed nodes per move (lower = weaker)')
    parser.add_argument('--time-limit', type=float, default=0.1, help='Time per move if nodes-limit omitted')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--num-games', type=int, default=4096)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--render-delay', type=float, default=0.3)
    parser.add_argument('--render-mode', type=str, default=None, choices=['human', 'ansi', 'rgb_array', 'raylib'])
    parser.add_argument('--output-plot', type=str, default='winrate_vs_epoch.png', help='Output plot filename')

    args = parser.parse_args()
    render = args.render_mode is not None

    try:
        from pufferlib.pufferl import load_config, load_env, load_policy
    except ImportError:
        from pufferl import load_config, load_env, load_policy

    # Find all .pt files in the folder
    pt_files = glob.glob(os.path.join(args.models_folder, '*.pt'))
    if not pt_files:
        print(f"No .pt files found in {args.models_folder}")
        return

    # Extract epochs and sort by epoch number
    model_data = []
    for pt_file in pt_files:
        epoch = extract_epoch(pt_file)
        if epoch is not None:
            model_data.append((epoch, pt_file))
        else:
            print(f"Warning: Could not extract epoch from {pt_file}, skipping")

    model_data.sort(key=lambda x: x[0])
    print(f"Found {len(model_data)} model files to evaluate")

    base_config = MaiaConfig(
        path=args.lc0_path,
        weights_path=args.weights_path,
        backend=args.backend,
        time_limit=args.time_limit,
        nodes_limit=args.nodes_limit,
        threads=args.threads,
    )

    epochs = []
    win_rates = []

    for epoch, model_path in model_data:
        print(f"\n{'='*50}")
        print(f"Evaluating epoch {epoch}: {model_path}")
        print(f"{'='*50}")

        # Load config and environment for each model
        pufferl_args = load_config(args.env_name)
        pufferl_args['load_model_path'] = model_path
        pufferl_args['train']['device'] = args.device
        pufferl_args['vec'] = dict(backend='Serial', num_envs=1)
        pufferl_args['env']['selfplay'] = 1

        if render:
            pufferl_args['render_mode'] = args.render_mode

        vecenv = load_env(args.env_name, pufferl_args)
        policy = load_policy(pufferl_args, vecenv, args.env_name)
        policy.eval()

        results = evaluate_vs_maia(
            policy=policy,
            vecenv=vecenv,
            maia_config=base_config,
            num_games=args.num_games,
            device=args.device,
            use_rnn=pufferl_args['train']['use_rnn'],
            verbose=args.verbose,
            render=render,
            render_delay=args.render_delay
        )

        epochs.append(epoch)
        win_rates.append(results['win_rate'])

        nodes_str = f"nodes {base_config.nodes_limit}" if base_config.nodes_limit is not None else f"time {base_config.time_limit}s"
        print(f"\nResults vs Maia ({nodes_str}) - Epoch {epoch}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"W/D/L: {results['wins']}/{results['draws']}/{results['losses']}")
        print(f"Total Games: {results['games']}")

        vecenv.close()
    import math
    
    def winrate_to_elo(winrate, opponent_elo=1100):
        """Convert winrate to estimated Elo given opponent's rating."""
        # Clamp winrate to avoid log(0) or division by zero
        winrate = max(0.001, min(0.999, winrate))
        elo_diff = -400 * math.log10((1 - winrate) / winrate)
        return opponent_elo + elo_diff
    
    estimated_elos = [winrate_to_elo(wr, 1100) for wr in win_rates]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    # Plot 1: Win Rate
    ax1.plot(epochs, win_rates, marker='o', linewidth=2, markersize=6, color='blue')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_title('Win Rate vs Maia by Training Epoch', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Plot 2: Estimated Elo
    ax2.plot(epochs, estimated_elos, marker='s', linewidth=2, markersize=6, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Estimated Elo', fontsize=12)
    ax2.set_title(f'Estimated Elo (vs Maia 1100 ) by Training Epoch', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1100, color='red', linestyle='--', alpha=0.7, label=f'Maia (1100)')
    ax2.legend()

    # Plot 3: Estimated Elo (Log x-axis)
    ax3.plot(epochs, estimated_elos, marker='s', linewidth=2, markersize=6, color='purple')
    ax3.set_xlabel('Epoch (log scale)', fontsize=12)
    ax3.set_ylabel('Estimated Elo', fontsize=12)
    ax3.set_title(f'Estimated Elo (vs Maia 1100) - Log Scale', fontsize=14)
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.axhline(y=1100, color='red', linestyle='--', alpha=0.7, label=f'Maia (1100)')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    print(f"\nPlot saved to {args.output_plot}")

    # Print summary table
    print(f"\n{'='*50}")
    print("Summary of Results")
    print(f"{'='*50}")
    print(f"{'Epoch':<10} {'Win Rate':<15} {'Est. Elo':<10}")
    print("-" * 35)
    for epoch, wr, elo in zip(epochs, win_rates, estimated_elos):
        print(f"{epoch:<10} {wr:.2%}          {elo:.0f}") 
if __name__ == '__main__':
    main()
