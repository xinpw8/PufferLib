#!/usr/bin/env python3
"""Evaluate Connect4 models against negamax and random opponents."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROWS, COLS = 6, 7

class MinGRUCell:
    """Minimal GRU inference cell matching C++ mingru_gate."""
    def __init__(self, weight, device):
        self.weight = weight.to(device)  # [3*H, H]
        self.hidden_size = weight.shape[0] // 3

    def forward(self, x, state):
        combined = x @ self.weight.T  # [B, 3*H]
        hidden, gate, proj = combined.chunk(3, dim=1)
        h = torch.where(hidden >= 0, hidden + 0.5, hidden.sigmoid())
        g = gate.sigmoid()
        mingru_out = torch.lerp(state, h, g)
        out = torch.sigmoid(proj) * mingru_out
        return out, mingru_out


class PufferPolicy:
    """Matches PufferLib 4.0 C++ model for inference."""
    def __init__(self, state_dict, device='cuda'):
        self.device = device
        self.enc_weight = state_dict['encoder.linear.weight'].to(device)  # [H, input]
        self.dec_weight = state_dict['decoder.linear.weight'].to(device)  # [out+1, H]
        self.hidden_size = self.enc_weight.shape[0]
        self.num_layers = sum(1 for k in state_dict if k.startswith('rnn.layer_'))
        self.rnn_layers = []
        for i in range(self.num_layers):
            w = state_dict[f'rnn.layer_{i}.weight']
            self.rnn_layers.append(MinGRUCell(w, device))
        self.act_size = self.dec_weight.shape[0] - 1  # last row is value head
        self.states = [torch.zeros(1, self.hidden_size, device=device) for _ in range(self.num_layers)]

    def reset_state(self):
        self.states = [torch.zeros(1, self.hidden_size, device=self.device) for _ in range(self.num_layers)]

    @torch.no_grad()
    def get_action(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Encoder
        h = x @ self.enc_weight.T  # [1, H]
        # MinGRU layers
        for i in range(self.num_layers):
            h, new_state = self.rnn_layers[i].forward(h, self.states[i])
            self.states[i] = new_state
        # Decoder
        out = h @ self.dec_weight.T  # [1, out+1]
        logits = out[0, :self.act_size]
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action


class Connect4Board:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.float32)
        self.heights = [0] * COLS
        self.current_player = 1
        self.num_moves = 0
        self.done = False
        self.winner = 0

    def can_play(self, col):
        return 0 <= col < COLS and self.heights[col] < ROWS

    def play(self, col):
        if not self.can_play(col):
            return False
        row = self.heights[col]
        self.board[row][col] = self.current_player
        self.heights[col] += 1
        self.num_moves += 1
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
        elif self.num_moves >= ROWS * COLS:
            self.done = True
            self.winner = 0
        self.current_player *= -1
        return True

    def _check_win(self, row, col):
        p = self.board[row][col]
        for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
            count = 1
            for d in [1, -1]:
                r, c = row + dr*d, col + dc*d
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == p:
                    count += 1
                    r += dr*d
                    c += dc*d
            if count >= 4:
                return True
        return False

    def get_obs(self, player):
        """42-float observation from player's perspective (column-major)."""
        obs = np.zeros(42, dtype=np.float32)
        for col in range(COLS):
            for row in range(ROWS):
                idx = col * ROWS + row
                if self.board[row][col] == player:
                    obs[idx] = 1.0
                elif self.board[row][col] == -player:
                    obs[idx] = -1.0
        return obs

    def valid_moves(self):
        return [c for c in range(COLS) if self.can_play(c)]


def negamax(board, depth, alpha, beta):
    if board.done:
        if board.winner == board.current_player * -1:
            return -(ROWS*COLS + 1 - board.num_moves) // 2
        return 0
    if depth <= 0:
        return 0
    valid = board.valid_moves()
    if not valid:
        return 0
    for col in valid:
        b2 = Connect4Board()
        b2.board = board.board.copy()
        b2.heights = board.heights.copy()
        b2.current_player = board.current_player
        b2.num_moves = board.num_moves
        b2.done = board.done
        b2.winner = board.winner
        b2.play(col)
        if b2.done and b2.winner == board.current_player:
            return (ROWS*COLS + 1 - board.num_moves) // 2
    best = -1000
    for col in valid:
        b2 = Connect4Board()
        b2.board = board.board.copy()
        b2.heights = board.heights.copy()
        b2.current_player = board.current_player
        b2.num_moves = board.num_moves
        b2.done = board.done
        b2.winner = board.winner
        b2.play(col)
        score = -negamax(b2, depth-1, -beta, -alpha)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def negamax_action(board, depth=3):
    valid = board.valid_moves()
    if not valid:
        return 0
    best_col = valid[0]
    best_score = -1000
    for col in valid:
        b2 = Connect4Board()
        b2.board = board.board.copy()
        b2.heights = board.heights.copy()
        b2.current_player = board.current_player
        b2.num_moves = board.num_moves
        b2.done = board.done
        b2.winner = board.winner
        b2.play(col)
        if b2.done and b2.winner == board.current_player:
            return col
        score = -negamax(b2, depth-1, -1000, 1000)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col


def play_game(p1_fn, p2_fn):
    board = Connect4Board()
    while not board.done:
        if board.current_player == 1:
            col = p1_fn(board)
        else:
            col = p2_fn(board)
        if not board.can_play(col):
            valid = board.valid_moves()
            col = valid[0] if valid else 0
        board.play(col)
    return board.winner


def evaluate(p1_name, p1_fn, p2_name, p2_fn, n_games=500):
    wins = {1: 0, -1: 0, 0: 0}
    for i in range(n_games):
        result = play_game(p1_fn, p2_fn)
        wins[result] += 1
    total = n_games
    print(f"  {p1_name} (P1) vs {p2_name} (P2): {n_games} games")
    print(f"    P1 wins: {wins[1]:4d} ({100*wins[1]/total:5.1f}%)")
    print(f"    P2 wins: {wins[-1]:4d} ({100*wins[-1]/total:5.1f}%)")
    print(f"    Draws:   {wins[0]:4d} ({100*wins[0]/total:5.1f}%)")
    return wins


def main():
    device = 'cuda'
    n_games = 1000

    selfplay_path = 'experiments/puffer_connect4/fgmc2wps.pt'
    baseline_path = 'experiments/puffer_connect4/37hl2sbn.pt'

    print("Loading models...")
    sp_sd = torch.load(selfplay_path, map_location='cpu')
    bl_sd = torch.load(baseline_path, map_location='cpu')
    sp_policy = PufferPolicy(sp_sd, device)
    bl_policy = PufferPolicy(bl_sd, device)

    def make_sp_fn(as_player):
        """Selfplay model: sees 42 obs from its perspective."""
        def fn(board):
            sp_policy.reset_state()
            obs = board.get_obs(board.current_player)
            return sp_policy.get_action(obs)
        return fn

    def make_bl_fn(as_player):
        """Baseline model: sees 84 obs (first 42 = own view, second 42 = zeros)."""
        def fn(board):
            bl_policy.reset_state()
            obs = np.zeros(84, dtype=np.float32)
            obs[:42] = board.get_obs(board.current_player)
            action = bl_policy.get_action(obs)
            return action % 7  # in case it outputs > 7
        return fn

    def random_fn(board):
        return np.random.choice(board.valid_moves())

    def negamax_fn(board):
        return negamax_action(board, depth=3)

    print("\n" + "="*60)
    print("CONNECT4 SELFPLAY VALIDATION RESULTS")
    print("="*60)

    print("\n--- 1. Selfplay Model vs Random ---")
    evaluate("Selfplay", make_sp_fn(1), "Random", random_fn, n_games)
    evaluate("Random", random_fn, "Selfplay", make_sp_fn(-1), n_games)

    print("\n--- 2. Baseline Model vs Random ---")
    evaluate("Baseline", make_bl_fn(1), "Random", random_fn, n_games)
    evaluate("Random", random_fn, "Baseline", make_bl_fn(-1), n_games)

    print("\n--- 3. Selfplay Model vs Negamax (depth 3) ---")
    evaluate("Selfplay", make_sp_fn(1), "Negamax", negamax_fn, n_games)
    evaluate("Negamax", negamax_fn, "Selfplay", make_sp_fn(-1), n_games)

    print("\n--- 4. Baseline Model vs Negamax (depth 3) ---")
    evaluate("Baseline", make_bl_fn(1), "Negamax", negamax_fn, n_games)
    evaluate("Negamax", negamax_fn, "Baseline", make_bl_fn(-1), n_games)

    print("\n--- 5. Selfplay Model vs Baseline Model ---")
    evaluate("Selfplay", make_sp_fn(1), "Baseline", make_bl_fn(-1), n_games)
    evaluate("Baseline", make_bl_fn(1), "Selfplay", make_sp_fn(-1), n_games)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
