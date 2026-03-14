"""
Centipawn Progress Monitor for Self-Play Training
==================================================
Measures whether a chess policy is learning during self-play by:
1. Loading the latest checkpoint and a recent historical checkpoint
2. Playing N games between them (current vs snapshot)
3. Evaluating terminal board positions with Stockfish (centipawn score)
4. Reporting whether the current policy consistently reaches better positions

Standalone usage:
  python centipawn_eval.py compare --current <ckpt> --opponent <ckpt> --num-games 50
  python centipawn_eval.py sweep --dir <experiment_dir>
  python centipawn_eval.py watch --dir <experiment_dir>  # auto-eval new checkpoints

Training integration (in pufferl.py):
  from centipawn_eval import CentipawnMonitor
  monitor = CentipawnMonitor(stockfish_path, experiment_dir, hidden_size=256)
  # In write_logs():
  cp_metrics = monitor.get_latest_metrics()
  if cp_metrics:
      logs.update(cp_metrics)
"""
import sys, os, argparse, glob, time, threading, statistics, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import chess.engine

STOCKFISH_PATH = "/home/spark-advantage/Stockfish/src/stockfish"

# ─── Observation layout (1082 per player, matches binding.h) ───
O_BOARD = 0        # 768 = 12 planes × 64 squares
O_SIDE = 768       # 2
O_CASTLE = 770     # 16
O_EP = 786         # 65
O_PHASE = 851      # 2
O_SELECTED = 853   # 64
O_VALID_PIECES = 917  # 64
O_VALID_DESTS = 981   # 64
O_VALID_PROMOS = 1045 # 32
O_SCALARS = 1077   # 5 (self_check, opp_check, rule50, repetition, pass_valid)
OBS_SIZE = 1082

PIECE_PLANE = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}


def board_to_obs(board, phase, selected_sq=None, learner_color=chess.WHITE):
    obs = np.zeros(OBS_SIZE, dtype=np.uint8)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            if learner_color == chess.BLACK:
                mapped_sq = chess.square_mirror(sq)
                color = not piece.color
            else:
                mapped_sq = sq
                color = piece.color
            plane = PIECE_PLANE[(piece.piece_type, color)]
            obs[O_BOARD + plane * 64 + mapped_sq] = 255

    side = board.turn
    if learner_color == chess.BLACK:
        side = not side
    obs[O_SIDE + (0 if side == chess.WHITE else 1)] = 255

    castle_idx = 0
    if learner_color == chess.WHITE:
        if board.has_kingside_castling_rights(chess.WHITE): castle_idx |= 1
        if board.has_queenside_castling_rights(chess.WHITE): castle_idx |= 2
        if board.has_kingside_castling_rights(chess.BLACK): castle_idx |= 4
        if board.has_queenside_castling_rights(chess.BLACK): castle_idx |= 8
    else:
        if board.has_kingside_castling_rights(chess.BLACK): castle_idx |= 1
        if board.has_queenside_castling_rights(chess.BLACK): castle_idx |= 2
        if board.has_kingside_castling_rights(chess.WHITE): castle_idx |= 4
        if board.has_queenside_castling_rights(chess.WHITE): castle_idx |= 8
    obs[O_CASTLE + castle_idx] = 255

    if board.ep_square is not None:
        ep = board.ep_square
        if learner_color == chess.BLACK:
            ep = chess.square_mirror(ep)
        obs[O_EP + ep] = 255
    else:
        obs[O_EP + 64] = 255

    obs[O_PHASE + phase] = 255

    if phase == 1 and selected_sq is not None:
        sq = selected_sq
        if learner_color == chess.BLACK:
            sq = chess.square_mirror(sq)
        obs[O_SELECTED + sq] = 255

    if phase == 0:
        for move in board.legal_moves:
            sq = move.from_square
            if learner_color == chess.BLACK:
                sq = chess.square_mirror(sq)
            obs[O_VALID_PIECES + sq] = 255

    if phase == 1 and selected_sq is not None:
        for move in board.legal_moves:
            if move.from_square == selected_sq:
                to_sq = move.to_square
                if learner_color == chess.BLACK:
                    to_sq = chess.square_mirror(to_sq)
                if move.promotion:
                    file_idx = chess.square_file(to_sq)
                    promo_type = move.promotion - 2
                    obs[O_VALID_PROMOS + promo_type * 8 + file_idx] = 255
                else:
                    obs[O_VALID_DESTS + to_sq] = 255

    if board.is_check():
        if board.turn == (chess.WHITE if learner_color == chess.WHITE else chess.BLACK):
            obs[O_SCALARS] = 255
        else:
            obs[O_SCALARS + 1] = 255
    obs[O_SCALARS + 2] = min(255, board.halfmove_clock)
    return obs


# ─── Architecture detection ───

def detect_architecture(state_dict):
    """Auto-detect checkpoint architecture from state_dict key names.

    Returns one of:
      'old_1x1'    -- original ChessSeven with 1x1 pointwise convolutions
      'chess_two'  -- ChessTwo with 4x conv3x3 (256ch), scalar_fc layers, 16 input channels
      'new_3x3'    -- revised ChessSeven with 3x3 full convolutions + skip connection
    """
    keys = set(state_dict.keys())

    # ChessTwo: has conv4 and scalar_fc layers, 16 spatial input channels
    has_conv4 = any('conv4' in k for k in keys)
    has_scalar_fc = any('scalar_fc' in k for k in keys)
    if has_conv4 and has_scalar_fc:
        return 'chess_two'

    # New ChessSeven 3x3: has skip connection, no conv4
    has_skip = any('skip' in k for k in keys)
    has_conv1 = any('conv1' in k for k in keys)
    if has_conv1 and has_skip and not has_conv4:
        return 'new_3x3'

    # Old ChessSeven 1x1: has square_embed
    if any('square_embed' in k for k in keys):
        return 'old_1x1'

    # Fallback for new_3x3 without skip (unlikely but safe)
    if has_conv1 and not has_conv4:
        return 'new_3x3'

    raise ValueError(f"Cannot detect architecture from keys: {sorted(keys)[:20]}")


# ─── Model: OLD architecture (1x1 pointwise convolutions) ───

class ChessEncoderOld(nn.Module):
    """Matches the original ChessSeven with 1x1 square_embed + channel_proj + depthwise spatial_mix."""
    SQUARE_DIM = 64
    PROJ_DIM = 8
    EMBED_DIM = 32
    SPATIAL_IN = 19  # 12 pieces + selected + valid_pieces + valid_dests + 4 geo

    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.square_embed = nn.Conv2d(self.SPATIAL_IN, self.SQUARE_DIM, 1)
        self.channel_proj = nn.Conv2d(self.SQUARE_DIM, self.PROJ_DIM, 1)
        self.spatial_mix = nn.Conv2d(self.PROJ_DIM, self.PROJ_DIM, 3, padding=1, groups=self.PROJ_DIM)
        self.side_embed = nn.Embedding(2, self.EMBED_DIM // 2)
        self.castle_embed = nn.Embedding(16, self.EMBED_DIM)
        self.ep_embed = nn.Embedding(65, self.EMBED_DIM)
        self.phase_embed = nn.Embedding(2, self.EMBED_DIM // 2)
        self.proj = nn.Linear(645, hidden_size)

        sqs = torch.arange(64, dtype=torch.float32)
        r = torch.div(sqs, 8, rounding_mode='floor')
        f = torch.fmod(sqs, 8)
        diag = (r + f) / 14.0
        anti = (r - f + 7) / 14.0
        cdist = (torch.where(r < 4, 3 - r, r - 4) + torch.where(f < 4, 3 - f, f - 4)) / 6.0
        sq_color = ((r + f).to(torch.int64) % 2).to(torch.float32)
        self.register_buffer('square_geo_planes',
            torch.stack([diag, anti, cdist, sq_color], 0).view(1, 4, 8, 8))

    def forward(self, x):
        B = x.shape[0]
        x = x.float()
        board = x[:, :768].view(B, 12, 8, 8)
        selected = x[:, 853:917].view(B, 1, 8, 8)
        valid_pieces = x[:, 917:981].view(B, 1, 8, 8)
        valid_dests = x[:, 981:1045].view(B, 1, 8, 8)
        geo = self.square_geo_planes.expand(B, -1, -1, -1)
        spatial = torch.cat([board, selected, valid_pieces, valid_dests, geo], dim=1)
        h = torch.relu(self.square_embed(spatial))
        h = torch.relu(self.channel_proj(h))
        h = h + torch.relu(self.spatial_mix(h))
        board_features = h.flatten(1)
        promos = (x[:, 1045:1077] > 0).float()
        side_f = self.side_embed(x[:, 768:770].argmax(1))
        castle_f = self.castle_embed(x[:, 770:786].argmax(1))
        ep_f = self.ep_embed(x[:, 786:851].argmax(1))
        phase_f = self.phase_embed(x[:, 851:853].argmax(1))
        scalars = x[:, 1077:1082] / 255.0
        features = torch.cat([board_features, promos, side_f, castle_f, ep_f, phase_f, scalars], 1)
        return torch.relu(self.proj(features))


class ChessPolicyOld(nn.Module):
    """Policy wrapper for old 1x1 architecture checkpoints."""
    def __init__(self, hidden_size=256, num_actions=98):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = ChessEncoderOld(hidden_size)
        self.rnn_weight = nn.Parameter(torch.zeros(3 * hidden_size, hidden_size))
        self.decoder = nn.Linear(hidden_size, num_actions, bias=False)

    def load_checkpoint(self, path, state_dict=None):
        if state_dict is None:
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_sd[k] = v
            elif k == 'decoder.linear.weight':
                new_sd['decoder.weight'] = v
            elif k == 'rnn.layer_0.weight':
                new_sd['rnn_weight'] = v
        self.load_state_dict(new_sd, strict=False)

    def forward(self, obs):
        h = self.encoder(obs)
        return self.decoder(h)


# ─── Model: NEW architecture (3x3 full convolutions with residual) ───

class ChessPolicyNew(nn.Module):
    """Policy for new 3x3 ChessSeven architecture.

    Matches the ChessSeven class in pufferlib/ocean/torch.py:
      conv1(19->32, 3x3) -> conv2(32->32, 3x3) + skip(19->32, 1x1) -> conv3(32->16, 3x3)
      -> flatten -> proj(total_features -> hidden_size) -> actor(hidden_size -> num_actions)
    """
    def __init__(self, hidden_size=256, num_actions=98, embed_dim=32):
        super().__init__()
        self.hidden_size = hidden_size

        sqs = torch.arange(64, dtype=torch.float32)
        r = torch.div(sqs, 8, rounding_mode='floor')
        f = torch.fmod(sqs, 8)
        diag = (r + f) / 14.0
        anti = (r - f + 7) / 14.0
        cdist = (torch.where(r < 4, 3 - r, r - 4) + torch.where(f < 4, 3 - f, f - 4)) / 6.0
        sq_color = ((r + f).to(torch.int64) % 2).to(torch.float32)
        self.register_buffer('square_geo_planes',
            torch.stack([diag, anti, cdist, sq_color], 0).view(1, 4, 8, 8))

        # 3x3 convolutional encoder (matches ChessSeven in torch.py)
        self.conv1 = nn.Conv2d(19, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(19, 32, kernel_size=1)  # residual projection
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.side_embed = nn.Embedding(2, embed_dim // 2)
        self.castle_embed = nn.Embedding(16, embed_dim)
        self.ep_embed = nn.Embedding(65, embed_dim)
        self.phase_embed = nn.Embedding(2, embed_dim // 2)

        # board_flat = 16*8*8 + 32(promos) = 1056
        # embeds = 3 * embed_dim = 96
        # scalars = 5
        board_flat = 16 * 8 * 8 + 32
        total_features = board_flat + (3 * embed_dim) + 5

        self.proj = nn.Sequential(
            nn.Linear(total_features, hidden_size),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def load_checkpoint(self, path, state_dict=None):
        if state_dict is None:
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
        # Checkpoints are saved with top-level keys (no 'encoder.' prefix)
        # because ChessSeven is the policy itself in training.
        # Our model has matching key names, so we can load directly.
        self.load_state_dict(state_dict, strict=False)

    def forward(self, obs):
        B = obs.shape[0]
        obs = obs.float()

        # Spatial features
        board = obs[:, :768].view(B, 12, 8, 8)
        selected = obs[:, 853:917].view(B, 1, 8, 8)
        valid_pieces = obs[:, 917:981].view(B, 1, 8, 8)
        valid_dests = obs[:, 981:1045].view(B, 1, 8, 8)
        geo = self.square_geo_planes.expand(B, -1, -1, -1)
        x = torch.cat([board, selected, valid_pieces, valid_dests, geo], dim=1)

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h) + self.skip(x))  # residual connection
        h = F.relu(self.conv3(h))
        board_features = h.flatten(1)

        promos = (obs[:, 1045:1077] > 0).float()
        side_f = self.side_embed(obs[:, 768:770].argmax(1))
        castle_f = self.castle_embed(obs[:, 770:786].argmax(1))
        ep_f = self.ep_embed(obs[:, 786:851].argmax(1))
        phase_f = self.phase_embed(obs[:, 851:853].argmax(1))
        scalars = obs[:, 1077:1082] / 255.0

        features = torch.cat([board_features, promos,
                               side_f, castle_f, ep_f, phase_f,
                               scalars], dim=1)
        h = self.proj(features)
        return self.actor(h)


# ─── Model: ChessTwo architecture (4x conv3x3, 256ch, scalar_fc layers) ───

class ChessPolicyTwo(nn.Module):
    """Policy for ChessTwo architecture.

    Matches checkpoints with encoder.conv1-conv4, encoder.scalar_fc1/fc2,
    and 16 spatial input channels (12 board + selected + valid_pieces + valid_dests + promos_padded).
    """
    def __init__(self, hidden_size=256, cnn_channels=256, num_actions=98, embed_dim=32):
        super().__init__()
        self.hidden_size = hidden_size

        self.encoder = ChessTwoEncoder(hidden_size, cnn_channels, embed_dim)
        self.rnn_weight = nn.Parameter(torch.zeros(3 * hidden_size, hidden_size))
        self.decoder = nn.Linear(hidden_size, num_actions, bias=False)

    def load_checkpoint(self, path, state_dict=None):
        if state_dict is None:
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_sd[k] = v
            elif k == 'decoder.linear.weight':
                new_sd['decoder.weight'] = v
            elif k == 'rnn.layer_0.weight':
                new_sd['rnn_weight'] = v
        self.load_state_dict(new_sd, strict=False)

    def forward(self, obs):
        h = self.encoder(obs)
        return self.decoder(h)


class ChessTwoEncoder(nn.Module):
    """Encoder matching ChessTwo's conv + scalar_fc architecture.

    Key differences from ChessSeven:
    - 16 spatial input channels (includes promos_padded as a spatial plane)
    - 4 conv layers (not 3), with residual between conv1 output and conv3 output
    - scalar_fc1 + scalar_fc2 instead of raw scalars concatenation
    - Embeddings use full embed_dim (not embed_dim//2 for side/phase)
    """
    def __init__(self, hidden_size=256, cnn_channels=256, embed_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(16, cnn_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(cnn_channels, hidden_size, kernel_size=3, stride=1, padding=1)

        self.side_embed = nn.Embedding(2, embed_dim)
        self.castle_embed = nn.Embedding(16, embed_dim)
        self.ep_embed = nn.Embedding(65, embed_dim)
        self.phase_embed = nn.Embedding(2, embed_dim)

        self.scalar_fc1 = nn.Linear(5, hidden_size)
        self.scalar_fc2 = nn.Linear(hidden_size, hidden_size)

        # cnn_flat = hidden_size * 8 * 8 = 16384
        # embeds = 4 * embed_dim = 128
        # scalar = hidden_size = 256
        # total = 16384 + 128 + 256 = 16768
        cnn_flat_size = hidden_size * 8 * 8
        total_features = cnn_flat_size + 4 * embed_dim + hidden_size
        self.proj = nn.Linear(total_features, hidden_size)

    def forward(self, obs):
        B = obs.shape[0]
        obs = obs.float()

        board = obs[:, :768].view(B, 12, 8, 8)
        selected_piece = obs[:, 853:917].view(B, 1, 8, 8)
        valid_pieces = obs[:, 917:981].view(B, 1, 8, 8)
        valid_dests = obs[:, 981:1045].view(B, 1, 8, 8)
        valid_promos = obs[:, 1045:1077].view(B, 1, 4, 8)
        valid_promos_padded = F.pad(valid_promos, (0, 0, 0, 4), value=0).view(B, 1, 8, 8)

        spatial_input = torch.cat([
            board, selected_piece, valid_pieces, valid_dests, valid_promos_padded
        ], dim=1)

        x = F.relu(self.conv1(spatial_input))
        residual = x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x + residual
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        spatial_features = x.flatten(1)

        side_f = self.side_embed(obs[:, 768:770].argmax(dim=1))
        castle_f = self.castle_embed(obs[:, 770:786].argmax(dim=1))
        ep_f = self.ep_embed(obs[:, 786:851].argmax(dim=1))
        phase_f = self.phase_embed(obs[:, 851:853].argmax(dim=1))

        scalars = torch.cat([
            obs[:, 1077:1078] / 255.0,
            obs[:, 1078:1079] / 255.0,
            obs[:, 1079:1080] / 255.0,
            obs[:, 1080:1081] / 255.0,
            obs[:, 1081:1082] / 255.0,
        ], dim=1)
        scalars = F.relu(self.scalar_fc1(scalars))
        scalars = F.relu(self.scalar_fc2(scalars))

        features = torch.cat([spatial_features, side_f, castle_f, ep_f, phase_f, scalars], dim=1)
        return F.relu(self.proj(features))


# ─── Backward-compatible alias ───
ChessEncoderPy = ChessEncoderOld
ChessPolicy = ChessPolicyOld


# ─── Game play ───

def get_model_move(model, board, learner_color, temperature=0.3):
    """Two-phase action: pick piece, then pick destination."""
    import random as rng

    obs = board_to_obs(board, phase=0, learner_color=learner_color)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(obs_t)[0]

    valid_from = {}
    for move in board.legal_moves:
        sq = move.from_square
        if learner_color == chess.BLACK:
            sq = chess.square_mirror(sq)
        if sq not in valid_from:
            valid_from[sq] = []
        valid_from[sq].append(move)

    if not valid_from:
        return None

    piece_logits = logits[:64].clone()
    mask = torch.full((64,), -1e8)
    for sq in valid_from:
        mask[sq] = 0
    piece_logits += mask
    if temperature > 0:
        probs = torch.softmax(piece_logits / temperature, dim=0)
        piece_sq = torch.multinomial(probs, 1).item()
    else:
        piece_sq = piece_logits.argmax().item()
    actual_from = piece_sq if learner_color == chess.WHITE else chess.square_mirror(piece_sq)

    if actual_from not in {m.from_square for m in board.legal_moves}:
        return rng.choice(list(board.legal_moves))

    obs = board_to_obs(board, phase=1, selected_sq=actual_from, learner_color=learner_color)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(obs_t)[0]

    dest_mask = torch.full((98,), -1e8)
    for move in board.legal_moves:
        if move.from_square == actual_from:
            if move.promotion:
                to_sq = move.to_square
                if learner_color == chess.BLACK:
                    to_sq = chess.square_mirror(to_sq)
                file_idx = chess.square_file(to_sq)
                promo_type = move.promotion - 2
                dest_mask[64 + promo_type * 8 + file_idx] = 0
            else:
                to_sq = move.to_square
                if learner_color == chess.BLACK:
                    to_sq = chess.square_mirror(to_sq)
                dest_mask[to_sq] = 0

    if dest_mask.max() < -1e7:
        return rng.choice(list(board.legal_moves))

    logits[:98] += dest_mask
    if temperature > 0:
        probs = torch.softmax(logits[:98] / temperature, dim=0)
        action = torch.multinomial(probs, 1).item()
    else:
        action = logits[:98].argmax().item()

    if action < 64:
        to_sq = action if learner_color == chess.WHITE else chess.square_mirror(action)
        move = chess.Move(actual_from, to_sq)
        if move not in board.legal_moves:
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                pm = chess.Move(actual_from, to_sq, promotion=promo)
                if pm in board.legal_moves:
                    return pm
            return rng.choice(list(board.legal_moves))
        return move
    elif action < 96:
        promo_idx = action - 64
        promo_type = promo_idx // 8
        file_idx = promo_idx % 8
        promo_piece = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][promo_type]
        for move in board.legal_moves:
            if move.from_square == actual_from and move.promotion == promo_piece:
                to_file = chess.square_file(move.to_square)
                if learner_color == chess.BLACK:
                    to_file = chess.square_file(chess.square_mirror(move.to_square))
                if to_file == file_idx:
                    return move
        return rng.choice(list(board.legal_moves))
    return rng.choice(list(board.legal_moves))


def evaluate_position_cp(engine, board, depth=12, clamp=2000):
    """Stockfish centipawn evaluation from White's perspective, clamped to +/-clamp."""
    if board.is_game_over():
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0
        return clamp if outcome.winner == chess.WHITE else -clamp
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info['score'].white()
        if score.is_mate():
            return clamp if score.mate() > 0 else -clamp
        return max(-clamp, min(clamp, score.score()))
    except Exception:
        return 0


def play_game_cp(model_a, model_b, engine, max_moves=200, eval_depth=12, temperature=0.3):
    """Play model_a (White) vs model_b (Black), return centipawn eval at end.
    
    For draw-by-repetition games: evaluates every position where new progress
    is made (halfmove_clock resets to 0 = a pawn push or capture happened),
    and returns the LAST such eval. This captures the position quality at
    the end of meaningful play, before models enter a repetition loop.
    
    For decisive games: evaluates the final position directly.
    """
    board = chess.Board()
    move_count = 0
    last_progress_cp = None  # CP at last non-repeating position

    while not board.is_game_over() and move_count < max_moves:
        # Track CP at positions where real progress is made
        # (halfmove_clock == 0 means a capture or pawn push just happened)
        if board.halfmove_clock == 0 and move_count > 0:
            last_progress_cp = evaluate_position_cp(engine, board, depth=eval_depth)

        if board.turn == chess.WHITE:
            move = get_model_move(model_a, board, chess.WHITE, temperature=temperature)
        else:
            move = get_model_move(model_b, board, chess.BLACK, temperature=temperature)

        if move is None or move not in board.legal_moves:
            import random
            move = random.choice(list(board.legal_moves))

        board.push(move)
        move_count += 1

    outcome = board.outcome(claim_draw=True)
    is_draw = outcome is None or outcome.winner is None

    if is_draw and last_progress_cp is not None:
        final_cp = last_progress_cp
    else:
        final_cp = evaluate_position_cp(engine, board, depth=eval_depth)

    if outcome is None or outcome.winner is None:
        result = 0
    elif outcome.winner == chess.WHITE:
        result = 1
    else:
        result = -1

    return {
        'final_cp': final_cp,
        'result': result,
        'num_moves': move_count,
        'final_fen': board.fen(),
        'used_progress_cp': is_draw and last_progress_cp is not None,
    }


# ─── Core evaluation ───

def load_model(checkpoint_path, hidden_size=256):
    """Load a checkpoint with auto-detected architecture.

    Inspects the state_dict keys to determine whether the checkpoint
    was saved from the old 1x1 architecture, ChessTwo, or the new 3x3
    ChessSeven architecture, then instantiates and loads the correct model.
    """
    sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    arch = detect_architecture(sd)

    if arch == 'chess_two':
        # Infer cnn_channels from conv1 weight shape
        conv1_key = next(k for k in sd if 'conv1.weight' in k)
        cnn_channels = sd[conv1_key].shape[0]
        proj_key = next(k for k in sd if k.endswith('proj.weight'))
        inferred_hidden = sd[proj_key].shape[0]
        model = ChessPolicyTwo(hidden_size=inferred_hidden, cnn_channels=cnn_channels)
        model.load_checkpoint(checkpoint_path, state_dict=sd)
    elif arch == 'new_3x3':
        model = ChessPolicyNew(hidden_size=hidden_size)
        model.load_checkpoint(checkpoint_path, state_dict=sd)
    else:
        model = ChessPolicyOld(hidden_size=hidden_size)
        model.load_checkpoint(checkpoint_path, state_dict=sd)

    model.eval()
    print(f"  [load_model] {os.path.basename(checkpoint_path)}: {arch} architecture, "
          f"hidden_size={hidden_size}")
    return model


def extract_epoch(path):
    base = os.path.basename(path)
    parts = base.replace('.pt', '').split('_')
    for part in reversed(parts):
        try:
            return int(part)
        except ValueError:
            continue
    return 0


def get_sorted_checkpoints(experiment_dir):
    files = glob.glob(os.path.join(experiment_dir, 'model_*.pt'))
    files.sort(key=extract_epoch)
    return files


def compare_checkpoints(ckpt_current, ckpt_opponent, num_games=50, eval_depth=12,
                        hidden_size=256, verbose=True):
    """Play current vs opponent, measure centipawn advantage."""
    model_current = load_model(ckpt_current, hidden_size)
    model_opponent = load_model(ckpt_opponent, hidden_size)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 4, "Hash": 256})

    current_cps = []
    results = []

    try:
        for i in range(num_games):
            if i % 2 == 0:
                game = play_game_cp(model_current, model_opponent, engine, eval_depth=eval_depth)
                cp_for_current = game['final_cp']
                res = game['result']
            else:
                game = play_game_cp(model_opponent, model_current, engine, eval_depth=eval_depth)
                cp_for_current = -game['final_cp']
                res = -game['result']

            current_cps.append(cp_for_current)
            results.append(res)

            if verbose:
                color = "W" if i % 2 == 0 else "B"
                res_str = {1: "WIN", -1: "LOSS", 0: "DRAW"}[results[-1]]
                print(f"  Game {i+1:3d}/{num_games} ({color}): {res_str:4s}  "
                      f"CP={cp_for_current:+6d}  moves={game['num_moves']}")
    finally:
        engine.quit()

    wins = sum(1 for r in results if r == 1)
    losses = sum(1 for r in results if r == -1)
    draws = sum(1 for r in results if r == 0)
    mean_cp = statistics.mean(current_cps) if current_cps else 0
    median_cp = statistics.median(current_cps) if current_cps else 0
    stdev_cp = statistics.stdev(current_cps) if len(current_cps) > 1 else 0

    return {
        'wins': wins, 'losses': losses, 'draws': draws,
        'mean_cp': mean_cp, 'median_cp': median_cp, 'stdev_cp': stdev_cp,
        'all_cps': current_cps, 'win_rate': wins / max(1, num_games),
    }


def sweep_checkpoints(experiment_dir, num_games=20, eval_depth=12, hidden_size=256):
    """Evaluate consecutive checkpoint pairs to show learning curve."""
    ckpts = get_sorted_checkpoints(experiment_dir)
    if len(ckpts) < 2:
        print(f"Need at least 2 checkpoints, found {len(ckpts)}")
        return

    print(f"Found {len(ckpts)} checkpoints in {experiment_dir}")
    n_pairs = min(10, len(ckpts) - 1)
    step = max(1, (len(ckpts) - 1) // n_pairs)
    pairs = [(ckpts[i], ckpts[i + step]) for i in range(0, len(ckpts) - step, step)]

    print(f"\n{'Opponent':>40s} | {'Current':>40s} | {'Mean CP':>8s} | {'Med CP':>8s} | {'Win%':>6s} | W/D/L")
    print("-" * 120)

    cp_trend = []
    cp_medians = []
    for ckpt_old, ckpt_new in pairs:
        old_name = os.path.basename(ckpt_old)
        new_name = os.path.basename(ckpt_new)

        results = compare_checkpoints(ckpt_new, ckpt_old, num_games=num_games,
                                      eval_depth=eval_depth, hidden_size=hidden_size,
                                      verbose=False)

        cp_trend.append(results['mean_cp'])
        cp_medians.append(results['median_cp'])
        print(f"{old_name:>40s} | {new_name:>40s} | {results['mean_cp']:+8.0f} | {results['median_cp']:+8.0f} | "
              f"{results['win_rate']*100:5.1f}% | {results['wins']}/{results['draws']}/{results['losses']}")

    print(f"\n{'='*60}")
    print(f"CP Trend (median): {' -> '.join(f'{cp:+.0f}' for cp in cp_medians)}")
    print(f"CP Trend (mean):   {' -> '.join(f'{cp:+.0f}' for cp in cp_trend)}")
    if len(cp_medians) >= 3:
        n = len(cp_medians)
        early = statistics.mean(cp_medians[:n//3])
        late = statistics.mean(cp_medians[-n//3:])
        print(f"Early median avg: {early:+.0f}  Late median avg: {late:+.0f}  Delta: {late-early:+.0f}")
        if late > early + 50:
            print("LEARNING: CP advantage increasing over training")
        elif late < early - 50:
            print("REGRESSION: CP advantage decreasing over training")
        else:
            print("FLAT: No clear learning signal (within noise margin)")


# ─── Background monitor for training integration ───

class CentipawnMonitor:
    """Background thread that periodically evaluates checkpoint pairs.

    Usage in pufferl.py:
        monitor = CentipawnMonitor(stockfish_path, experiment_dir)
        # In write_logs():
        cp_metrics = monitor.get_latest_metrics()
        if cp_metrics:
            logs.update(cp_metrics)
    """
    def __init__(self, stockfish_path, experiment_dir, hidden_size=256,
                 num_games=20, eval_depth=10, snapshot_gap=2,
                 eval_interval_epochs=50):
        self.stockfish_path = stockfish_path
        self.experiment_dir = experiment_dir
        self.hidden_size = hidden_size
        self.num_games = num_games
        self.eval_depth = eval_depth
        self.snapshot_gap = snapshot_gap
        self.eval_interval_epochs = eval_interval_epochs

        self._latest_metrics = {}
        self._metrics_history = []
        self._lock = threading.Lock()
        self._last_evaluated_ckpt = None
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        """Background loop: check for new checkpoints, evaluate."""
        while self._running:
            try:
                self._check_and_evaluate()
            except Exception as e:
                print(f"[CentipawnMonitor] Error: {e}")
            time.sleep(30)

    def _check_and_evaluate(self):
        ckpts = get_sorted_checkpoints(self.experiment_dir)
        if len(ckpts) < 2:
            return

        latest = ckpts[-1]
        if latest == self._last_evaluated_ckpt:
            return

        latest_epoch = extract_epoch(latest)
        # Only eval every N epochs
        if latest_epoch % self.eval_interval_epochs != 0:
            return

        gap = min(self.snapshot_gap, len(ckpts) - 1)
        snapshot = ckpts[-1 - gap]
        snapshot_epoch = extract_epoch(snapshot)

        if latest_epoch == snapshot_epoch:
            return

        print(f"[CentipawnMonitor] Evaluating epoch {latest_epoch} vs {snapshot_epoch} "
              f"({self.num_games} games, depth {self.eval_depth})...")

        model_current = load_model(latest, self.hidden_size)
        model_snapshot = load_model(snapshot, self.hidden_size)

        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        engine.configure({"Threads": 2, "Hash": 128})

        current_cps = []
        results_list = []
        try:
            for i in range(self.num_games):
                if i % 2 == 0:
                    game = play_game_cp(model_current, model_snapshot, engine,
                                       eval_depth=self.eval_depth)
                    cp = game['final_cp']
                    res = game['result']
                else:
                    game = play_game_cp(model_snapshot, model_current, engine,
                                       eval_depth=self.eval_depth)
                    cp = -game['final_cp']
                    res = -game['result']
                current_cps.append(cp)
                results_list.append(res)
        finally:
            engine.quit()

        mean_cp = statistics.mean(current_cps) if current_cps else 0
        wins = sum(1 for r in results_list if r == 1)
        draws = sum(1 for r in results_list if r == 0)
        losses = sum(1 for r in results_list if r == -1)
        wr = wins / max(1, self.num_games)
        pct_better = sum(1 for cp in current_cps if cp > 0) / max(1, len(current_cps))

        metrics = {
            'centipawn/mean_cp_advantage': mean_cp,
            'centipawn/win_rate_vs_snapshot': wr,
            'centipawn/pct_positions_better': pct_better,
            'centipawn/epoch_current': latest_epoch,
            'centipawn/epoch_snapshot': snapshot_epoch,
            'centipawn/wins': wins,
            'centipawn/draws': draws,
            'centipawn/losses': losses,
        }

        # Compute trend from history
        with self._lock:
            self._metrics_history.append({
                'epoch': latest_epoch,
                'mean_cp': mean_cp,
                'wr': wr,
            })
            if len(self._metrics_history) >= 3:
                recent = [h['mean_cp'] for h in self._metrics_history[-3:]]
                early = [h['mean_cp'] for h in self._metrics_history[:max(1, len(self._metrics_history)//3)]]
                metrics['centipawn/cp_trend_recent'] = statistics.mean(recent)
                metrics['centipawn/cp_trend_early'] = statistics.mean(early)
                metrics['centipawn/cp_trend_delta'] = statistics.mean(recent) - statistics.mean(early)

            self._latest_metrics = metrics
            self._last_evaluated_ckpt = latest

        learning = "YES" if mean_cp > 0 and pct_better > 0.55 else "NO" if mean_cp < -100 else "UNCLEAR"
        print(f"[CentipawnMonitor] epoch {latest_epoch} vs {snapshot_epoch}: "
              f"CP={mean_cp:+.0f}  WR={wr:.2f}  W/D/L={wins}/{draws}/{losses}  "
              f"Learning={learning}")

    def get_latest_metrics(self):
        """Thread-safe: return latest metrics dict for logging."""
        with self._lock:
            return dict(self._latest_metrics)

    def stop(self):
        self._running = False


# ─── Watch mode (standalone) ───

def watch_checkpoints(experiment_dir, num_games=20, eval_depth=12, hidden_size=256,
                      snapshot_gap=2, interval=60):
    """Continuously watch for new checkpoints and evaluate."""
    last_ckpt = None
    print(f"Watching {experiment_dir} for new checkpoints (every {interval}s)...")

    while True:
        ckpts = get_sorted_checkpoints(experiment_dir)
        latest = ckpts[-1] if ckpts else None

        if latest and latest != last_ckpt and len(ckpts) >= snapshot_gap + 1:
            gap = min(snapshot_gap, len(ckpts) - 1)
            snapshot = ckpts[-1 - gap]

            epoch_new = extract_epoch(latest)
            epoch_old = extract_epoch(snapshot)

            if epoch_new != epoch_old:
                print(f"\n{'='*60}")
                print(f"  New checkpoint: epoch {epoch_new} vs {epoch_old}")
                print(f"{'='*60}")

                results = compare_checkpoints(
                    latest, snapshot, num_games=num_games,
                    eval_depth=eval_depth, hidden_size=hidden_size, verbose=True)

                delta = results['mean_cp']
                learning = "YES" if delta > 0 and results['win_rate'] > 0.55 else \
                           "NO" if delta < -100 else "UNCLEAR"
                print(f"\n  Mean CP: {delta:+.0f}  WR: {results['win_rate']:.2f}  "
                      f"Learning: {learning}\n")

                last_ckpt = latest

        time.sleep(interval)


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(description="Centipawn progress monitor for selfplay")
    sub = parser.add_subparsers(dest='cmd')

    cmp = sub.add_parser('compare', help='Compare two checkpoints')
    cmp.add_argument('--current', required=True)
    cmp.add_argument('--opponent', required=True)
    cmp.add_argument('--num-games', type=int, default=50)
    cmp.add_argument('--eval-depth', type=int, default=12)
    cmp.add_argument('--hidden-size', type=int, default=256)

    sw = sub.add_parser('sweep', help='Sweep consecutive checkpoint pairs')
    sw.add_argument('--dir', required=True)
    sw.add_argument('--num-games', type=int, default=20)
    sw.add_argument('--eval-depth', type=int, default=12)
    sw.add_argument('--hidden-size', type=int, default=256)

    wt = sub.add_parser('watch', help='Watch for new checkpoints and evaluate')
    wt.add_argument('--dir', required=True)
    wt.add_argument('--num-games', type=int, default=20)
    wt.add_argument('--eval-depth', type=int, default=12)
    wt.add_argument('--hidden-size', type=int, default=256)
    wt.add_argument('--snapshot-gap', type=int, default=2)
    wt.add_argument('--interval', type=int, default=60)

    args = parser.parse_args()

    if args.cmd == 'compare':
        print(f"Current:  {args.current}")
        print(f"Opponent: {args.opponent}")
        results = compare_checkpoints(args.current, args.opponent,
                                      num_games=args.num_games,
                                      eval_depth=args.eval_depth,
                                      hidden_size=args.hidden_size)
        print(f"\n{'='*60}")
        print(f"  W/D/L: {results['wins']}/{results['draws']}/{results['losses']}  "
              f"WR: {results['win_rate']*100:.1f}%")
        print(f"  Mean CP: {results['mean_cp']:+.0f}  "
              f"Median CP: {results['median_cp']:+.0f}  "
              f"Stdev: {results['stdev_cp']:.0f}")
        if results['mean_cp'] > 100:
            print("  -> SIGNIFICANTLY STRONGER")
        elif results['mean_cp'] > 0:
            print("  -> Slightly stronger")
        elif results['mean_cp'] > -100:
            print("  -> Roughly equal")
        else:
            print("  -> WEAKER")

    elif args.cmd == 'sweep':
        sweep_checkpoints(args.dir, num_games=args.num_games,
                          eval_depth=args.eval_depth, hidden_size=args.hidden_size)

    elif args.cmd == 'watch':
        watch_checkpoints(args.dir, num_games=args.num_games,
                          eval_depth=args.eval_depth, hidden_size=args.hidden_size,
                          snapshot_gap=args.snapshot_gap, interval=args.interval)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
