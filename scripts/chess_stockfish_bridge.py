import ast
import configparser
import os
import sys
from types import SimpleNamespace

import chess
import chess.engine
import gymnasium
import numpy as np
import torch

SQ_FEATURES = 17
O_SQUARES = 0
O_VALID_PROMOS = 1088
O_SIDE = 1120
O_CASTLE = 1121
O_EP = 1122
O_PICK_PHASE = 1123
O_SELF_CHECK = 1124
O_OPP_CHECK = 1125
O_RULE50 = 1126
O_REPETITION = 1127
O_PASS_VALID = 1128
OBS_SIZE = 1129
PASS_ACTION = 96
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_chess_model_config():
    root = repo_root()
    parser = configparser.ConfigParser()
    parser.read([
        os.path.join(root, "pufferlib", "config", "default.ini"),
        os.path.join(root, "pufferlib", "config", "ocean", "chess.ini"),
    ])

    data = {}
    for section in ("base", "policy", "rnn"):
        values = {}
        for key, value in parser[section].items():
            try:
                values[key] = ast.literal_eval(value)
            except Exception:
                values[key] = value
        data[section] = values
    return data


def control_map(board, color):
    bb = 0
    for sq in chess.SQUARES:
        if board.is_attacked_by(color, sq):
            bb |= (1 << sq)
    return bb


def count_repetitions(board):
    key = board._transposition_key()
    count = 0
    history = board.copy(stack=True)
    while history.move_stack:
        history.pop()
        if history._transposition_key() == key:
            count += 1
            if count >= 2:
                return count
    return count


def build_observation(board, observer_color, pick_phase, selected_sq,
                      legal_moves, valid_destinations, repetition_count):
    obs = np.zeros(OBS_SIZE, dtype=np.uint8)
    flip = 56 if observer_color == chess.BLACK else 0
    us = observer_color
    them = not us

    valid_from_bb = 0
    valid_to_bb = 0
    if board.turn == us:
        marks = valid_destinations if pick_phase == 1 else legal_moves
        for move in marks:
            if pick_phase == 1:
                valid_to_bb |= (1 << move.to_square)
            else:
                valid_from_bb |= (1 << move.from_square)

    us_control = control_map(board, us)
    them_control = control_map(board, them)
    selected_bb = 0 if selected_sq is None else (1 << selected_sq)

    for sq in range(64):
        view_sq = sq ^ flip
        feat_offset = O_SQUARES + view_sq * SQ_FEATURES
        piece = board.piece_at(sq)
        if piece is not None:
            channel = (piece.piece_type - 1) if piece.color == us else (6 + piece.piece_type - 1)
            obs[feat_offset + channel] = 1
        bb = 1 << sq
        obs[feat_offset + 12] = 1 if (pick_phase == 1 and (selected_bb & bb)) else 0
        obs[feat_offset + 13] = 1 if (valid_from_bb & bb) else 0
        obs[feat_offset + 14] = 1 if (valid_to_bb & bb) else 0
        obs[feat_offset + 15] = 1 if (us_control & bb) else 0
        obs[feat_offset + 16] = 1 if (them_control & bb) else 0

    if pick_phase == 1:
        for move in valid_destinations:
            if move.promotion is not None:
                type_idx = chess.QUEEN - move.promotion
                file_idx = chess.square_file(move.to_square)
                obs[O_VALID_PROMOS + type_idx * 8 + file_idx] = 1

    obs[O_SIDE] = 0 if board.turn == us else 1

    castle_rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castle_rights |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castle_rights |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castle_rights |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castle_rights |= 8
    if observer_color == chess.BLACK:
        flipped = 0
        if castle_rights & 4:
            flipped |= 1
        if castle_rights & 8:
            flipped |= 2
        if castle_rights & 1:
            flipped |= 4
        if castle_rights & 2:
            flipped |= 8
        castle_rights = flipped
    obs[O_CASTLE] = castle_rights

    if board.ep_square is not None:
        obs[O_EP] = board.ep_square ^ flip
    else:
        obs[O_EP] = 64

    obs[O_PICK_PHASE] = pick_phase
    obs[O_SELF_CHECK] = 255 if board.is_check() and board.turn == us else 0
    obs[O_OPP_CHECK] = 255 if board.is_check() and board.turn == them else 0
    obs[O_RULE50] = min(255, int(board.halfmove_clock * 255 / 100))
    obs[O_REPETITION] = 0 if repetition_count >= 2 else (128 if repetition_count == 1 else 255)
    obs[O_PASS_VALID] = 255 if board.turn != us else 0
    return obs


def move_to_actions(move, observer_color):
    flip = 56 if observer_color == chess.BLACK else 0
    piece_action = move.from_square ^ flip
    if move.promotion is not None:
        promo_row = chess.QUEEN - move.promotion
        file_idx = chess.square_file(move.to_square)
        dest_action = 64 + promo_row * 8 + file_idx
    else:
        dest_action = move.to_square ^ flip
    return piece_action, dest_action


def actions_to_move(actions, legal_moves):
    piece_action, dest_action = actions
    matches = [move for move in legal_moves if move.from_square == piece_action]
    if dest_action >= 64:
        promo_row = (dest_action - 64) // 8
        file_idx = (dest_action - 64) % 8
        desired_promo = chess.QUEEN - promo_row
        for move in matches:
            if move.promotion == desired_promo and chess.square_file(move.to_square) == file_idx:
                return move
        return None

    for move in matches:
        if move.to_square == dest_action:
            return move
    return None


def legal_destinations_for_source(legal_moves, from_square):
    return [move for move in legal_moves if move.from_square == from_square]


def build_phase_examples(board, observer_color, move):
    legal_moves = list(board.legal_moves)
    repetition_count = count_repetitions(board)
    piece_action, dest_action = move_to_actions(move, observer_color)
    phase0 = build_observation(board, observer_color, 0, None, legal_moves, [], repetition_count)
    destinations = legal_destinations_for_source(legal_moves, move.from_square)
    phase1 = build_observation(
        board, observer_color, 1, move.from_square, legal_moves, destinations, repetition_count)
    return phase0, piece_action, phase1, dest_action


def make_mock_env():
    env = SimpleNamespace()
    env.single_action_space = gymnasium.spaces.Discrete(97)
    env.single_observation_space = gymnasium.spaces.Box(
        low=0, high=255, shape=(OBS_SIZE * 2,), dtype=np.uint8)
    env.selfplay = True
    return env


def load_policy(checkpoint_path=None, device="cpu"):
    root = repo_root()
    sys.path.insert(0, root)
    import pufferlib.models
    from pufferlib.ocean import torch as ocean_torch

    config = load_chess_model_config()
    env = make_mock_env()

    policy_cls = getattr(ocean_torch, config["base"]["policy_name"])
    model = policy_cls(env, **config["policy"])

    rnn_name = config["base"]["rnn_name"]
    if rnn_name is not None:
        rnn_cls = getattr(ocean_torch, rnn_name)
        model = rnn_cls(env, model, **config["rnn"])

    model.to(device)
    model.eval()

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    return model, config


def init_recurrent_state(model, device):
    if hasattr(model, "lstm"):
        return {
            "lstm_h": torch.zeros(1, model.hidden_size, device=device),
            "lstm_c": torch.zeros(1, model.hidden_size, device=device),
        }
    return {}


def select_action(model, obs, state, device="cpu", mode="greedy"):
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model.forward_eval(obs_t, state)
        if mode == "sample":
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()
        return torch.argmax(logits, dim=-1).item()


def stockfish_limit(depth=None, movetime_ms=None):
    if depth is not None:
        return chess.engine.Limit(depth=depth)
    if movetime_ms is not None:
        return chess.engine.Limit(time=movetime_ms / 1000.0)
    return chess.engine.Limit(time=0.01)
