#!/usr/bin/env python3
"""Evaluate a 3.0 chess checkpoint against Stockfish using python-chess as bridge.

Since 3.0 has no native stockfish support, this script:
1. Loads the 3.0 ChessSeven+LSTM model
2. Uses python-chess to maintain the board and run stockfish
3. Constructs 3.0-format observations from python-chess board state
4. Runs the two-step pick-piece/pick-dest action cycle
"""

import argparse
import sys
import os
import time
import numpy as np
import torch
import chess
import chess.engine

# ── 3.0 observation constants ──
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

# Piece type mapping: chess.PAWN=1..chess.KING=6 matches the C code


def control_map(board, color):
    """Compute bitboard of all squares attacked by `color`."""
    bb = 0
    for sq in chess.SQUARES:
        if board.is_attacked_by(color, sq):
            bb |= (1 << sq)
    return bb


def build_observation(board, observer_color, pick_phase, selected_sq,
                      legal_moves, valid_destinations, repetition_count):
    """Build a 1129-byte observation in the 3.0 format."""
    obs = np.zeros(OBS_SIZE, dtype=np.uint8)
    flip = 56 if observer_color == chess.BLACK else 0
    us = observer_color
    them = not us

    # Compute valid source/dest bitboards
    valid_from_bb = 0
    valid_to_bb = 0
    if board.turn == us:
        if pick_phase == 1:
            for m in valid_destinations:
                valid_to_bb |= (1 << m.to_square)
        else:
            for m in legal_moves:
                valid_from_bb |= (1 << m.from_square)

    us_control = control_map(board, us)
    them_control = control_map(board, them)

    selected_bb = (1 << selected_sq) if (pick_phase == 1 and selected_sq is not None) else 0

    # Encode squares
    for sq in range(64):
        view_sq = sq ^ flip
        feat_offset = O_SQUARES + view_sq * SQ_FEATURES
        piece = board.piece_at(sq)
        if piece is not None:
            pt = piece.piece_type  # 1-6
            c = piece.color
            channel = (pt - 1) if (c == us) else (6 + pt - 1)
            obs[feat_offset + channel] = 1
        bb = 1 << sq
        obs[feat_offset + 12] = 1 if (selected_bb & bb) else 0
        obs[feat_offset + 13] = 1 if (valid_from_bb & bb) else 0
        obs[feat_offset + 14] = 1 if (valid_to_bb & bb) else 0
        obs[feat_offset + 15] = 1 if (us_control & bb) else 0
        obs[feat_offset + 16] = 1 if (them_control & bb) else 0

    # Valid promotions
    if pick_phase == 1 and valid_destinations:
        for m in valid_destinations:
            if m.promotion is not None:
                type_idx = chess.QUEEN - m.promotion  # Q=0, R=1, B=2, N=3
                file_idx = chess.square_file(m.to_square)
                obs[O_VALID_PROMOS + type_idx * 8 + file_idx] = 1

    # Scalar features
    obs[O_SIDE] = 0 if (board.turn == us) else 1

    # Castling rights
    cr = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        cr |= 1  # WHITE_OO
    if board.has_queenside_castling_rights(chess.WHITE):
        cr |= 2  # WHITE_OOO
    if board.has_kingside_castling_rights(chess.BLACK):
        cr |= 4  # BLACK_OO
    if board.has_queenside_castling_rights(chess.BLACK):
        cr |= 8  # BLACK_OOO
    if observer_color == chess.BLACK:
        flipped = 0
        if cr & 4: flipped |= 1
        if cr & 8: flipped |= 2
        if cr & 1: flipped |= 4
        if cr & 2: flipped |= 8
        cr = flipped
    obs[O_CASTLE] = cr

    # En passant
    if board.ep_square is not None:
        ep = board.ep_square
        if observer_color == chess.BLACK:
            ep ^= 56
        obs[O_EP] = ep
    else:
        obs[O_EP] = 64

    obs[O_PICK_PHASE] = pick_phase
    obs[O_SELF_CHECK] = 255 if board.is_check() and board.turn == us else 0
    obs[O_OPP_CHECK] = 255 if board.is_check() and board.turn == them else 0
    obs[O_RULE50] = min(255, int(board.halfmove_clock * 255 / 100))
    obs[O_PASS_VALID] = 255 if (board.turn != us) else 0

    # Repetition
    if repetition_count >= 2:
        obs[O_REPETITION] = 0
    elif repetition_count == 1:
        obs[O_REPETITION] = 128
    else:
        obs[O_REPETITION] = 255

    return obs


def action_to_square(action, observer_color):
    """Convert action index (0-63) to board square, accounting for flip."""
    if observer_color == chess.BLACK:
        return action ^ 56
    return action


def move_to_actions(move, observer_color, legal_from_sq):
    """Convert a chess.Move to (piece_action, dest_action) pair."""
    from_sq = move.from_square
    to_sq = move.to_square
    flip = 56 if observer_color == chess.BLACK else 0

    piece_action = from_sq ^ flip
    if move.promotion is not None:
        promo_row = chess.QUEEN - move.promotion  # Q=0,R=1,B=2,N=3
        file_idx = chess.square_file(to_sq)
        dest_action = 64 + promo_row * 8 + file_idx
    else:
        dest_action = to_sq ^ flip

    return piece_action, dest_action


def count_repetitions(board):
    """Count how many times the current position has occurred before."""
    key = board._transposition_key()
    count = 0
    b = board.copy()
    while b.move_stack:
        b.pop()
        if b._transposition_key() == key:
            count += 1
            if count >= 2:
                return count
    return count


def load_3p0_model(checkpoint_path, device="cpu"):
    """Load the 3.0 ChessSeven + LSTMWrapper model."""
    # Add 3.0 to path
    sys.path.insert(0, "/home/spark-advantage/pufferlib-3.0-chess")
    import pufferlib
    import pufferlib.models
    from pufferlib.ocean.torch import ChessSeven

    # Create a mock env to initialize the policy
    class MockEnv:
        class SingleActionSpace:
            n = 97
        class SingleObsSpace:
            shape = (OBS_SIZE * 2,)
        single_action_space = SingleActionSpace()
        single_observation_space = SingleObsSpace()
        selfplay = True

    env = MockEnv()
    policy = ChessSeven(env, square_dim=64, proj_dim=8, hidden_size=256,
                        embed_dim=64, use_action_masking=1)
    model = pufferlib.models.LSTMWrapper(env, policy, input_size=256, hidden_size=256)

    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def model_select_action(model, obs, state, device="cpu"):
    """Run the model on a single observation and sample an action."""
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        # Use the full model (LSTMWrapper) to include LSTM state
        logits, _ = model.forward_eval(obs_t, state)
        # Get action mask set by ChessSeven.encode_observations
        mask = model.policy.current_mask
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e8)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
    return action


def play_game(model, engine, stockfish_elo, stockfish_movetime_ms,
              learner_color, device="cpu", max_moves=500, verbose=False):
    """Play one game. Returns 1.0 for learner win, 0.0 for loss, 0.5 for draw."""
    board = chess.Board()

    # LSTM state
    state = {
        "lstm_h": torch.zeros(1, 256, device=device),
        "lstm_c": torch.zeros(1, 256, device=device),
    }

    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        if board.turn == learner_color:
            # Model's turn - two-step action
            legal = list(board.legal_moves)
            if not legal:
                break
            rep_count = count_repetitions(board)

            # Phase 0: pick piece
            obs = build_observation(board, learner_color, 0, None,
                                    legal, [], rep_count)
            piece_action = model_select_action(model, obs, state, device)

            if piece_action >= 64:
                # Invalid piece selection, try random legal
                move = np.random.choice(legal)
                board.push(move)
                move_count += 1
                if verbose:
                    print(f"  Model ({'W' if learner_color == chess.WHITE else 'B'}) invalid piece, random: {move}")
                continue

            from_sq = action_to_square(piece_action, learner_color)

            # Find legal moves from this square
            valid_dests = [m for m in legal if m.from_square == from_sq]
            if not valid_dests:
                # Invalid piece (no legal moves from that square), try random
                move = np.random.choice(legal)
                board.push(move)
                move_count += 1
                if verbose:
                    print(f"  Model ({'W' if learner_color == chess.WHITE else 'B'}) no moves from {chess.square_name(from_sq)}, random: {move}")
                continue

            # Phase 1: pick destination
            obs = build_observation(board, learner_color, 1, from_sq,
                                    legal, valid_dests, rep_count)
            dest_action = model_select_action(model, obs, state, device)

            # Decode dest action
            chosen_move = None
            if dest_action < 64:
                to_sq = action_to_square(dest_action, learner_color)
                for m in valid_dests:
                    if m.to_square == to_sq and m.promotion is None:
                        chosen_move = m
                        break
                # If no exact match, try with promotion (auto-queen)
                if chosen_move is None:
                    for m in valid_dests:
                        if m.to_square == to_sq:
                            chosen_move = m
                            break
            elif dest_action < 96:
                # Promotion
                promo_idx = dest_action - 64
                promo_row = promo_idx // 8
                file_idx = promo_idx % 8
                desired_promo = chess.QUEEN - promo_row
                for m in valid_dests:
                    if m.promotion == desired_promo and chess.square_file(m.to_square) == file_idx:
                        chosen_move = m
                        break

            if chosen_move is None:
                move = np.random.choice(legal)
                board.push(move)
                move_count += 1
                if verbose:
                    print(f"  Model ({'W' if learner_color == chess.WHITE else 'B'}) invalid dest, random: {move}")
                continue

            board.push(chosen_move)
            move_count += 1
            if verbose:
                print(f"  Model ({'W' if learner_color == chess.WHITE else 'B'}): {chosen_move}")
        else:
            # Stockfish's turn
            result = engine.play(board, chess.engine.Limit(time=stockfish_movetime_ms / 1000.0))
            board.push(result.move)
            move_count += 1
            if verbose:
                print(f"  Stockfish ({'W' if learner_color == chess.BLACK else 'B'}): {result.move}")

    # Determine result
    if board.is_game_over():
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.5
        return 1.0 if outcome.winner == learner_color else 0.0
    else:
        return 0.5  # Timeout = draw


def main():
    parser = argparse.ArgumentParser(description="Eval 3.0 chess checkpoint vs Stockfish")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--stockfish-elo", type=int, default=800)
    parser.add_argument("--stockfish-movetime-ms", type=int, default=50)
    parser.add_argument("--stockfish-path", type=str, default="/usr/games/stockfish")
    parser.add_argument("--max-moves", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = load_3p0_model(args.checkpoint, args.device)
    print("Model loaded.")

    print(f"Starting Stockfish at ELO {args.stockfish_elo}...")
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    # Stockfish 16 min UCI_Elo is 1320. For lower, use Skill Level.
    if args.stockfish_elo >= 1320:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.stockfish_elo})
    else:
        # Map rough ELO to Skill Level (0-20). Level 0 ≈ 1000-1100 ELO.
        # Use level 0 + very short movetime for weakest play.
        skill = max(0, min(20, (args.stockfish_elo - 600) // 50))
        print(f"  ELO {args.stockfish_elo} < 1320, using Skill Level {skill}")
        engine.configure({"Skill Level": skill})

    wins, draws, losses = 0, 0, 0
    t0 = time.time()

    for g in range(args.games):
        learner_color = chess.WHITE if g % 2 == 0 else chess.BLACK
        result = play_game(model, engine, args.stockfish_elo,
                           args.stockfish_movetime_ms, learner_color,
                           args.device, args.max_moves, args.verbose)
        if result == 1.0:
            wins += 1
        elif result == 0.0:
            losses += 1
        else:
            draws += 1

        total = g + 1
        elapsed = time.time() - t0
        wr = wins / total
        print(f"Game {total}/{args.games}: "
              f"W={wins} D={draws} L={losses} "
              f"WR={wr:.3f} "
              f"({elapsed:.0f}s, {elapsed/total:.1f}s/game)")

    engine.quit()
    total = wins + draws + losses
    print(f"\n=== Final Results vs Stockfish ELO {args.stockfish_elo} ===")
    print(f"Games: {total}  W={wins} D={draws} L={losses}")
    print(f"Win rate: {wins/total:.3f}  Draw rate: {draws/total:.3f}")


if __name__ == "__main__":
    main()
