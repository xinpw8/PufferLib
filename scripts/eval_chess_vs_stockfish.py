#!/usr/bin/env python3
import argparse
import os
import time

import chess
import chess.engine

from chess_stockfish_bridge import (
    STARTING_FEN,
    build_observation,
    count_repetitions,
    init_recurrent_state,
    legal_destinations_for_source,
    load_policy,
    select_action,
    stockfish_limit,
)


def play_game(model, engine, learner_color, device, mode, max_moves, limit, verbose=False):
    board = chess.Board(STARTING_FEN)
    state = init_recurrent_state(model, device)
    full_moves = 0

    while not board.is_game_over() and full_moves < max_moves:
        if board.turn == learner_color:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            repetition_count = count_repetitions(board)
            phase0 = build_observation(board, learner_color, 0, None, legal_moves, [], repetition_count)
            piece_action = select_action(model, phase0, state, device=device, mode=mode)

            chosen_source = piece_action ^ (56 if learner_color == chess.BLACK else 0)
            destinations = legal_destinations_for_source(legal_moves, chosen_source)
            if not destinations:
                piece_action = max(legal_moves, key=lambda mv: mv.from_square).from_square
                chosen_source = piece_action
                destinations = legal_destinations_for_source(legal_moves, chosen_source)

            phase1 = build_observation(
                board, learner_color, 1, chosen_source, legal_moves, destinations, repetition_count)
            dest_action = select_action(model, phase1, state, device=device, mode=mode)

            selected_move = None
            if dest_action >= 64:
                promo_row = (dest_action - 64) // 8
                file_idx = (dest_action - 64) % 8
                promotion = chess.QUEEN - promo_row
                for move in destinations:
                    if move.promotion == promotion and chess.square_file(move.to_square) == file_idx:
                        selected_move = move
                        break
            else:
                chosen_target = dest_action ^ (56 if learner_color == chess.BLACK else 0)
                for move in destinations:
                    if move.to_square == chosen_target:
                        selected_move = move
                        break

            if selected_move is None:
                selected_move = destinations[0]
            board.push(selected_move)
            if verbose:
                print(f"learner: {selected_move}")
        else:
            result = engine.play(board, limit)
            board.push(result.move)
            if verbose:
                print(f"stockfish: {result.move}")

        if board.turn == chess.WHITE:
            full_moves += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.5
    if outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == learner_color else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stockfish-path", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--stockfish-elo", type=int, default=1325)
    parser.add_argument("--stockfish-depth", type=int, default=None)
    parser.add_argument("--stockfish-movetime-ms", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    if not os.path.exists(args.stockfish_path):
        raise FileNotFoundError(args.stockfish_path)

    model, _ = load_policy(args.checkpoint, device=args.device)
    limit = stockfish_limit(depth=args.stockfish_depth, movetime_ms=args.stockfish_movetime_ms)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    engine.configure({
        "UCI_LimitStrength": True,
        "UCI_Elo": args.stockfish_elo,
    })

    wins = 0
    draws = 0
    losses = 0
    start = time.time()
    try:
        for idx in range(args.games):
            learner_color = chess.WHITE if idx % 2 == 0 else chess.BLACK
            result = play_game(
                model=model,
                engine=engine,
                learner_color=learner_color,
                device=args.device,
                mode=args.mode,
                max_moves=args.max_moves,
                limit=limit,
                verbose=args.verbose,
            )
            if result == 1.0:
                wins += 1
            elif result == 0.5:
                draws += 1
            else:
                losses += 1

            games = idx + 1
            win_rate = (wins + 0.5 * draws) / games
            elapsed = time.time() - start
            print(
                f"Game {games}/{args.games}: W={wins} D={draws} L={losses} "
                f"WR={win_rate:.3f} ({elapsed:.0f}s)")
    finally:
        engine.quit()

    total = wins + draws + losses
    win_rate = (wins + 0.5 * draws) / total if total else 0.0
    print("\n=== Final Results vs Stockfish ===")
    print(f"Games: {total}  W={wins} D={draws} L={losses}")
    print(f"Win rate: {win_rate:.3f}")


if __name__ == "__main__":
    main()
