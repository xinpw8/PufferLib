#!/usr/bin/env python3
import argparse
import os
import random

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from chess_stockfish_bridge import (
    STARTING_FEN,
    build_phase_examples,
    load_policy,
    repo_root,
    stockfish_limit,
)


def generate_engine_opening(engine, limit, max_prefix_plies, rng):
    board = chess.Board(STARTING_FEN)
    prefix_len = rng.randint(0, max_prefix_plies)
    for _ in range(prefix_len):
        if board.is_game_over():
            break
        board.push(engine.play(board, limit).move)
    return board


def generate_engine_trajectory(engine, limit, max_plies, random_opening_plies, rng):
    board = chess.Board(STARTING_FEN)
    warmup = rng.randint(0, random_opening_plies)
    for _ in range(warmup):
        if board.is_game_over():
            break
        legal_moves = list(board.legal_moves)
        board.push(rng.choice(legal_moves))

    examples = []
    for _ in range(max_plies):
        if board.is_game_over():
            break
        move = engine.play(board, limit).move
        examples.append(build_phase_examples(board, board.turn, move))
        board.push(move)
    return examples


def load_seed_positions(fens_file, requested):
    with open(fens_file, "r", encoding="utf-8") as handle:
        positions = [line.strip() for line in handle if line.strip()]
    if requested >= len(positions):
        return positions
    rng = random.Random(0)
    rng.shuffle(positions)
    return positions[:requested]


def build_dataset(engine, limit, fens_file, fen_positions, opening_positions,
                  max_prefix_plies, trajectory_games, trajectory_max_plies,
                  random_opening_plies):
    rng = random.Random(0)
    observations = []
    targets = []

    for fen in load_seed_positions(fens_file, fen_positions):
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        move = engine.play(board, limit).move
        phase0, piece_action, phase1, dest_action = build_phase_examples(board, board.turn, move)
        observations.extend([phase0, phase1])
        targets.extend([piece_action, dest_action])

    for _ in range(opening_positions):
        board = generate_engine_opening(engine, limit, max_prefix_plies, rng)
        if board.is_game_over():
            continue
        move = engine.play(board, limit).move
        phase0, piece_action, phase1, dest_action = build_phase_examples(board, board.turn, move)
        observations.extend([phase0, phase1])
        targets.extend([piece_action, dest_action])

    for _ in range(trajectory_games):
        for phase0, piece_action, phase1, dest_action in generate_engine_trajectory(
                engine, limit, trajectory_max_plies, random_opening_plies, rng):
            observations.extend([phase0, phase1])
            targets.extend([piece_action, dest_action])

    obs_tensor = torch.from_numpy(np.stack(observations))
    target_tensor = torch.tensor(targets, dtype=torch.long)
    return TensorDataset(obs_tensor, target_tensor)


def train_policy(model, dataset, device, epochs, batch_size, learning_rate):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    amp_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for obs, targets in loader:
            obs = obs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.startswith("cuda")):
                logits, _ = model(obs)
                loss = F.cross_entropy(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * targets.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_examples += targets.size(0)

        print(
            f"epoch={epoch + 1} loss={total_loss / total_examples:.4f} "
            f"acc={total_correct / total_examples:.4f}")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--load", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--fen-positions", type=int, default=10000)
    parser.add_argument("--opening-positions", type=int, default=4000)
    parser.add_argument("--max-prefix-plies", type=int, default=8)
    parser.add_argument("--trajectory-games", type=int, default=0)
    parser.add_argument("--trajectory-max-plies", type=int, default=24)
    parser.add_argument("--random-opening-plies", type=int, default=4)
    parser.add_argument("--stockfish-depth", type=int, default=None)
    parser.add_argument("--stockfish-movetime-ms", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.stockfish_path):
        raise FileNotFoundError(args.stockfish_path)

    model, _ = load_policy(args.load, device=args.device)
    limit = stockfish_limit(depth=args.stockfish_depth, movetime_ms=args.stockfish_movetime_ms)
    fens_file = os.path.join(repo_root(), "pufferlib", "ocean", "chess", "fens2.txt")

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    try:
        dataset = build_dataset(
            engine=engine,
            limit=limit,
            fens_file=fens_file,
            fen_positions=args.fen_positions,
            opening_positions=args.opening_positions,
            max_prefix_plies=args.max_prefix_plies,
            trajectory_games=args.trajectory_games,
            trajectory_max_plies=args.trajectory_max_plies,
            random_opening_plies=args.random_opening_plies,
        )
    finally:
        engine.quit()

    print(f"dataset_examples={len(dataset)}")
    model = train_policy(
        model=model,
        dataset=dataset,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
