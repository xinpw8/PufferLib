"""
Policy-vs-Policy Gold Standard Evaluation
==========================================
The definitive test for selfplay training: does the current policy beat
past snapshotted policies?

This script:
1. Finds all checkpoints in an experiment directory
2. Plays the latest checkpoint against selected historical snapshots
3. Reports win rate, centipawn advantage, and ELO-like relative strength
4. Produces a clear PASS/FAIL verdict

Usage:
  # Evaluate latest checkpoint against all historical snapshots
  python eval_policy_vs_policy.py --dir experiments/chess_selfplay/

  # Evaluate specific checkpoint vs all others
  python eval_policy_vs_policy.py --dir experiments/chess_selfplay/ --checkpoint model_500.pt

  # Quick test (fewer games per matchup)
  python eval_policy_vs_policy.py --dir experiments/chess_selfplay/ --quick

  # Full tournament (every checkpoint vs every other)
  python eval_policy_vs_policy.py --dir experiments/chess_selfplay/ --tournament

Gold Standard Criteria (PASS requires ALL):
  - Current beats earliest checkpoint with >60% win rate
  - Current beats checkpoint from 50% through training with >55% win rate
  - Mean CP advantage vs historical snapshots is positive
  - No regression: never loses >60% to any past snapshot
"""

import sys
import os
import argparse
import glob
import math
import statistics
import time
import json
from collections import defaultdict
from datetime import datetime

import torch

# Import from centipawn_eval (same directory)
from centipawn_eval import (
    load_model,
    play_game_cp,
    evaluate_position_cp,
    extract_epoch,
    get_sorted_checkpoints,
    STOCKFISH_PATH,
)
import chess
import chess.engine


# ─── ELO estimation ───

def expected_score(elo_a, elo_b):
    """Expected score for player A given ELO ratings."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def estimate_elo_from_winrate(win_rate, draw_rate=0.0):
    """Estimate ELO difference from observed win rate.

    Uses the standard ELO formula:
      score = wins + 0.5 * draws
      expected = 1 / (1 + 10^(-delta/400))
    """
    score = win_rate + 0.5 * draw_rate
    if score <= 0.001:
        return -800
    if score >= 0.999:
        return 800
    return -400 * math.log10(1.0 / score - 1.0)


# ─── Match evaluation ───

def play_match(model_a, model_b, engine, num_games=20, temperature=0.3,
               eval_depth=12, verbose=False):
    """Play a match between two models. Returns detailed results.

    model_a and model_b alternate colors each game for fairness.
    """
    cps_for_a = []
    results_for_a = []  # +1 win, 0 draw, -1 loss from model_a's perspective
    game_details = []

    for i in range(num_games):
        if i % 2 == 0:
            # model_a plays White
            game = play_game_cp(model_a, model_b, engine,
                               eval_depth=eval_depth, temperature=temperature)
            cp_for_a = game['final_cp']
            res_for_a = game['result']
        else:
            # model_a plays Black
            game = play_game_cp(model_b, model_a, engine,
                               eval_depth=eval_depth, temperature=temperature)
            cp_for_a = -game['final_cp']
            res_for_a = -game['result']

        cps_for_a.append(cp_for_a)
        results_for_a.append(res_for_a)

        game_details.append({
            'game_num': i + 1,
            'a_color': 'W' if i % 2 == 0 else 'B',
            'result_for_a': res_for_a,
            'cp_for_a': cp_for_a,
            'num_moves': game['num_moves'],
            'final_fen': game['final_fen'],
        })

        if verbose:
            color = "W" if i % 2 == 0 else "B"
            res_str = {1: "WIN", -1: "LOSS", 0: "DRAW"}[res_for_a]
            print(f"    Game {i+1:3d}/{num_games} ({color}): {res_str:4s}  "
                  f"CP={cp_for_a:+6d}  moves={game['num_moves']}")

    wins = sum(1 for r in results_for_a if r == 1)
    losses = sum(1 for r in results_for_a if r == -1)
    draws = sum(1 for r in results_for_a if r == 0)
    n = max(1, num_games)

    win_rate = wins / n
    loss_rate = losses / n
    draw_rate = draws / n
    mean_cp = statistics.mean(cps_for_a) if cps_for_a else 0
    median_cp = statistics.median(cps_for_a) if cps_for_a else 0
    stdev_cp = statistics.stdev(cps_for_a) if len(cps_for_a) > 1 else 0
    elo_diff = estimate_elo_from_winrate(win_rate, draw_rate)

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'draw_rate': draw_rate,
        'mean_cp': mean_cp,
        'median_cp': median_cp,
        'stdev_cp': stdev_cp,
        'elo_diff': elo_diff,
        'game_details': game_details,
    }


# ─── Checkpoint selection ───

def select_opponent_checkpoints(all_checkpoints, target_checkpoint, max_opponents=8):
    """Select a representative set of historical checkpoints to play against.

    Strategy:
    - Always include the earliest checkpoint (baseline)
    - Always include the most recent before target (immediate predecessor)
    - Sample evenly from the remaining checkpoints
    - Never include the target itself
    """
    if not all_checkpoints:
        return []

    # Filter out the target
    opponents = [c for c in all_checkpoints if c != target_checkpoint]
    if not opponents:
        return []

    if len(opponents) <= max_opponents:
        return opponents

    selected = set()

    # Always include earliest
    selected.add(opponents[0])

    # Always include latest (immediate predecessor)
    selected.add(opponents[-1])

    # Include checkpoint from ~25%, ~50%, ~75% through training
    for pct in [0.25, 0.5, 0.75]:
        idx = int(pct * (len(opponents) - 1))
        selected.add(opponents[idx])

    # Fill remaining slots evenly
    remaining = max_opponents - len(selected)
    if remaining > 0:
        step = max(1, len(opponents) // (remaining + 1))
        for i in range(step, len(opponents), step):
            if len(selected) >= max_opponents:
                break
            selected.add(opponents[i])

    # Sort by epoch
    result = sorted(selected, key=extract_epoch)
    return result


# ─── Main evaluation ───

def evaluate_policy(experiment_dir, target_checkpoint=None, num_games=20,
                    eval_depth=12, hidden_size=256, max_opponents=8,
                    temperature=0.3, verbose=True):
    """Run the gold-standard policy-vs-policy evaluation.

    Returns a dict with:
    - matchup_results: list of per-opponent results
    - overall_verdict: PASS/FAIL/INCONCLUSIVE
    - summary: human-readable summary string
    """
    all_ckpts = get_sorted_checkpoints(experiment_dir)
    if len(all_ckpts) < 2:
        print(f"ERROR: Need at least 2 checkpoints, found {len(all_ckpts)} in {experiment_dir}")
        return None

    # Determine target checkpoint
    if target_checkpoint:
        target_path = os.path.join(experiment_dir, target_checkpoint) \
            if not os.path.isabs(target_checkpoint) else target_checkpoint
        if target_path not in all_ckpts:
            # Try matching by basename
            target_path = next((c for c in all_ckpts
                                if os.path.basename(c) == os.path.basename(target_checkpoint)), None)
        if target_path is None:
            print(f"ERROR: Checkpoint {target_checkpoint} not found in {experiment_dir}")
            return None
    else:
        target_path = all_ckpts[-1]

    target_epoch = extract_epoch(target_path)
    print(f"\n{'='*70}")
    print(f"  POLICY-VS-POLICY GOLD STANDARD EVALUATION")
    print(f"  Target: {os.path.basename(target_path)} (epoch {target_epoch})")
    print(f"  Games per matchup: {num_games}  |  Eval depth: {eval_depth}")
    print(f"  Temperature: {temperature}")
    print(f"{'='*70}\n")

    # Select opponents
    opponents = select_opponent_checkpoints(all_ckpts, target_path, max_opponents)
    print(f"  Opponents ({len(opponents)}):")
    for opp in opponents:
        print(f"    - {os.path.basename(opp)} (epoch {extract_epoch(opp)})")
    print()

    # Load target model once
    print(f"  Loading target model...")
    model_target = load_model(target_path, hidden_size)

    # Start Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 4, "Hash": 256})

    matchup_results = []
    try:
        for i, opp_path in enumerate(opponents):
            opp_epoch = extract_epoch(opp_path)
            opp_name = os.path.basename(opp_path)
            print(f"\n  [{i+1}/{len(opponents)}] epoch {target_epoch} vs epoch {opp_epoch} "
                  f"({opp_name})")
            print(f"  {'-'*50}")

            model_opp = load_model(opp_path, hidden_size)

            result = play_match(
                model_target, model_opp, engine,
                num_games=num_games, temperature=temperature,
                eval_depth=eval_depth, verbose=verbose,
            )
            result['opponent_path'] = opp_path
            result['opponent_epoch'] = opp_epoch
            result['opponent_name'] = opp_name
            matchup_results.append(result)

            print(f"  Result: W/D/L = {result['wins']}/{result['draws']}/{result['losses']}  "
                  f"WR={result['win_rate']*100:.1f}%  "
                  f"CP={result['mean_cp']:+.0f}  "
                  f"ELO≈{result['elo_diff']:+.0f}")

            del model_opp
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    finally:
        engine.quit()

    # ─── Verdict ───
    verdict, details = compute_verdict(matchup_results, all_ckpts, target_path)

    # ─── Summary ───
    print(f"\n{'='*70}")
    print(f"  SUMMARY: epoch {target_epoch}")
    print(f"{'='*70}")
    print(f"\n  {'Opponent':>30s} | {'Epoch':>6s} | {'W/D/L':>8s} | {'WR%':>6s} | {'CP':>7s} | {'ELO':>6s}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}")

    for r in matchup_results:
        print(f"  {r['opponent_name']:>30s} | {r['opponent_epoch']:>6d} | "
              f"{r['wins']}/{r['draws']}/{r['losses']:>4s} | "
              f"{r['win_rate']*100:5.1f}% | {r['mean_cp']:+6.0f} | {r['elo_diff']:+5.0f}")

    # Overall stats
    all_wrs = [r['win_rate'] for r in matchup_results]
    all_cps = [r['mean_cp'] for r in matchup_results]
    overall_wr = statistics.mean(all_wrs) if all_wrs else 0
    overall_cp = statistics.mean(all_cps) if all_cps else 0

    print(f"\n  Overall avg WR: {overall_wr*100:.1f}%  |  Overall avg CP: {overall_cp:+.0f}")

    # Verdict
    verdict_color = {"PASS": "✅", "FAIL": "❌", "INCONCLUSIVE": "⚠️"}
    print(f"\n  {verdict_color.get(verdict, '❓')} VERDICT: {verdict}")
    for d in details:
        print(f"    {d}")

    print(f"\n{'='*70}\n")

    # Save results to JSON
    results_file = os.path.join(experiment_dir, f"pvp_eval_epoch_{target_epoch}.json")
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'target_checkpoint': target_path,
        'target_epoch': target_epoch,
        'num_games_per_matchup': num_games,
        'eval_depth': eval_depth,
        'temperature': temperature,
        'verdict': verdict,
        'verdict_details': details,
        'overall_win_rate': overall_wr,
        'overall_mean_cp': overall_cp,
        'matchups': [
            {
                'opponent': r['opponent_name'],
                'opponent_epoch': r['opponent_epoch'],
                'wins': r['wins'],
                'draws': r['draws'],
                'losses': r['losses'],
                'win_rate': r['win_rate'],
                'mean_cp': r['mean_cp'],
                'median_cp': r['median_cp'],
                'elo_diff': r['elo_diff'],
            }
            for r in matchup_results
        ],
    }
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved to: {results_file}")

    return {
        'matchup_results': matchup_results,
        'verdict': verdict,
        'verdict_details': details,
        'overall_win_rate': overall_wr,
        'overall_mean_cp': overall_cp,
    }


def compute_verdict(matchup_results, all_ckpts, target_path):
    """Compute PASS/FAIL/INCONCLUSIVE verdict based on gold standard criteria.

    PASS requires ALL of:
    1. Beats earliest checkpoint with >60% win rate
    2. Beats midpoint checkpoint with >55% win rate
    3. Mean CP advantage across all matchups is positive
    4. Never loses >60% to any past snapshot (no regression)

    FAIL if ANY of:
    1. Loses to earliest checkpoint
    2. Mean CP is negative across all matchups
    3. Regression detected (loses >60% to any snapshot)
    """
    if not matchup_results:
        return "INCONCLUSIVE", ["No matchup results"]

    details = []
    passed = True
    failed = False

    all_epochs = sorted([extract_epoch(c) for c in all_ckpts])
    target_epoch = extract_epoch(target_path)

    # Criterion 1: Beat earliest checkpoint
    earliest_result = matchup_results[0]  # opponents are sorted by epoch
    if earliest_result['opponent_epoch'] == all_epochs[0]:
        if earliest_result['win_rate'] > 0.60:
            details.append(f"✅ Beats earliest (epoch {earliest_result['opponent_epoch']}) "
                          f"with {earliest_result['win_rate']*100:.0f}% WR")
        elif earliest_result['win_rate'] > 0.50:
            details.append(f"⚠️ Only slightly beats earliest (epoch {earliest_result['opponent_epoch']}) "
                          f"with {earliest_result['win_rate']*100:.0f}% WR (need >60%)")
            passed = False
        else:
            details.append(f"❌ LOSES to earliest (epoch {earliest_result['opponent_epoch']}) "
                          f"with {earliest_result['win_rate']*100:.0f}% WR")
            passed = False
            failed = True

    # Criterion 2: Beat midpoint checkpoint
    mid_results = [r for r in matchup_results
                   if r['opponent_epoch'] >= all_epochs[len(all_epochs)//2 - 1]
                   and r['opponent_epoch'] <= all_epochs[len(all_epochs)//2 + 1]]
    if mid_results:
        mid_wr = statistics.mean([r['win_rate'] for r in mid_results])
        mid_epoch = mid_results[0]['opponent_epoch']
        if mid_wr > 0.55:
            details.append(f"✅ Beats midpoint (epoch ~{mid_epoch}) "
                          f"with {mid_wr*100:.0f}% WR")
        elif mid_wr > 0.45:
            details.append(f"⚠️ Roughly equal to midpoint (epoch ~{mid_epoch}) "
                          f"with {mid_wr*100:.0f}% WR (need >55%)")
            passed = False
        else:
            details.append(f"❌ LOSES to midpoint (epoch ~{mid_epoch}) "
                          f"with {mid_wr*100:.0f}% WR")
            passed = False
            failed = True

    # Criterion 3: Mean CP positive across all matchups
    overall_cp = statistics.mean([r['mean_cp'] for r in matchup_results])
    if overall_cp > 0:
        details.append(f"✅ Overall CP advantage is positive: {overall_cp:+.0f}")
    elif overall_cp > -50:
        details.append(f"⚠️ Overall CP is near zero: {overall_cp:+.0f} (need positive)")
        passed = False
    else:
        details.append(f"❌ Overall CP is NEGATIVE: {overall_cp:+.0f}")
        passed = False
        failed = True

    # Criterion 4: No regression (never lose >60% to any snapshot)
    regressions = [r for r in matchup_results if r['loss_rate'] > 0.60]
    if regressions:
        for r in regressions:
            details.append(f"❌ REGRESSION: Loses {r['loss_rate']*100:.0f}% to "
                          f"epoch {r['opponent_epoch']}")
        passed = False
        failed = True
    else:
        details.append(f"✅ No regression detected (never loses >60% to any snapshot)")

    # ELO trend
    elo_diffs = [r['elo_diff'] for r in matchup_results]
    if len(elo_diffs) >= 2:
        # Check if ELO advantage is increasing (later opponents harder)
        early_elo = statistics.mean(elo_diffs[:len(elo_diffs)//2])
        late_elo = statistics.mean(elo_diffs[len(elo_diffs)//2:])
        if late_elo < early_elo:
            details.append(f"  INFO: ELO advantage decreases for later opponents "
                          f"(early avg: {early_elo:+.0f}, late avg: {late_elo:+.0f}) — expected")
        else:
            details.append(f"  INFO: ELO advantage stable/increasing "
                          f"(early avg: {early_elo:+.0f}, late avg: {late_elo:+.0f})")

    if passed:
        return "PASS", details
    elif failed:
        return "FAIL", details
    else:
        return "INCONCLUSIVE", details


# ─── Tournament mode ───

def run_tournament(experiment_dir, num_games=10, eval_depth=10, hidden_size=256,
                   max_players=10, temperature=0.3):
    """Round-robin tournament: every selected checkpoint vs every other.

    Produces a full ELO table using iterative rating computation.
    """
    all_ckpts = get_sorted_checkpoints(experiment_dir)

    # Select evenly spaced checkpoints
    if len(all_ckpts) > max_players:
        step = max(1, (len(all_ckpts) - 1) // (max_players - 1))
        players = [all_ckpts[i] for i in range(0, len(all_ckpts), step)]
        if all_ckpts[-1] not in players:
            players.append(all_ckpts[-1])
    else:
        players = all_ckpts

    n = len(players)
    total_matches = n * (n - 1) // 2

    print(f"\n{'='*70}")
    print(f"  TOURNAMENT: {n} players, {total_matches} matches, {num_games} games each")
    print(f"{'='*70}\n")

    # Load all models
    models = {}
    for p in players:
        epoch = extract_epoch(p)
        print(f"  Loading epoch {epoch}...")
        models[p] = load_model(p, hidden_size)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 4, "Hash": 256})

    # Scores for ELO computation: scores[i][j] = score of player i vs player j
    scores = defaultdict(lambda: defaultdict(float))
    match_count = 0

    try:
        for i in range(n):
            for j in range(i + 1, n):
                match_count += 1
                epoch_i = extract_epoch(players[i])
                epoch_j = extract_epoch(players[j])
                print(f"\n  Match {match_count}/{total_matches}: "
                      f"epoch {epoch_i} vs epoch {epoch_j}")

                result = play_match(
                    models[players[i]], models[players[j]], engine,
                    num_games=num_games, temperature=temperature,
                    eval_depth=eval_depth, verbose=False,
                )

                score_i = result['wins'] + 0.5 * result['draws']
                score_j = result['losses'] + 0.5 * result['draws']
                scores[i][j] = score_i / max(1, num_games)
                scores[j][i] = score_j / max(1, num_games)

                print(f"    W/D/L = {result['wins']}/{result['draws']}/{result['losses']}  "
                      f"CP={result['mean_cp']:+.0f}")
    finally:
        engine.quit()

    # Compute ELO ratings via iterative method
    elos = [1500.0] * n  # Start everyone at 1500
    for iteration in range(100):
        new_elos = list(elos)
        for i in range(n):
            opponents = [j for j in range(n) if j != i and j in scores[i]]
            if not opponents:
                continue
            expected_total = sum(expected_score(elos[i], elos[j]) for j in opponents)
            actual_total = sum(scores[i][j] for j in opponents)
            k = 32
            new_elos[i] = elos[i] + k * (actual_total - expected_total)
        elos = new_elos

    # Normalize so earliest = 1000
    min_elo = min(elos)
    elos = [e - min_elo + 1000 for e in elos]

    # Print results table
    print(f"\n{'='*70}")
    print(f"  TOURNAMENT RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Rank':>4s}  {'Checkpoint':>30s}  {'Epoch':>6s}  {'ELO':>6s}  {'Δ from prev':>11s}")
    print(f"  {'-'*4}  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*11}")

    ranked = sorted(range(n), key=lambda i: elos[i], reverse=True)
    prev_elo = None
    for rank, idx in enumerate(ranked, 1):
        epoch = extract_epoch(players[idx])
        delta = f"{elos[idx] - prev_elo:+.0f}" if prev_elo is not None else ""
        print(f"  {rank:>4d}  {os.path.basename(players[idx]):>30s}  {epoch:>6d}  "
              f"{elos[idx]:>6.0f}  {delta:>11s}")
        prev_elo = elos[idx]

    # Save tournament results
    results_file = os.path.join(experiment_dir, "tournament_results.json")
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'num_games_per_match': num_games,
        'eval_depth': eval_depth,
        'players': [
            {
                'checkpoint': os.path.basename(players[i]),
                'epoch': extract_epoch(players[i]),
                'elo': elos[i],
            }
            for i in range(n)
        ],
    }
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {results_file}")

    # Check if ELO is monotonically increasing with epoch
    epoch_elo_pairs = [(extract_epoch(players[i]), elos[i]) for i in range(n)]
    epoch_elo_pairs.sort(key=lambda x: x[0])
    monotonic = all(epoch_elo_pairs[i][1] <= epoch_elo_pairs[i+1][1]
                    for i in range(len(epoch_elo_pairs) - 1))

    if monotonic:
        print(f"\n  ✅ ELO is monotonically increasing with training epoch — LEARNING CONFIRMED")
    else:
        inversions = []
        for i in range(len(epoch_elo_pairs) - 1):
            if epoch_elo_pairs[i][1] > epoch_elo_pairs[i+1][1]:
                inversions.append(
                    f"epoch {epoch_elo_pairs[i][0]} ({epoch_elo_pairs[i][1]:.0f}) > "
                    f"epoch {epoch_elo_pairs[i+1][0]} ({epoch_elo_pairs[i+1][1]:.0f})"
                )
        print(f"\n  ⚠️ ELO is NOT monotonic — {len(inversions)} inversion(s):")
        for inv in inversions:
            print(f"    {inv}")

    print()


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(
        description="Policy-vs-Policy Gold Standard Evaluation for Chess Selfplay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dir experiments/chess_selfplay/
  %(prog)s --dir experiments/chess_selfplay/ --quick
  %(prog)s --dir experiments/chess_selfplay/ --tournament
  %(prog)s --dir experiments/chess_selfplay/ --checkpoint model_500.pt --num-games 50
        """,
    )
    parser.add_argument('--dir', required=True,
                        help='Experiment directory containing model_*.pt checkpoints')
    parser.add_argument('--checkpoint', default=None,
                        help='Specific checkpoint to evaluate (default: latest)')
    parser.add_argument('--num-games', type=int, default=20,
                        help='Games per matchup (default: 20)')
    parser.add_argument('--eval-depth', type=int, default=12,
                        help='Stockfish evaluation depth (default: 12)')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Model hidden size (default: 256)')
    parser.add_argument('--max-opponents', type=int, default=8,
                        help='Max historical opponents to play against (default: 8)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Sampling temperature for move selection (default: 0.3)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 10 games, depth 8, 4 opponents')
    parser.add_argument('--tournament', action='store_true',
                        help='Round-robin tournament mode')
    parser.add_argument('--tournament-players', type=int, default=10,
                        help='Max players in tournament (default: 10)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Show individual game results')
    parser.add_argument('--quiet', action='store_true',
                        help='Only show summary, not individual games')

    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    if args.quick:
        args.num_games = 10
        args.eval_depth = 8
        args.max_opponents = 4
        print("  [Quick mode: 10 games, depth 8, 4 opponents]")

    if args.tournament:
        run_tournament(
            args.dir,
            num_games=args.num_games,
            eval_depth=args.eval_depth,
            hidden_size=args.hidden_size,
            max_players=args.tournament_players,
            temperature=args.temperature,
        )
    else:
        evaluate_policy(
            args.dir,
            target_checkpoint=args.checkpoint,
            num_games=args.num_games,
            eval_depth=args.eval_depth,
            hidden_size=args.hidden_size,
            max_opponents=args.max_opponents,
            temperature=args.temperature,
            verbose=args.verbose,
        )


if __name__ == '__main__':
    main()
