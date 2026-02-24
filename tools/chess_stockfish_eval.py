#!/usr/bin/env python3
"""Evaluate a chess checkpoint against Stockfish via the native 4.0 chess env.

This script uses the C++ backend path (PuffeRL + static env binding) and requires
`stockfish_bot=1` support in `pufferlib/ocean/chess/chess.h`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

import pufferlib._C as _C
from pufferlib.pufferl import PuffeRL, load_config


@dataclass
class EvalSummary:
    games: int
    wins: float
    draws: float
    losses: float
    win_rate: float
    draw_rate: float
    loss_rate: float
    stockfish_elo: int
    stockfish_movetime_ms: int
    stockfish_depth: int
    stockfish_random_pct: int
    stockfish_query_pct: int
    total_agents: int
    duration_sec: float
    model_path: str | None


def _safe_load_config(env_name: str):
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        return load_config(env_name)
    finally:
        sys.argv = saved_argv


def _load_checkpoint(pufferl_cpp, model_path: str) -> None:
    """Load a checkpoint into both fp32 (optimizer) and bf16 (inference) policies.

    PufferLib 4.0 maintains two policy copies when USE_BF16=true:
      - policy_fp32: master weights, backed by muon.weight_buffer (params are views)
      - policy_bf16: inference weights, a SEPARATE model in bfloat16

    Rollouts use policy_bf16 for forward passes. The sync from fp32 -> bf16 only
    happens inside _C.train(). So for eval-only mode, we must explicitly sync
    after loading into the muon weight buffer.

    BUG HISTORY: Prior to 2026-02-23, this function only wrote to muon.weight_buffer,
    which updated policy_fp32 (via views) but left policy_bf16 with random init weights.
    All previous Stockfish eval results were evaluating RANDOM WEIGHTS.
    """
    state_dict = torch.load(model_path, map_location="cpu")
    wb = pufferl_cpp.muon.weight_buffer
    offset = 0
    with torch.no_grad():
        for name, param in pufferl_cpp.policy_fp32.named_parameters():
            size = param.numel()
            if name not in state_dict:
                raise KeyError(f"Missing parameter '{name}' in checkpoint: {model_path}")
            src = state_dict[name].view(-1).to(device=wb.device, dtype=wb.dtype)
            wb.narrow(0, offset, size).copy_(src)
            offset += size

    # Sync bf16 inference policy from fp32 master weights.
    # This is critical — without it, policy_bf16 retains random init weights.
    _sync_bf16_from_fp32(pufferl_cpp)

    # Validate the sync worked
    _validate_checkpoint_load(pufferl_cpp, state_dict)


def _sync_bf16_from_fp32(pufferl_cpp) -> None:
    """Copy weights from fp32 master policy to bf16 inference policy.

    When USE_BF16=false, policy_bf16 IS policy_fp32 (same object), so this is a no-op.
    When USE_BF16=true, they are separate models and explicit sync is required.
    """
    bf16 = pufferl_cpp.policy_bf16
    fp32 = pufferl_cpp.policy_fp32
    if bf16 is fp32:
        return  # Same object (USE_BF16=false), no sync needed
    with torch.no_grad():
        for p_bf16, p_fp32 in zip(bf16.parameters(), fp32.parameters()):
            p_bf16.data.copy_(p_fp32.data)


def _validate_checkpoint_load(pufferl_cpp, state_dict: dict) -> None:
    """Sanity check that bf16 inference weights match the checkpoint.

    Compares the first parameter's values between the loaded state_dict and
    the bf16 policy that will actually be used for inference.
    """
    bf16_params = dict(pufferl_cpp.policy_bf16.named_parameters())
    first_name = next(iter(state_dict))
    if first_name not in bf16_params:
        return  # Can't validate, different param names
    src = state_dict[first_name].to(device="cuda", dtype=torch.bfloat16)
    dst = bf16_params[first_name].data.to(dtype=torch.bfloat16)
    max_diff = (src - dst).abs().max().item()
    if max_diff > 0.01:
        raise RuntimeError(
            f"Checkpoint validation FAILED: bf16 policy param '{first_name}' "
            f"differs from checkpoint by max={max_diff:.6f}. "
            f"Weight sync from fp32->bf16 did not work correctly."
        )
    print(f"Checkpoint validation OK: bf16 param '{first_name}' max_diff={max_diff:.6f}")


def _require_stockfish_binary(explicit_path: str | None) -> str:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists() or not os.access(str(p), os.X_OK):
            raise FileNotFoundError(f"Stockfish binary is not executable: {explicit_path}")
        return str(p)

    for cand in ("/usr/games/stockfish", shutil.which("stockfish")):
        if cand and os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand

    raise FileNotFoundError(
        "Stockfish binary not found. Install stockfish and/or pass --stockfish-path."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint vs Stockfish")
    parser.add_argument("--model-path", type=str, default=None, help="Checkpoint .pt file")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--target-win-rate", type=float, default=0.70)
    parser.add_argument("--stockfish-elo", type=int, default=2200)
    parser.add_argument("--stockfish-movetime-ms", type=int, default=30)
    parser.add_argument("--stockfish-depth", type=int, default=0,
                        help="Stockfish search depth (0=use movetime instead)")
    parser.add_argument("--stockfish-path", type=str, default=None)
    parser.add_argument("--total-agents", type=int, default=64)
    parser.add_argument("--num-buffers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="puffer4")
    parser.add_argument("--wandb-group", type=str, default="chess-stockfish-eval")
    parser.add_argument("--wandb-tag", type=str, default="stockfish-gate")
    parser.add_argument("--stockfish-random-pct", type=int, default=0,
                        help="Pct of SF moves replaced with random (0=full strength)")
    parser.add_argument("--stockfish-query-pct", type=int, default=100,
                        help="Pct of opponent turns that actually query Stockfish (100=always)")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--log-pgn", action="store_true",
                        help="Enable PGN game logging (written by C env)")
    args = parser.parse_args()

    sf_path = _require_stockfish_binary(args.stockfish_path)
    os.environ["PUFFER_STOCKFISH_PATH"] = sf_path

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            tags=[args.wandb_tag],
            config=vars(args),
            save_code=False,
        )

    cfg = _safe_load_config("puffer_chess")
    cfg["vec"]["total_agents"] = int(args.total_agents)
    cfg["vec"]["num_buffers"] = int(args.num_buffers)

    env_cfg = cfg["env"]
    env_cfg["selfplay"] = 1
    env_cfg["random_bot"] = 0
    env_cfg["stockfish_bot"] = 1
    env_cfg["stockfish_random_pct"] = int(args.stockfish_random_pct)
    env_cfg["stockfish_query_pct"] = int(args.stockfish_query_pct)
    if args.log_pgn:
        env_cfg["log_pgn"] = 1
    env_cfg["stockfish_limit_strength"] = 1
    env_cfg["stockfish_elo"] = int(args.stockfish_elo)
    env_cfg["stockfish_movetime_ms"] = int(args.stockfish_movetime_ms)
    if args.stockfish_depth > 0:
        env_cfg["stockfish_depth"] = int(args.stockfish_depth)

    train_cfg = dict(cfg["train"])
    train_cfg["env_name"] = cfg["env_name"]
    train_cfg["env"] = cfg["env_name"]
    train_cfg["seed"] = int(args.seed)
    train_cfg["cudagraphs"] = -1
    train_cfg["total_timesteps"] = 10**18

    p = PuffeRL(train_cfg, cfg["vec"], env_cfg, cfg["policy"], verbose=False)
    if args.model_path:
        _load_checkpoint(p.pufferl_cpp, args.model_path)
    else:
        print("WARNING: No --model-path specified. Evaluating with random weights.")

    total_games = 0.0
    wins = 0.0
    draws = 0.0
    losses = 0.0
    t0 = time.time()

    while total_games < args.games:
        p.evaluate()
        torch.cuda.synchronize()
        logs = _C.log_environments(p.pufferl_cpp)
        if not logs:
            continue

        n = float(logs.get("n", 0.0))
        if n <= 0:
            continue

        perf = float(logs.get("perf", 0.5))
        draw_rate = float(logs.get("draw_rate", 0.0))

        chunk_wins = n * max(0.0, perf - 0.5 * draw_rate)
        chunk_draws = n * max(0.0, draw_rate)
        chunk_losses = max(0.0, n - chunk_wins - chunk_draws)

        remaining = float(args.games) - total_games
        scale = 1.0 if n <= remaining else (remaining / n)

        total_games += n * scale
        wins += chunk_wins * scale
        draws += chunk_draws * scale
        losses += chunk_losses * scale

        win_rate = wins / max(total_games, 1.0)
        draw_frac = draws / max(total_games, 1.0)
        loss_frac = losses / max(total_games, 1.0)
        elapsed = time.time() - t0

        print(
            f"games={total_games:.1f}/{args.games} "
            f"W={wins:.1f} D={draws:.1f} L={losses:.1f} "
            f"win_rate={win_rate:.3f} draw_rate={draw_frac:.3f} "
            f"elapsed={elapsed:.1f}s"
        )

        if run is not None:
            run.log(
                {
                    "stockfish_eval/games": total_games,
                    "stockfish_eval/wins": wins,
                    "stockfish_eval/draws": draws,
                    "stockfish_eval/losses": losses,
                    "stockfish_eval/win_rate": win_rate,
                    "stockfish_eval/draw_rate": draw_frac,
                    "stockfish_eval/loss_rate": loss_frac,
                    "stockfish_eval/stockfish_elo": args.stockfish_elo,
                    "stockfish_eval/movetime_ms": args.stockfish_movetime_ms,
                },
                step=int(total_games),
            )

    duration = time.time() - t0
    sf_depth = int(env_cfg.get("stockfish_depth", args.stockfish_depth))
    summary = EvalSummary(
        games=int(round(total_games)),
        wins=wins,
        draws=draws,
        losses=losses,
        win_rate=wins / max(total_games, 1.0),
        draw_rate=draws / max(total_games, 1.0),
        loss_rate=losses / max(total_games, 1.0),
        stockfish_elo=args.stockfish_elo,
        stockfish_movetime_ms=args.stockfish_movetime_ms,
        stockfish_depth=sf_depth,
        stockfish_random_pct=int(args.stockfish_random_pct),
        stockfish_query_pct=int(args.stockfish_query_pct),
        total_agents=args.total_agents,
        duration_sec=duration,
        model_path=args.model_path,
    )

    print("\n=== Stockfish Gate Summary ===")
    print(json.dumps(asdict(summary), indent=2))

    passed = summary.win_rate >= args.target_win_rate
    print(
        f"Gate: win_rate >= {args.target_win_rate:.2f} -> "
        f"{'PASS' if passed else 'FAIL'}"
    )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote summary: {out_path}")

    p.rollouts = None
    p.policy_fp32 = None
    torch.cuda.synchronize()
    _C.close(p.pufferl_cpp)
    p.pufferl_cpp = None
    torch.cuda.empty_cache()
    torch._C._cuda_clearCublasWorkspaces()

    if run is not None:
        run.summary.update(asdict(summary))
        run.summary["stockfish_eval/passed"] = bool(passed)
        run.finish()

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
