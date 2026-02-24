#!/usr/bin/env python3
"""Play human vs a trained chess policy using the native 4.0 backend."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import torch

import pufferlib._C as _C
from pufferlib.pufferl import PuffeRL, load_config


def _safe_load_config(env_name: str):
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        return load_config(env_name)
    finally:
        sys.argv = saved_argv


def _sync_bf16_from_fp32(pufferl_cpp) -> None:
    bf16 = pufferl_cpp.policy_bf16
    fp32 = pufferl_cpp.policy_fp32
    if bf16 is fp32:
        return
    with torch.no_grad():
        for p_bf16, p_fp32 in zip(bf16.parameters(), fp32.parameters()):
            p_bf16.data.copy_(p_fp32.data)


def _load_checkpoint(pufferl_cpp, model_path: str) -> None:
    state_dict = torch.load(model_path, map_location="cpu")
    wb = pufferl_cpp.muon.weight_buffer

    def adapt_param(name: str, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        if src.shape == dst.shape:
            return src

        # Bridge selfplay-trained chess checkpoints (single action head) into
        # human-play eval config (two action heads). Duplicate policy logits
        # into both heads and keep the value head as the last row.
        if name.endswith("decoder.linear.weight") and src.ndim == 2 and dst.ndim == 2:
            if src.shape[1] == dst.shape[1] and (src.shape[0] * 2 - 1) == dst.shape[0]:
                policy_rows = src.shape[0] - 1
                out = torch.empty_like(dst, device=src.device, dtype=src.dtype)
                out[:policy_rows] = src[:policy_rows]
                out[policy_rows : 2 * policy_rows] = src[:policy_rows]
                out[-1] = src[-1]
                return out

        if name.endswith("decoder.linear.bias") and src.ndim == 1 and dst.ndim == 1:
            if (src.shape[0] * 2 - 1) == dst.shape[0]:
                policy_rows = src.shape[0] - 1
                out = torch.empty_like(dst, device=src.device, dtype=src.dtype)
                out[:policy_rows] = src[:policy_rows]
                out[policy_rows : 2 * policy_rows] = src[:policy_rows]
                out[-1] = src[-1]
                return out

        raise RuntimeError(
            f"Incompatible checkpoint for '{name}': "
            f"checkpoint shape {tuple(src.shape)} vs model shape {tuple(dst.shape)}"
        )

    offset = 0
    with torch.no_grad():
        for name, param in pufferl_cpp.policy_fp32.named_parameters():
            size = param.numel()
            if name not in state_dict:
                raise KeyError(f"Missing parameter '{name}' in checkpoint: {model_path}")
            src_full = adapt_param(name, state_dict[name], param)
            src = src_full.view(-1).to(device=wb.device, dtype=wb.dtype)
            wb.narrow(0, offset, size).copy_(src)
            offset += size
    _sync_bf16_from_fp32(pufferl_cpp)


def _checkpoint_candidates() -> list[str]:
    paths = glob.glob("experiments/puffer_chess/*/model_*.pt")
    if not paths:
        raise FileNotFoundError("No checkpoints found under experiments/puffer_chess/*/model_*.pt")
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Human vs policy chess (native 4.0 backend)")
    parser.add_argument(
        "--model-path",
        type=str,
        default="latest",
        help='Checkpoint path, or "latest" to auto-pick newest model under experiments/puffer_chess',
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--log-pgn", action="store_true", help="Append games to a timestamped PGN file")
    parser.add_argument("--print-every", type=int, default=0, help="Print env stats every N loop steps")
    args = parser.parse_args()

    requested_model_path = args.model_path

    cfg = _safe_load_config("puffer_chess")
    cfg["vec"]["total_agents"] = 1
    cfg["vec"]["num_buffers"] = 1
    cfg["vec"]["num_threads"] = 1

    env_cfg = cfg["env"]
    env_cfg["selfplay"] = 0
    env_cfg["human_play"] = 1
    env_cfg["random_bot"] = 0
    env_cfg["stockfish_bot"] = 0
    env_cfg["render_fps"] = int(args.fps)
    env_cfg["log_pgn"] = 1 if args.log_pgn else 0
    env_cfg["fen_curric_pct"] = 0.0
    env_cfg["deepmind_fen_pct"] = 0.0

    train_cfg = dict(cfg["train"])
    train_cfg["env_name"] = cfg["env_name"]
    train_cfg["env"] = cfg["env_name"]
    train_cfg["seed"] = int(args.seed)
    train_cfg["horizon"] = 1
    train_cfg["minibatch_size"] = 1
    train_cfg["total_timesteps"] = 10**18
    train_cfg["cudagraphs"] = -1

    p = PuffeRL(train_cfg, cfg["vec"], env_cfg, cfg["policy"], verbose=False)

    if requested_model_path == "latest":
        model_path = None
        errors: list[str] = []
        for candidate in _checkpoint_candidates():
            resolved = str(Path(candidate).expanduser().resolve())
            try:
                _load_checkpoint(p.pufferl_cpp, resolved)
                model_path = resolved
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{resolved}: {exc}")

        if model_path is None:
            raise RuntimeError(
                "Could not find a compatible checkpoint for current chess policy config.\n"
                + "\n".join(errors[:5])
            )
    else:
        model_path = str(Path(requested_model_path).expanduser().resolve())
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        _load_checkpoint(p.pufferl_cpp, model_path)

    print(f"Loaded model: {model_path}")
    print("Chess window controls: mouse to move, choose white/black on start screen, ESC to quit.")

    steps = 0
    try:
        while True:
            _C.render(p.pufferl_cpp, 0)
            p.evaluate()
            steps += 1

            if args.print_every > 0 and steps % args.print_every == 0:
                logs = _C.log_environments(p.pufferl_cpp)
                if logs:
                    print(logs)
    finally:
        p.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
