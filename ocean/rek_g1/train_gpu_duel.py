"""Train the CUDA REK G1 semantic duel with native PufferLib PPO."""

from __future__ import annotations

import argparse
import ast
import configparser
import hashlib
import json
from pathlib import Path
import socket
import time

import torch

from gpu_native_puffer import create_rek_g1_native_puffer
from gpu_semantic_duel import GpuSemanticDuel
from verify_gpu_duel import load_config as load_gpu_duel_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_value(raw: str):
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def load_native_config(default_path: Path, environment_path: Path) -> dict:
    parser = configparser.ConfigParser()
    loaded = parser.read((default_path, environment_path))
    if loaded != [str(default_path), str(environment_path)]:
        raise FileNotFoundError(
            f"failed to load native configs: {default_path}, {environment_path}"
        )
    result: dict = {}
    for section in parser.sections():
        target = result if section == "base" else result.setdefault(section, {})
        for key, raw in parser[section].items():
            target[key] = _config_value(raw)
    result["env_name"] = "rek_g1"
    return result


def _checkpoint_record(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def train(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    if args.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if args.horizon <= 1:
        raise ValueError("horizon must be greater than one")
    if args.log_every <= 0:
        raise ValueError("log_every must be positive")
    if args.reward_clip < 0.0:
        raise ValueError("reward_clip must be nonnegative")

    native_args = load_native_config(args.default_config, args.native_config)
    duel_config = load_gpu_duel_config(args.gpu_duel_config)
    total_agents = args.total_agents or int(native_args["vec"]["total_agents"])
    if total_agents <= 0 or total_agents % 2:
        raise ValueError("total_agents must be a positive even fighter count")
    batch_steps = total_agents * args.horizon
    if args.total_timesteps % batch_steps:
        raise ValueError(
            f"total_timesteps must be divisible by agents*horizon={batch_steps}"
        )
    minibatch_size = args.minibatch_size or batch_steps
    if minibatch_size % args.horizon or minibatch_size > batch_steps:
        raise ValueError(
            "minibatch_size must be divisible by horizon and no larger than one rollout"
        )

    native_args["vec"]["total_agents"] = total_agents
    native_args["vec"]["num_buffers"] = 1
    native_args["vec"]["num_threads"] = 0
    native_args["train"]["total_timesteps"] = args.total_timesteps
    native_args["train"]["horizon"] = args.horizon
    native_args["train"]["minibatch_size"] = minibatch_size
    native_args["train"]["gpus"] = 1
    native_args["gpu_id"] = torch.device(duel_config.device).index or 0
    native_args["world_size"] = 1
    native_args["rank"] = 0
    native_args["nccl_id"] = b""

    args.run_dir.mkdir(parents=True)
    initial_path = args.run_dir / "initial.bin"
    final_path = args.run_dir / f"{args.total_timesteps:016d}.bin"
    setup_start = time.perf_counter()
    environment = GpuSemanticDuel(duel_config)
    if environment.rows != total_agents:
        environment.close()
        raise ValueError(
            f"total_agents={total_agents} does not match GPU duel rows={environment.rows}"
        )
    environment.capture_step()
    trainer = create_rek_g1_native_puffer(
        native_args,
        environment,
        reward_clip=args.reward_clip,
    )
    try:
        if args.load_checkpoint is not None:
            trainer.load_weights(args.load_checkpoint)
        initial_manifest = trainer.save_weights(initial_path)
        torch.cuda.synchronize(environment.actions.device)
        setup_seconds = time.perf_counter() - setup_start

        train_wall_start = time.perf_counter()
        train_cpu_start = time.process_time()
        epochs = args.total_timesteps // batch_steps
        latest_log: dict = {}
        for epoch in range(1, epochs + 1):
            trainer.rollouts()
            trainer.train()
            if epoch % args.log_every == 0 or epoch == epochs:
                latest_log = trainer.log(clear_metrics=False)
                elapsed = time.perf_counter() - train_wall_start
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "agent_steps": trainer.global_step,
                            "measured_cumulative_agent_steps_per_second": (
                                trainer.global_step / elapsed
                            ),
                            "completed_rounds": latest_log["env"]["n"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        torch.cuda.synchronize(environment.actions.device)
        train_wall_seconds = time.perf_counter() - train_wall_start
        host_cpu_seconds = time.process_time() - train_cpu_start
        if not latest_log:
            latest_log = trainer.log(clear_metrics=False)
        final_manifest = trainer.save_weights(final_path)
        environment.check_status()

        extension_path = Path(trainer.backend.__file__).resolve()
        initial_record = _checkpoint_record(initial_path)
        final_record = _checkpoint_record(final_path)
        report = {
            "schema": "rek.g1_native_external_gpu_training.v1",
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(environment.actions.device),
            "cuda_device": str(environment.actions.device),
            "native_extension": {
                "path": str(extension_path),
                "sha256": _sha256(extension_path),
                "precision_bytes": int(trainer.backend.precision_bytes),
            },
            "inputs": {
                "gpu_duel_config": {
                    "path": str(args.gpu_duel_config),
                    "sha256": _sha256(args.gpu_duel_config),
                },
                "default_config": {
                    "path": str(args.default_config),
                    "sha256": _sha256(args.default_config),
                },
                "native_config": {
                    "path": str(args.native_config),
                    "sha256": _sha256(args.native_config),
                },
                "loaded_checkpoint": (
                    None
                    if args.load_checkpoint is None
                    else _checkpoint_record(args.load_checkpoint)
                ),
            },
            "training": {
                "backend": "pufferlib-native-cuda",
                "policy": "native",
                "ppo": "native",
                "optimizer": "native-muon",
                "prioritized_replay": "native",
                "total_agents": total_agents,
                "arenas": total_agents // 2,
                "horizon": args.horizon,
                "epochs": epochs,
                "agent_steps": trainer.global_step,
                "setup_seconds": setup_seconds,
                "wall_seconds": train_wall_seconds,
                "host_cpu_seconds": host_cpu_seconds,
                "host_cpu_to_wall_ratio": host_cpu_seconds / train_wall_seconds,
                "agent_steps_per_second": trainer.global_step / train_wall_seconds,
                "cpu_physics_steps": 0,
                "cpu_controller_inferences": 0,
                "native_cpu_environment_count": int(
                    trainer.pufferl.native_env_count
                ),
                "native_environment_threads": bool(trainer.pufferl.has_env_threads),
                "reward_transform": (
                    "none" if args.reward_clip == 0.0 else "symmetric_clamp"
                ),
                "reward_clip": args.reward_clip,
            },
            "checkpoints": {
                "initial": initial_record,
                "final": final_record,
                "weights_changed": initial_record["sha256"] != final_record["sha256"],
                "initial_manifest": initial_manifest,
                "final_manifest": final_manifest,
            },
            "final_native_log": latest_log,
            "completed_combat_metrics": latest_log.get("env", {}),
            "claim_limits": {
                "shared_policy_self_play": True,
                "human_baseline_measured": False,
                "authentic_rek_parity_established": False,
                "superhuman_claim_supported": False,
            },
        }
    finally:
        trainer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-duel-config", required=True, type=Path)
    parser.add_argument(
        "--default-config", type=Path, default=repo_root / "config" / "default.ini"
    )
    parser.add_argument(
        "--native-config", type=Path, default=repo_root / "config" / "rek_g1.ini"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--total-timesteps", required=True, type=int)
    parser.add_argument("--total-agents", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--reward-clip", type=float, default=0.0)
    parser.add_argument("--load-checkpoint", type=Path)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
