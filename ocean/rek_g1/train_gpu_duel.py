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
import traceback

import torch

from gpu_native_puffer import NativeExternalGpuPuffer, create_rek_g1_native_puffer
from gpu_semantic_duel import GpuSemanticDuel
from profile_gpu_duel import TrainingPhaseTimer
from verify_gpu_duel import load_config as load_gpu_duel_config
from gpu_policy_observation_encoder import (
    ENCODER_CHOICES, INITIALIZATION_CHOICES, load_policy_encoder_checkpoint,
    policy_encoder_report, save_policy_weights,
)


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


def save_failure(args, environment, trainer, error, epoch, checked_ticks) -> dict:
    """Preserve a failed trajectory for diagnosis; never label it a valid run."""
    import numpy as np

    report = {
        "schema": "rek.g1_gpu_training_failure.v1",
        "status": "failed", "host": socket.gethostname(),
        "error": repr(error), "traceback": traceback.format_exc(),
        "epoch": epoch, "agent_steps": trainer.global_step,
        "diagnostic_checked_ticks": checked_ticks,
        "diagnostic_step_checks": getattr(args, "check_every_step", False),
        "diagnostic_only": True, "capture_errors": [],
        "sources": {}, "snapshots": {},
        "device_addresses": {
            "matchers": environment.motion.matchers.tensor.data_ptr(),
            "slots": environment.motion.slots.tensor.data_ptr(),
        },
    }
    for name in ("train_gpu_duel.py", "gpu_semantic_scheduler.py",
                 "gpu_native_motion.py", "gpu_semantic_duel.py"):
        report["sources"][name] = _checkpoint_record(Path(__file__).with_name(name))
    tensors = {
        "scheduler_rows": environment.scheduler.rows.tensor,
        "scheduler_status": environment.scheduler.statuses,
        "composers": environment.motion.composers.tensor,
        "matchers": environment.motion.matchers.tensor,
        "slots": environment.motion.slots.tensor,
        "route_commands": environment.motion.route_commands,
        "matcher_status": environment.motion.matchers.field("last_status"),
        "actions": environment.actions,
        "action_mask": environment.action_mask,
        "observations": environment.observations,
        "qpos": environment.physics.qpos,
        "qvel": environment.physics.qvel,
    }
    for name, tensor in tensors.items():
        try:
            path = args.run_dir / f"failure-{name}.npy"
            with path.open("xb") as stream:
                np.save(stream, tensor.detach().cpu().numpy(), allow_pickle=False)
            report["snapshots"][name] = _checkpoint_record(path)
        except Exception as capture_error:
            report["capture_errors"].append(f"{name}: {capture_error!r}")
    try:
        report["diagnostic_policy"] = save_policy_weights(
            trainer, args.run_dir / "failed-policy.bin",
            getattr(args, "policy_observation_encoder", "raw"),
            getattr(args, "policy_observation_warm_start", "matching-checkpoint"),
        )
    except Exception as capture_error:
        report["capture_errors"].append(f"policy: {capture_error!r}")
    with (args.run_dir / "failure.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


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
    checkpoint_every = getattr(args, "checkpoint_every", 16)
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if args.reward_clip < 0.0:
        raise ValueError("reward_clip must be nonnegative")
    policy_encoding = getattr(args, "policy_observation_encoder", "raw")
    policy_initialization = getattr(args, "policy_observation_warm_start", "matching-checkpoint")
    policy_manifest, policy_checkpoint_sha = load_policy_encoder_checkpoint(
        args.load_checkpoint, policy_encoding, initialization=policy_initialization,
        expected_sha256=getattr(args, "load_checkpoint_sha256", None),
    )

    native_args = load_native_config(args.default_config, args.native_config)
    duel_config = load_gpu_duel_config(args.gpu_duel_config)
    total_agents = args.total_agents or int(native_args["vec"]["total_agents"])
    if total_agents <= 0 or total_agents % 2:
        raise ValueError("total_agents must be a positive even fighter count")
    opponent = getattr(args, "opponent", "self-play")
    if opponent not in ("self-play", "candidate-dummy"):
        raise ValueError("unknown opponent")
    if policy_encoding != "raw" and opponent != "candidate-dummy":
        raise ValueError("policy observation encoders currently require candidate-dummy")
    learning_agents = total_agents // 2 if opponent == "candidate-dummy" else total_agents
    batch_steps = learning_agents * args.horizon
    if args.total_timesteps % batch_steps:
        raise ValueError(
            f"total_timesteps must be divisible by agents*horizon={batch_steps}"
        )
    minibatch_size = args.minibatch_size or batch_steps
    if minibatch_size % args.horizon or minibatch_size > batch_steps:
        raise ValueError(
            "minibatch_size must be divisible by horizon and no larger than one rollout"
        )

    native_args["vec"]["total_agents"] = learning_agents
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
    if opponent == "candidate-dummy":
        from gpu_candidate_dummy import DUMMY_LABEL, GpuCandidateDummyDuel
        from gpu_puffer_env import CudaTensorEnvAdapter

        candidate_environment = GpuCandidateDummyDuel(
            environment, policy_observation_encoder=policy_encoding,
            checkpoint_manifest=policy_manifest, checkpoint_sha256=policy_checkpoint_sha,
            policy_observation_initialization=policy_initialization,
        )
        adapter = CudaTensorEnvAdapter(
            candidate_environment, (33,),
            metric_plugins=(candidate_environment.metric_plugin,),
        )
        opponent_label = DUMMY_LABEL
        opponent_sources = {
            "human_eval_reference": _checkpoint_record(
                Path(__file__).resolve().with_name("human_eval_server.py")
            ),
            "cuda_opponent": _checkpoint_record(
                Path(__file__).resolve().with_name("gpu_candidate_dummy.py")
            ),
        }
        trainer = NativeExternalGpuPuffer(
            native_args, adapter, reward_clip=args.reward_clip,
        )
    else:
        trainer = create_rek_g1_native_puffer(
            native_args, environment, reward_clip=args.reward_clip,
        )
        opponent_label = "shared_policy_self_play"
        opponent_sources = None
    epoch = 0
    checked_ticks = 0
    failed = False
    periodic_checkpoints = []
    if getattr(args, "check_every_step", False):
        original_step = trainer.env.step

        def diagnostic_step(actions):
            nonlocal checked_ticks
            original_step(actions)
            checked_ticks += 1
            environment.check_status()

        trainer.env.step = diagnostic_step
    try:
        if args.load_checkpoint is not None:
            trainer.load_weights(args.load_checkpoint)
        initial_manifest = save_policy_weights(trainer, initial_path, policy_encoding, policy_initialization)
        torch.cuda.synchronize(environment.actions.device)
        setup_seconds = time.perf_counter() - setup_start

        phase_timer = TrainingPhaseTimer(environment.actions.device)
        train_wall_start = time.perf_counter()
        train_cpu_start = time.process_time()
        epochs = args.total_timesteps // batch_steps
        latest_log: dict = {}
        for epoch in range(1, epochs + 1):
            phase_timer.call("rollout", trainer.rollouts)
            phase_timer.call("ppo_update", trainer.train)
            if epoch % args.log_every == 0 or epoch == epochs:
                latest_log = phase_timer.call("reporting", trainer.log, clear_metrics=False)
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
            if checkpoint_every and epoch % checkpoint_every == 0 and epoch != epochs:
                environment.check_status()
                periodic_checkpoints.append(phase_timer.call(
                    "checkpoint", save_policy_weights, trainer,
                    args.run_dir / f"verified-{trainer.global_step:016d}.bin",
                    policy_encoding, policy_initialization,
                ))

        torch.cuda.synchronize(environment.actions.device)
        train_wall_seconds = time.perf_counter() - train_wall_start
        host_cpu_seconds = time.process_time() - train_cpu_start
        if not latest_log:
            latest_log = phase_timer.call("reporting", trainer.log, clear_metrics=False)
        phase_timing = phase_timer.snapshot()
        final_manifest = save_policy_weights(trainer, final_path, policy_encoding, policy_initialization)
        environment.check_status()

        extension_path = Path(trainer.backend.__file__).resolve()
        initial_record = _checkpoint_record(initial_path)
        final_record = _checkpoint_record(final_path)
        report = {
            "schema": "rek.g1_native_external_gpu_training.v1",
            **policy_encoder_report(policy_encoding, policy_initialization),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(environment.actions.device),
            "cuda_device": str(environment.actions.device),
            "native_extension": {
                "path": str(extension_path),
                "sha256": _sha256(extension_path),
                "precision_bytes": int(trainer.backend.precision_bytes),
            },
            "inputs": {
                "fused_combat_library": (
                    None if environment.config.fused_combat_library is None
                    else _checkpoint_record(environment.config.fused_combat_library)
                ),
                "opponent_sources": opponent_sources,
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
                "physical_fighters": total_agents,
                "learning_agents": learning_agents,
                "arenas": total_agents // 2,
                "opponent": opponent_label,
                "opponent_is_bot_1": False,
                "opponent_samples_in_ppo": opponent == "self-play",
                "horizon": args.horizon,
                "epochs": epochs,
                "agent_steps": trainer.global_step,
                "physical_fighter_steps": trainer.global_step * total_agents // learning_agents,
                "setup_seconds": setup_seconds,
                "wall_seconds": train_wall_seconds,
                "host_cpu_seconds": host_cpu_seconds,
                "host_cpu_to_wall_ratio": host_cpu_seconds / train_wall_seconds,
                "agent_steps_per_second": trainer.global_step / train_wall_seconds,
                "physical_fighter_steps_per_second": (
                    trainer.global_step * total_agents / learning_agents / train_wall_seconds
                ),
                "cpu_physics_steps": 0,
                "cpu_controller_inferences": 0,
                "physics_sparse_jacobian": bool(environment.physics.model.is_sparse),
                "conditional_reset_forward": environment.reset_forward_gate is not None,
                "combat_measurement_backend": type(environment.measurement).__name__,
                "native_cpu_environment_count": int(
                    trainer.pufferl.native_env_count
                ),
                "native_environment_threads": bool(trainer.pufferl.has_env_threads),
                "diagnostic_step_checks": getattr(args, "check_every_step", False),
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
                "periodic": periodic_checkpoints,
                "periodic_resume_scope": "policy weights only; optimizer and environment state excluded",
            },
            "final_native_log": latest_log,
            "phase_timing": phase_timing,
            "completed_combat_metrics": latest_log.get("env", {}),
            "claim_limits": {
                "shared_policy_self_play": opponent == "self-play",
                "human_baseline_measured": False,
                "authentic_rek_parity_established": False,
                "superhuman_claim_supported": False,
            },
        }
    except Exception as error:
        failed = True
        try:
            save_failure(args, environment, trainer, error, epoch, checked_ticks)
        except Exception as capture_error:
            print(f"failure evidence capture also failed: {capture_error!r}", flush=True)
        raise
    finally:
        try:
            trainer.close()
        except Exception as close_error:
            if not failed:
                raise
            print(f"failed-run cleanup also failed: {close_error!r}", flush=True)

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
    parser.add_argument("--checkpoint-every", type=int, default=16,
                        help="save verified policy weights every N epochs; zero disables")
    parser.add_argument("--check-every-step", action="store_true",
                        help="synchronize status each tick for fault diagnosis, invalidating throughput comparisons")
    parser.add_argument("--reward-clip", type=float, default=0.0)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--load-checkpoint-sha256", default=None,
                        help="required pinned raw checkpoint hash for raw-initial-weights")
    parser.add_argument("--policy-observation-encoder", choices=ENCODER_CHOICES, default="raw")
    parser.add_argument("--policy-observation-warm-start", choices=INITIALIZATION_CHOICES,
                        default="matching-checkpoint",
                        help="transformed policy initialization must be explicit; raw-initial-weights is a deliberate coordinate change")
    parser.add_argument(
        "--opponent", choices=("self-play", "candidate-dummy"), default="self-play",
        help="candidate-dummy trains even learner rows against the human-eval opponent",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
