"""Explicit CPU-simulation baseline with native CUDA Puffer training.

The reference simulator, Sonic ONNX inference, and evaluation dummy run on
CPU. Per-step action downloads and observation uploads are intentional and
measured here. This module is not a CUDA environment implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import socket
import time
import traceback

import numpy as np
import torch

from human_eval_server import (
    CandidateApproachDummy, ConservativeActionPlanner, NativeVectorBoundary,
    RuntimeIdentity, ROBOT_ROWS, OBSERVATION_FLOATS, verify_runtime_identity,
)
from gpu_metrics import RekG1GpuMetricCollector
from gpu_puffer_env import CudaTensorEnvAdapter
from gpu_native_puffer import NativeExternalGpuPuffer
from train_gpu_duel import load_native_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_identity(path: Path) -> RuntimeIdentity:
    record = json.loads(path.read_text(encoding="utf-8"))
    for key in ("extension", "semantic_assets", "encoder", "decoder"):
        record[key] = Path(record[key])
    return RuntimeIdentity(**record)


def _cpu_worker(connection, identity, max_steps, physics_workers):
    """Keep old and new pybind11 Policy registries in separate processes."""
    boundary = None
    try:
        torch.set_num_threads(1)
        boundary = NativeVectorBoundary.open(identity, max_steps=max_steps,
                                             physics_workers=physics_workers)
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            if command == "step":
                boundary.step(payload)
            elif command == "reset":
                boundary.reset()
            else:
                raise ValueError(f"unknown worker command: {command}")
            native_seconds = time.perf_counter() - wall_start
            cpu_seconds = time.process_time() - cpu_start
            connection.send(("ok", (boundary.observations.copy(),
                boundary.rewards.copy(), boundary.terminals.copy(),
                boundary.action_masks.copy(), native_seconds, cpu_seconds)))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        if boundary is not None:
            boundary.close()
        connection.close()


class CpuWorkerBoundary:
    def __init__(self, identity, *, max_steps, physics_workers):
        self.identity = verify_runtime_identity(identity)
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_cpu_worker,
            args=(child, identity, max_steps, physics_workers))
        self.process.start()
        child.close()
        self.native_seconds = 0.0
        self.worker_cpu_seconds = 0.0
        self.reset()

    def _request(self, command, payload):
        self.connection.send((command, payload))
        kind, result = self.connection.recv()
        if kind != "ok":
            raise RuntimeError(f"CPU reference worker failed:\n{result}")
        (self.observations, self.rewards, self.terminals, self.action_masks,
         native_seconds, worker_cpu_seconds) = result
        if command == "step":
            self.native_seconds += native_seconds
            self.worker_cpu_seconds += worker_cpu_seconds

    def reset(self):
        self._request("reset", None)

    def step(self, actions):
        self._request("step", actions)

    def close(self):
        if self.process.is_alive():
            self.connection.send(("close", None))
        self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.connection.close()


class CpuBaselineUploadBoundary:
    """Expose four learning rows while running four full CPU reference duels."""

    def __init__(self, boundary: NativeVectorBoundary, *, behavior_metrics=True):
        self.boundary = boundary
        self.learning_agents = ROBOT_ROWS // 2
        self.device = torch.device("cuda:0")
        self.observations = torch.empty(
            (self.learning_agents, OBSERVATION_FLOATS), device=self.device)
        self.rewards = torch.empty(self.learning_agents, device=self.device)
        self.terminals = torch.empty_like(self.rewards)
        self.action_mask = torch.empty(
            (self.learning_agents, 33), dtype=torch.uint8, device=self.device)
        self.full_actions = np.ones((ROBOT_ROWS, 1), dtype=np.float32)
        self.dummies = [CandidateApproachDummy() for _ in range(self.learning_agents)]
        self.planners = [ConservativeActionPlanner() for _ in self.dummies]
        self.metrics = RekG1GpuMetricCollector(ROBOT_ROWS, "cpu")
        self.behavior = None
        if behavior_metrics:
            from gpu_behavior_metrics import GpuBehaviorMetricCollector
            self.behavior = GpuBehaviorMetricCollector(
                ROBOT_ROWS, "cpu", learner_rows=tuple(range(0, ROBOT_ROWS, 2)))
        self.clear_timing()
        self.reset()

    def clear_timing(self):
        self.timings = dict.fromkeys((
            "action_download", "dummy_decision", "native_cpu_step",
            "metrics_and_validation", "observation_upload"), 0.0)
        self.control_ticks = 0

    def _upload(self):
        self.observations.copy_(torch.from_numpy(self.boundary.observations[0::2]))
        self.rewards.copy_(torch.from_numpy(self.boundary.rewards[0::2]))
        self.terminals.copy_(torch.from_numpy(self.boundary.terminals[0::2]))
        self.action_mask.copy_(torch.from_numpy(self.boundary.action_masks[0::2]))

    def reset(self):
        self.boundary.reset()
        for dummy, planner in zip(self.dummies, self.planners):
            dummy.reset()
            planner.reset()
        self.metrics.reset()
        if self.behavior is not None:
            self.behavior.reset()
        self._upload()

    def step(self, actions):
        start = time.perf_counter()
        learned = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        self.full_actions[0::2] = learned
        self.timings["action_download"] += time.perf_counter() - start

        start = time.perf_counter()
        for arena, (dummy, planner) in enumerate(zip(self.dummies, self.planners)):
            row = 2 * arena + 1
            preferred = dummy.preferred_action(self.boundary.observations[row])
            choice = planner.select(preferred, 1, self.boundary.observations[row],
                                    self.boundary.action_masks[row])
            dummy.note_selected(choice.category)
            self.full_actions[row, 0] = choice.category
        self.timings["dummy_decision"] += time.perf_counter() - start

        start = time.perf_counter()
        # NativeVectorBoundary validates exact masks and observations. This
        # elapsed component includes those checks with the full native step.
        self.boundary.step(self.full_actions)
        self.timings["native_cpu_step"] += time.perf_counter() - start

        start = time.perf_counter()
        observations = torch.from_numpy(self.boundary.observations)
        terminals = torch.from_numpy(self.boundary.terminals)
        self.metrics.update(observations, terminals)
        if self.behavior is not None:
            self.behavior.update(observations, terminals,
                torch.from_numpy(self.full_actions), None, None)
        for arena, (dummy, planner) in enumerate(zip(self.dummies, self.planners)):
            if self.boundary.terminals[2 * arena + 1]:
                dummy.reset()
                planner.reset()
        self.timings["metrics_and_validation"] += time.perf_counter() - start

        start = time.perf_counter()
        self._upload()
        self.timings["observation_upload"] += time.perf_counter() - start
        self.control_ticks += 1

    def log(self):
        record = self.metrics.snapshot(clear=False)
        record.update({"opponent": CandidateApproachDummy.LABEL,
                       "opponent_is_bot_1": False})
        if self.behavior is not None:
            record["behavior"] = self.behavior.snapshot(clear=False)
        return record

    def close(self):
        self.boundary.close()


def benchmark(args):
    if args.output.exists() or args.run_dir.exists():
        raise FileExistsError("baseline output or run directory already exists")
    learning_agents = ROBOT_ROWS // 2
    rollout_steps = learning_agents * args.horizon
    if args.total_timesteps <= 0 or args.total_timesteps % rollout_steps:
        raise ValueError("timesteps must be positive and divisible by 4*horizon")
    torch.set_num_threads(1)
    native_args = load_native_config(args.default_config, args.native_config)
    native_args["vec"].update(total_agents=learning_agents, num_buffers=1, num_threads=0)
    native_args["train"].update(total_timesteps=args.total_timesteps,
        horizon=args.horizon, minibatch_size=rollout_steps, gpus=1)
    native_args.update(gpu_id=0, rank=0, world_size=1, nccl_id=b"")
    setup_start = time.perf_counter()
    cpu = CpuWorkerBoundary(load_identity(args.identity),
        max_steps=args.max_steps, physics_workers=args.physics_workers)
    try:
        env = CpuBaselineUploadBoundary(cpu, behavior_metrics=not args.no_behavior_metrics)
        adapter = CudaTensorEnvAdapter(env, (33,))
        trainer = NativeExternalGpuPuffer(native_args, adapter, reward_clip=0.0)
    except BaseException:
        cpu.close()
        raise
    args.run_dir.mkdir(parents=True)
    try:
        if args.load_checkpoint:
            trainer.load_weights(args.load_checkpoint)
        initial = trainer.save_weights(args.run_dir / "initial.bin")
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - setup_start
        env.clear_timing()
        cpu.native_seconds = 0.0
        cpu.worker_cpu_seconds = 0.0
        rollout_seconds = 0.0
        train_seconds = 0.0
        report_seconds = 0.0
        start = time.perf_counter()
        cpu_start = time.process_time()
        epochs = args.total_timesteps // rollout_steps
        latest = {}
        for epoch in range(1, epochs + 1):
            phase = time.perf_counter()
            trainer.rollouts()
            torch.cuda.synchronize()
            rollout_seconds += time.perf_counter() - phase
            phase = time.perf_counter()
            trainer.train()
            torch.cuda.synchronize()
            train_seconds += time.perf_counter() - phase
            if epoch % args.log_every == 0 or epoch == epochs:
                phase = time.perf_counter()
                latest = trainer.log(clear_metrics=False)
                print(json.dumps({"epoch": epoch, "learner_steps": trainer.global_step,
                    "learner_steps_per_second": trainer.global_step / (time.perf_counter() - start),
                    "completed_rounds": latest["env"]["n"]}), flush=True)
                report_seconds += time.perf_counter() - phase
        wall_seconds = time.perf_counter() - start
        host_cpu_seconds = time.process_time() - cpu_start
        final = trainer.save_weights(args.run_dir / "final.bin")
        measured = dict(env.timings)
        measured["cpu_worker_transport"] = measured["native_cpu_step"] - cpu.native_seconds
        measured["native_cpu_step"] = cpu.native_seconds
        measured["native_policy_rollout_and_dispatch"] = rollout_seconds - sum(env.timings.values())
        measured["native_ppo_training"] = train_seconds
        measured["reporting"] = report_seconds
        measured["other_host_loop"] = wall_seconds - rollout_seconds - train_seconds - report_seconds
        report = {
            "schema": "rek.g1_cpu_baseline_fixed_dummy_training.v1",
            "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
            "cpu_reference_identity": cpu.identity.report(),
            "opponent": CandidateApproachDummy.LABEL, "opponent_is_bot_1": False,
            "execution": {"simulation": "native-cpu", "sonic_controller": "cpu-onnxruntime",
                "opponent_decisions": "cpu", "policy_ppo_optimizer": "native-puffer-cuda",
                "explicit_per_step_cpu_gpu_copies": True, "cuda_production_path": False,
                "physics_workers": args.physics_workers, "cpu_fighter_rows": ROBOT_ROWS,
                "learning_agents": learning_agents, "arenas": learning_agents},
            "training": {"learner_steps": trainer.global_step,
                "physical_fighter_steps": trainer.global_step * 2,
                "control_ticks": env.control_ticks, "horizon": args.horizon,
                "epochs": epochs, "setup_seconds": setup_seconds,
                "wall_seconds": wall_seconds,
                "host_orchestration_cpu_seconds": host_cpu_seconds,
                "native_environment_cpu_seconds": cpu.worker_cpu_seconds,
                "host_cpu_seconds": host_cpu_seconds + cpu.worker_cpu_seconds,
                "host_cpu_to_wall_ratio": (host_cpu_seconds + cpu.worker_cpu_seconds) / wall_seconds,
                "learner_steps_per_second": trainer.global_step / wall_seconds,
                "physical_fighter_steps_per_second": trainer.global_step * 2 / wall_seconds,
                "simulated_seconds_per_arena": env.control_ticks * 0.02},
            "component_wall_seconds": measured,
            "component_wall_percent": {key: 100 * value / wall_seconds for key, value in measured.items()},
            "timing_scope": {"method": "synchronized-wall-clock-sequential-baseline",
                "native_cpu_step": "physics+Sonic+motion+combat+native mask and finite validation",
                "physics_vs_sonic_breakdown": None,
                "unmeasured_breakdown_reason": "existing reference extension exposes no internal phase timers",
                "profile_barrier_overhead_included": True},
            "combat_metrics": latest["env"], "native_log": latest,
            "initial_checkpoint": initial, "final_checkpoint": final,
            "weights_changed": initial["checkpoint"]["sha256"] != final["checkpoint"]["sha256"],
            "source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
            "claim_limits": {"authentic_rek_parity_established": False,
                "human_baseline_measured": False, "superhuman_claim_supported": False},
        }
    finally:
        trainer.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--default-config", required=True, type=Path)
    parser.add_argument("--native-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--total-timesteps", required=True, type=int)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=10000000)
    parser.add_argument("--physics-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=16)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--no-behavior-metrics", action="store_true")
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
