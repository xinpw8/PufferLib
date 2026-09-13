"""Bounded native PPO benchmark for the complete CUDA semantic combat duel.

Run each physics arm in its own process with the same configs, checkpoint,
opponent, horizon and update counts. The primary denominator is the sum of
completed rollout + native PPO + final synchronization wall intervals. Setup,
real training warmup, status checks, logs, hashes and checkpoint I/O are
reported separately. CUDA events measure stream intervals, including waits;
they do not measure kernel busy time. No environment reset occurs at a horizon.

The optional factory must return a complete GpuSemanticDuel-compatible object:
    create_training_duel(config, *, library, export_path, solver_mode)
Both arms then use the existing GpuCandidateDummyDuel and native PPO bridge.
"""

from __future__ import annotations

import argparse
import array
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import socket
import sys
import time
import traceback


# Honor the caller's explicitly selected runtime before the checkout fallback.
DUEL_SOURCE = Path(__file__).resolve().parent.parent
if str(DUEL_SOURCE) not in sys.path:
    sys.path.append(str(DUEL_SOURCE))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve(strict=True)
    if any(part.lower().startswith("onedrive") for part in path.parts):
        raise ValueError("OneDrive paths are excluded from this benchmark")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def json_value(value):
    """Keep failure evidence serializable while preserving nonfinite identity."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, (Path, bytes)):
        return str(value) if isinstance(value, Path) else {"hex": value.hex()}
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as destination:
        json.dump(json_value(payload), destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")


def finite_numbers(value):
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_numbers(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def loaded_source_records():
    records = {"benchmark": file_record(__file__)}
    for name, module in sorted(sys.modules.copy().items()):
        if name.startswith(("gpu_", "gear_sonic_", "puffysics_")) or name in (
                "train_gpu_duel", "verify_gpu_duel", "profile_gpu_duel"):
            path = getattr(module, "__file__", None)
            if path and Path(path).is_file():
                records[name] = file_record(path)
    return records


def input_records(args, config):
    paths = {
        "gpu_duel_config": args.gpu_duel_config,
        "default_config": args.default_config,
        "native_config": args.native_config,
        "model": config.model,
        "motion_manifest": config.assets / "semantic_duel_assets_manifest.json",
        "motion_feature_manifest": config.motion_features / "foot_features_manifest.json",
        "controller_manifest": config.controller_manifest,
        "motion_library": config.motion_library,
        "combat_library": config.combat_library,
    }
    for key, path in (("fused_combat_library", config.fused_combat_library),
                      ("loaded_checkpoint", args.load_checkpoint),
                      ("puffysics_library", args.puffysics_library),
                      ("model_export", args.model_export)):
        if path is not None:
            paths[key] = path
    records = {key: file_record(path) for key, path in paths.items()}
    if args.load_checkpoint is not None:
        manifest = args.load_checkpoint.with_suffix(args.load_checkpoint.suffix + ".manifest.json")
        if manifest.exists():
            records["loaded_checkpoint_manifest"] = file_record(manifest)
    return records


class UpdateTimer:
    """Preallocated events around existing native API calls and their joins."""

    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        # Torch events allocate their CUDA resources lazily, outside timing.
        for event in self.events:
            event.record(torch.cuda.current_stream(device))
        self.events[-1].synchronize()

    def run(self, trainer, record):
        torch = self.torch
        stream = torch.cuda.current_stream(self.device)
        start, after_rollout, after_update = self.events
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        stage = "rollout"
        try:
            start.record(stream)
            phase_start = time.perf_counter()
            torch.cuda.nvtx.range_push("rollout")
            try:
                trainer.rollouts()
            finally:
                torch.cuda.nvtx.range_pop()
            record["rollout_host_call_seconds"] = time.perf_counter() - phase_start
            after_rollout.record(stream)
            record["rollout_returned"] = True
            stage = "ppo_update"
            phase_start = time.perf_counter()
            torch.cuda.nvtx.range_push("ppo_update")
            try:
                trainer.train()
            finally:
                torch.cuda.nvtx.range_pop()
            record["ppo_update_host_call_seconds"] = time.perf_counter() - phase_start
            record["ppo_update_returned"] = True
            after_update.record(stream)
            stage = "final_synchronization"
            phase_start = time.perf_counter()
            torch.cuda.synchronize(self.device)
            record["final_synchronization_seconds"] = time.perf_counter() - phase_start
            record["completed"] = True
        finally:
            if not record.get("completed"):
                sync_started = time.perf_counter()
                try:
                    torch.cuda.synchronize(self.device)
                    record["failure_synchronized"] = True
                except Exception as sync_error:
                    record["failure_synchronization_error"] = repr(sync_error)
                record["failure_synchronization_seconds"] = time.perf_counter() - sync_started
            record["wall_seconds"] = time.perf_counter() - start_wall
            record["process_cpu_seconds"] = time.process_time() - start_cpu
            record["last_timed_stage"] = stage
        record["rollout_cuda_stream_ms"] = start.elapsed_time(after_rollout)
        record["ppo_update_cuda_stream_ms"] = after_rollout.elapsed_time(after_update)
        record["cuda_stream_envelope_ms"] = start.elapsed_time(after_update)


class RolloutEventProxy:
    """Optional outer-API events, preserving native and environment streams.

    external_rollout_* receive the current trainer stream explicitly. The
    adapter step joins the duel stream before returning to that same stream.
    These intervals are nested within the rollout phase, never added to it.
    """

    NATIVE_PHASES = {
        "external_rollout_begin": "rollout_begin",
        "external_rollout_step": "policy_inference_and_storage",
        "external_actions_to_int32": "action_conversion",
        "external_rollout_finish": "bootstrap_and_rollout_finish",
    }

    def __init__(self, trainer, torch, horizon):
        self.trainer, self.torch = trainer, torch
        self.backend = trainer.backend
        self.original_step = trainer.env.step
        self.active = False
        self.position = 0
        self.records = []
        self.pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(3 * horizon + 2)]
        for pair in self.pairs:
            for event in pair:
                event.record(torch.cuda.current_stream(trainer.env.device))
        self.pairs[-1][-1].synchronize()
        self.functions = {
            name: self.wrap(phase, getattr(self.backend, name), native=True)
            for name, phase in self.NATIVE_PHASES.items()
        }
        trainer.backend = self
        trainer.env.step = self.wrap("environment_with_dummy_and_metrics", self.original_step)

    def __getattr__(self, name):
        if name in self.functions:
            return self.functions[name]
        return getattr(self.backend, name)

    def wrap(self, name, function, native=False):
        def call(*args, **kwargs):
            if not self.active:
                return function(*args, **kwargs)
            stream = self.torch.cuda.current_stream(self.trainer.env.device)
            if native and int(args[-1]) != int(stream.cuda_stream):
                raise RuntimeError("native API stream differs from event stream")
            before, after = self.pairs[self.position]
            self.position += 1
            before.record(stream)
            wall = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                duration = time.perf_counter() - wall
                after.record(stream)
                self.records.append((name, before, after, duration))
        return call

    def begin(self):
        self.position = 0
        self.records.clear()
        self.active = True

    def snapshot(self):
        self.active = False
        totals = defaultdict(lambda: {"calls": 0, "cuda_stream_elapsed_ms": 0., "host_call_seconds": 0.})
        for name, before, after, host in self.records:
            row = totals[name]
            row["calls"] += 1
            row["cuda_stream_elapsed_ms"] += before.elapsed_time(after)
            row["host_call_seconds"] += host
        return dict(totals)

    def restore(self):
        self.active = False
        self.trainer.backend = self.backend
        self.trainer.env.step = self.original_step


def summarize_updates(records, *, valid):
    completed = [row for row in records if row.get("completed")]
    wall = sum(row.get("wall_seconds", 0.) for row in records)
    transitions = sum(row.get("learner_transitions", 0) for row in completed)
    totals = {key: sum(row.get(key, 0.) for row in records) for key in (
        "wall_seconds", "process_cpu_seconds", "rollout_host_call_seconds",
        "ppo_update_host_call_seconds", "final_synchronization_seconds",
        "failure_synchronization_seconds",
        "rollout_cuda_stream_ms", "ppo_update_cuda_stream_ms", "cuda_stream_envelope_ms",
        "validation_and_logging_seconds")}
    envelope = totals["cuda_stream_envelope_ms"]
    return {
        "attempted_updates": len(records), "completed_updates": len(completed),
        "verified_updates": sum(bool(row.get("verified")) for row in records),
        "learner_transitions": transitions,
        "verified_learner_transitions": sum(row.get("learner_transitions", 0)
                                             for row in records if row.get("verified")),
        "training_learner_transitions_per_second": transitions / wall if valid and wall > 0 and completed else None,
        "cuda_stream_phase_percentages": {
            name: 100. * totals[key] / envelope if envelope > 0 else None
            for name, key in (("rollout", "rollout_cuda_stream_ms"), ("ppo_update", "ppo_update_cuda_stream_ms"))},
        "host_process_cpu_percent_of_one_core": 100. * totals["process_cpu_seconds"] / wall if wall > 0 else None,
        "timing": totals,
        "records": records,
    }


def physics_status(duel, wrapper, torch):
    duel.check_status()
    wrapper.log()
    for name in ("qpos", "qvel", "ctrl"):
        if not bool(torch.isfinite(getattr(duel.physics, name)).all().item()):
            raise RuntimeError(f"physics {name} contains nonfinite values")
    for name in ("rewards", "terminals"):
        if not bool(torch.isfinite(getattr(wrapper, name)).all().item()):
            raise RuntimeError(f"learner {name} contains nonfinite values")
    stats = getattr(duel.physics, "stats", None)
    if stats is None:
        return None
    result = stats()
    for key in ("nonfinite_arenas", "articulated_solver_failure_arenas", "contact_capacity_reached_arenas"):
        if result.get(key, 0):
            raise RuntimeError(f"invalid physics status: {json_value(result)}")
    return result


def checkpoint(trainer, path, args, reward_transform, save_training_weights):
    manifest = save_training_weights(trainer, path, args.policy_observation_encoder,
                                     args.policy_observation_warm_start, reward_transform)
    values = array.array("f")
    raw = Path(path).read_bytes()
    values.frombytes(raw)
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"checkpoint contains invalid native float32 weights: {path}")
    return {**file_record(path), "all_weights_finite": True, "manifest": manifest}


def component_probe(duel, wrapper, torch, steps):
    from profile_gpu_duel import ComponentEventProfiler

    profiler = ComponentEventProfiler(duel)
    actions = torch.ones((wrapper.rows, 1), device=wrapper.observations.device, dtype=torch.int32)
    try:
        duel.capture_step()
        wrapper.reset()
        for _ in range(steps):
            actions[:, 0].copy_(torch.where(wrapper.action_mask[:, 1] != 0, 1, 0))
            wrapper.step(actions)
            profiler.sample()
        physics_status(duel, wrapper, torch)
        return {"status": "passed", "scenario": "neutral learner versus candidate dummy",
                "after_primary_timing_and_final_checkpoint": True,
                "training_throughput_claim": False, **profiler.snapshot()}
    finally:
        profiler.restore()


def benchmark(args):
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    if any(part.lower().startswith("onedrive") for part in args.run_dir.resolve().parts):
        raise ValueError("OneDrive output is prohibited")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.run_dir / "report.json"
    report = {
        "schema": "rek.native_ppo_physics_comparison.v1", "status": "starting",
        "backend": args.backend, "host": socket.gethostname(),
        "command": {"argv": [sys.executable, *sys.argv], "posix_shell": shlex.join([sys.executable, *sys.argv]),
                    "cwd": os.getcwd(), "pythonpath": os.environ.get("PYTHONPATH", "")},
        "arguments": vars(args), "checkpoints": {}, "errors": [],
        "timing_contract": {
            "numerator": "learner rows times horizon for completed native PPO updates",
            "denominator": "sum of rollout plus native PPO plus final device synchronization wall intervals",
            "excluded": ["setup", "warmup updates", "status checks", "logs", "checkpoint I/O", "hashing", "diagnostic capture"],
            "environment_reset_each_horizon": False,
            "cuda_events": "elapsed sequential stream intervals including dependencies and host submission gaps",
            "kernel_busy_time_measured": False,
            "step_events_instrument_primary_run": args.step_timing,
            "measurement_regime": "cold_start_diagnostic" if args.warmup_updates == 0 else "after_warmup",
        },
        "claim_limits": {"authentic_rek_parity_established": False, "superhuman_claim_supported": False},
    }
    write_json(args.run_dir / "invocation.json", report)
    started = time.perf_counter()
    setup_start = started
    trainer = duel = wrapper = proxy = None
    profiler_active = False
    warmup_records, measured_records = [], []
    measured_envelope_start = None
    failure_stage = "setup"
    try:
        import torch
        from gpu_candidate_dummy import DUMMY_LABEL, GpuCandidateDummyDuel
        from gpu_native_puffer import NativeExternalGpuPuffer
        from gpu_puffer_env import CudaTensorEnvAdapter
        from gpu_policy_observation_encoder import load_policy_encoder_checkpoint
        from gpu_round_win_reward import resolve_reward_objective
        from train_gpu_duel import load_native_config, save_training_weights
        from verify_gpu_duel import load_config

        config = load_config(args.gpu_duel_config)
        overrides = {}
        if args.conditional_reset_forward is not None:
            overrides["conditional_reset_forward"] = args.conditional_reset_forward
        if args.fused_combat is False:
            overrides["fused_combat_library"] = None
        if args.fused_combat is True and config.fused_combat_library is None:
            raise ValueError("--fused-combat requires a fused library in the duel config")
        if args.defer_substep_combat_observations is not None:
            overrides["defer_substep_combat_observations"] = args.defer_substep_combat_observations
        config = replace(config, **overrides)
        report["effective_duel_config"] = asdict(config)
        report["inputs"] = input_records(args, config)
        native = load_native_config(args.default_config, args.native_config)
        policy_manifest, checkpoint_sha = load_policy_encoder_checkpoint(
            args.load_checkpoint, args.policy_observation_encoder,
            initialization=args.policy_observation_warm_start,
            expected_sha256=args.load_checkpoint_sha256)
        facing, round_win, reward_transform = resolve_reward_objective(
            native, objective=args.reward_objective, facing_scale=args.facing_potential_scale,
            margin_potential_scale=args.margin_potential_scale, margin_points=args.margin_points,
            reward_clip=args.reward_clip)
        report["training_reward_transform"] = reward_transform
        if args.backend == "mujoco":
            from gpu_semantic_duel import GpuSemanticDuel
            duel = GpuSemanticDuel(config)
        else:
            module_name, separator, factory_name = args.puffysics_factory.partition(":")
            if not separator:
                raise ValueError("puffysics factory must use module:function syntax")
            factory = getattr(importlib.import_module(module_name), factory_name)
            duel = factory(config, library=args.puffysics_library,
                           export_path=args.model_export, solver_mode=args.solver_mode)
        duel.capture_step()
        wrapper = GpuCandidateDummyDuel(
            duel, policy_observation_encoder=args.policy_observation_encoder,
            checkpoint_manifest=policy_manifest, checkpoint_sha256=checkpoint_sha,
            policy_observation_initialization=args.policy_observation_warm_start,
            facing_potential_config=facing, round_win_reward_config=round_win)
        adapter = CudaTensorEnvAdapter(wrapper, (33,), metric_plugins=(wrapper.metric_plugin,))
        batch = wrapper.rows * args.horizon
        minibatch = args.minibatch_size or batch
        if minibatch < args.horizon or minibatch > batch or minibatch % args.horizon:
            raise ValueError("minibatch must be horizon-divisible and no larger than learner rows times horizon")
        total_steps = (args.warmup_updates + args.epochs) * batch
        schedule_steps = args.schedule_total_timesteps or total_steps
        if schedule_steps < total_steps:
            raise ValueError("learning-rate schedule must cover warmup and measured transitions")
        native["vec"].update(total_agents=wrapper.rows, num_buffers=1, num_threads=0)
        native["train"].update(total_timesteps=schedule_steps, horizon=args.horizon,
                                minibatch_size=minibatch, gpus=1)
        native.update(gpu_id=torch.device(config.device).index or 0, world_size=1, rank=0, nccl_id=b"")
        if float(native["train"]["replay_ratio"]) * batch / minibatch < 1:
            raise ValueError("native config would execute zero PPO minibatches")
        report["effective_native_config"] = native
        trainer = NativeExternalGpuPuffer(native, adapter, reward_clip=args.reward_clip)
        if args.load_checkpoint is not None:
            trainer.load_weights(args.load_checkpoint)
        active_reward = round_win or facing
        if active_reward is not None:
            active_reward.validate_training_discount(float(trainer.pufferl.hypers.gamma),
                                                       reward_clip=float(trainer.pufferl.hypers.reward_clip))
        report["native_manifest"] = trainer.manifest()
        report["native_extension"] = {**file_record(trainer.backend.__file__),
                                      "precision_bytes": int(trainer.backend.precision_bytes)}
        report["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                             "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(adapter.device),
                             "device": str(adapter.device), "platform": platform.platform()}
        report["training"] = {
            "learner_rows": wrapper.rows, "physical_fighters": duel.rows, "arenas": duel.rows // 2,
            "horizon": args.horizon, "minibatch_size": minibatch, "opponent": DUMMY_LABEL,
            "opponent_samples_in_ppo": False, "policy_parameters": trainer.num_params(),
            "native_cpu_environment_count": int(trainer.pufferl.native_env_count),
            "native_environment_threads": bool(trainer.pufferl.has_env_threads),
            "cpu_physics_steps": 0, "cpu_controller_inferences": 0,
            "physics_metadata": getattr(duel.physics, "metadata", None),
        }
        report["controller_models"] = {role: file_record(getattr(duel.controller, role + "_identity").path)
                                       for role in ("encoder", "decoder")}
        report["checkpoints"]["initial"] = checkpoint(
            trainer, args.run_dir / "initial.bin", args, reward_transform, save_training_weights)
        physics_status(duel, wrapper, torch)
        # Clear native init-time capture losses before the genuine warmup.
        trainer.log(clear_metrics=False)
        timer = UpdateTimer(torch, adapter.device)
        if args.step_timing:
            proxy = RolloutEventProxy(trainer, torch, args.horizon)
        torch.cuda.synchronize(adapter.device)
        report["setup_seconds"] = time.perf_counter() - setup_start

        def update(phase, index, records):
            row = {"index": index, "completed": False, "verified": False,
                   "global_step_before": trainer.global_step, "native_epoch_before": trainer.epoch}
            records.append(row)
            if proxy is not None and phase == "measured":
                proxy.begin()
            timer.run(trainer, row)
            row["global_step_after"] = trainer.global_step
            row["native_epoch_after"] = trainer.epoch
            row["learner_transitions"] = trainer.global_step - row["global_step_before"]
            check_started = time.perf_counter()
            try:
                if proxy is not None and phase == "measured":
                    row["rollout_subphases"] = proxy.snapshot()
                if row["learner_transitions"] != batch or trainer.epoch - row["native_epoch_before"] != 1:
                    raise RuntimeError("native update counters do not match one complete learner rollout and PPO update")
                row["physics_status"] = physics_status(duel, wrapper, torch)
                log = trainer.log(clear_metrics=False)
                row["native_loss"] = log.get("loss", {})
                if not row["native_loss"] or not finite_numbers(row["native_loss"]):
                    raise RuntimeError("native PPO produced missing or nonfinite loss metrics")
                report["latest_native_log"] = log
                row["verified"] = True
            finally:
                row["validation_and_logging_seconds"] = time.perf_counter() - check_started
            print(json.dumps({"phase": phase, "update": index, "learner_transitions": row["learner_transitions"],
                              "training_wall_seconds": row["wall_seconds"], "native_loss": row["native_loss"]},
                             sort_keys=True, allow_nan=False), flush=True)

        failure_stage = "warmup"
        warmup_start = time.perf_counter()
        for index in range(args.warmup_updates):
            update("warmup", index + 1, warmup_records)
        report["warmup_envelope_seconds"] = time.perf_counter() - warmup_start
        report["checkpoints"]["measured_start"] = checkpoint(
            trainer, args.run_dir / "measured-start.bin", args, reward_transform, save_training_weights)
        report["metrics_at_measured_start"] = wrapper.metric_plugin.snapshot(clear=False)
        # Keep cumulative metric state and environment state intact across the boundary.
        torch.cuda.synchronize(adapter.device)
        failure_stage = "measured"
        if args.cuda_profiler_range:
            result = torch.cuda.cudart().cudaProfilerStart()
            if result is not None and int(result) != 0:
                raise RuntimeError(f"cudaProfilerStart failed: {result}")
            profiler_active = True
        measured_envelope_start = time.perf_counter()
        for index in range(args.epochs):
            update("measured", index + 1, measured_records)
        report["measured_envelope_seconds"] = time.perf_counter() - measured_envelope_start
        measured_envelope_start = None
        if profiler_active:
            result = torch.cuda.cudart().cudaProfilerStop()
            profiler_active = False
            if result is not None and int(result) != 0:
                raise RuntimeError(f"cudaProfilerStop failed: {result}")
        failure_stage = "final_verification"
        report["metrics_at_measured_end"] = wrapper.metric_plugin.snapshot(clear=False)
        report["metrics_scope"] = "cumulative changing-policy combat; start and end snapshots delimit measured interval"
        report["checkpoints"]["measured_end"] = checkpoint(
            trainer, args.run_dir / "measured-end.bin", args, reward_transform, save_training_weights)
        report["checkpoints"]["measured_weights_changed"] = (
            report["checkpoints"]["measured_start"]["sha256"] != report["checkpoints"]["measured_end"]["sha256"])
        if not report["checkpoints"]["measured_weights_changed"]:
            raise RuntimeError("native weights did not change during measured PPO updates")
        report["final_physics_status"] = physics_status(duel, wrapper, torch)
        report["status"] = "passed"
        if proxy is not None:
            proxy.restore()
            proxy = None
        if args.component_steps:
            diagnostic_start = time.perf_counter()
            try:
                report["component_diagnostic"] = component_probe(duel, wrapper, torch, args.component_steps)
            except Exception as error:
                report["component_diagnostic"] = {"status": "failed", "error": repr(error),
                                                  "traceback": traceback.format_exc()}
            report["component_diagnostic"]["wall_seconds"] = time.perf_counter() - diagnostic_start
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"stage": failure_stage, "error": repr(error), "traceback": traceback.format_exc()})
        if failure_stage == "setup":
            report["setup_seconds"] = time.perf_counter() - setup_start
        if measured_envelope_start is not None:
            report["measured_envelope_seconds"] = time.perf_counter() - measured_envelope_start
        if trainer is not None:
            report["native_counters_at_failure"] = {"global_step": trainer.global_step, "epoch": trainer.epoch}
            capture_started = time.perf_counter()
            try:
                # The native log is independent of the invalid environment's log.
                # Preserve the actual learner loss without clearing its status latch.
                report["native_log_at_failure"] = dict(trainer.backend.log(trainer.pufferl))
            except Exception as capture_error:
                report["errors"].append({"stage": "failure_loss", "error": repr(capture_error)})
            try:
                report["checkpoints"]["failed_diagnostic"] = checkpoint(
                    trainer, args.run_dir / "failed-diagnostic.bin", args, reward_transform, save_training_weights)
                previous = report["checkpoints"].get("measured_start", report["checkpoints"].get("initial"))
                report["checkpoints"]["failed_diagnostic_weights_changed"] = (
                    previous["sha256"] != report["checkpoints"]["failed_diagnostic"]["sha256"] if previous else None)
            except Exception as capture_error:
                report["errors"].append({"stage": "failure_checkpoint", "error": repr(capture_error)})
            report["failure_capture_seconds"] = time.perf_counter() - capture_started
        if duel is not None and hasattr(duel.physics, "stats"):
            try:
                report["physics_failure_evidence"] = duel.physics.stats()
            except Exception as capture_error:
                report["errors"].append({"stage": "failure_stats", "error": repr(capture_error)})
    finally:
        if profiler_active:
            try:
                torch.cuda.cudart().cudaProfilerStop()
            except Exception as stop_error:
                report["errors"].append({"stage": "profiler_stop", "error": repr(stop_error)})
        if proxy is not None:
            proxy.restore()
        try:
            if trainer is not None:
                trainer.close()
            elif wrapper is not None:
                wrapper.close()
            elif duel is not None:
                duel.close()
        except Exception as close_error:
            report["errors"].append({"stage": "cleanup", "error": repr(close_error)})
        report["warmup"] = summarize_updates(warmup_records, valid=False)
        report["measured"] = summarize_updates(measured_records, valid=report["status"] == "passed")
        try:
            report["sources"] = loaded_source_records()
        except Exception as source_error:
            report["errors"].append({"stage": "source_hashes", "error": repr(source_error)})
        report["total_process_work_seconds"] = time.perf_counter() - started
        write_json(report_path, report)
    print(json.dumps({"report": str(report_path.resolve()), "status": report["status"],
                      "completed_updates": report["measured"]["completed_updates"],
                      "training_learner_transitions_per_second": report["measured"]["training_learner_transitions_per_second"]},
                     sort_keys=True, allow_nan=False), flush=True)
    return report


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("mujoco", "puffysics"), required=True)
    parser.add_argument("--gpu-duel-config", "--config", dest="gpu_duel_config", type=Path, required=True)
    parser.add_argument("--default-config", type=Path, default=root / "config" / "default.ini")
    parser.add_argument("--native-config", type=Path, default=root / "config" / "rek_g1.ini")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup-updates", type=int, default=2,
                        help="zero explicitly selects a cold-start diagnostic, not steady-state throughput")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--schedule-total-timesteps", type=int, default=0,
                        help="optional fixed annealing schedule, at least warmup plus measured transitions")
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--load-checkpoint-sha256")
    parser.add_argument("--policy-observation-encoder", default="raw")
    parser.add_argument("--policy-observation-warm-start", default="matching-checkpoint")
    parser.add_argument("--reward-objective", choices=("score-delta", "round-win"), default="score-delta")
    parser.add_argument("--reward-clip", type=float, default=0.)
    parser.add_argument("--facing-potential-scale", type=float, default=0.)
    parser.add_argument("--margin-potential-scale", type=float, default=0.5)
    parser.add_argument("--margin-points", type=float, default=5.)
    parser.add_argument("--puffysics-library", "--library", dest="puffysics_library", type=Path)
    parser.add_argument("--model-export", type=Path)
    parser.add_argument("--solver-mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--puffysics-factory", default="puffysics_training_duel:create_training_duel")
    parser.add_argument("--conditional-reset-forward", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fused-combat", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--defer-substep-combat-observations", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--step-timing", action="store_true", help="instrument primary rollout API calls with CUDA events")
    parser.add_argument("--cuda-profiler-range", action="store_true",
                        help="CUDA profiler start/stop around measured training for Nsight capture-range=cudaProfilerApi")
    parser.add_argument("--component-steps", type=int, default=0,
                        help="separate neutral-action diagnostic capture after final training checkpoint")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.warmup_updates < 0 or args.horizon < 2:
        parser.error("epochs must be positive, warmup-updates nonnegative, and horizon must exceed one")
    if args.minibatch_size < 0 or args.schedule_total_timesteps < 0 or not 0 <= args.component_steps <= 500:
        parser.error("minibatch and schedule must be nonnegative; component steps must be 0 to 500")
    if args.backend == "puffysics" and (args.puffysics_library is None or args.model_export is None):
        parser.error("Puffysics requires --puffysics-library and --model-export")
    return args


def main(argv=None):
    return 0 if benchmark(parse_args(argv))["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
