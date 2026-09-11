"""Bounded CUDA stream timing for native training and candidate components.

CUDA events measure elapsed stream intervals, not GPU kernel busy time. Host
call durations overlap those intervals and must never be added to them. The
component probe inserts event nodes in a separate diagnostic capture and is
explicitly distinguished from the uninstrumented throughput measurement.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import socket
import time

import torch


class TrainingPhaseTimer:
    """Time sequential, nonnested API phases without per-phase synchronization.

    Calls must establish their GPU dependencies with the current torch stream,
    as NativeExternalGpuPuffer does. snapshot is the synchronization boundary.
    Initialization, asset loading and CUDA compilation precede this timer.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("training phase timer requires a CUDA device")
        self.records = []
        self._inside = False
        torch.cuda.synchronize(self.device)
        self.wall_start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.begin = torch.cuda.Event(enable_timing=True)
        self.begin.record(torch.cuda.current_stream(self.device))

    def call(self, name, function, *args, **kwargs):
        if self._inside:
            raise RuntimeError("training phases must be sequential, not nested")
        self._inside = True
        before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before.record(torch.cuda.current_stream(self.device))
        wall = time.perf_counter()
        cpu = time.process_time()
        try:
            return function(*args, **kwargs)
        finally:
            host_cpu = time.process_time() - cpu
            host_wall = time.perf_counter() - wall
            after.record(torch.cuda.current_stream(self.device))
            self.records.append((str(name), before, after, host_wall, host_cpu))
            self._inside = False

    def snapshot(self):
        if self._inside:
            raise RuntimeError("cannot snapshot inside a timing phase")
        end = torch.cuda.Event(enable_timing=True)
        end.record(torch.cuda.current_stream(self.device))
        end.synchronize()
        wall = time.perf_counter() - self.wall_start
        cpu = time.process_time() - self.cpu_start
        envelope_ms = self.begin.elapsed_time(end)
        totals = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
        for name, before, after, host_wall, host_cpu in self.records:
            row = totals[name]
            row[0] += 1
            row[1] += before.elapsed_time(after)
            row[2] += host_wall
            row[3] += host_cpu
        measured_ms = sum(row[1] for row in totals.values())
        return {
            "schema": "rek.g1_cuda_training_phase_timing.v1",
            "wall_seconds": wall,
            "process_cpu_seconds": cpu,
            "process_cpu_percent_of_one_core": 100 * cpu / wall,
            "cuda_stream_envelope_ms": envelope_ms,
            "phases": {
                name: {
                    "calls": row[0], "cuda_stream_elapsed_ms": row[1],
                    "percent_cuda_stream_envelope": 100 * row[1] / envelope_ms,
                    "host_call_wall_seconds": row[2],
                    "percent_host_wall": 100 * row[2] / wall,
                    "host_process_cpu_seconds": row[3],
                }
                for name, row in sorted(totals.items())
            },
            "unscoped_cuda_stream_elapsed_ms": max(0.0, envelope_ms - measured_ms),
            "unscoped_host_wall_seconds": max(0.0, wall - sum(row[2] for row in totals.values())),
            "kernel_busy_time_ms": None,
            "interpretation": [
                "CUDA percentages use one sequential current-stream envelope, including stream waits and host submission gaps.",
                "Host-call wall durations and process CPU durations overlap CUDA execution; do not add these percentages.",
                "Rollout includes policy inference, environment, fixed opponent, observations and metrics.",
                "Nested component proportions come from a separate instrumented environment probe, not an additive training breakdown.",
                "Kernel busy time and hardware utilization require a separate device activity trace and are not inferred here.",
            ],
        }


class ComponentEventProfiler:
    """Insert external event nodes around leaf operations in one CUDA graph."""

    METHODS = {
        "physics_step": (("physics", "step"),),
        "reset_forward_selected": (("physics", "forward_selected"),),
        "controller_inference": (("controller", "encode"), ("controller", "decode")),
        "semantic_motion": (("scheduler", "pre_step"), ("scheduler", "post_step"), ("scheduler", "reset_rows")),
        "combat_measurement_and_referee": (
            ("measurement", "sample"), ("measurement", "sample_fall"),
            ("combat", "begin_tick"), ("combat", "post_step"), ("combat", "observe"),
        ),
        "state_and_observation": (
            ("observer", "gather_kinematics"), ("observer", "pack"),
            ("robot_state", "prepare"), ("robot_state", "decoder_input"),
            ("robot_state", "apply_actions"), ("robot_state", "reset"),
        ),
        "actuator_drive": (
            ("drive", "prepare"), ("drive", "set_dampened"),
            ("drive", "begin_reset"), ("drive", "complete_reset"),
        ),
    }

    def __init__(self, duel):
        self.duel = duel
        self.device = duel.actions.device
        self.originals = []
        self.active = False
        self.depth = 0
        self.positions = {}
        self.layout = []
        self.totals = defaultdict(float)
        self.calls = defaultdict(int)
        self.tick_ms = 0.0
        self.samples = 0
        self.tick_events = self._pair()
        for category, entries in self.METHODS.items():
            for object_name, method_name in entries:
                owner = getattr(duel, object_name)
                original = getattr(owner, method_name)
                key = object_name + "." + method_name
                # All events exist before capture. Each method runs at most
                # eleven times in the fixed ten-substep control graph.
                pairs = [self._pair() for _ in range(16)]
                self.originals.append((owner, method_name, original))
                setattr(owner, method_name, self._wrap(original, key, category, pairs))
        original_step = duel._step_impl
        self.originals.append((duel, "_step_impl", original_step))

        def step():
            self.positions.clear()
            self.layout.clear()
            self.active = True
            self.tick_events[0].record()
            try:
                return original_step()
            finally:
                self.tick_events[1].record()
                self.active = False

        duel._step_impl = step

    @staticmethod
    def _pair():
        try:
            return tuple(torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
        except TypeError as error:
            raise RuntimeError("component graph profiling needs torch CUDA external events") from error

    def _wrap(self, original, key, category, pairs):
        def wrapped(*args, **kwargs):
            if not self.active or self.depth:
                return original(*args, **kwargs)
            index = self.positions.get(key, 0)
            if index >= len(pairs):
                raise RuntimeError(f"component event capacity exceeded for {key}")
            self.positions[key] = index + 1
            before, after = pairs[index]
            self.layout.append((category, before, after))
            self.depth += 1
            before.record()
            try:
                return original(*args, **kwargs)
            finally:
                after.record()
                self.depth -= 1
        return wrapped

    def sample(self):
        """Read the most recent replay after an explicit diagnostic boundary."""
        self.tick_events[1].synchronize()
        self.tick_ms += self.tick_events[0].elapsed_time(self.tick_events[1])
        for category, before, after in self.layout:
            self.totals[category] += before.elapsed_time(after)
            self.calls[category] += 1
        self.samples += 1

    def restore(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)

    def snapshot(self):
        if not self.samples:
            raise RuntimeError("no instrumented graph samples")
        attributed = sum(self.totals.values())
        if attributed > self.tick_ms + 0.01:
            raise RuntimeError("component timings overlap the enclosing graph interval")
        return {
            "samples": self.samples, "cuda_graph_interval_ms": self.tick_ms,
            "components": {
                name: {"cuda_stream_elapsed_ms": value,
                       "percent_graph_interval": 100 * value / self.tick_ms,
                       "measured_calls": self.calls[name]}
                for name, value in sorted(self.totals.items())
            },
            "unattributed_cuda_stream_elapsed_ms": max(0.0, self.tick_ms - attributed),
            "unattributed_percent_graph_interval": 100 * max(0.0, self.tick_ms - attributed) / self.tick_ms,
            "limits": [
                "Timing event nodes instrument a separate CUDA graph and perturb its duration.",
                "Components are nonoverlapping leaf intervals; nested measurement calls are counted once.",
                "Unattributed time includes reset tensor operations, control copies and instrumentation overhead.",
                "Environment-only denominator excludes learner PPO, opponent selection and wrapper metric graphs.",
            ],
        }


def profile(args):
    from gpu_candidate_dummy import GpuCandidateDummyDuel
    from gpu_semantic_duel import GpuSemanticDuel
    from verify_gpu_duel import load_config

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1 or args.steps > 500:
        raise ValueError("bounded profiling requires 1 to 500 control steps")
    nsight_range_only = getattr(args, "nsight_range_only", False)
    duel = GpuSemanticDuel(load_config(args.gpu_duel_config))
    duel.capture_step()
    wrapper = GpuCandidateDummyDuel(duel)
    actions = torch.ones((wrapper.rows, 1), device=duel.actions.device, dtype=torch.int32)

    def step():
        # A declared neutral learner isolates the same standard dummy used by
        # the human evaluator. Respect the native continue-only busy mask.
        actions[:, 0].copy_(torch.where(wrapper.action_mask[:, 1] != 0, 1, 0))
        wrapper.step(actions)

    try:
        for _ in range(10):
            step()
        wrapper.reset()
        torch.cuda.synchronize(duel.actions.device)
        if nsight_range_only:
            torch.cuda.nvtx.range_push("rek_baseline_rollout")
        started, cpu_started = time.perf_counter(), time.process_time()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize(duel.actions.device)
        wall = time.perf_counter() - started
        cpu = time.process_time() - cpu_started
        if nsight_range_only:
            torch.cuda.nvtx.range_pop()
        baseline_metrics = wrapper.behavior_metrics.snapshot(clear=False)
        components = None
        if not nsight_range_only:
            instrumentation = ComponentEventProfiler(duel)
            try:
                duel.capture_step()
                wrapper.reset()
                for _ in range(args.steps):
                    step()
                    instrumentation.sample()
                components = instrumentation.snapshot()
            finally:
                instrumentation.restore()
        duel.check_status()
        wrapper.dummy.check_status()
        report = {
            "schema": "rek.g1_cuda_component_profile.v1", "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(duel.actions.device),
            "config_path": str(args.gpu_duel_config),
            "config_sha256": hashlib.sha256(args.gpu_duel_config.read_bytes()).hexdigest(),
            "scenario": "neutral learner versus human-evaluator candidate approach dummy",
            "nsight_capture_range": "rek_baseline_rollout" if nsight_range_only else None,
            "external_profiler_capture_requested": nsight_range_only,
            ("external_profiler_baseline" if nsight_range_only else "uninstrumented_baseline"): {
                "control_ticks": args.steps, "learner_rows": wrapper.rows,
                "simulated_fighter_rows": duel.rows,
                "learner_control_steps_per_second": wrapper.rows * args.steps / wall,
                "fighter_control_steps_per_second": duel.rows * args.steps / wall,
                "wall_seconds": wall, "host_process_cpu_seconds": cpu,
                "training_steps_per_second": None,
                "behavior": baseline_metrics,
            },
            "instrumented_environment_graph": components,
            "claim_limits": "This is a bounded candidate environment diagnostic, not a policy training benchmark or authentic REK parity proof.",
        }
    finally:
        wrapper.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-duel-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--nsight-range-only", action="store_true")
    profile(parser.parse_args())


if __name__ == "__main__":
    main()
