"""Frozen native policies sharing one complete CUDA candidate physics batch."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import socket
from types import SimpleNamespace
import time


@dataclass(frozen=True)
class PolicyGroup:
    name: str
    arenas: int
    checkpoint: Path
    checkpoint_sha256: str
    horizon: int = 64
    greedy: bool = False
    encoder: str = "raw"
    seed: int = 73


def validate_groups(specs, arenas, ticks):
    if not specs or sum(spec.arenas for spec in specs) != arenas:
        raise ValueError("policy groups must partition the shared arenas exactly")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("policy group names must be unique")
    for spec in specs:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", spec.name) or spec.arenas < 1:
            raise ValueError("group needs a safe name and positive arena count")
        if spec.horizon < 4 or spec.horizon % 4 or ticks <= 0 or ticks % spec.horizon:
            raise ValueError("ticks must contain complete native four-aligned horizons")


def grouped_ticks(groups, *, ticks, stream_pointer, physical_step, after_step):
    """One physical advance after every group has selected this tick's actions."""
    for tick in range(ticks):
        for group in groups:
            trainer, local = group.trainer, tick % group.spec.horizon
            native = trainer.backend
            if local == 0:
                native.external_rollout_begin(trainer.pufferl, stream_pointer)
            native.external_rollout_step(trainer.pufferl, local, stream_pointer)
            native.external_actions_to_int32(trainer.pufferl, group.actions.data_ptr(), stream_pointer)
        physical_step()
        after_step(tick)
        for group in groups:
            if (tick + 1) % group.spec.horizon == 0:
                group.trainer.backend.external_rollout_finish(group.trainer.pufferl, stream_pointer)


class _PolicyRows:
    def __init__(self, candidate, first, count, encoder):
        rows = slice(first, first + count)
        self.raw = candidate.observations[rows]
        self.encoder = encoder
        self.observations = self.raw if encoder is None else encoder.encode(self.raw)
        self.rewards, self.terminals = candidate.rewards[rows], candidate.terminals[rows]
        self.action_mask = candidate.action_mask[rows]

    def refresh(self):
        if self.encoder is not None:
            self.encoder.encode(self.raw)

    def reset(self):
        raise RuntimeError("individual policy groups cannot reset shared physics")

    def step(self, actions):
        raise RuntimeError("use grouped_ticks to advance all groups together")

    def close(self):
        pass


def evaluate_grouped(candidate, specs, native_args, *, ticks, output, provenance=None):
    """Run an already-created raw candidate; caller retains physics ownership."""
    import torch
    from evaluate_gpu_dummy import (RoundJournal, evaluation_config, file_record,
                                   ongoing_snapshot, pinned_checkpoint, save_frozen_weights)
    from gpu_behavior_metrics import GpuBehaviorMetricCollector
    from gpu_candidate_dummy import DUMMY_LABEL
    from gpu_metrics import RekG1GpuMetricCollector
    from gpu_native_puffer import NativeExternalGpuPuffer
    from gpu_policy_observation_encoder import (GpuPolarXYPolicyEncoder, GpuScaledPolarXYPolicyEncoder,
                                               load_policy_encoder_checkpoint, policy_encoder_report)
    from gpu_puffer_env import CudaTensorEnvAdapter

    validate_groups(specs, candidate.rows, ticks)
    if candidate.policy_encoder is not None:
        raise ValueError("shared candidate must publish raw observations for per-policy encoding")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    checked = [(pinned_checkpoint(s.checkpoint, s.checkpoint_sha256),
                load_policy_encoder_checkpoint(s.checkpoint, s.encoder, expected_sha256=s.checkpoint_sha256)) for s in specs]
    device, duel = candidate.observations.device, candidate.duel
    encoders = {"polar_xy_v1": GpuPolarXYPolicyEncoder, "scaled_polar_xy_v1": GpuScaledPolarXYPolicyEncoder}
    output.mkdir(parents=True)
    candidate.reset()
    actions = torch.empty_like(candidate.learner_actions)
    groups, reports, events_file = [], [], (output / "rounds.jsonl").open("x", encoding="utf-8")
    setup_start = time.perf_counter()
    try:
        first = 0
        for spec, (checkpoint, (manifest, sha)) in zip(specs, checked):
            encoded = None if spec.encoder == "raw" else encoders[spec.encoder](
                spec.arenas, device, checkpoint_manifest=manifest, checkpoint_sha256=sha)
            rows = _PolicyRows(candidate, first, spec.arenas, encoded)
            args = evaluation_config(native_args, fighters=2 * spec.arenas, horizon=spec.horizon,
                                     ticks=ticks, seed=spec.seed, greedy=spec.greedy)
            trainer = NativeExternalGpuPuffer(args, CudaTensorEnvAdapter(rows, (33,)),
                                             reward_clip=0.0, reset_environment=False)
            view = slice(2 * first, 2 * (first + spec.arenas))
            group = SimpleNamespace(spec=spec, trainer=trainer, actions=actions[first:first + spec.arenas],
                                    rows=rows, view=view, first=first, checkpoint=checkpoint, events=[],
                                    combat=RekG1GpuMetricCollector(2 * spec.arenas, device),
                                    behavior=GpuBehaviorMetricCollector(2 * spec.arenas, device, learner_rows=tuple(range(0, 2 * spec.arenas, 2))),
                                    journal=RoundJournal(spec.arenas, spec.horizon, device))
            groups.append(group)
            trainer.load_weights(spec.checkpoint)
            group.before = save_frozen_weights(trainer, output / spec.name / "policy-before.bin", sha, spec.encoder)
            group.initial_qpos_sha = __import__("hashlib").sha256(duel.physics.qpos[first:first + spec.arenas].detach().cpu().numpy().tobytes()).hexdigest()
            first += spec.arenas
        def publish_groups():
            for group in groups:
                sl = group.view
                obs, terminal = duel.observations[sl], duel.terminals[sl]
                group.rows.refresh()
                group.combat.update(obs, terminal)
                group.behavior.update(obs, terminal, candidate.full_actions[sl], duel.scheduler.move_start_edge[sl], duel.combat.tick_score_delta[sl])

        publish_stream = torch.cuda.Stream(device=device)
        publish_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(publish_stream):
            publish_groups()
            publish_stream.synchronize()
            publication = torch.cuda.CUDAGraph()
            with torch.cuda.graph(publication, stream=publish_stream):
                publish_groups()
        publish_stream.synchronize()
        for group in groups:
            group.combat.reset()
            group.behavior.reset()
            if group.rows.encoder is not None:
                group.rows.encoder.reset_status()
        torch.cuda.synchronize(device)
        setup_seconds = time.perf_counter() - setup_start
        stream = torch.cuda.current_stream(device)

        def after_step(tick):
            publication.replay()
            for group in groups:
                obs, terminal = duel.observations[group.view], duel.terminals[group.view]
                group.journal.record(obs, terminal)
                if (tick + 1) % group.spec.horizon == 0:
                    for event in group.journal.drain(tick + 1 - group.spec.horizon):
                        event.update(group=group.spec.name, global_arena=group.first + event["arena"])
                        group.events.append(event)
                        events_file.write(json.dumps(event, sort_keys=True) + "\n")
            if (tick + 1) % 256 == 0:
                duel.check_status()
                candidate.dummy.check_status()
                events_file.flush()
                print(json.dumps({"control_ticks_per_arena": tick + 1,
                                  "completed_rounds_by_group": {g.spec.name: len(g.events) for g in groups}}, sort_keys=True), flush=True)

        start, cpu_start = time.perf_counter(), time.process_time()
        grouped_ticks(groups, ticks=ticks, stream_pointer=int(stream.cuda_stream),
                      physical_step=lambda: candidate.step(actions), after_step=after_step)
        torch.cuda.synchronize(device)
        wall, cpu = time.perf_counter() - start, time.process_time() - cpu_start
        for group in groups:
            spec = group.spec
            if group.trainer.global_step != ticks * spec.arenas:
                raise RuntimeError("group native learner-step count differs from executed ticks")
            duel.check_status()
            candidate.dummy.check_status()
            if group.rows.encoder is not None:
                group.rows.encoder.check_status()
            after = save_frozen_weights(group.trainer, output / spec.name / "policy-after.bin", spec.checkpoint_sha256, spec.encoder)
            pinned_checkpoint(spec.checkpoint, spec.checkpoint_sha256)
            metrics, behavior = group.combat.snapshot(clear=False), group.behavior.snapshot(clear=False)
            if len(group.events) != metrics["n"] or len(group.events) != behavior["completed_rounds"]:
                raise RuntimeError("group round journal disagrees with canonical metrics")
            partial_duel = SimpleNamespace(observations=duel.observations[group.view], terminals=duel.terminals[group.view])
            report = {"schema": "rek.g1_frozen_native_candidate_dummy_evaluation.v1", "status": "completed", "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(device),
                      **(provenance or {}), **policy_encoder_report(spec.encoder), "group": {"name": spec.name, "first_arena": group.first, "arenas": spec.arenas},
                      "checkpoints": {"before": group.before, "after": after},
                      "inputs": {**((provenance or {}).get("inputs", {})), "checkpoint": group.checkpoint},
                      "evaluation": {"arenas": spec.arenas, "physical_fighters": 2 * spec.arenas, "horizon": spec.horizon, "seed": spec.seed,
                                     "sampling": "native masked argmax" if spec.greedy else "native stochastic categorical", "opponent": DUMMY_LABEL,
                                     "weights_unchanged": True, "policy_updates": 0, "reset_state_each_horizon": bool(native_args["reset_state"]),
                                     "native_cpu_environment_count": 0, "native_environment_threads": False, "cpu_physics_steps": 0, "cpu_controller_inferences": 0,
                                     "actual_control_ticks_per_arena": ticks, "requested_control_ticks_per_arena": ticks, "learner_control_steps": ticks * spec.arenas,
                                     "simulated_seconds_per_arena": ticks * .02, "initial_qpos_sha256": group.initial_qpos_sha,
                                     "instrumented_evaluation_wall_seconds": wall, "host_cpu_seconds": cpu, "setup_seconds": setup_seconds,
                                     "instrumented_evaluation_learner_steps_per_second": ticks * spec.arenas / wall,
                                     "timing_scope": "shared concurrent batch; do not sum group wall times or interpret as standalone speed"},
                      "completed_rounds": len(group.events), "round_events": group.events,
                      "cutoff_ongoing_rounds": ongoing_snapshot(partial_duel, group.journal),
                      "final_native_log": {"env": {**metrics, "behavior": behavior}},
                      "native_extension": file_record(group.trainer.backend.__file__),
                      "claim_limits": ["Frozen grouped evaluation of the candidate dummy, not authentic REK or human performance."]}
            (output / spec.name / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
            reports.append(report)
        return reports
    finally:
        events_file.close()
        for group in reversed(groups):
            group.trainer.close()
