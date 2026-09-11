"""Same-graph reset-skip diagnostic with fully restored temporary storage.

No epsilon is added. Report exact invariants and the existing paired-repeat
variance criterion separately. This script performs GPU work only when invoked.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import socket

import numpy as np


def paired_error(a, b, c):
    """Use the prior diagnostic's paired bootstrap rule without a floor."""
    a, b, c = [value.astype(np.float64).reshape(len(value), -1) for value in (a, b, c)]
    if not all(np.isfinite(value).all() for value in (a, b, c)):
        return {"finite": False, "accepted": False}
    baseline = a - b
    candidate = a - c
    baseline_ms = np.mean(baseline * baseline, axis=1)
    candidate_ms = np.mean(candidate * candidate, axis=1)
    baseline_rms = float(np.sqrt(np.mean(baseline_ms)))
    candidate_rms = float(np.sqrt(np.mean(candidate_ms)))
    indices = np.random.default_rng(5217).integers(0, len(a), size=(2000, len(a)))
    extra = candidate_ms - baseline_ms
    upper = float(np.sqrt(max(0.0, np.quantile(extra[indices].mean(axis=1), .95))))
    return {
        "finite": True,
        "baseline_max_abs": float(np.abs(baseline).max()),
        "candidate_max_abs": float(np.abs(candidate).max()),
        "baseline_rms": baseline_rms,
        "candidate_rms": candidate_rms,
        "bootstrap95_upper_extra_rms": upper,
        "accepted": upper <= baseline_rms,
        "all_exact": bool(np.array_equal(a, b) and np.array_equal(a, c)),
    }


def contact_fingerprint(snapshot):
    """Separate raw slot ordering, within-world ordering, and pair content."""
    count = int(snapshot["nacon"].reshape(-1)[0])
    worlds = snapshot["contact.worldid"].reshape(-1)
    geometry = snapshot["contact.geom"].reshape(-1, 2)
    if not 0 <= count <= len(worlds):
        return {"valid_capacity": False, "count": count}
    keys = np.column_stack((worlds[:count], geometry[:count])).astype(np.int64)
    canonical = keys[np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))]
    grouped = keys[np.argsort(keys[:, 0], kind="stable")]
    digest = lambda value: hashlib.sha256(value.tobytes()).hexdigest()
    return {
        "valid_capacity": True,
        "count": count,
        "raw_slots_sha256": digest(keys),
        "within_world_order_sha256": digest(grouped),
        "pair_multiset_sha256": digest(canonical),
    }


def run(args):
    import torch

    from gpu_duel_physics import GpuDuelPhysics
    from gpu_duel_reset import GpuDuelReset
    from gpu_motion_assets import GpuMotionAssets
    from gpu_reset_forward_gate import PersistentWarpCall, ResetForwardGate
    from verify_gpu_duel import load_config

    raw_path = args.out.with_suffix(".npz")
    if args.out.exists() or raw_path.exists():
        raise FileExistsError("diagnostic output already exists")
    config = load_config(args.config)
    stream = torch.cuda.Stream(device=config.device)
    stream.wait_stream(torch.cuda.current_stream(config.device))
    captures = []
    with torch.cuda.stream(stream):
        physics = GpuDuelPhysics(config.model, config.model_sha256, arenas=4, device=config.device)
        reset = GpuDuelReset(physics, GpuMotionAssets(config.assets, config.assets_sha256, config.device))
        all_arenas = torch.ones(4, dtype=torch.bool, device=config.device)
        begin = torch.zeros_like(all_arenas)
        complete = torch.zeros_like(all_arenas)
        selected = torch.zeros_like(all_arenas)
        velocity = torch.linspace(-.01, .01, physics.qvel.numel(), device=config.device).reshape_as(physics.qvel)

        def initialize():
            reset.full(all_arenas, reset_clock=True)
            physics.qvel.copy_(velocity)
            physics.ctrl.copy_(physics.qpos[:, reset.qindices].reshape_as(physics.ctrl))
            physics.forward()
            physics.reset_reader_fields["qacc_warmstart"].zero_()

        initialize()
        gate = ResetForwardGate(physics)
        physics.forward_selected = gate.forward_selected
        # The next-step graph also owns persistent scratch, so every temporary
        # byte can be restored identically instead of relying on graph malloc.
        step_call = PersistentWarpCall(physics, lambda: physics.mjw.step(physics.model, physics.data))

        def capture(operation):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                with physics.wp.ScopedCapture(stream=physics.stream, external=True) as scope:
                    operation()
            captures.extend((graph, scope))
            return graph

        step_graph = capture(step_call.capture)
        reset_graphs = {
            "after_substep": capture(lambda: reset.after_substep(begin, complete)),
            "full": capture(lambda: reset.full(selected, reset_clock=True)),
        }
        initialize()
        arrays = {}

        def gather(value, prefix=""):
            for field in dataclasses.fields(value):
                item = getattr(value, field.name)
                name = prefix + field.name
                if isinstance(item, physics.wp.array):
                    tensor = physics.wp.to_torch(item)
                    if tensor.numel():
                        arrays[name] = tensor
                elif dataclasses.is_dataclass(item):
                    gather(item, name + ".")

        gather(physics.data)
        state = {name: value.clone() for name, value in arrays.items()}
        forward_scratch = gate.body.snapshot_scratch()
        step_scratch = step_call.snapshot_scratch()
        persistent = tuple(name for name in (
            "time", "qpos", "qvel", "act", "qacc_warmstart", "ctrl",
            "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
        ) if name in arrays)
        measured = tuple(dict.fromkeys((*persistent, *physics.reset_reader_fields,
                                      "nacon", "nefc", "solver_niter")))
        measured = tuple(name for name in measured if name in arrays)
        contact_names = ("nacon", "contact.geom", "contact.worldid")
        for name in contact_names:
            if name not in arrays:
                raise RuntimeError(f"required contact-order diagnostic field missing: {name}")
        scenarios = {
            "false_mask": ([], [], [], "after_substep"),
            "mixed_begin_complete": ([1], [2], [], "after_substep"),
            "full_reset": ([], [], [0, 1, 2, 3], "full"),
        }
        raw = {}
        contact_reports = {}
        strict_checks = {}
        order_rng = np.random.default_rng(73019)
        for scenario, (begins, completes, full, operation) in scenarios.items():
            branches = ("forced_a", "forced_b", "automatic")
            trials = {branch: {name: [] for name in measured} for branch in branches}
            contact_traces = {branch: [] for branch in branches}
            checks = []
            for repeat in range(args.repeats):
                for branch in order_rng.permutation(branches):
                    for name, value in arrays.items():
                        value.copy_(state[name])
                    gate.body.restore_scratch(forward_scratch)
                    step_call.restore_scratch(step_scratch)
                    reset.pending.zero_()
                    begin.zero_()
                    complete.zero_()
                    selected.zero_()
                    if begins:
                        begin[begins] = True
                    if completes:
                        complete[completes] = True
                        reset.pending[completes] = True
                    if full:
                        selected[full] = True
                    gate.force_forward.fill_(branch != "automatic")
                    reset_graphs[operation].replay()
                    snapshots = {name: [arrays[name].clone()] for name in measured}
                    if scenario == "false_mask":
                        # Exact semantic identity before another physics step.
                        checks.append(torch.stack([
                            torch.all(arrays[name] == state[name])
                            for name in dict.fromkeys((*persistent, *physics.reset_reader_fields))
                        ]))
                    contacts = [{name: arrays[name].clone() for name in contact_names}]
                    for _ in range(args.follow_steps):
                        step_graph.replay()
                    for name in measured:
                        snapshots[name].append(arrays[name].clone())
                    contacts.append({name: arrays[name].clone() for name in contact_names})
                    for name, values in snapshots.items():
                        trials[branch][name].append(torch.stack(values))
                    contact_traces[branch].append(contacts)
            stream.synchronize()
            strict_checks[scenario] = {
                "false_mask_persistent_inputs_and_readers_exact":
                    bool(torch.stack(checks).all().item()) if checks else None,
            }
            contact_reports[scenario] = {}
            for branch in branches:
                for name, records in trials[branch].items():
                    raw[f"{scenario}/{branch}/{name}"] = torch.stack(records).cpu().numpy()
                contact_reports[scenario][branch] = [
                    [contact_fingerprint({name: value.cpu().numpy() for name, value in snapshot.items()})
                     for snapshot in pair] for pair in contact_traces[branch]
                ]
        gate.force_forward.zero_()

    comparisons = {}
    failures = []
    for scenario in scenarios:
        comparisons[scenario] = {}
        for name in measured:
            values = [raw[f"{scenario}/{branch}/{name}"] for branch in ("forced_a", "forced_b", "automatic")]
            comparisons[scenario][name] = {}
            for index, boundary in enumerate(("immediately_after_reset", "after_shared_physics")):
                result = paired_error(*(value[:, index] for value in values))
                comparisons[scenario][name][boundary] = result
                if not result["accepted"]:
                    failures.append(f"{scenario}/{name}/{boundary}")
    if strict_checks["false_mask"]["false_mask_persistent_inputs_and_readers_exact"] is not True:
        failures.append("false_mask_exact_invariants")
    report = {
        "schema": "rek.shared_graph_reset_forward_gate_diagnostic.v1",
        "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(config.device),
        "config_path": str(args.config), "model_sha256": config.model_sha256,
        "arenas": 4, "repeats_per_branch": args.repeats, "follow_physics_steps": args.follow_steps,
        "all_branches_share_reset_graph": True, "all_branches_share_next_step_graph": True,
        "restored_data_arrays": len(arrays),
        "restored_forward_scratch_arrays": len(forward_scratch),
        "restored_step_scratch_arrays": len(step_scratch),
        "gate": gate.metadata(), "strict_checks": strict_checks,
        "comparisons": comparisons, "contact_order_fingerprints": contact_reports,
        "failures": failures, "paired_criterion_passed": not failures,
        "production_enabled": False,
        "limits": [
            "No epsilon or relaxed tolerance: prior paired bootstrap upper-extra-RMS criterion is retained.",
            "Forced and automatic nonempty masks invoke identical captured kernels and persistent addresses.",
            "Contact fingerprints distinguish slot order from pair contents but do not prove the cause of a float difference.",
            "Raw scratch is restored to one recorded snapshot; dedicated poison tests are needed to establish absence of uninitialized reads.",
            "This diagnostic does not establish long-run combat/trajectory or authentic REK equivalence.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **raw)
    report["raw_arrays_path"] = str(raw_path)
    report["raw_arrays_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: report[key] for key in (
        "schema", "host", "arenas", "gate", "strict_checks", "failures", "paired_criterion_passed"
    )}, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", default=20, type=int)
    parser.add_argument("--follow-steps", default=1, type=int)
    args = parser.parse_args()
    if args.repeats < 2 or args.follow_steps < 1:
        parser.error("repeats must be at least two and follow-steps positive")
    result = run(args)
    raise SystemExit(0 if result["paired_criterion_passed"] else 1)
