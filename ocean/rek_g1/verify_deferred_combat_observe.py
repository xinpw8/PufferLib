"""Explicit CUDA native-state comparison and optional real-duel timing probe.

Synthetic substep tests compare the existing native entrypoint plus observe
against the new entrypoint, including its preserved pending-reset validation.
All arena state/status/reset flags must match byte-for-byte. Packed outputs
must match for valid arenas. A failed arena retains different stale diagnostic
values when intermediate packing is deferred; its status aborts the run and
prevents an accepted result. These values are never shared with another arena.
The optional timing mode runs the actual asset-pinned duel with identical
neutral learner/native-mask handling and unchanged candidate opponent. It is
environment throughput, never training SPS or authentic-game parity evidence.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from gpu_native_combat import GpuNativeCombat
from gpu_native_motion import Composer, DeviceStructs
from test_gpu_combat_measurement import synthetic_fixture, set_contacts


STATE_ARRAYS = ("states", "tick_fall_events", "tick_signals", "tick_referee_calls", "tick_score_delta",
                "tick_attributed_contacts", "tick_scored_contacts", "_arena_terminals", "input_reset",
                "dampened", "begin_reset", "complete_reset", "episode_reset", "clear_contacts", "statuses")
PACKED_ARRAYS = ("fall", "fight", "rewards", "terminals")


def fixture(library):
    measurement, source = synthetic_fixture("cuda")
    motion = SimpleNamespace(composers=DeviceStructs(Composer, 4, torch.device("cuda"), zero=True))
    routes = torch.zeros(4, dtype=torch.int32, device="cuda")
    combat = GpuNativeCombat(library, measurement, motion, routes)
    set_contacts(source, [])
    can_get_up = torch.zeros(4, dtype=torch.int64, device="cuda")
    batch = measurement.sample(0, can_get_up)
    batch.hits.arena_time_seconds.fill_(.002)
    return combat, batch, source


def equal_bytes(left, right, name):
    a, b = left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
    if not torch.equal(a, b):
        raise AssertionError(f"native consumed array differs: {name}")


def native_comparison(library):
    cases = ["valid", "pending-valid", "terminal-valid", "prelatched", "normal-invalid", "pending-invalid-time",
             "pending-invalid-validflag", "pending-nan-float"]
    cases += [f"pending-invalid-int-{column}-{row}" for column in range(7) for row in (0, 1)]
    reports = []
    for case in cases:
        reference, ref_batch, _ = fixture(library)
        deferred, new_batch, _ = fixture(library)
        # State padding and initially unused output rows get a common baseline.
        for name in (*STATE_ARRAYS, *PACKED_ARRAYS):
            getattr(deferred, name).copy_(getattr(reference, name))
        for combat, batch in ((reference, ref_batch), (deferred, new_batch)):
            combat.begin_tick()
            if case.startswith("pending"):
                # Checked by static_assert in this compiled native library.
                combat.states[0, 248] = 1
            if case == "terminal-valid": combat._arena_terminals[0] = 1
            if case == "prelatched": combat.statuses[0] = 3
            if case in ("normal-invalid", "pending-invalid-validflag"): batch.fall.valid[0] = False
            if case == "pending-invalid-time": batch.hits.arena_time_seconds[0] = float("nan")
            if case == "pending-nan-float": batch.fall.floats[0, 0] = float("nan")
            if case.startswith("pending-invalid-int-"):
                _, _, _, column, row = case.split("-")
                batch.fall.integers[int(row), int(column)] = 2 if int(column) != 3 else -1
        first_status = None
        first_complete = None
        stale_diagnostics = []
        for substep in range(10):
            reference.post_step(ref_batch)
            deferred.post_step(new_batch, pack_observation=False)
            for name in STATE_ARRAYS:
                equal_bytes(getattr(reference, name), getattr(deferred, name), name)
            if substep == 0:
                first_status = reference.statuses.cpu().tolist()
                first_complete = reference.complete_reset.cpu().tolist()
            # Match final refresh at the actual control-step consumption point.
            if substep == 9:
                reference.observe(ref_batch.fall)
                deferred.observe(new_batch.fall)
                for name in STATE_ARRAYS:
                    equal_bytes(getattr(reference, name), getattr(deferred, name), name)
                valid_rows = (reference.statuses == 0).repeat_interleave(2)
                for name in PACKED_ARRAYS:
                    equal_bytes(getattr(reference, name)[valid_rows], getattr(deferred, name)[valid_rows], name)
                    if not torch.equal(getattr(reference, name).contiguous().view(torch.uint8),
                                       getattr(deferred, name).contiguous().view(torch.uint8)):
                        stale_diagnostics.append(name)
            ref_batch.hits.arena_time_seconds.add_(.002)
            new_batch.hits.arena_time_seconds.add_(.002)
        if case.startswith("pending-invalid-int-") or case == "pending-invalid-validflag":
            assert first_status == [1, 0] and first_complete == [True, False]
        if case == "normal-invalid": assert first_status == [2, 0]
        failed = bool((reference.statuses != 0).any().item())
        if failed:
            for combat in (reference, deferred):
                try:
                    combat.check_status()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("failed arena did not abort status checking")
        reports.append({"case": case, "substeps": 10, "first_status": first_status,
                        "first_complete_reset": first_complete, "all_native_state_status_flags_exact": True,
                        "valid_arena_packed_outputs_exact": True, "failed_run_status_check_aborts": failed,
                        "failed_arena_stale_diagnostic_fields_differ": stale_diagnostics})
    return reports


def timing(config_path, library, ticks):
    from gpu_candidate_dummy import GpuCandidateDummyDuel
    from gpu_semantic_duel import GpuSemanticDuel
    from verify_gpu_duel import load_config

    config = load_config(config_path)
    results = []
    for deferred in (False, True, False, True):
        duel = GpuSemanticDuel(replace(config, combat_library=library,
                                     defer_substep_combat_observations=deferred))
        duel.capture_step()
        env = GpuCandidateDummyDuel(duel)
        actions = torch.ones((env.rows, 1), dtype=torch.int32, device=duel.actions.device)
        def step():
            actions[:, 0].copy_(torch.where(env.action_mask[:, 1] != 0, 1, 0))
            env.step(actions)
        for _ in range(4): step()
        env.reset()
        torch.cuda.synchronize()
        before, cpu = time.perf_counter(), time.process_time()
        for _ in range(ticks): step()
        torch.cuda.synchronize()
        elapsed, cpu = time.perf_counter()-before, time.process_time()-cpu
        env.log()
        results.append({"deferred": deferred, "arenas": env.rows, "control_ticks": ticks,
                        "wall_seconds": elapsed, "host_process_cpu_seconds": cpu,
                        "learner_control_sps": env.rows*ticks/elapsed,
                        "training_sps": None, "behavior": env.behavior_metrics.snapshot(clear=False)})
        env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ticks", type=int, default=128)
    args = parser.parse_args()
    if args.report.exists(): raise FileExistsError(args.report)
    if not 1 <= args.ticks <= 512: raise ValueError("bounded timing needs 1..512 ticks")
    report = {"status": "passed", "gpu": torch.cuda.get_device_name(),
              "native_cases": native_comparison(args.library),
              "real_duel_timings": timing(args.config, args.library, args.ticks) if args.config else None,
              "library": {"path": str(args.library), "sha256": hashlib.sha256(args.library.read_bytes()).hexdigest()},
              "claim_limits": "native boundary regression and optional candidate control throughput; no training SPS or authentic parity"}
    report["sources"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ("g1_native_combat_cuda.cu", "gpu_native_combat.py", "gpu_semantic_duel.py",
                                      "verify_deferred_combat_observe.py")}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
