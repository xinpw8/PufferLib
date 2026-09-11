"""CPU/CUDA differential fixtures and opt-in captured measurement benchmark.

Build separately with g++ -x c++ -shared -fPIC -O2 -ffp-contract=off, or
nvcc -shared -Xcompiler=-fPIC -O2 --fmad=false -arch=sm_121. Set
REK_G1_FUSED_MEASUREMENT_LIBRARY and REK_G1_FUSED_TEST_DEVICE=cpu|cuda.
The benchmark measures measurement only, not training or environment SPS.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
import unittest

import torch

from gpu_combat_measurement import RekG1GpuCombatMeasurement
from gpu_combat_measurement_fused import FusedGpuCombatMeasurement
from test_gpu_combat_measurement import synthetic_fixture, set_contacts


LIBRARY = os.environ.get("REK_G1_FUSED_MEASUREMENT_LIBRARY")
DEVICE = os.environ.get("REK_G1_FUSED_TEST_DEVICE", "cpu")


def compare(test, reference, candidate, samplers):
    for name in ("integers", "floats", "valid", "floor_body_contacts"):
        torch.testing.assert_close(getattr(reference.fall, name), getattr(candidate.fall, name),
                                   rtol=0, atol=0, equal_nan=True)
    left, right = reference.hits, candidate.hits
    for name in ("candidate_valid", "candidate_counts", "candidate_offsets",
                 "arena_scan_valid", "contact_capacity_overflow", "arena_time_seconds"):
        torch.testing.assert_close(getattr(left, name), getattr(right, name),
                                   rtol=0, atol=0, equal_nan=True)
    valid = left.candidate_valid
    for name in ("integers", "floats", "candidate_processing_key"):
        torch.testing.assert_close(getattr(left, name)[valid], getattr(right, name)[valid],
                                   rtol=0, atol=0, equal_nan=True)
    count = int(left.candidate_counts.sum().cpu())
    test.assertTrue(torch.equal(left.candidate_order[:count], right.candidate_order[:count]))
    for name in ("_previous_pairs", "_expected_substep"):
        test.assertTrue(torch.equal(getattr(samplers[0], name), getattr(samplers[1], name)))


@unittest.skipUnless(LIBRARY, "explicit native library required")
class FusedMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.reference, self.source = synthetic_fixture(DEVICE)
        self.fused = FusedGpuCombatMeasurement(
            self.reference.model_map, self.source, LIBRARY, host_test=DEVICE == "cpu")
        self.can = torch.zeros(4, dtype=torch.int64, device=DEVICE)
        self.reference.reset()
        self.fused.reset()

    def sample(self, substep):
        left = self.reference.sample(substep, self.can)
        right = self.fused.sample(substep, self.can)
        compare(self, left, right, (self.reference, self.fused))

    def test_duplicates_reversal_arena_order_history_and_reset(self):
        sequences = (
            [(1,10,4),(0,4,10),(0,10,4),(0,4,10),(0,0,2)],
            [(0,10,4),(1,4,10)], [], [(1,4,10),(0,10,4)],
        )
        for tick in range(24):
            set_contacts(self.source, sequences[tick % 4])
            if tick == 6:
                mask = torch.tensor([True, False], device=DEVICE)
                self.reference.clear_arena_contacts(mask)
                self.fused.clear_arena_contacts(mask)
            self.sample(tick % 10)

    def test_invalid_inputs_preserve_per_arena_transaction(self):
        cases = ("world", "geom", "same", "dist", "pos", "frame", "time",
                 "velocity", "position", "slot", "negative_count", "overflow")
        for case in cases:
            with self.subTest(case=case):
                self.setUp()
                set_contacts(self.source, [(0,4,8),(1,10,1)])
                s = self.source
                if case == "world": s.contact_worldid[0] = -1
                elif case == "geom": s.contact_geom[0,0] = 11
                elif case == "same": s.contact_geom[0,1] = 4
                elif case == "dist": s.contact_dist[0] = float("nan")
                elif case == "pos": s.contact_pos[0,0] = float("nan")
                elif case == "frame": s.contact_frame[0,0,0] = float("inf")
                elif case == "time": s.time[0] = float("nan")
                elif case == "velocity": s.cvel[0,4,3] = float("inf")
                elif case == "position": s.xpos[0,4,0] = float("nan")
                elif case == "slot": self.reference.model_map.striker_slot[4] = 7
                elif case == "negative_count": s.nacon.fill_(-1)
                elif case == "overflow": s.nacon.fill_(9)
                self.sample(0)

    def test_inactive_stale_invalid_slots_do_not_invalidate(self):
        self.source.contact_worldid.fill_(-10)
        self.source.contact_geom.fill_(999)
        self.source.contact_dist.fill_(float("nan"))
        self.source.nacon.zero_()
        self.sample(0)

    def test_random_active_facts_exact_float_payloads(self):
        generator = torch.Generator().manual_seed(20260911)
        for tick in range(100):
            for name in ("xipos", "subtree_com", "cvel"):
                tensor = getattr(self.source, name)
                tensor.copy_(torch.randn(tensor.shape, generator=generator).to(DEVICE))
            set_contacts(self.source, [(0,4,8),(1,10,1)] if tick % 2 == 0 else [])
            self.sample(tick % 10)

    @unittest.skipUnless(DEVICE == "cuda", "CUDA graph fixture")
    def test_graph_replay_after_contact_changes(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2): self.fused.sample(0, self.can)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.fused.sample(0, self.can)
        stream.synchronize()
        for entries in ([(0,4,8)], [(1,10,1),(0,8,4)], []):
            self.reference.reset()
            self.fused.reset()
            set_contacts(self.source, entries)
            expected = self.reference.sample(0, self.can)
            graph.replay()
            compare(self, expected, output, (self.reference, self.fused))


def benchmark(args):
    if DEVICE != "cuda" or not LIBRARY:
        raise ValueError("benchmark requires explicitly selected CUDA and library")
    fixture, source = synthetic_fixture("cuda")
    arenas, capacity = args.arenas, args.arenas*128
    if arenas < 1 or args.repeats < 1:
        raise ValueError("arenas and repeats must be positive")
    # Match production body/geom dimensions while retaining a synthetic map.
    # This is not a replacement model and never advances any physics.
    model_updates = {"body_count":63, "geom_count":91}
    for name in ("body_root_ids", "body_owner", "body_zone", "striker_part", "striker_side", "striker_slot"):
        original = getattr(fixture.model_map, name)
        value = torch.full((63,), -1 if name in ("body_owner","striker_side","striker_slot") else 0,
                           device="cuda", dtype=original.dtype)
        value[:original.numel()].copy_(original)
        model_updates[name] = value
    for name in ("geom_body_ids", "geom_zone", "geom_contype", "geom_conaffinity"):
        original = getattr(fixture.model_map, name)
        value = torch.zeros(91, device="cuda", dtype=original.dtype)
        value[:original.numel()].copy_(original)
        model_updates[name] = value
    model_map = replace(fixture.model_map, **model_updates)
    tiled = {}
    for name in ("qpos", "xpos", "xipos", "subtree_com", "cvel", "geom_xpos", "geom_xmat", "time"):
        original = getattr(source, name)
        value = original[:1].repeat((arenas,)+(1,)*(original.ndim-1)).contiguous()
        if name in ("xpos", "xipos", "subtree_com", "cvel", "geom_xpos", "geom_xmat"):
            count = 91 if name.startswith("geom_") else 63
            larger = torch.zeros((arenas,count)+tuple(value.shape[2:]),device="cuda",dtype=value.dtype)
            larger[:,:value.shape[1]].copy_(value)
            value = larger
        tiled[name] = value
    for name in ("contact_geom", "contact_worldid", "contact_dist", "contact_pos", "contact_frame"):
        original = getattr(source, name)
        tiled[name] = torch.zeros((capacity,)+tuple(original.shape[1:]), dtype=original.dtype, device="cuda")
    tiled["nacon"] = torch.zeros_like(source.nacon)
    source = replace(source, **tiled)
    entries = [(a, x, y) for a in range(arenas) for x,y in ((0,2),(0,3),(0,6),(0,7),(4,8),(10,1),(8,4),(1,10))]
    set_contacts(source, entries)
    reference = RekG1GpuCombatMeasurement(model_map, source)
    fused = FusedGpuCombatMeasurement(model_map, source, LIBRARY)
    can = torch.zeros(2*arenas, dtype=torch.int64, device="cuda")
    reference.reset(); fused.reset()
    expected, actual = reference.sample(0, can), fused.sample(0, can)
    compare(unittest.TestCase(), expected, actual, (reference,fused))
    reports = []
    for label, sampler in (("original",reference),("fused",fused)):
        sampler.reset()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for i in range(10): sampler.sample(i, can)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                sampler._previous_pairs.zero_()
                for i in range(10): sampler.sample(i, can)
        stream.synchronize()
        # Clear pair history before each 10-substep repeat so one sampled
        # contact-enter burst occurs per control interval for both candidates.
        sampler.reset()
        for _ in range(5): graph.replay()
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        begin.record()
        for _ in range(args.repeats): graph.replay()
        end.record(); end.synchronize()
        elapsed = time.perf_counter()-wall
        reports.append({"label":label,"wall_seconds":elapsed,
            "gpu_event_ms":begin.elapsed_time(end),"measurement_substeps":10*args.repeats,
            "arena_measurement_substeps_per_second":arenas*10*args.repeats/elapsed})
    return {"scope":"captured measurement only; synthetic contact workload; not training SPS",
        "arenas":arenas,"bodies":63,"geoms":91,"capacity":capacity,"live_contacts":len(entries),
        "library_sha256":hashlib.sha256(Path(LIBRARY).read_bytes()).hexdigest(),"timings":reports}


if __name__ == "__main__":
    if "--benchmark" in __import__("sys").argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--benchmark", action="store_true")
        parser.add_argument("--arenas",type=int,default=512)
        parser.add_argument("--repeats",type=int,default=50)
        args = parser.parse_args()
        print(json.dumps(benchmark(args), indent=2))
    else:
        unittest.main()
