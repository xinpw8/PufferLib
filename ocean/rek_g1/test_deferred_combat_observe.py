"""CPU-only opt-in/API checks; CUDA comparisons are a separate explicit probe."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gpu_native_combat import GpuNativeCombat
from gpu_semantic_duel import GpuDuelConfig


class DeferredObservationTests(unittest.TestCase):
    def fixture(self, *, has_deferred=True):
        owner = object.__new__(GpuNativeCombat)
        calls = []
        owner.device = torch.device("cpu")
        owner.arenas, owner.candidate_capacity = 2, 16
        owner._validate_measurement = lambda batch: calls.append("validate-shapes")
        owner.observe = lambda fall: calls.append("observe") or owner.outputs
        owner.measurement = SimpleNamespace(clear_arena_contacts=lambda mask: calls.append("clear-contacts"))
        owner.library = SimpleNamespace(rek_g1_cuda_native_combat_post_step=lambda *args: calls.append("reference-native") or 0)
        owner._deferred_post_step = ((lambda *args: calls.append("deferred-native") or 0) if has_deferred else None)
        for field in ("states", "composers", "active_route_ids_source", "active_route_ids", "impact_events",
                      "route_event_offsets", "route_event_counts", "packed_contacts", "tick_fall_events",
                      "tick_signals", "tick_referee_calls", "tick_score_delta", "tick_attributed_contacts",
                      "tick_scored_contacts", "_arena_terminals", "input_reset", "dampened", "begin_reset",
                      "complete_reset", "clear_contacts", "statuses"):
            setattr(owner, field, torch.zeros(4))
        owner.outputs = SimpleNamespace(marker="same owned output object")
        fall = SimpleNamespace(floats=torch.zeros(1), integers=torch.zeros(1), valid=torch.zeros(1))
        hits = SimpleNamespace(**{name: torch.zeros(1) for name in ("integers", "floats", "candidate_valid",
            "candidate_order", "candidate_offsets", "candidate_counts", "arena_scan_valid", "arena_time_seconds")})
        return owner, SimpleNamespace(fall=fall, hits=hits), calls

    def call(self, owner, batch, **kwargs):
        with patch("torch.cuda.device", side_effect=lambda device: nullcontext()), patch(
                "torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=0)):
            return owner.post_step(batch, **kwargs)

    def test_reference_remains_default_and_supports_old_library(self):
        owner, batch, calls = self.fixture(has_deferred=False)
        self.assertIs(self.call(owner, batch), owner.outputs)
        self.assertEqual(calls, ["validate-shapes", "reference-native", "clear-contacts", "observe"])

    def test_opt_in_keeps_state_and_contact_calls_but_defers_only_pack(self):
        owner, batch, calls = self.fixture()
        self.assertIs(self.call(owner, batch, pack_observation=False), owner.outputs)
        self.assertEqual(calls, ["validate-shapes", "deferred-native", "clear-contacts"])

    def test_missing_new_abi_is_rejected_before_any_substep(self):
        owner, batch, calls = self.fixture(has_deferred=False)
        with self.assertRaisesRegex(RuntimeError, "opt-in native"):
            self.call(owner, batch, pack_observation=False)
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            self.call(owner, batch, pack_observation=0)

    def test_config_defaults_reference_and_production_loop_reads_no_packed_values(self):
        self.assertIs(GpuDuelConfig.__dataclass_fields__["defer_substep_combat_observations"].default, False)
        tree = ast.parse(Path(__file__).with_name("gpu_semantic_duel.py").read_text())
        step = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_step_impl")
        substeps = next(node for node in step.body if isinstance(node, ast.For))
        consumed = {node.attr for node in ast.walk(substeps) if isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name) and node.value.id == "output"}
        self.assertEqual(consumed, {"dampened", "begin_reset", "complete_reset"})
        final_observe = [node for node in step.body if isinstance(node, ast.Assign)
                         and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                         and node.value.func.attr == "observe"]
        self.assertEqual(len(final_observe), 1)


if __name__ == "__main__":
    unittest.main()
