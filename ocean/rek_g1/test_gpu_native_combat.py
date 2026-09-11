import os
from pathlib import Path
import unittest

import torch

from gpu_native_combat import GpuNativeCombat
from gpu_native_motion import Composer, DeviceStructs
from test_gpu_combat_measurement import synthetic_fixture, set_contacts


LIBRARY = Path(os.environ.get(
    "REK_G1_COMBAT_CUDA_LIBRARY",
    "/tmp/rek-native-combat-20260910/build/librek_g1_combat_cuda.so",
))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(LIBRARY.is_file(), "CUDA combat library is unavailable")
class GpuNativeCombatTests(unittest.TestCase):
    def setUp(self):
        self.measurement, self.source = synthetic_fixture("cuda")
        self.motion = type("Motion", (), {})()
        self.motion.composers = DeviceStructs(
            Composer, 4, torch.device("cuda"), zero=True)
        # Deliberately strided, matching a scheduler-owned struct field view.
        self.route_storage = torch.zeros(
            (4, 2), dtype=torch.int32, device="cuda")
        self.routes = self.route_storage[:, 1]
        self.combat = GpuNativeCombat(
            LIBRARY, self.measurement, self.motion, self.routes)
        self.can_get_up = torch.zeros(4, dtype=torch.int64, device="cuda")
        self.combat.check_status()

    def _enable_move(self, row, route, cursor, move_id):
        composer = self.motion.composers
        composer.field("current_layer.cursor")[row] = cursor
        composer.field("current_layer.clip.fps")[row] = 50.0
        composer.field("current_layer.has_clip")[row] = 1
        composer.field("current_layer.has_config")[row] = 1
        composer.field("current_layer.active")[row] = 1
        composer.field("current_layer.config.loop")[row] = 0
        composer.field("action_playing")[row] = 1
        composer.field("action_move_id")[row] = move_id
        self.routes[row] = route

    def test_native_abi_sizes_are_build_pinned(self):
        self.assertEqual(self.combat.states.shape[1], 252)
        self.assertEqual(self.combat.packed_contacts.shape[1], 120)
        self.assertEqual(self.combat.impact_events.shape, (29, 20))
        self.assertEqual(self.motion.composers.tensor.shape[1], 336)

    def test_reset_and_empty_substep_export_exact_row_shapes(self):
        output = self.combat.outputs
        self.assertEqual(tuple(output.fall.shape), (4, 15))
        self.assertEqual(tuple(output.fight.shape), (4, 39))
        self.assertEqual(output.fight[:, 1].cpu().tolist(), [2.0] * 4)
        self.assertEqual(output.fight[:, 2].cpu().tolist(), [1.0] * 4)
        self.assertEqual(output.fight[:, 5].cpu().tolist(), [120.0] * 4)

        self.combat.begin_tick()
        self.source.time.fill_(0.002)
        set_contacts(self.source, [])
        output = self.combat.sample_and_post_step(0, self.can_get_up)
        self.combat.check_status()
        torch.testing.assert_close(
            output.fight[:, 5].cpu(), torch.full((4,), 119.998),
            rtol=0.0, atol=2e-5)
        self.assertEqual(output.rewards.cpu().tolist(), [0.0] * 4)
        self.assertEqual(output.terminals.cpu().tolist(), [0.0] * 4)

    def test_left_foot_apex_scores_two_points_and_one_hit(self):
        composer = self.motion.composers
        composer.field("current_layer.cursor").fill_(55.0)
        composer.field("current_layer.clip.fps").fill_(50.0)
        composer.field("current_layer.has_clip").fill_(1)
        composer.field("current_layer.has_config").fill_(1)
        composer.field("current_layer.active").fill_(1)
        composer.field("current_layer.config.loop").zero_()
        composer.field("action_playing").fill_(1)
        composer.field("action_move_id").copy_(
            torch.arange(100, 104, dtype=torch.int32, device="cuda"))
        self.routes.fill_(7)

        self.source.xpos[0, 2] = torch.tensor(
            [0.0, 0.0, 1.0], device="cuda")
        self.source.xpos[0, 8] = torch.tensor(
            [1.0, 0.0, 1.0], device="cuda")
        self.source.cvel[0, 2, 3:6] = torch.tensor(
            [3.0, 0.0, 0.0], device="cuda")
        self.source.cvel[0, 8, 3:6].zero_()
        self.source.time.fill_(1.1)
        set_contacts(self.source, [(0, 2, 9)])

        self.combat.begin_tick()
        output = self.combat.sample_and_post_step(0, self.can_get_up)
        self.combat.check_status()
        self.assertEqual(output.tick_score_delta.cpu().tolist(), [2, 0, 0, 0])
        self.assertEqual(output.tick_scored_contacts.cpu().tolist(), [1, 0])
        self.assertEqual(output.tick_attributed_contacts.cpu().tolist(), [1, 0])
        self.assertEqual(output.fight[:, 6].cpu().tolist(), [2.0, 0.0, 0.0, 0.0])
        self.assertEqual(output.rewards.cpu().tolist(), [2.0, -2.0, 0.0, 0.0])

    def test_interleaved_multi_arena_contacts_are_grouped_without_drop(self):
        self._enable_move(0, 7, 55.0, 201)       # left lower-body event
        self._enable_move(3, 14, 24.5, 202)      # right upper-body event
        self.source.xpos[0, 2] = torch.tensor([0.0, 0.0, 1.0], device="cuda")
        self.source.xpos[0, 8] = torch.tensor([1.0, 0.0, 1.0], device="cuda")
        self.source.cvel[0, 2, 3] = 3.0
        self.source.xpos[1, 9] = torch.tensor([0.0, 0.0, 1.0], device="cuda")
        self.source.xpos[1, 1] = torch.tensor([1.0, 0.0, 1.0], device="cuda")
        self.source.cvel[1, 9, 3] = 3.0
        self.source.time.fill_(1.1)
        # Warp slots are deliberately arena1 then arena0.
        set_contacts(self.source, [(1, 10, 1), (0, 2, 9)])

        self.combat.begin_tick()
        batch = self.measurement.sample(0, self.can_get_up)
        valid_count = int(batch.hits.candidate_counts.sum().cpu())
        self.assertEqual(batch.hits.candidate_order[:valid_count].cpu().tolist(), [2, 0])
        output = self.combat.post_step(batch)
        self.combat.check_status()
        self.assertEqual(output.tick_score_delta.cpu().tolist(), [2, 0, 0, 1])
        self.assertEqual(output.tick_scored_contacts.cpu().tolist(), [1, 1])
        self.assertEqual(output.tick_attributed_contacts.cpu().tolist(), [1, 1])

    def test_contact_during_blocked_move_is_attributed_but_not_scored(self):
        self.source.xpos[0, 2] = torch.tensor([0.0, 0.0, 1.0], device="cuda")
        self.source.xpos[0, 8] = torch.tensor([1.0, 0.0, 1.0], device="cuda")
        self.source.cvel[0, 2, 3] = 3.0
        self.source.time.fill_(1.1)
        set_contacts(self.source, [(0, 2, 9)])
        self.routes[0] = 7
        self.motion.composers.field("action_playing")[0] = 0

        self.combat.begin_tick()
        output = self.combat.sample_and_post_step(0, self.can_get_up)
        self.combat.check_status()
        self.assertEqual(output.tick_attributed_contacts.cpu().tolist(), [1, 0])
        self.assertEqual(output.tick_scored_contacts.cpu().tolist(), [0, 0])
        self.assertEqual(output.tick_score_delta.cpu().tolist(), [0, 0, 0, 0])

    def test_counted_fall_reset_begins_then_completes_one_substep_later(self):
        set_contacts(self.source, [])
        batch = self.measurement.sample(0, self.can_get_up)
        batch.fall.floats[0, 0] = 90.0
        batch.fall.floats[0, 1] = 0.2
        batch.fall.floats[0, 2] = 0.002
        batch.fall.integers[0] = torch.tensor(
            [1, 1, 0, 1, 0, 0, 0], dtype=torch.int64, device="cuda")
        begin_history = torch.zeros((1800, 2), dtype=torch.bool, device="cuda")
        complete_history = torch.zeros_like(begin_history)
        dampen_history = torch.zeros((1800, 4), dtype=torch.bool, device="cuda")

        self.combat.begin_tick()
        for step in range(1800):
            batch.hits.arena_time_seconds.fill_((step + 1) * 0.002)
            output = self.combat.post_step(batch)
            begin_history[step].copy_(output.begin_reset)
            complete_history[step].copy_(output.complete_reset)
            dampen_history[step].copy_(output.dampened)
        self.combat.check_status()

        begin_indices = torch.nonzero(begin_history[:, 0]).flatten().cpu().tolist()
        complete_indices = torch.nonzero(complete_history[:, 0]).flatten().cpu().tolist()
        self.assertEqual(len(begin_indices), 1)
        self.assertEqual(complete_indices, [begin_indices[0] + 1])
        self.assertEqual(int(dampen_history[:, 0].sum().cpu()), 1)
        self.assertEqual(output.input_reset.cpu().tolist(), [True, True, False, False])
        self.assertEqual(output.tick_score_delta.cpu().tolist(), [0, 5, 0, 0])
        self.assertTrue(bool(output.tick_signals[0].cpu() & 32))
        self.assertGreater(float(output.fall[0, 12].cpu()), 1.8)


if __name__ == "__main__":
    unittest.main()
