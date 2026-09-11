from __future__ import annotations

import ctypes as ct
import unittest

import numpy as np

from inspect_gpu_kick_contacts import (
    CONTACT_BYTES,
    IMPACT_EVENT_BYTES,
    HitContact,
    ImpactEvent,
    analyze,
)


class KickContactDiagnosticTests(unittest.TestCase):
    def test_pre_apex_attribution_replays_native_counters(self):
        event_base = 0x1000
        event = ImpactEvent(1.1, 0.3, 0.5, 2.0, 3)
        contact = HitContact()
        contact.strike_intent.impact_events = event_base
        contact.strike_intent.impact_event_count = 1
        contact.strike_intent.clip_cursor_frames = 41.0
        contact.strike_intent.clip_fps = 50.0
        contact.strike_intent.move_id = 1
        contact.strike_intent.action_playing = 1
        contact.strike_intent.layer_active = 1
        contact.striker_body_position_world[:] = (0.0, 0.0, 1.0)
        contact.target_body_position_world[:] = (1.0, 0.0, 1.0)
        contact.striker_body_linear_velocity_world[:] = (6.0, 0.0, 0.0)
        contact.target_body_linear_velocity_world[:] = (0.0, 0.0, 0.0)
        contact.relative_speed_mps = 6.0
        contact.time_seconds = 3.194
        contact.striker_part = 2
        contact.striker_side = 0
        contact.target_zone = 13
        contact.striker_fighter = 0
        contact.target_fighter = 1
        contact.striker_body_slot = 2
        contact.is_enter = 1
        contact.round_active = 1
        contact.striker_upright = 1
        contact.target_upright = 1
        contact.target_standing = 1

        shape = (1, 10, 1)
        arrays = {
            "packed_contacts": np.zeros(shape + (1, CONTACT_BYTES), np.uint8),
            "contact_valid": np.zeros(shape + (1,), np.bool_),
            "contact_counts": np.zeros(shape, np.int64),
            "attributed_cumulative": np.ones(shape, np.uint32),
            "scored_cumulative": np.zeros(shape, np.uint32),
            "points_cumulative": np.zeros((1, 10, 2), np.int32),
            "begin_reset": np.zeros(shape, np.bool_),
            "complete_reset": np.zeros(shape, np.bool_),
            "episode_reset": np.zeros(shape, np.bool_),
            "terminal": np.zeros(shape, np.uint8),
        }
        arrays["contact_counts"][0, 0, 0] = 1
        arrays["contact_valid"][0, 0, 0, 0] = True
        arrays["packed_contacts"][0, 0, 0, 0] = np.frombuffer(
            ct.string_at(ct.addressof(contact), CONTACT_BYTES), dtype=np.uint8)
        events = np.frombuffer(
            ct.string_at(ct.addressof(event), IMPACT_EVENT_BYTES),
            dtype=np.uint8,
        ).reshape(1, IMPACT_EVENT_BYTES)

        result = analyze(arrays, events, event_base, per_arena=1)

        self.assertTrue(result["conclusive"])
        self.assertEqual(result["arena_totals"][0]["attributed"], 1)
        self.assertEqual(result["arena_totals"][0]["scored"], 0)
        rejected = result["unscored_attributed_contacts"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["score_gate"], "apex_ramp_below_threshold")
        self.assertAlmostEqual(rejected[0]["apex_ramp"], 0.0127407, places=6)


if __name__ == "__main__":
    unittest.main()
