import base64
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import human_eval_server as human


def observation_rows() -> np.ndarray:
    rows = np.zeros(
        (human.ROBOT_ROWS, human.OBSERVATION_FLOATS), dtype=np.float32
    )
    for row in rows:
        row[3] = 1.0
        row[human.ENTITY_FLOATS + 0] = 1.0
        row[human.ENTITY_FLOATS + 3] = 1.0
    return rows


class FakeBoundary:
    def __init__(self, expose_masks: bool = True) -> None:
        self.observations = observation_rows()
        self.rewards = np.zeros(human.ROBOT_ROWS, dtype=np.float32)
        self.terminals = np.zeros(human.ROBOT_ROWS, dtype=np.float32)
        self.action_masks = (
            np.zeros(
                (human.ROBOT_ROWS, human.ACTION_CATEGORIES), dtype=np.uint8
            )
            if expose_masks
            else None
        )
        self.last_actions = np.ones(
            (human.ROBOT_ROWS, human.ACTION_HEADS), dtype=np.float32
        )
        self.reset_calls = 0
        self.step_calls = 0
        self.active_kicks = np.zeros(human.ROBOT_ROWS, dtype=np.int32)
        self.reset()

    def _write_masks(self) -> None:
        if self.action_masks is None:
            return
        self.action_masks.fill(0)
        for row in range(human.ROBOT_ROWS):
            if self.active_kicks[row] > 0:
                self.action_masks[
                    row,
                    [
                        human.CONTINUE_CATEGORY,
                        human.NEUTRAL_CATEGORY,
                        human.YAW_LEFT_CATEGORY,
                        human.YAW_RIGHT_CATEGORY,
                    ],
                ] = 1
            else:
                self.action_masks[row, 1:] = 1

    def reset(self) -> None:
        self.reset_calls += 1
        self.observations[:] = observation_rows()
        self.rewards.fill(0)
        self.terminals.fill(0)
        self.active_kicks.fill(0)
        self._write_masks()

    def step(self, actions: np.ndarray) -> None:
        self.step_calls += 1
        self.last_actions = actions.copy()
        categories = actions[:, 0].astype(np.int32)
        for row, category in enumerate(categories.tolist()):
            if self.active_kicks[row] > 0:
                self.active_kicks[row] -= 1
            elif category >= 16:
                self.active_kicks[row] = 2
        self._write_masks()


class FakeMaskVector:
    def __init__(self) -> None:
        self.storage = (
            np.arange(
                human.ROBOT_ROWS * human.ACTION_CATEGORIES, dtype=np.uint8
            )
            % 2
        ).reshape(human.ROBOT_ROWS, human.ACTION_CATEGORIES)
        self.action_mask_ptr = self.storage.ctypes.data
        self.action_mask_size = human.ACTION_CATEGORIES


class BrowserInputTests(unittest.TestCase):
    def test_held_combinations_and_kick_edge_are_explicit(self) -> None:
        state = human.BrowserInputState()
        self.assertTrue(
            state.update({"sequence": 1, "held": ["W", "Q"], "kick_move": 8})
        )
        self.assertEqual(
            human.HELD_CATEGORY_BY_SYMBOLS[state.held], 8
        )
        self.assertEqual(state.take_kick_edge(), 8)
        self.assertIsNone(state.take_kick_edge())

    def test_duplicate_or_stale_sequence_cannot_reinject_an_edge(self) -> None:
        state = human.BrowserInputState()
        self.assertTrue(
            state.update({"sequence": 1, "held": ["Q"], "kick_move": None})
        )
        self.assertFalse(
            state.update({"sequence": 1, "held": ["E"], "kick_move": 9})
        )
        self.assertEqual(state.held, frozenset({"Q"}))
        self.assertIsNone(state.take_kick_edge())
        state.reset()
        self.assertEqual(state.sequence, 0)

    def test_conflicting_translation_is_rejected(self) -> None:
        state = human.BrowserInputState()
        with self.assertRaisesRegex(human.HumanEvalFailure, "multiple translation"):
            state.update(
                {"sequence": 1, "held": ["W", "S"], "kick_move": None}
            )


class NativeMaskTests(unittest.TestCase):
    def test_host_mask_pointer_is_zero_copy(self) -> None:
        vector = FakeMaskVector()
        view = human._optional_action_mask(vector)
        self.assertIsNotNone(view)
        np.testing.assert_array_equal(view, vector.storage)
        vector.storage[3, 7] = 1 - vector.storage[3, 7]
        self.assertEqual(view[3, 7], vector.storage[3, 7])

    def test_partial_mask_abi_is_rejected(self) -> None:
        vector = FakeMaskVector()
        del vector.action_mask_size
        with self.assertRaisesRegex(human.HumanEvalFailure, "incomplete"):
            human._optional_action_mask(vector)


class TrackingCameraTests(unittest.TestCase):
    def test_lookat_tracks_fighter_midpoint_without_mutating_roots(self) -> None:
        roots = np.asarray(
            [[-1.4, 0.5, 0.4], [0.8, -0.3, 0.8]], dtype=np.float64
        )
        original = roots.copy()

        lookat = human._tracking_camera_lookat(roots)

        np.testing.assert_array_equal(roots, original)
        np.testing.assert_allclose(
            lookat,
            [-0.3, 0.1, human.RENDER_CAMERA_MIN_LOOKAT_Z_M],
        )
        elevated_roots = roots.copy()
        elevated_roots[:, 2] = [1.1, 1.3]
        self.assertAlmostEqual(
            human._tracking_camera_lookat(elevated_roots)[2], 1.2
        )

    def test_elevation_clears_center_sightline_over_arena_wall(self) -> None:
        # The pinned XML walls have radius 2.34 m, half-thickness 0.2 m,
        # center height 1 m, and half-height 1 m.
        wall_inner_face_radius_m = 2.34 - 0.2
        wall_top_m = 1.0 + 1.0
        sightline_height_at_wall_m = wall_inner_face_radius_m * np.tan(
            np.deg2rad(abs(human.RENDER_CAMERA_ELEVATION_DEGREES))
        )

        self.assertGreater(sightline_height_at_wall_m, wall_top_m)
        self.assertEqual(human.RENDER_CAMERA_DISTANCE_M, 3.8)
        horizontal_camera_offset_m = human.RENDER_CAMERA_DISTANCE_M * np.cos(
            np.deg2rad(abs(human.RENDER_CAMERA_ELEVATION_DEGREES))
        )
        self.assertLess(horizontal_camera_offset_m, wall_inner_face_radius_m)


class ShutdownSignalTests(unittest.TestCase):
    def test_int_and_term_request_one_graceful_server_shutdown(self) -> None:
        installed = {}
        restored = []

        class FakeSignal:
            SIGINT = 2
            SIGTERM = 15

            @staticmethod
            def getsignal(signum):
                return f"previous-{signum}"

            @staticmethod
            def signal(signum, handler):
                if callable(handler):
                    installed[signum] = handler
                else:
                    restored.append((signum, handler))

        class FakeServer:
            shutdown_calls = 0

            def shutdown(self):
                self.shutdown_calls += 1

        class InlineThread:
            def __init__(self, *, target, name, daemon):
                self.target = target
                self.name = name
                self.daemon = daemon

            def start(self):
                self.target()

        server = FakeServer()
        restore = human._install_shutdown_signal_handlers(
            server,
            signal_module=FakeSignal,
            thread_factory=InlineThread,
        )
        installed[FakeSignal.SIGTERM](FakeSignal.SIGTERM, None)
        installed[FakeSignal.SIGINT](FakeSignal.SIGINT, None)

        self.assertEqual(server.shutdown_calls, 1)
        restore()
        self.assertEqual(
            restored,
            [(2, "previous-2"), (15, "previous-15")],
        )


class HumanEvalCoreTests(unittest.TestCase):
    def test_q_to_kick_is_dispatched_without_neutral_and_q_updates_kick(self) -> None:
        boundary = FakeBoundary()
        core = human.SemanticHumanEvalCore(boundary)
        self.assertTrue(
            core.update_input(
                {"sequence": 1, "held": ["Q"], "kick_move": None}
            )
        )
        first = core.step_once()
        self.assertEqual(first[0, 0], human.YAW_LEFT_CATEGORY)

        self.assertTrue(
            core.update_input(
                {"sequence": 2, "held": ["Q"], "kick_move": 9}
            )
        )
        second = core.step_once()
        self.assertEqual(second[0, 0], human.KICK_MOVE_9_CATEGORY)
        self.assertEqual(core.last_kick_disposition, "accepted")

        self.assertTrue(
            core.update_input(
                {"sequence": 3, "held": ["Q"], "kick_move": None}
            )
        )
        third = core.step_once()
        self.assertEqual(third[0, 0], human.YAW_LEFT_CATEGORY)

    def test_translation_held_discards_kick_edge(self) -> None:
        core = human.SemanticHumanEvalCore(FakeBoundary())
        core.update_input({"sequence": 1, "held": ["W"], "kick_move": 7})
        actions = core.step_once()
        self.assertEqual(actions[0, 0], 2)
        self.assertEqual(core.last_kick_disposition, "discarded_translation_held")

    def test_only_first_arena_is_interactive(self) -> None:
        core = human.SemanticHumanEvalCore(FakeBoundary())
        core.update_input({"sequence": 1, "held": ["D", "E"], "kick_move": None})
        actions = core.step_once()
        self.assertEqual(actions[0, 0], 15)
        self.assertEqual(actions[1, 0], 16)
        np.testing.assert_array_equal(actions[2:, 0], 1.0)

    def test_state_disclaims_bot_identity_training_and_parity(self) -> None:
        core = human.SemanticHumanEvalCore(FakeBoundary())
        state = core.state()
        self.assertFalse(state["opponent_is_bot_1"])
        self.assertFalse(state["training_enabled"])
        self.assertFalse(state["rek_parity_claim"])
        self.assertEqual(state["runtime_get_up_authority"], "unknown")
        self.assertIsNone(state["paired_replay_trace"])
        self.assertEqual(
            state["robot_identity"],
            {
                "build_catalog_id": "g1",
                "build_catalog_display_name": "L100",
                "build_catalog_type_label": "Lightweight",
            },
        )
        self.assertEqual(
            state["move_coverage"],
            {
                "build_catalog_discrete_moves": 17,
                "evaluator_exposed_moves": 4,
                "scope": "measured_kick_subset",
            },
        )


class TraceTests(unittest.TestCase):
    def test_trace_records_exact_applied_tick_action_and_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.jsonl"
            writer = human.JsonlTraceWriter(path, {"extension": {"sha256": "0" * 64}})
            core = human.SemanticHumanEvalCore(FakeBoundary(), writer)
            core.update_input({"sequence": 1, "held": ["E"], "kick_move": None})
            actions = core.step_once()
            core.close_trace()
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(
            [record["event"] for record in records],
            [
                "trace_start",
                "environment_reset",
                "browser_input_received",
                "control_step",
                "trace_end",
            ],
        )
        self.assertEqual(
            [record["trace_sequence"] for record in records], list(range(5))
        )
        step = records[3]
        self.assertEqual(step["tick"], 1)
        self.assertEqual(step["actions"], actions[:, 0].astype(int).tolist())
        self.assertEqual(step["input_sequence"], 1)
        encoded = base64.b64decode(step["arena_0_observation_f32_le_b64"])
        self.assertEqual(len(encoded), human.OBSERVATION_FLOATS * 4)

    def test_trace_path_is_create_new(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exists.jsonl"
            path.write_text("preserve\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                human.JsonlTraceWriter(path, {})
            self.assertEqual(path.read_text(encoding="utf-8"), "preserve\n")


class HttpBoundaryTests(unittest.TestCase):
    def test_browser_reset_uses_json_and_resynchronizes_input_sequence(self) -> None:
        reset_handler = human.INDEX_HTML.split(
            "document.getElementById('reset').addEventListener", 1
        )[1].split("const image", 1)[0]
        self.assertIn("headers:{'content-type':'application/json'}", reset_handler)
        self.assertIn("body:'{}'", reset_handler)
        self.assertIn("sequence = 0", reset_handler)

    def test_exact_loopback_host_and_origin_are_accepted(self) -> None:
        human._require_http_request_boundary(
            {
                "Host": "127.0.0.1:18766",
                "Origin": "http://127.0.0.1:18766",
                "Content-Type": "application/json; charset=utf-8",
            },
            18766,
            require_json=True,
        )

    def test_dns_rebinding_host_is_rejected(self) -> None:
        with self.assertRaisesRegex(human.HumanEvalFailure, "Host"):
            human._require_http_request_boundary(
                {"Host": "attacker.invalid:18766"},
                18766,
                require_json=False,
            )

    def test_cross_origin_post_is_rejected(self) -> None:
        with self.assertRaisesRegex(human.HumanEvalFailure, "Origin"):
            human._require_http_request_boundary(
                {
                    "Host": "127.0.0.1:18766",
                    "Origin": "https://attacker.invalid",
                    "Content-Type": "application/json",
                },
                18766,
                require_json=True,
            )

    def test_simple_cross_origin_content_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(human.HumanEvalFailure, "content type"):
            human._require_http_request_boundary(
                {
                    "Host": "127.0.0.1:18766",
                    "Content-Type": "text/plain",
                },
                18766,
                require_json=True,
            )


if __name__ == "__main__":
    unittest.main()
