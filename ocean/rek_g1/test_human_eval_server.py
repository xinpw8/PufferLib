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
        self.active_moves = np.zeros(human.ROBOT_ROWS, dtype=np.int32)
        self.reset()

    def _write_masks(self) -> None:
        self.observations[:, human.ACTION_PLAYING_INDEX] = self.active_moves > 0
        if self.action_masks is None:
            return
        self.action_masks.fill(0)
        for row in range(human.ROBOT_ROWS):
            if self.active_moves[row] > 0:
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
        self.active_moves.fill(0)
        self._write_masks()

    def step(self, actions: np.ndarray) -> None:
        self.step_calls += 1
        self.last_actions = actions.copy()
        categories = actions[:, 0].astype(np.int32)
        for row, category in enumerate(categories.tolist()):
            if self.active_moves[row] > 0:
                self.active_moves[row] -= 1
            elif category >= 16:
                self.active_moves[row] = 2
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
    def test_held_combinations_and_move_edge_are_explicit(self) -> None:
        state = human.BrowserInputState()
        self.assertTrue(
            state.update({"sequence": 1, "held": ["W", "Q"], "move_index": 16})
        )
        self.assertEqual(
            human.HELD_CATEGORY_BY_SYMBOLS[state.held], 8
        )
        self.assertEqual(state.take_move_edge(), 16)
        self.assertIsNone(state.take_move_edge())
        self.assertIsNone(state.peek_pending_move())
        state.buffer_yaw_move(16)
        state.consume_pending_move(16)
        self.assertIsNone(state.peek_pending_move())

    def test_repeated_move_edges_neither_stack_nor_replace(self) -> None:
        state = human.BrowserInputState()
        self.assertTrue(
            state.update({"sequence": 1, "held": ["Q"], "move_index": 7})
        )
        self.assertTrue(
            state.update({"sequence": 2, "held": ["E"], "move_index": 8})
        )
        self.assertEqual(state.take_move_edge(), 7)
        self.assertIsNone(state.peek_pending_move())
        state.buffer_yaw_move(7)
        self.assertFalse(
            state.update({"sequence": 1, "held": ["Q"], "move_index": 9})
        )
        self.assertTrue(
            state.update({"sequence": 3, "held": [], "move_index": None})
        )
        self.assertIsNone(state.take_move_edge())
        self.assertEqual(state.peek_pending_move(), 7)
        state.update({"sequence": 4, "held": ["E"], "move_index": 9})
        self.assertIsNone(state.take_move_edge())
        self.assertEqual(state.peek_pending_move(), 7)
        state.reset()
        self.assertIsNone(state.take_move_edge())
        self.assertIsNone(state.peek_pending_move())

    def test_duplicate_or_stale_sequence_cannot_reinject_an_edge(self) -> None:
        state = human.BrowserInputState()
        self.assertTrue(
            state.update({"sequence": 1, "held": ["Q"], "move_index": None})
        )
        self.assertFalse(
            state.update({"sequence": 1, "held": ["E"], "move_index": 9})
        )
        self.assertEqual(state.held, frozenset({"Q"}))
        self.assertIsNone(state.take_move_edge())
        state.reset()
        self.assertEqual(state.sequence, 0)

    def test_conflicting_translation_is_rejected(self) -> None:
        state = human.BrowserInputState()
        with self.assertRaisesRegex(human.HumanEvalFailure, "multiple translation"):
            state.update(
                {"sequence": 1, "held": ["W", "S"], "move_index": None}
            )

    def test_legacy_or_out_of_range_move_identity_is_rejected(self) -> None:
        state = human.BrowserInputState()
        with self.assertRaisesRegex(human.HumanEvalFailure, "legacy kick_move"):
            state.update({"sequence": 1, "held": [], "kick_move": 7})
        with self.assertRaisesRegex(human.HumanEvalFailure, "0 through 16"):
            state.update({"sequence": 2, "held": [], "move_index": 17})


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

    def test_candidate_no_get_up_guard_checks_self_and_opponent_rows(self) -> None:
        rows = observation_rows()
        human._require_candidate_no_get_up(rows)
        for offset in (
            human.BUILD_PINNED_CAN_GET_UP_OFFSET,
            human.OPPONENT_BUILD_PINNED_CAN_GET_UP_OFFSET,
        ):
            contradictory = rows.copy()
            contradictory[3, offset] = 1.0
            with self.assertRaisesRegex(
                human.HumanEvalFailure,
                "contradicts candidate no-get-up",
            ):
                human._require_candidate_no_get_up(contradictory)


class FullMoveContractTests(unittest.TestCase):
    def test_all_17_runtime_moves_have_exact_categories_and_durations(self) -> None:
        self.assertEqual(human.ACTION_CATEGORIES, 33)
        self.assertEqual(
            human.MOVE_REGISTRY_ORDER,
            (6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16),
        )
        self.assertEqual(set(human.MOVE_TO_CATEGORY), set(range(17)))
        self.assertEqual(set(human.MOVE_TO_CATEGORY.values()), set(range(16, 33)))
        self.assertEqual(
            human.MOVE_DURATION_TICKS,
            (35, 27, 31, 45, 32, 45, 157, 145, 158, 139, 134, 138, 73, 75, 68, 71, 103),
        )
        self.assertEqual(len(human.MOVE_METADATA), 17)
        for metadata in human.MOVE_METADATA:
            self.assertEqual(
                metadata["category"], human.MOVE_TO_CATEGORY[metadata["move"]]
            )

    def test_vector_arguments_supply_every_required_move_duration(self) -> None:
        arguments = human.vector_arguments(123, 7)
        self.assertEqual(arguments["vec"], {"total_agents": 8, "num_buffers": 1})
        self.assertEqual(arguments["env"]["max_steps"], 123)
        self.assertEqual(arguments["env"]["physics_workers"], 7)
        self.assertEqual(arguments["env"]["locomotion_segment_ticks"], 1)
        for move_index, duration in enumerate(human.MOVE_DURATION_TICKS):
            self.assertEqual(
                arguments["env"][f"move_{move_index}_duration_ticks"], duration
            )
        self.assertFalse(
            any(key.startswith("kick_move_") for key in arguments["env"])
        )


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

    def test_oblique_camera_hides_walls_and_distinguishes_fighters(self) -> None:
        self.assertEqual(human.RENDER_CAMERA_MIN_DISTANCE_M, 3.8)
        self.assertEqual(human.RENDER_CAMERA_ELEVATION_DEGREES, -45.0)
        self.assertEqual(
            human._render_geom_rgba_override("arena_Collider_Wall_01"),
            (0.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(
            human._render_geom_rgba_override("arena_Collider_Pillar_08"),
            (0.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(
            human._render_geom_rgba_override("player__mjgeom_3021"),
            human.RENDER_PLAYER_RGBA,
        )
        self.assertEqual(
            human._render_geom_rgba_override("opponent__mjgeom_3021"),
            human.RENDER_OPPONENT_RGBA,
        )
        self.assertIsNone(
            human._render_geom_rgba_override("arena_Collider_Floor_Rektagon")
        )

    def test_distance_expands_to_keep_separated_fighters_in_frame(self) -> None:
        close = np.asarray([[-0.5, 0.0, 0.9], [0.5, 0.0, 0.9]])
        wide = np.asarray([[-2.3, 0.0, 0.9], [2.3, 0.0, 0.9]])
        self.assertEqual(human._tracking_camera_distance(close, 45.0), 3.8)
        wide_distance = human._tracking_camera_distance(wide, 45.0)
        visible_vertical_span_m = 2.0 * wide_distance * np.tan(
            np.deg2rad(22.5)
        )
        self.assertGreaterEqual(
            visible_vertical_span_m,
            4.6 + 2.0 * human.RENDER_CAMERA_FIT_MARGIN_M,
        )


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
    def test_q_to_move_is_dispatched_without_neutral_and_q_resumes(self) -> None:
        boundary = FakeBoundary()
        core = human.SemanticHumanEvalCore(boundary)
        self.assertTrue(
            core.update_input(
                {"sequence": 1, "held": ["Q"], "move_index": None}
            )
        )
        first = core.step_once()
        self.assertEqual(first[0, 0], human.YAW_LEFT_CATEGORY)

        self.assertTrue(
            core.update_input(
                {"sequence": 2, "held": ["Q"], "move_index": 16}
            )
        )
        second = core.step_once()
        self.assertEqual(second[0, 0], human.MOVE_TO_CATEGORY[16])
        self.assertEqual(core.last_move_disposition, "accepted")

        self.assertTrue(
            core.update_input(
                {"sequence": 3, "held": ["Q"], "move_index": None}
            )
        )
        third = core.step_once()
        self.assertEqual(third[0, 0], human.YAW_LEFT_CATEGORY)

    def test_translation_held_discards_move_without_deferred_attack(self) -> None:
        core = human.SemanticHumanEvalCore(FakeBoundary())
        core.update_input({"sequence": 1, "held": ["W", "Q"], "move_index": 7})
        actions = core.step_once()
        self.assertEqual(actions[0, 0], 8)
        self.assertEqual(core.last_move_disposition, "discarded_translation_held")
        self.assertIsNone(core.browser_input.peek_pending_move())

        core.update_input({"sequence": 2, "held": ["Q"], "move_index": None})
        actions = core.step_once()
        self.assertEqual(actions[0, 0], human.YAW_LEFT_CATEGORY)
        self.assertIsNone(core.browser_input.peek_pending_move())

    def test_move_during_active_action_is_discarded_even_with_held_yaw(self) -> None:
        for symbol, yaw_category, first_move, buffered_move in (
            ("Q", human.YAW_LEFT_CATEGORY, 0, 1),
            ("E", human.YAW_RIGHT_CATEGORY, 8, 9),
        ):
            with self.subTest(symbol=symbol):
                boundary = FakeBoundary()
                core = human.SemanticHumanEvalCore(boundary)
                core.update_input(
                    {"sequence": 1, "held": [symbol], "move_index": first_move}
                )
                self.assertEqual(
                    core.step_once()[0, 0], human.MOVE_TO_CATEGORY[first_move]
                )

                core.update_input(
                    {"sequence": 2, "held": [symbol], "move_index": buffered_move}
                )
                for _ in range(2):
                    actions = core.step_once()
                    self.assertEqual(actions[0, 0], yaw_category)
                    self.assertEqual(
                        core.last_move_disposition,
                        "discarded_move_in_progress",
                    )
                    self.assertIsNone(core.browser_input.peek_pending_move())

                actions = core.step_once()
                self.assertEqual(
                    actions[0, 0], yaw_category
                )
                self.assertEqual(core.last_move_disposition, "discarded_move_in_progress")
                self.assertIsNone(core.browser_input.peek_pending_move())

                core.update_input(
                    {"sequence": 3, "held": [symbol], "move_index": None}
                )
                self.assertEqual(core.step_once()[0, 0], yaw_category)

    def test_only_yaw_interruption_survives_a_masked_dispatch(self) -> None:
        for held in ([], ["Q"], ["E"]):
            with self.subTest(held=held):
                boundary = FakeBoundary()
                core = human.SemanticHumanEvalCore(boundary)
                boundary.action_masks[0, 16:] = 0
                core.update_input({"sequence": 1, "held": held, "move_index": 7})
                self.assertEqual(core.step_once()[0, 0], human.NEUTRAL_CATEGORY)
                self.assertEqual(core.browser_input.peek_pending_move(), 7 if held else None)
                core.update_input({"sequence": 2, "held": held, "move_index": None})
                expected = human.MOVE_TO_CATEGORY[7] if held else human.NEUTRAL_CATEGORY
                self.assertEqual(core.step_once()[0, 0], expected)
                self.assertIsNone(core.browser_input.peek_pending_move())

    def test_yaw_pending_move_cannot_be_replaced_by_another_attack(self) -> None:
        boundary = FakeBoundary()
        core = human.SemanticHumanEvalCore(boundary)
        boundary.action_masks[0, 16:] = 0
        core.update_input({"sequence": 1, "held": ["Q"], "move_index": 7})
        core.step_once()
        core.update_input({"sequence": 2, "held": ["Q"], "move_index": 8})
        self.assertEqual(core.step_once()[0, 0], human.MOVE_TO_CATEGORY[7])
        for _ in range(4):
            self.assertLess(core.step_once()[0, 0], 16)
        self.assertIsNone(core.browser_input.peek_pending_move())

    def test_terminal_clears_a_masked_move_buffer(self) -> None:
        boundary = FakeBoundary()
        core = human.SemanticHumanEvalCore(boundary)
        boundary.terminals[0] = 1.0
        boundary._write_masks()
        boundary.action_masks[0, 16:] = 0
        core.update_input({"sequence": 1, "held": ["Q"], "move_index": 7})

        self.assertEqual(core.step_once()[0, 0], human.NEUTRAL_CATEGORY)
        self.assertIsNone(core.browser_input.peek_pending_move())
        self.assertEqual(core.last_move_disposition, "cleared_terminal")

        boundary.terminals[0] = 0.0
        self.assertNotEqual(
            core.step_once()[0, 0],
            human.MOVE_TO_CATEGORY[7],
        )

    def test_only_first_arena_is_interactive(self) -> None:
        core = human.SemanticHumanEvalCore(FakeBoundary())
        core.update_input({"sequence": 1, "held": ["D", "E"], "move_index": None})
        actions = core.step_once()
        self.assertEqual(actions[0, 0], 15)
        self.assertEqual(actions[1, 0], 16)
        np.testing.assert_array_equal(actions[2:, 0], 1.0)

    def test_state_disclaims_bot_identity_training_and_parity(self) -> None:
        core = human.SemanticHumanEvalCore(FakeBoundary())
        state = core.state()
        self.assertEqual(state["schema"], "rek.g1_human_eval_state.v2")
        self.assertFalse(state["opponent_is_bot_1"])
        self.assertFalse(state["training_enabled"])
        self.assertFalse(state["rek_parity_claim"])
        self.assertEqual(
            state["runtime_get_up_authority"],
            "user_observed_l100_no_getup_and_candidate_observation_guard",
        )
        self.assertEqual(state["authentic_three_down_terminal_rule"], "unknown")
        self.assertIsNone(state["pending_move_index"])
        self.assertEqual(
            state["opponent_move_scope"],
            "combat moves 0 through 15; move 16 emote excluded",
        )
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
                "evaluator_exposed_moves": 17,
                "scope": "complete_build_pinned_static_discrete_catalog",
            },
        )
        self.assertEqual(len(state["move_controls"]), 17)
        self.assertIn("player_score", state["fight"])
        self.assertIn("opponent_score", state["fight"])
        self.assertNotIn("player_clean_hits", state["fight"])
        self.assertNotIn("opponent_clean_hits", state["fight"])


class TraceTests(unittest.TestCase):
    def test_trace_records_one_edge_while_buffer_survives_until_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "buffered.jsonl"
            writer = human.JsonlTraceWriter(path, {"extension": {"sha256": "0" * 64}})
            boundary = FakeBoundary()
            core = human.SemanticHumanEvalCore(boundary, writer)
            boundary.action_masks[0, 16:] = 0
            core.update_input({"sequence": 1, "held": ["Q"], "move_index": 0})
            core.step_once()
            core.update_input({"sequence": 2, "held": ["Q"], "move_index": 1})
            core.step_once()
            core.step_once()
            core.step_once()
            core.close_trace()
            records = [json.loads(line) for line in path.read_text().splitlines()]

        steps = [record for record in records if record["event"] == "control_step"]
        self.assertEqual([step["move_index_edge"] for step in steps], [0, None, None, None])
        self.assertEqual(
            [step["move_index_buffer_before_step"] for step in steps],
            [None, 0, None, None],
        )
        self.assertEqual(
            [step["move_index_buffer_after_step"] for step in steps],
            [0, None, None, None],
        )
        self.assertEqual(
            [step["move_disposition"] for step in steps],
            [
                "buffered_yaw_interruption",
                "accepted",
                "accepted",
                "accepted",
            ],
        )

    def test_trace_records_exact_applied_tick_action_and_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.jsonl"
            writer = human.JsonlTraceWriter(path, {"extension": {"sha256": "0" * 64}})
            core = human.SemanticHumanEvalCore(FakeBoundary(), writer)
            core.update_input({"sequence": 1, "held": ["E"], "move_index": None})
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
        self.assertTrue(
            all(
                record["schema"] == "rek.g1_human_eval_trace.v2"
                for record in records
            )
        )
        step = records[3]
        self.assertEqual(step["tick"], 1)
        self.assertEqual(step["actions"], actions[:, 0].astype(int).tolist())
        self.assertEqual(step["input_sequence"], 1)
        self.assertIsNone(step["move_index_edge"])
        self.assertIsNone(step["move_index_buffer_before_step"])
        self.assertIsNone(step["move_index_buffer_after_step"])
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
        self.assertIn("enqueueControl('/reset', () => {", reset_handler)
        self.assertIn("sequence = 0", reset_handler)

    def test_browser_serializes_control_posts(self) -> None:
        self.assertIn("let controlQueue = synchronizeSequence()", human.INDEX_HTML)
        self.assertIn("controlQueue = controlQueue.catch", human.INDEX_HTML)
        self.assertIn("const request = enqueueControl('/input', () => ({", human.INDEX_HTML)
        self.assertIn("moveIndex !== null && moveRequestPending", human.INDEX_HTML)
        self.assertIn("return request.finally(() => {", human.INDEX_HTML)

    def test_browser_reload_synchronizes_server_input_sequence(self) -> None:
        self.assertIn("sequence = state.input_sequence", human.INDEX_HTML)
        self.assertIn("fetch('/state', {cache:'no-store'})", human.INDEX_HTML)
        self.assertIn("const initialHeldSnapshot = Array.from(held).sort()", human.INDEX_HTML)
        self.assertIn("held: initialHeldSnapshot", human.INDEX_HTML)
        self.assertIn("reload input clear was rejected", human.INDEX_HTML)
        self.assertIn("result.accepted !== true", human.INDEX_HTML)
        self.assertIn("body = {...body, sequence: ++sequence}", human.INDEX_HTML)

    def test_browser_exposes_all_moves_with_explicit_edge_field(self) -> None:
        self.assertEqual(human.INDEX_HTML.count('data-move="'), 17)
        self.assertIn("move_index: moveIndex", human.INDEX_HTML)
        self.assertIn("['Digit0',10]", human.INDEX_HTML)
        self.assertIn("['Digit6',16]", human.INDEX_HTML)
        self.assertNotIn("kick_move: kickMove", human.INDEX_HTML)

    def test_browser_suppresses_held_movement_key_autorepeat(self) -> None:
        self.assertIn(
            "if (event.repeat || held.has(symbol)) return;", human.INDEX_HTML
        )

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
