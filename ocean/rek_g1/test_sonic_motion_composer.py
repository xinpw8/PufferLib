import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sonic_motion_composer as composer


CONTRACT = HERE.parent / "rek" / "evidence" / "g1_motion_clip_contract.v1.json"


class SonicMotionComposerTests(unittest.TestCase):
    def test_contract_idle_installs_exact_native_layer_fields(self):
        measured = composer.load_contract_clip(CONTRACT, "idle")
        layer = composer.Layer()
        composer.install_contract_clip(layer, measured)

        self.assertEqual(measured.clip.frame_count, 39)
        self.assertEqual(measured.clip.fps, 50.0)
        self.assertEqual(measured.clip.filename, "idle_processed.npz")
        self.assertEqual(
            measured.clip.sha256,
            "2fd9ca753183566cceaff3ff10ae54b1a7167986297524a9d24d970d57aee969",
        )
        self.assertEqual(layer.start_frame, 0)
        self.assertEqual(layer.end_frame, 38)
        self.assertEqual(layer.speed, 1.0)
        self.assertEqual(layer.per_tick, 1.0)
        self.assertEqual(layer.cursor, 0.0)
        self.assertTrue(layer.loop)
        self.assertTrue(layer.active)
        self.assertFalse(layer.heading_valid)
        self.assertFalse(layer.heading_resync)
        self.assertEqual(layer.feature_instance_id, 377)
        self.assertEqual(measured.config.blend_in_time, 1.0)
        self.assertEqual(measured.config.blend_out_time, 0.5139999985694885)
        self.assertEqual(measured.config.yaw_blend, 0.0)

    def test_install_clamps_start_end_and_end_before_start(self):
        clip = composer.NpzClip(frame_count=10, fps=100.0)
        layer = composer.Layer()
        composer.install_layer(
            layer,
            clip,
            None,
            mirror=True,
            loop=False,
            speed=0.5,
            start_frame=-4,
            end_frame=99,
            controller_rate_hz=50,
        )
        self.assertEqual((layer.start_frame, layer.end_frame), (0, 9))
        self.assertEqual(layer.per_tick, 1.0)
        self.assertTrue(layer.mirror)

        composer.install_layer(
            layer,
            clip,
            None,
            mirror=False,
            loop=False,
            speed=1.0,
            start_frame=7,
            end_frame=3,
        )
        self.assertEqual((layer.start_frame, layer.end_frame), (7, 7))

    def test_speed_sanitize_preserves_direction_and_selects_entry_cursor(self):
        clip = composer.NpzClip(frame_count=10, fps=50.0)
        layer = composer.Layer()
        composer.install_layer(
            layer,
            clip,
            None,
            mirror=False,
            loop=False,
            speed=-0.001,
            start_frame=2,
            end_frame=7,
        )
        self.assertEqual(layer.speed, -composer.MIN_ABS_PLAYBACK_SPEED)
        self.assertEqual(layer.per_tick, -composer.MIN_ABS_PLAYBACK_SPEED)
        self.assertEqual(layer.cursor, 7.0)

        composer.install_layer(
            layer,
            clip,
            None,
            mirror=False,
            loop=False,
            speed=0.0,
            start_frame=2,
            end_frame=7,
        )
        self.assertEqual(layer.speed, composer.MIN_ABS_PLAYBACK_SPEED)
        self.assertEqual(layer.cursor, 2.0)

    def test_wrap_loop_cursor_uses_inclusive_frame_span(self):
        layer = composer.Layer(start_frame=2, end_frame=4, cursor=8.25)
        self.assertTrue(composer.wrap_loop_cursor(layer))
        self.assertEqual(layer.cursor, 2.25)

        layer.cursor = -4.0
        self.assertTrue(composer.wrap_loop_cursor(layer))
        self.assertEqual(layer.cursor, 2.0)
        self.assertFalse(composer.wrap_loop_cursor(layer))

    def test_resolve_frames_interpolates_and_clamps_non_loop(self):
        layer = composer.Layer(
            clip=composer.NpzClip(frame_count=10, fps=50.0),
            loop=False,
            per_tick=0.5,
            cursor=3.25,
            start_frame=2,
            end_frame=6,
        )
        self.assertEqual(composer.resolve_frames(layer, 2), (4, 5, 0.25))
        self.assertEqual(composer.resolve_frames(layer, 20), (6, 6, 0.0))

        layer.cursor = 1.25
        self.assertEqual(composer.resolve_frames(layer, 0), (2, 3, 0.0))

    def test_resolve_frames_wraps_interpolation_across_loop_seam(self):
        layer = composer.Layer(
            clip=composer.NpzClip(frame_count=10, fps=50.0),
            loop=True,
            per_tick=1.0,
            cursor=4.75,
            start_frame=2,
            end_frame=4,
        )
        self.assertEqual(composer.resolve_frames(layer, 0), (4, 2, 0.75))
        self.assertEqual(composer.resolve_frames(layer, 1), (2, 3, 0.75))
        self.assertEqual(composer.resolve_frames(layer, -4), (3, 4, 0.75))

    def test_advance_wraps_loop_cursor(self):
        layer = composer.Layer(
            clip=composer.NpzClip(frame_count=5, fps=50.0),
            active=True,
            loop=True,
            per_tick=1.5,
            cursor=3.75,
            start_frame=0,
            end_frame=4,
        )
        self.assertEqual(composer.advance_layer(layer), (True, False))
        self.assertEqual(layer.cursor, 0.25)

    def test_non_loop_completion_occurs_at_endpoint_and_callback_runs_once(self):
        calls = []
        layer = composer.Layer(
            clip=composer.NpzClip(frame_count=3, fps=50.0),
            active=True,
            loop=False,
            per_tick=0.5,
            cursor=1.0,
            start_frame=0,
            end_frame=2,
            on_complete=lambda: calls.append("done"),
        )
        runtime = composer.SingleLayerComposer(layer, action_playing=True)

        self.assertEqual(runtime.advance(), (False, False))
        self.assertEqual(layer.cursor, 1.5)
        self.assertEqual(runtime.advance(), (False, True))
        self.assertEqual(layer.cursor, 2.0)
        self.assertFalse(runtime.action_playing)
        self.assertEqual(calls, ["done"])
        self.assertEqual(runtime.advance(), (False, True))
        self.assertEqual(calls, ["done"])
        self.assertTrue(layer.active)

    def test_reverse_non_loop_completion_occurs_at_start_endpoint(self):
        layer = composer.Layer(
            clip=composer.NpzClip(frame_count=3, fps=50.0),
            active=True,
            loop=False,
            per_tick=-0.5,
            cursor=1.0,
            start_frame=0,
            end_frame=2,
        )
        self.assertEqual(composer.advance_layer(layer), (False, False))
        self.assertEqual(composer.advance_layer(layer), (False, True))
        self.assertEqual(layer.cursor, 0.0)

    def test_non_loop_advance_clamps_overshoot_before_completion(self):
        forward = composer.Layer(
            clip=composer.NpzClip(frame_count=4, fps=50.0),
            active=True,
            loop=False,
            per_tick=7.0,
            cursor=1.0,
            start_frame=0,
            end_frame=3,
        )
        reverse = composer.Layer(
            clip=forward.clip,
            active=True,
            loop=False,
            per_tick=-7.0,
            cursor=2.0,
            start_frame=0,
            end_frame=3,
        )
        self.assertEqual(composer.advance_layer(forward), (False, True))
        self.assertEqual(forward.cursor, 3.0)
        self.assertEqual(composer.advance_layer(reverse), (False, True))
        self.assertEqual(reverse.cursor, 0.0)

    def test_contract_non_loop_kick_duration_matches_cursor_gate(self):
        measured = composer.load_contract_clip(CONTRACT, "kick_left_front")
        layer = composer.Layer()
        composer.install_contract_clip(layer, measured)
        runtime = composer.SingleLayerComposer(layer, action_playing=True)

        for _ in range(144):
            self.assertFalse(runtime.advance().completed)
        self.assertTrue(runtime.advance().completed)
        self.assertEqual(layer.cursor, 145.0)

    def test_weight_current_uses_asymmetric_renormalization(self):
        self.assertEqual(composer.weight_current(0.0, 2, 26), 0.0)
        self.assertAlmostEqual(composer.weight_current(1.0, 2, 26), 0.34210527)
        self.assertAlmostEqual(composer.weight_current(2.0, 2, 26), 0.52)
        self.assertEqual(composer.weight_current(26.0, 2, 26), 1.0)
        self.assertEqual(composer.xfade_at(1, -1, 2, 26), 0.0)

    def test_copy_layer_copies_heading_state_but_clears_callback(self):
        callback = lambda: None
        source = composer.Layer(
            clip=composer.NpzClip(frame_count=4, fps=50.0, path_id=8),
            mirror=True,
            loop=True,
            speed=-1.0,
            per_tick=-1.0,
            cursor=2.5,
            start_frame=1,
            end_frame=3,
            feature_instance_id=8,
            on_complete=callback,
            active=False,
            prev_heading=0.25,
            last_heading_delta=-0.125,
            heading_valid=True,
            heading_resync=True,
        )
        destination = composer.Layer(on_complete=callback)
        composer.copy_layer(source, destination)
        self.assertTrue(destination.active)
        self.assertIsNone(destination.on_complete)
        self.assertEqual(destination.clip, source.clip)
        self.assertEqual(destination.cursor, 2.5)
        self.assertEqual(destination.feature_instance_id, 8)
        self.assertEqual(destination.prev_heading, 0.25)
        self.assertEqual(destination.last_heading_delta, -0.125)
        self.assertTrue(destination.heading_valid)
        self.assertTrue(destination.heading_resync)

    def test_g1_mirror_table_matches_current_build_name_rules(self):
        names = (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        )
        tables = composer.build_mirror_tables(names)
        self.assertEqual(
            tables.source_indices,
            (
                6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 14,
                22, 23, 24, 25, 26, 27, 28, 15, 16, 17, 18, 19, 20, 21,
            ),
        )
        self.assertEqual(
            tables.negate,
            (
                False, True, True, False, False, True,
                False, True, True, False, False, True,
                True, True, False,
                False, True, True, False, True, False, True,
                False, True, True, False, True, False, True,
            ),
        )

    def test_play_action_builds_measured_idle_to_kick_transition(self):
        idle = composer.load_contract_clip(CONTRACT, "idle")
        kick = composer.load_contract_clip(CONTRACT, "kick_left_front")
        runtime = composer.TransitionComposer(
            root_heading_sampler=lambda _layer: composer.LayerRootHeadingResult(0.0, False)
        )

        self.assertTrue(runtime.play_action(idle.clip, idle.config))
        runtime.advance()
        self.assertTrue(runtime.play_action(kick.clip, kick.config))
        self.assertEqual((runtime.w_in, runtime.w_out, runtime.w_total), (2, 26, 26))
        self.assertEqual(runtime.xt, 0)
        self.assertTrue(runtime.from_layer.active)
        self.assertEqual(runtime.from_layer.clip, idle.clip)
        self.assertIsNone(runtime.from_layer.on_complete)
        self.assertTrue(runtime.action_playing)
        self.assertEqual(runtime.action_move_id, 1)

        first = runtime.advance()
        self.assertEqual(runtime.xt, 1)
        self.assertAlmostEqual(first.weight_current, 0.34210527)
        self.assertEqual(runtime.current_layer.cursor, 1.0)
        self.assertEqual(runtime.from_layer.cursor, 2.0)
        for _ in range(25):
            final = runtime.advance()
        self.assertEqual(runtime.xt, 26)
        self.assertFalse(runtime.from_layer.active)
        self.assertEqual(final.weight_current, 1.0)

    def test_completion_callback_transition_is_processed_in_same_advance(self):
        idle = composer.load_contract_clip(CONTRACT, "idle")
        kick = composer.load_contract_clip(CONTRACT, "kick_left_front")
        runtime = composer.TransitionComposer(
            entry_matcher=lambda _target, _outgoing: 4.0,
            root_heading_sampler=lambda _layer: composer.LayerRootHeadingResult(0.0, False),
        )
        calls = []

        def return_to_idle():
            calls.append("done")
            runtime.play_action(idle.clip, idle.config)

        runtime.play_action(kick.clip, kick.config, return_to_idle)
        runtime.current_layer.cursor = 144.0
        result = runtime.advance()
        self.assertEqual(calls, ["done"])
        self.assertEqual(runtime.current_layer.clip, idle.clip)
        self.assertEqual(runtime.current_layer.cursor, 4.0)
        self.assertEqual(runtime.from_layer.clip, kick.clip)
        self.assertEqual(runtime.from_layer.cursor, 145.0)
        self.assertEqual(runtime.xt, 1)
        self.assertTrue(runtime.from_layer.active)
        self.assertTrue(result.current.completed)
        self.assertFalse(runtime.action_playing)

    def test_active_loop_entry_requires_injected_matcher_and_immediate_still_matches(self):
        first_clip = composer.NpzClip(frame_count=10, fps=50.0, name="first")
        next_clip = composer.NpzClip(frame_count=10, fps=50.0, name="next")
        config = composer.MocapClipConfig(False, True, 1.0, 0, 9)
        runtime = composer.TransitionComposer()
        runtime.play_action(first_clip, config)
        runtime.current_layer.cursor = 3.0

        with self.assertRaises(composer.UnsupportedComposerSemantics):
            runtime.play_action(next_clip, config)
        self.assertEqual(runtime.current_layer.clip, first_clip)
        self.assertEqual(runtime.current_layer.cursor, 3.0)
        self.assertFalse(runtime.from_layer.active)

        calls = []
        runtime.entry_matcher = lambda target, outgoing: (
            calls.append((target.clip.name, outgoing.clip.name)) or 7.25
        )
        self.assertTrue(runtime.play_action_immediate(next_clip, config))
        self.assertEqual(calls, [("next", "first")])
        self.assertEqual(runtime.current_layer.cursor, 7.25)
        self.assertFalse(runtime.from_layer.active)
        self.assertEqual(runtime.xt, 0)

    def test_cancel_preserves_current_cursor_pending_heading_and_blend_widths(self):
        callback = lambda: None
        runtime = composer.TransitionComposer(
            current_layer=composer.Layer(
                active=True,
                cursor=4.5,
                on_complete=callback,
                heading_valid=True,
                heading_resync=True,
            ),
            from_layer=composer.Layer(active=True),
            xt=7,
            w_in=2,
            w_out=26,
            w_total=26,
            action_playing=True,
            pending_heading_delta=0.75,
        )
        runtime.cancel_action()
        self.assertTrue(runtime.current_layer.active)
        self.assertEqual(runtime.current_layer.cursor, 4.5)
        self.assertIsNone(runtime.current_layer.on_complete)
        self.assertFalse(runtime.from_layer.active)
        self.assertFalse(runtime.action_playing)
        self.assertEqual(runtime.xt, 0)
        self.assertEqual((runtime.w_in, runtime.w_out, runtime.w_total), (2, 26, 26))
        self.assertEqual(runtime.pending_heading_delta, 0.75)
        self.assertFalse(runtime.current_layer.heading_valid)
        self.assertFalse(runtime.current_layer.heading_resync)

    def test_set_locomotion_speed_only_rescales_active_current_loop(self):
        clip = composer.NpzClip(frame_count=10, fps=100.0)
        current = composer.Layer(clip=clip, active=True, loop=True, speed=-0.5, per_tick=-1.0)
        outgoing = composer.Layer(clip=clip, active=True, loop=True, speed=1.0, per_tick=9.0)
        runtime = composer.TransitionComposer(current_layer=current, from_layer=outgoing)
        runtime.set_locomotion_speed(-10.0)
        self.assertAlmostEqual(current.per_tick, -0.05)
        self.assertEqual(outgoing.per_tick, 9.0)
        runtime.set_locomotion_speed(2.0)
        self.assertEqual(current.per_tick, -2.0)
        current.loop = False
        runtime.set_locomotion_speed(7.0)
        self.assertEqual(current.per_tick, -2.0)

    def test_sample_layer_applies_dof_and_root_mirror_after_interpolation(self):
        clip = composer.NpzClip(frame_count=2, fps=50.0)
        layer = composer.Layer(
            clip=clip,
            config=composer.MocapClipConfig(True, False, 1.0, 0, 1),
            mirror=True,
            loop=False,
            cursor=0.5,
            start_frame=0,
            end_frame=1,
        )
        samples = composer.ClipSamples(
            ((1.0, 10.0, 100.0), (3.0, 14.0, 104.0)),
            ((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)),
        )
        tables = composer.build_mirror_tables(("left_roll", "right_roll", "waist_pitch"))

        def linear_slerp(a, b, t):
            return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))

        result = composer.sample_layer(
            layer,
            samples,
            mirror_tables=tables,
            quaternion_slerp=linear_slerp,
        )
        self.assertEqual(result.joint_positions, (-12.0, -2.0, 102.0))
        self.assertEqual(result.root_quaternion_wxyz, (3.0, -4.0, 5.0, -6.0))

    def test_sample_layer_yaw_removal_uses_recovered_left_premultiply(self):
        calls = {}
        clip = composer.NpzClip(frame_count=1, fps=50.0)
        layer = composer.Layer(
            clip=clip,
            config=composer.MocapClipConfig(
                False, False, 1.0, 0, 0, yaw_blend=0.5
            ),
            cursor=0.0,
            start_frame=0,
            end_frame=0,
        )
        samples = composer.ClipSamples(((1.0,),), ((0.5, 0.1, 0.2, 0.3),))

        def atan2f(y, x):
            calls["atan2"] = (y, x)
            return 0.4

        def sin_cos_f(angle):
            calls["half_angle"] = angle
            return 0.2, 0.8

        result = composer.sample_layer(
            layer,
            samples,
            quaternion_slerp=lambda a, _b, _t: a,
            atan2f=atan2f,
            sin_cos_f=sin_cos_f,
        )
        self.assertAlmostEqual(calls["atan2"][0], 0.34)
        self.assertAlmostEqual(calls["atan2"][1], 0.74)
        self.assertAlmostEqual(calls["half_angle"], -0.1)
        for actual, expected in zip(
            result.root_quaternion_wxyz, (0.34, 0.04, 0.18, 0.34)
        ):
            self.assertAlmostEqual(actual, expected)

    def test_reference_frame_blends_dofs_and_root_with_xfade_at(self):
        old_clip = composer.NpzClip(frame_count=1, fps=50.0, name="old")
        new_clip = composer.NpzClip(frame_count=1, fps=50.0, name="new")
        config = composer.MocapClipConfig(False, False, 1.0, 0, 0)
        current = composer.Layer(
            clip=new_clip, config=config, active=True, start_frame=0, end_frame=0
        )
        outgoing = composer.Layer(
            clip=old_clip, config=config, active=True, start_frame=0, end_frame=0
        )
        by_name = {
            "old": composer.ClipSamples(((0.0,),), ((1.0, 0.0, 0.0, 0.0),)),
            "new": composer.ClipSamples(((10.0,),), ((0.0, 1.0, 0.0, 0.0),)),
        }

        def linear_slerp(a, b, t):
            return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))

        result = composer.get_reference_frame(
            current,
            outgoing,
            0,
            samples_for_clip=lambda clip: by_name[clip.name],
            num_dofs=1,
            w_in=2,
            w_out=2,
            xt=1,
            quaternion_slerp=linear_slerp,
        )
        self.assertEqual(result.joint_positions, (5.0,))
        self.assertEqual(result.root_quaternion_wxyz, (0.5, 0.5, 0.0, 0.0))

    def test_heading_contribution_repeats_delta_across_seam_and_resync_tick(self):
        layer = composer.Layer(
            config=composer.MocapClipConfig(False, True, 1.0, 0, 4, yaw_blend=1.0),
            active=True,
        )
        samples = iter(
            (
                composer.LayerRootHeadingResult(0.0, False),
                composer.LayerRootHeadingResult(0.1, False),
                composer.LayerRootHeadingResult(-3.0, True),
                composer.LayerRootHeadingResult(0.2, False),
                composer.LayerRootHeadingResult(0.3, False),
            )
        )
        sampler = lambda _layer: next(samples)
        values = [
            composer.layer_heading_contribution(
                layer, False, root_heading_sampler=sampler
            )
            for _ in range(5)
        ]
        self.assertAlmostEqual(values[0], 0.0)
        for value in values[1:]:
            self.assertAlmostEqual(value, 0.1)
        self.assertFalse(layer.heading_resync)
        self.assertAlmostEqual(layer.last_heading_delta, 0.1)

    def test_unknown_runtime_math_and_feature_matching_fail_closed(self):
        clip = composer.NpzClip(frame_count=1, fps=50.0)
        config = composer.MocapClipConfig(False, False, 1.0, 0, 0)
        layer = composer.Layer(
            clip=clip, config=config, active=True, start_frame=0, end_frame=0
        )
        samples = composer.ClipSamples(((0.0,),), ((1.0, 0.0, 0.0, 0.0),))
        with self.assertRaises(composer.UnsupportedComposerSemantics):
            composer.sample_layer(layer, samples)
        with self.assertRaises(composer.UnsupportedComposerSemantics):
            composer.calc_heading_wxyz((1.0, 0.0, 0.0, 0.0), atan2f=None)

    def test_crossfade_fails_closed(self):
        with self.assertRaises(composer.UnsupportedComposerSemantics):
            composer.SingleLayerComposer().begin_crossfade()


if __name__ == "__main__":
    unittest.main()
