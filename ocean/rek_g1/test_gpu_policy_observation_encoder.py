"""CPU geometry and metadata fixtures; no CUDA execution or game changes."""

from copy import deepcopy
import math
import unittest

import torch

from gpu_policy_observation_encoder import (
    GpuPolarXYPolicyEncoder, GpuScaledPolarXYPolicyEncoder, SCALED_ENCODER_NAME,
    GpuStrikeAgeScaledPolarXYPolicyEncoder, STRIKE_AGE_ENCODER_NAME,
    encoder_fingerprint, encoder_metadata, reconstruct_opponent_world_xy,
    reconstruct_raw_observations, require_checkpoint_encoder_metadata,
)


CHECKPOINT_SHA = "a" * 64


def manifest(encoder_name="polar_xy_v1"):
    return {"checkpoint": {"sha256": CHECKPOINT_SHA},
            "policy_observation_encoder": encoder_metadata(encoder_name),
            "policy_observation_encoder_sha256": encoder_fingerprint(encoder_name)}


def make_encoder(rows):
    return GpuPolarXYPolicyEncoder(rows, "cpu", checkpoint_manifest=manifest(),
                                   checkpoint_sha256=CHECKPOINT_SHA, allow_cpu_for_tests=True)


def raw_fixture(rows):
    generator = torch.Generator().manual_seed(671)
    obs = torch.randn((rows, 223), generator=generator)
    obs[:, 3:7] = torch.tensor([1., 0., 0., 0.])
    return obs


class EncoderMetadataTests(unittest.TestCase):
    def test_exact_metadata_required_even_with_matching_checkpoint_width(self):
        correct = manifest()
        require_checkpoint_encoder_metadata(correct, CHECKPOINT_SHA)
        for key in ("checkpoint", "policy_observation_encoder", "policy_observation_encoder_sha256"):
            bad = deepcopy(correct)
            bad.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                require_checkpoint_encoder_metadata(bad, CHECKPOINT_SHA)
        bad = deepcopy(correct)
        bad["policy_observation_encoder"]["name"] = "raw_223"
        with self.assertRaises(ValueError):
            require_checkpoint_encoder_metadata(bad, CHECKPOINT_SHA)
        with self.assertRaises(ValueError):
            require_checkpoint_encoder_metadata(correct, "b" * 64)

    def test_descriptor_is_not_mutable_global_state(self):
        original_hash = encoder_fingerprint()
        descriptor = encoder_metadata()
        descriptor["replaced_policy_columns"]["86"] = "wrong"
        self.assertEqual(encoder_fingerprint(), original_hash)
        self.assertNotEqual(descriptor, encoder_metadata())

    def test_cpu_requires_explicit_test_permission(self):
        with self.assertRaises(ValueError):
            GpuPolarXYPolicyEncoder(1, "cpu", checkpoint_manifest=manifest(), checkpoint_sha256=CHECKPOINT_SHA)


class PolarXYTests(unittest.TestCase):
    def test_known_yaws_and_bearings(self):
        raw = raw_fixture(5)
        raw[:, :2] = 0
        raw[:, 86:88] = torch.tensor([[2., 0.], [0., 3.], [0., -4.], [-5., 0.], [2., 0.]])
        raw[4, 3] = math.sqrt(0.5)
        raw[4, 6] = math.sqrt(0.5)
        encoder = make_encoder(5)
        output = encoder.encode(raw)
        torch.testing.assert_close(output[:, 86], torch.tensor([2., 3., 4., 5., 2.]), rtol=0, atol=0)
        torch.testing.assert_close(output[:, 87], torch.tensor([0., math.pi/2, -math.pi/2, math.pi, -math.pi/2]), rtol=0, atol=1e-6)
        encoder.check_status()

    def test_raw_input_other_columns_and_output_pointer_are_unchanged(self):
        raw = raw_fixture(8)
        original = raw.clone()
        encoder = make_encoder(8)
        pointer = encoder.observations.data_ptr()
        for _ in range(3):
            output = encoder.encode(raw)
            self.assertEqual(output.data_ptr(), pointer)
            self.assertEqual(output.shape, (8, 223))
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(torch.equal(output[:, :86], original[:, :86]))
            self.assertTrue(torch.equal(output[:, 88:], original[:, 88:]))
            self.assertTrue(torch.equal(raw, original))
        with self.assertRaises(ValueError):
            encoder.encode(output)

    def test_random_quaternions_round_trip_xy(self):
        raw = raw_fixture(1024)
        generator = torch.Generator().manual_seed(987)
        raw[:, 3:7] = torch.randn((1024, 4), generator=generator)
        raw[:, :2] *= 10
        raw[:, 86:88] *= 10
        encoded = make_encoder(1024).encode(raw)
        reconstructed = reconstruct_opponent_world_xy(encoded)
        torch.testing.assert_close(reconstructed, raw[:, 86:88], rtol=0, atol=1e-5)

    def test_xy_translation_and_global_yaw_invariance(self):
        raw = raw_fixture(32)
        yaw = torch.linspace(-2.9, 2.9, 32)
        raw[:, 3] = torch.cos(yaw / 2)
        raw[:, 6] = torch.sin(yaw / 2)
        base = make_encoder(32).encode(raw).clone()
        translated = raw.clone()
        translated[:, :2] += torch.tensor([4., -7.])
        translated[:, 86:88] += torch.tensor([4., -7.])
        torch.testing.assert_close(make_encoder(32).encode(translated)[:, 86:88], base[:, 86:88], rtol=0, atol=2e-6)
        angle = 0.37
        c, s = math.cos(angle), math.sin(angle)
        rotation = torch.tensor([[c, -s], [s, c]])
        rotated = raw.clone()
        rotated[:, :2] = raw[:, :2] @ rotation.T
        rotated[:, 86:88] = raw[:, 86:88] @ rotation.T
        rotated[:, 3] = torch.cos((yaw + angle) / 2)
        rotated[:, 6] = torch.sin((yaw + angle) / 2)
        torch.testing.assert_close(make_encoder(32).encode(rotated)[:, 86:88], base[:, 86:88], rtol=0, atol=2e-6)

    def test_quaternion_sign_and_scale_invariance(self):
        raw = raw_fixture(16)
        raw[:, 3:7] = torch.randn((16, 4), generator=torch.Generator().manual_seed(876))
        base = make_encoder(16).encode(raw).clone()
        raw[:, 3:7] *= -2
        changed = make_encoder(16).encode(raw)
        torch.testing.assert_close(changed[:, 86:88], base[:, 86:88], rtol=0, atol=0)

    def test_coincident_xy_canonical_angle_and_round_trip(self):
        raw = raw_fixture(2)
        raw[:, 86:88] = raw[:, :2]
        encoder = make_encoder(2)
        output = encoder.encode(raw)
        self.assertTrue(torch.equal(output[:, 86:88], torch.zeros((2, 2))))
        self.assertTrue(torch.equal(reconstruct_opponent_world_xy(output), raw[:, :2]))
        encoder.check_status()

    def test_invalid_input_latches_status_without_guessing_direction(self):
        for reason in ("zero_quaternion", "vertical_heading", "nonfinite"):
            raw = raw_fixture(1)
            if reason == "zero_quaternion":
                raw[:, 3:7] = 0
            elif reason == "vertical_heading":
                raw[:, 3:7] = torch.tensor([0.5, 0.5, -0.5, 0.5])
            else:
                raw[:, 42] = float("nan")
            encoder = make_encoder(1)
            output = encoder.encode(raw)
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                encoder.check_status()
            if reason == "vertical_heading":
                self.assertTrue(torch.isnan(output[:, 87]).all())
            encoder.encode(raw_fixture(1))
            with self.assertRaises(RuntimeError):
                encoder.check_status()
            encoder.reset_status()
            encoder.encode(raw_fixture(1))
            encoder.check_status()

    def test_invalid_width_or_precision_is_rejected(self):
        encoder = make_encoder(1)
        with self.assertRaises(ValueError):
            encoder.encode(torch.zeros((1, 224)))
        with self.assertRaises(ValueError):
            encoder.encode(raw_fixture(1).double())


class ScaledPolarXYTests(unittest.TestCase):
    def encoder(self, rows):
        return GpuScaledPolarXYPolicyEncoder(rows, "cpu", checkpoint_manifest=manifest(SCALED_ENCODER_NAME),
                                            checkpoint_sha256=CHECKPOINT_SHA, allow_cpu_for_tests=True)

    def test_scaled_metadata_rejects_raw_polar_checkpoint(self):
        with self.assertRaises(ValueError):
            GpuScaledPolarXYPolicyEncoder(1, "cpu", checkpoint_manifest=manifest(),
                                          checkpoint_sha256=CHECKPOINT_SHA, allow_cpu_for_tests=True)
        with self.assertRaises(ValueError):
            make_metadata = manifest(SCALED_ENCODER_NAME)
            GpuPolarXYPolicyEncoder(1, "cpu", checkpoint_manifest=make_metadata,
                                    checkpoint_sha256=CHECKPOINT_SHA, allow_cpu_for_tests=True)
        self.assertNotEqual(encoder_fingerprint(), encoder_fingerprint(SCALED_ENCODER_NAME))

    def test_only_declared_columns_rescaled_without_clipping(self):
        raw = raw_fixture(3)
        raw[:, :2] = 0
        raw[:, 86:88] = torch.tensor([[0., 2.], [0., -2.], [-2., 0.]])
        raw[:, 72] = torch.tensor([90., 180., 270.])
        raw[:, 158] = torch.tensor([45., 0., -90.])
        raw[:, 188] = torch.tensor([120., 60., 240.])
        raw[:, 189] = torch.tensor([117., 30., -12.])
        original = raw.clone()
        encoder = self.encoder(3)
        output = encoder.encode(raw)
        expected = torch.tensor([[.5, .5, .25, 1., .975], [1., -.5, 0., .5, .25], [1.5, 1., -.5, 2., -.1]])
        torch.testing.assert_close(output[:, [72, 87, 158, 188, 189]], expected, rtol=0, atol=1e-7)
        changed = {72, 86, 87, 158, 188, 189}
        preserved = [i for i in range(223) if i not in changed]
        self.assertTrue(torch.equal(output[:, preserved], raw[:, preserved]))
        self.assertTrue(torch.equal(raw, original))
        encoder.check_status()

    def test_fixed_units_round_trip_and_polar_geometry(self):
        raw = raw_fixture(1024)
        raw[:, 72] *= 90
        raw[:, 158] *= 90
        raw[:, 188] = 120
        raw[:, 189] = torch.linspace(0, 120, 1024)
        output = self.encoder(1024).encode(raw)
        reconstructed = reconstruct_raw_observations(output, encoder_name=SCALED_ENCODER_NAME)
        torch.testing.assert_close(reconstructed, raw, rtol=0, atol=3.1e-5)
        torch.testing.assert_close(reconstruct_opponent_world_xy(output, encoder_name=SCALED_ENCODER_NAME), raw[:, 86:88], rtol=0, atol=2e-6)


class StrikeAgeScaledPolarXYTests(unittest.TestCase):
    def encoder(self, rows):
        return GpuStrikeAgeScaledPolarXYPolicyEncoder(rows, "cpu", checkpoint_manifest=manifest(STRIKE_AGE_ENCODER_NAME),
                                                      checkpoint_sha256=CHECKPOINT_SHA, allow_cpu_for_tests=True)

    def test_new_descriptor_preserves_existing_descriptor_and_changes_only_age_units(self):
        self.assertEqual(encoder_fingerprint(SCALED_ENCODER_NAME),
                         "8b24ff625aa2f3a2916920255ea38efb54e597095078bb298257d99f5b512c6b")
        old, new = encoder_metadata(SCALED_ENCODER_NAME), encoder_metadata(STRIKE_AGE_ENCODER_NAME)
        self.assertNotEqual(encoder_fingerprint(STRIKE_AGE_ENCODER_NAME), encoder_fingerprint(SCALED_ENCODER_NAME))
        self.assertEqual(new["fixed_scale_divisors"], {**old["fixed_scale_divisors"], "198":120., "199":120.})
        unchanged = [i for low, high in new["unchanged_column_intervals_half_open"] for i in range(low, high)]
        self.assertEqual(unchanged, [i for i in range(223) if i not in (72,86,87,158,188,189,198,199)])

    def test_only_two_additional_columns_change_and_scores_remain_exact(self):
        raw = raw_fixture(5)
        raw[:, 198] = torch.tensor([0., 1., 42.76975, 120., 240.])
        raw[:, 199] = torch.tensor([118.02839, .002, 60., 180., 360.])
        raw[:, 190:192] = torch.tensor([23., 14.])
        original = raw.clone()
        base = ScaledPolarXYTests().encoder(5).encode(raw).clone()
        encoder = self.encoder(5)
        pointer = encoder.observations.data_ptr()
        for _ in range(3):
            output = encoder.encode(raw)
            self.assertEqual(output.data_ptr(), pointer)
            self.assertTrue(torch.equal(raw, original))
            cols = [i for i in range(223) if i not in (198,199)]
            self.assertTrue(torch.equal(output[:, cols], base[:, cols]))
            self.assertTrue(torch.equal(output[:, 198:200], (raw[:, 198:200].double()/120).float()))
            self.assertEqual(output[4, 199].item(), 3.)
            encoder.check_status()
        torch.testing.assert_close(reconstruct_raw_observations(output, encoder_name=STRIKE_AGE_ENCODER_NAME),
                                   raw, rtol=0, atol=3.1e-5)
        torch.testing.assert_close(reconstruct_opponent_world_xy(output, encoder_name=STRIKE_AGE_ENCODER_NAME),
                                   raw[:,86:88], rtol=0, atol=2e-6)

    def test_old_checkpoint_is_not_matching_resume(self):
        with self.assertRaises(ValueError):
            GpuStrikeAgeScaledPolarXYPolicyEncoder(1, "cpu", checkpoint_manifest=manifest(SCALED_ENCODER_NAME),
                                                  checkpoint_sha256=CHECKPOINT_SHA, allow_cpu_for_tests=True)
        with self.assertRaises(ValueError):
            GpuScaledPolarXYPolicyEncoder(1, "cpu", checkpoint_manifest=manifest(STRIKE_AGE_ENCODER_NAME),
                                         checkpoint_sha256=CHECKPOINT_SHA, allow_cpu_for_tests=True)


if __name__ == "__main__":
    unittest.main()
