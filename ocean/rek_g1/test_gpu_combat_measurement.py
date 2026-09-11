import math
import unittest

import torch

import gpu_combat_measurement as measurement


def synthetic_fixture(device: str):
    arenas = 2
    bodies = 10
    geoms = 11
    dtype = torch.float32
    body_owner = [-1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    body_zone = [0, 3, 16, 17, 8, 3, 16, 17, 2, 9]
    striker_part = [0, 0, 2, 2, 1, 0, 2, 2, 0, 1]
    striker_side = [-1, -1, 0, 1, 0, -1, 0, 1, -1, 1]
    striker_slot = [-1, -1, 2, 3, 0, -1, 2, 3, -1, 1]
    geom_body = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9]
    geom_zone = [body_zone[body] for body in geom_body]
    geom_zone[9] = measurement.BODY_ZONE_HEAD
    model_map = measurement.GpuCombatModelMap(
        body_count=bodies,
        geom_count=geoms,
        floor_geom_id=0,
        floor_half_height=0.1,
        root_qpos_addresses=(0, 7),
        root_body_ids=torch.tensor([1, 5], dtype=torch.long, device=device),
        left_foot_body_ids=torch.tensor([2, 6], dtype=torch.long, device=device),
        right_foot_body_ids=torch.tensor([3, 7], dtype=torch.long, device=device),
        geom_body_ids=torch.tensor(geom_body, dtype=torch.long, device=device),
        body_root_ids=torch.tensor(
            [0, 1, 1, 1, 1, 5, 5, 5, 5, 5],
            dtype=torch.long,
            device=device,
        ),
        body_owner=torch.tensor(body_owner, dtype=torch.long, device=device),
        body_zone=torch.tensor(body_zone, dtype=torch.long, device=device),
        geom_zone=torch.tensor(geom_zone, dtype=torch.long, device=device),
        striker_part=torch.tensor(striker_part, dtype=torch.long, device=device),
        striker_side=torch.tensor(striker_side, dtype=torch.long, device=device),
        striker_slot=torch.tensor(striker_slot, dtype=torch.long, device=device),
        geom_contype=torch.ones(geoms, dtype=torch.long, device=device),
        geom_conaffinity=torch.ones(geoms, dtype=torch.long, device=device),
    )
    qpos = torch.zeros((arenas, 14), dtype=dtype, device=device)
    qpos[:, 3] = 1.0
    qpos[:, 10] = 1.0
    xpos = torch.zeros((arenas, bodies, 3), dtype=dtype, device=device)
    xpos[:, 1, 2] = 1.0
    xpos[:, 5, 2] = 1.0
    xipos = xpos.clone()
    subtree_com = torch.zeros_like(xpos)
    cvel = torch.zeros((arenas, bodies, 6), dtype=dtype, device=device)
    geom_xpos = torch.zeros((arenas, geoms, 3), dtype=dtype, device=device)
    geom_xpos[:, 0, 2] = -0.1
    geom_xmat = torch.eye(3, dtype=dtype, device=device).expand(
        arenas, geoms, 3, 3).clone()
    capacity = 8
    contact_geom = torch.zeros((capacity, 2), dtype=torch.int32, device=device)
    contact_worldid = torch.zeros(capacity, dtype=torch.int32, device=device)
    contact_dist = torch.zeros(capacity, dtype=dtype, device=device)
    contact_pos = torch.zeros((capacity, 3), dtype=dtype, device=device)
    contact_frame = torch.eye(3, dtype=dtype, device=device).expand(
        capacity, 3, 3).clone()
    nacon = torch.zeros(1, dtype=torch.int32, device=device)
    time = torch.zeros(arenas, dtype=dtype, device=device)
    source = measurement.GpuCombatTensorSource(
        qpos=qpos,
        xpos=xpos,
        xipos=xipos,
        subtree_com=subtree_com,
        cvel=cvel,
        geom_xpos=geom_xpos,
        geom_xmat=geom_xmat,
        contact_geom=contact_geom,
        contact_worldid=contact_worldid,
        contact_dist=contact_dist,
        contact_pos=contact_pos,
        contact_frame=contact_frame,
        nacon=nacon,
        time=time,
    )
    sampler = measurement.RekG1GpuCombatMeasurement(model_map, source)
    return sampler, source


def set_contacts(source, entries):
    source.contact_geom.zero_()
    source.contact_worldid.zero_()
    source.contact_dist.zero_()
    source.contact_pos.zero_()
    source.contact_frame.copy_(torch.eye(
        3, dtype=source.contact_frame.dtype, device=source.contact_frame.device
    ).expand_as(source.contact_frame))
    for index, (arena, geom0, geom1) in enumerate(entries):
        source.contact_worldid[index] = arena
        source.contact_geom[index, 0] = geom0
        source.contact_geom[index, 1] = geom1
    source.nacon[0] = len(entries)


def valid_candidate_rows(batch):
    return batch.integers[batch.candidate_valid], batch.floats[batch.candidate_valid]


class GpuCombatMeasurementTests(unittest.TestCase):
    def test_first_slot_presence_matches_count_reduction(self):
        generator = torch.Generator().manual_seed(20260910)
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            for capacity in (1, 8, 128, 4096):
                for active_count in (0, capacity // 2, capacity):
                    keys = torch.randint(0, 91 * 91, (capacity,), generator=generator).to(device)
                    # Include many duplicate keys and stale inactive slots.
                    keys[::3] = 17
                    slots = torch.arange(capacity, device=device, dtype=torch.int32)
                    valid = slots < active_count
                    counts = torch.zeros(91 * 91, dtype=torch.int32, device=device)
                    counts.scatter_add_(0, keys, valid.to(torch.int32))
                    first = torch.full_like(counts, capacity)
                    first.scatter_reduce_(0, keys, torch.where(valid, slots, capacity),
                                          reduce="amin", include_self=True)
                    self.assertTrue(torch.equal(first < capacity, counts > 0))

    def test_duplicate_and_reversed_contacts_use_only_first_enter_slot(self):
        sampler, source = synthetic_fixture("cpu")
        sampler.reset()
        can_get_up = torch.zeros(4, dtype=torch.long)
        set_contacts(source, [(0, 4, 8), (0, 4, 8), (0, 8, 4), (1, 10, 1)])
        first = sampler.sample(0, can_get_up)
        self.assertEqual(first.hits.candidate_counts.tolist(), [1, 1])
        self.assertEqual(first.hits.candidate_valid.nonzero().flatten().tolist(), [0, 6])
        set_contacts(source, [])
        empty = sampler.sample(1, can_get_up)
        self.assertEqual(empty.hits.candidate_counts.tolist(), [0, 0])
        set_contacts(source, [(0, 8, 4)])
        reenter = sampler.sample(2, can_get_up)
        self.assertEqual(reenter.hits.candidate_counts.tolist(), [1, 0])

    def test_fall_rows_match_native_tilt_height_and_distinct_body_facts(self):
        sampler, source = synthetic_fixture("cpu")
        self.assertTrue(bool(sampler.reset().all()))
        half = math.sqrt(0.5)
        source.qpos[0, 3:7] = torch.tensor([half, half, 0.0, 0.0])
        source.xpos[0, 1, 2] = 0.5
        set_contacts(source, [(0, 0, 2), (0, 0, 4), (0, 0, 4)])
        result = sampler.sample(0, torch.zeros(4, dtype=torch.long))

        fall = result.fall
        self.assertTrue(bool(fall.valid.all()))
        self.assertAlmostEqual(
            float(fall.floats[0, measurement.FALL_TILT_DEGREES]), 90.0,
            places=4,
        )
        self.assertAlmostEqual(
            float(fall.floats[0, measurement.FALL_PELVIS_HEIGHT_RATIO]), 0.5,
            places=5,
        )
        self.assertEqual(
            int(fall.integers[0, measurement.FALL_LEFT_FOOT_BODY_CONTACT]), 1)
        self.assertEqual(
            int(fall.integers[0, measurement.FALL_RIGHT_FOOT_BODY_CONTACT]), 0)
        self.assertEqual(
            int(fall.integers[
                0, measurement.FALL_DISTINCT_NONFOOT_BODY_CONTACT_COUNT]), 1)
        self.assertEqual(
            int(fall.integers[1, measurement.FALL_BOTH_FEET_OFF_FLOOR]), 1)

    def test_contact_enter_head_override_and_mj_object_velocity_formula(self):
        sampler, source = synthetic_fixture("cpu")
        sampler.reset()
        source.xpos[0, 4] = torch.tensor([0.0, 0.0, 1.0])
        source.xpos[0, 8] = torch.tensor([0.0, 1.0, 1.0])
        source.xipos[0, 4] = torch.tensor([1.0, 0.0, 0.0])
        source.cvel[0, 4, 2] = 2.0
        source.cvel[0, 8, 3:6] = torch.tensor([0.0, -1.0, 0.0])
        set_contacts(source, [(0, 4, 9), (0, 9, 4)])
        result = sampler.sample(0, torch.zeros(4, dtype=torch.long))
        integers, floats = valid_candidate_rows(result.hits)

        self.assertTrue(bool(result.hits.arena_scan_valid.all()))
        self.assertEqual(result.hits.candidate_counts.tolist(), [1, 0])
        self.assertEqual(integers.shape[0], 1)
        self.assertEqual(int(integers[0, measurement.HIT_STRIKER_FIGHTER]), 0)
        self.assertEqual(int(integers[0, measurement.HIT_STRIKER_PART]), 1)
        self.assertEqual(int(integers[0, measurement.HIT_STRIKER_SIDE]), 0)
        self.assertEqual(int(integers[0, measurement.HIT_STRIKER_BODY_SLOT]), 0)
        self.assertEqual(int(integers[0, measurement.HIT_TARGET_ZONE]), 1)
        torch.testing.assert_close(
            floats[0, measurement.HIT_STRIKER_LINEAR_VELOCITY],
            torch.tensor([0.0, 2.0, 0.0]),
        )
        self.assertAlmostEqual(
            float(floats[0, measurement.HIT_RELATIVE_SPEED_MPS]), 3.0,
            places=5,
        )

    def test_u_kick_foot_geometry_passes_through_with_held_root_motion(self):
        sampler, source = synthetic_fixture("cpu")
        sampler.reset()
        source.cvel[0, 2, 3:6] = torch.tensor([1.25, -0.5, 0.0])
        source.cvel[0, 8, 3:6] = torch.tensor([0.25, -0.5, 0.0])
        set_contacts(source, [(0, 2, 9)])
        result = sampler.sample(0, torch.zeros(4, dtype=torch.long))
        integers, floats = valid_candidate_rows(result.hits)

        self.assertEqual(integers.shape[0], 1)
        self.assertEqual(int(integers[0, measurement.HIT_STRIKER_GEOM_ID]), 2)
        self.assertEqual(int(integers[0, measurement.HIT_TARGET_GEOM_ID]), 9)
        self.assertEqual(int(integers[0, measurement.HIT_STRIKER_PART]), 2)
        self.assertEqual(int(integers[0, measurement.HIT_STRIKER_SIDE]), 0)
        self.assertEqual(int(integers[0, measurement.HIT_STRIKER_BODY_SLOT]), 2)
        self.assertEqual(int(integers[0, measurement.HIT_TARGET_ZONE]), 1)
        torch.testing.assert_close(
            floats[0, measurement.HIT_STRIKER_LINEAR_VELOCITY],
            torch.tensor([1.25, -0.5, 0.0]),
        )
        self.assertAlmostEqual(
            float(floats[0, measurement.HIT_RELATIVE_SPEED_MPS]), 1.0,
            places=5,
        )

    def test_pair_persistence_exit_reentry_and_arena_clear(self):
        sampler, source = synthetic_fixture("cpu")
        sampler.reset()
        can_get_up = torch.zeros(4, dtype=torch.long)
        set_contacts(source, [(0, 4, 8), (1, 10, 1)])
        first = sampler.sample(0, can_get_up)
        self.assertEqual(first.hits.candidate_counts.tolist(), [1, 1])

        second = sampler.sample(1, can_get_up)
        self.assertEqual(second.hits.candidate_counts.tolist(), [0, 0])
        set_contacts(source, [])
        empty = sampler.sample(2, can_get_up)
        self.assertEqual(empty.hits.candidate_counts.tolist(), [0, 0])
        set_contacts(source, [(0, 8, 4), (1, 1, 10)])
        reentered = sampler.sample(3, can_get_up)
        self.assertEqual(reentered.hits.candidate_counts.tolist(), [1, 1])

        sampler.clear_arena_contacts(torch.tensor([1, 0], dtype=torch.bool))
        cleared = sampler.sample(4, can_get_up)
        self.assertEqual(cleared.hits.candidate_counts.tolist(), [1, 0])

    def test_bidirectional_strikers_have_stable_keys_and_substeps_wrap(self):
        sampler, source = synthetic_fixture("cpu")
        sampler.reset()
        can_get_up = torch.zeros(4, dtype=torch.long)
        set_contacts(source, [(1, 10, 4), (0, 4, 10)])
        first = sampler.sample(0, can_get_up)
        valid = first.hits.candidate_valid
        self.assertEqual(first.hits.candidate_counts.tolist(), [2, 2])
        keys = first.hits.candidate_processing_key[valid]
        self.assertEqual(torch.unique(keys).numel(), 4)
        ordered = first.hits.candidate_order[:int(first.hits.candidate_counts.sum())]
        ordered_keys = first.hits.candidate_processing_key[ordered]
        self.assertEqual(ordered_keys.tolist(), sorted(ordered_keys.tolist()))
        self.assertEqual(first.hits.candidate_offsets.tolist(), [0, 2])
        integer_rows = first.hits.integers[valid]
        self.assertEqual(
            sorted(integer_rows[:, measurement.HIT_STRIKER_FIGHTER].tolist()),
            [0, 0, 1, 1],
        )

        set_contacts(source, [])
        for substep in range(1, 10):
            wrapped = sampler.sample(substep, can_get_up)
            self.assertTrue(bool(wrapped.hits.arena_scan_valid.all()))
        self.assertTrue(bool(sampler.sample(0, can_get_up).hits.arena_scan_valid.all()))
        repeated = sampler.sample(0, can_get_up)
        self.assertFalse(bool(repeated.hits.arena_scan_valid.any()))

    def test_overflow_and_nonfinite_contact_fail_closed_without_committing(self):
        sampler, source = synthetic_fixture("cpu")
        sampler.reset()
        can_get_up = torch.zeros(4, dtype=torch.long)
        set_contacts(source, [(0, 4, 8)])
        source.nacon[0] = source.contact_geom.shape[0] + 1
        overflow = sampler.sample(0, can_get_up)
        self.assertTrue(bool(overflow.hits.contact_capacity_overflow))
        self.assertFalse(bool(overflow.hits.arena_scan_valid.any()))
        self.assertFalse(bool(overflow.hits.candidate_valid.any()))

        source.nacon[0] = 1
        source.contact_pos[0, 0] = float("nan")
        invalid = sampler.sample(0, can_get_up)
        self.assertFalse(bool(invalid.hits.arena_scan_valid[0]))
        self.assertTrue(bool(invalid.hits.arena_scan_valid[1]))
        self.assertFalse(bool(invalid.hits.candidate_valid.any()))
        source.contact_pos[0, 0] = 0.0
        recovered = sampler.sample(0, can_get_up)
        self.assertEqual(recovered.hits.candidate_counts.tolist(), [1, 0])

    def test_nonfinite_arena_time_fails_only_that_arena_transactionally(self):
        sampler, source = synthetic_fixture("cpu")
        sampler.reset()
        can_get_up = torch.zeros(4, dtype=torch.long)
        source.time[0] = float("nan")
        set_contacts(source, [(0, 4, 8), (1, 10, 1)])
        invalid = sampler.sample(0, can_get_up)
        self.assertFalse(bool(invalid.hits.arena_scan_valid[0]))
        self.assertTrue(bool(invalid.hits.arena_scan_valid[1]))
        self.assertEqual(invalid.hits.candidate_counts.tolist(), [0, 1])

        source.time[0] = 0.002
        recovered = sampler.sample(0, can_get_up)
        self.assertTrue(bool(recovered.hits.arena_scan_valid[0]))
        self.assertFalse(bool(recovered.hits.arena_scan_valid[1]))
        self.assertEqual(recovered.hits.candidate_counts.tolist(), [1, 0])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_matches_cpu_synthetic_measurement(self):
        outputs = []
        for device in ("cpu", "cuda"):
            sampler, source = synthetic_fixture(device)
            sampler.reset()
            source.cvel[0, 2, 3] = 2.0
            source.cvel[0, 8, 3] = -1.0
            set_contacts(source, [(0, 0, 2), (0, 2, 9), (1, 10, 1)])
            result = sampler.sample(
                0, torch.zeros(4, dtype=torch.long, device=device))
            integers, floats = valid_candidate_rows(result.hits)
            outputs.append((
                result.fall.floats.cpu(),
                result.fall.integers.cpu(),
                integers.cpu(),
                floats.cpu(),
                result.hits.candidate_counts.cpu(),
            ))
        for cpu_value, cuda_value in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(cpu_value, cuda_value)


if __name__ == "__main__":
    unittest.main()
