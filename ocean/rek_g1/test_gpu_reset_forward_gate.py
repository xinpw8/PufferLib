"""CPU-only ownership and layout regressions for opt-in persistent scratch."""

import unittest

import numpy as np

from gpu_reset_forward_gate import PersistentScratch
from verify_gpu_reset_forward_gate import contact_fingerprint, paired_error


class Buffer:
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class PersistentScratchTests(unittest.TestCase):
    def setUp(self):
        self.created = []

        def allocate(size):
            value = Buffer(1000 + len(self.created) * 256)
            self.created.append((size, value))
            return value

        self.scratch = PersistentScratch(allocate)

    def test_replay_preserves_distinct_pointers_and_ownership(self):
        first = [self.scratch.alloc(size) for size in (16, 32, 16)]
        for _ in range(3):
            self.scratch.begin_replay()
            self.assertEqual(first, [self.scratch.alloc(size) for size in (16, 32, 16)])
            self.scratch.end_replay()
        self.assertEqual(3, len(self.created))
        self.scratch.free(first[0], 16)
        self.assertEqual(3, len(self.scratch.buffers))

    def test_size_change_rejected(self):
        self.scratch.alloc(16)
        self.scratch.begin_replay()
        with self.assertRaisesRegex(RuntimeError, "sequence"):
            self.scratch.alloc(32)
        with self.assertRaisesRegex(RuntimeError, "count"):
            self.scratch.end_replay()
        self.assertIsNone(self.scratch.cursor)

    def test_missing_allocation_rejected(self):
        self.scratch.alloc(16)
        self.scratch.begin_replay()
        with self.assertRaisesRegex(RuntimeError, "count"):
            self.scratch.end_replay()

    def test_extra_allocation_rejected(self):
        self.scratch.alloc(16)
        self.scratch.begin_replay()
        self.scratch.alloc(16)
        with self.assertRaisesRegex(RuntimeError, "sequence"):
            self.scratch.alloc(16)
        self.scratch.end_replay()

    def test_nested_replay_rejected(self):
        self.scratch.begin_replay()
        with self.assertRaisesRegex(RuntimeError, "already active"):
            self.scratch.begin_replay()
        self.scratch.end_replay()


class DiagnosticMathTests(unittest.TestCase):
    def test_zero_baseline_requires_exact_candidate(self):
        value = np.zeros((8, 2), dtype=np.float32)
        self.assertTrue(paired_error(value, value, value)["accepted"])
        candidate = value.copy()
        candidate[0, 0] = np.nextafter(np.float32(0), np.float32(1))
        self.assertFalse(paired_error(value, value, candidate)["accepted"])

    def test_nonfinite_rejected(self):
        value = np.zeros((8, 2), dtype=np.float32)
        candidate = value.copy()
        candidate[0, 0] = np.nan
        self.assertFalse(paired_error(value, value, candidate)["accepted"])

    def test_contacts_distinguish_global_and_within_world_order(self):
        def fingerprint(rows):
            rows = np.asarray(rows)
            return contact_fingerprint({
                "nacon": np.array([len(rows)]),
                "contact.worldid": rows[:, 0], "contact.geom": rows[:, 1:],
            })

        first = fingerprint([[0, 1, 2], [1, 3, 4], [0, 5, 6]])
        across = fingerprint([[1, 3, 4], [0, 1, 2], [0, 5, 6]])
        within = fingerprint([[0, 5, 6], [1, 3, 4], [0, 1, 2]])
        self.assertNotEqual(first["raw_slots_sha256"], across["raw_slots_sha256"])
        self.assertEqual(first["within_world_order_sha256"], across["within_world_order_sha256"])
        self.assertNotEqual(first["within_world_order_sha256"], within["within_world_order_sha256"])
        self.assertEqual(first["pair_multiset_sha256"], within["pair_multiset_sha256"])

    def test_contact_overflow_is_not_fingerprinted_as_valid(self):
        result = contact_fingerprint({
            "nacon": np.array([2]), "contact.worldid": np.array([0]),
            "contact.geom": np.array([[1, 2]]),
        })
        self.assertFalse(result["valid_capacity"])


if __name__ == "__main__":
    unittest.main()
