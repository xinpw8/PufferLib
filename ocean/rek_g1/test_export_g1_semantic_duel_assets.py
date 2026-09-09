from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import export_g1_semantic_duel_assets as subject


HERE = Path(__file__).resolve().parent
ROUTE_CONTRACT = HERE.parent / "rek" / "evidence" / "g1_motion_route_contract.v1.json"


class SemanticDuelAssetTests(unittest.TestCase):
    def test_checked_in_route_contract_is_complete(self) -> None:
        contract, raw = subject.load_route_contract(ROUTE_CONTRACT)
        self.assertGreater(len(raw), 0)
        self.assertEqual([route["route_id"] for route in contract["routes"]], list(range(11)))
        self.assertEqual(
            [route["runtime_move_index"] for route in contract["routes"][7:]],
            [6, 7, 8, 9],
        )

    def test_route_config_projection_preserves_binary32_values(self) -> None:
        contract, _ = subject.load_route_contract(ROUTE_CONTRACT)
        backward = subject._route_config(contract["routes"][2])
        self.assertEqual(backward["playback_speed"], -1.0)
        self.assertEqual(backward["loop"], 1)
        left_side = subject._route_config(contract["routes"][7])
        self.assertEqual(left_side["blend_in_seconds"], 0.035999998450279236)
        self.assertEqual(left_side["blend_out_seconds"], 0.6899999976158142)

    def test_root_order_conversion(self) -> None:
        xyzw = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        actual = subject.root_xyzw_to_wxyz(xyzw)
        np.testing.assert_array_equal(
            actual,
            np.array([[4.0, 1.0, 2.0, 3.0]], dtype=np.float32),
        )
        self.assertTrue(actual.flags.c_contiguous)

    def test_route_contract_rejects_changed_identity(self) -> None:
        contract = json.loads(ROUTE_CONTRACT.read_text(encoding="utf-8"))
        contract["routes"][0]["route_id"] = 99
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(subject.SemanticDuelAssetError, "ordered"):
                subject.load_route_contract(path)


if __name__ == "__main__":
    unittest.main()
