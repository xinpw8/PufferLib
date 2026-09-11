import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import client_fixed_import
import raw_bone_validate
from test_client_fixed_import import (
    control_records,
    inventory_record,
    schedule_manifest,
)
from test_client_fixed_import_v6 import v6_schedule_records
from test_raw_bone_validate_v7 import upgrade_to_v7
from trace import Trace


class ClientFixedImportV7RoutingTests(unittest.TestCase):
    def test_v7_uses_protocol_import_and_accepts_motion_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl"
            output = root / "trace.jsonl"
            raw.write_text(
                json.dumps({
                    "event": "capture_start",
                    "schema": client_fixed_import.SCHEMA_V7,
                }) + "\n",
                encoding="utf-8",
            )
            expected = {"raw_recorder_schema": client_fixed_import.SCHEMA_V7}
            with mock.patch.object(
                client_fixed_import, "_convert_v5", return_value=expected
            ) as protocol_import:
                observed = client_fixed_import.convert(
                    raw,
                    root / "inventory.json",
                    output,
                    control_log_path=root / "control.jsonl",
                    schedule_manifest_path=root / "schedule.json",
                    motion_edge="walk_forward.press.1",
                )
            self.assertEqual(observed, expected)
            protocol_import.assert_called_once()
            self.assertEqual(
                protocol_import.call_args.kwargs["motion_edge"],
                "walk_forward.press.1",
            )


class ClientFixedImportV7EndToEndTests(unittest.TestCase):
    def test_imports_exact_g1_root_stream_and_preserves_v7_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "capture.jsonl"
            inventory = root / "inventory.json"
            control = root / "control.jsonl"
            manifest_path = root / "schedule.json"
            output = root / "v7.trace"
            manifest = schedule_manifest()
            records = upgrade_to_v7(
                v6_schedule_records(("g1_30", "g1_30")),
                runtime_model="g1",
                semantic_ids=(None, "t800"),
                copy_records=False,
            )
            raw.write_text(
                "".join(
                    json.dumps(record, separators=(",", ":")) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            inventory.write_text(
                json.dumps(inventory_record()), encoding="utf-8"
            )
            control.write_text(
                "".join(
                    json.dumps(record, separators=(",", ":")) + "\n"
                    for record in control_records(manifest)
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            validation = raw_bone_validate.validate(raw)
            result = client_fixed_import.convert(
                raw,
                inventory,
                output,
                control_log_path=control,
                schedule_manifest_path=manifest_path,
                motion_edge="walk_forward.press.1",
            )
            trace = Trace.load(output)

            self.assertEqual(
                validation["claims"]["validated_runtime_model"], "g1"
            )
            self.assertTrue(
                validation["claims"]["exact_g1_vs_g1_validated"]
            )
            self.assertEqual(
                result["raw_recorder_schema"], client_fixed_import.SCHEMA_V7
            )
            self.assertEqual(result["ticks"], 26010)
            self.assertEqual(
                result["tick_domain"],
                "controlled_schedule_client_fixed_substep_500hz",
            )
            self.assertEqual(trace.header["tick_rate_hz"], 500)
            self.assertEqual(
                trace.header["fighter_pairing"]["runtime_model"], "g1"
            )
            self.assertTrue(
                trace.header["fighter_pairing"]["exact_g1_vs_g1"]
            )
            self.assertEqual(
                trace.header["fighter_pairing"]["fighters"]["0"]["layout_id"],
                "g1_30",
            )
            self.assertEqual(
                trace.header["private_ai_scope"]["solo_route_reason"],
                "solo_route_proven",
            )
            self.assertTrue(
                trace.header["private_ai_scope"]["solo_route_proven"]
            )
            self.assertFalse(
                trace.header["private_ai_scope"]["server_private_proven"]
            )
            self.assertEqual(
                trace.header["private_ai_scope"]["server_private_status"],
                "unknown",
            )
            self.assertEqual(
                trace.header["selected_command_edge"]["selector"],
                "walk_forward.press.1",
            )


if __name__ == "__main__":
    unittest.main()
