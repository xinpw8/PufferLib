from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import held_trace_candidate_replay as replay


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class HeldTraceCandidateReplayTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        trace = root / "trace.jsonl"
        pose = root / "pose.json"
        coverage = root / "coverage.json"
        raw_sha = "a" * 64
        records: list[dict[str, object]] = [
            {
                "event": "trace_start",
                "schema": replay.TRACE_SCHEMA,
                "trace_rate_hz": 50,
                "trace_grid_stride_client_fixed_ticks": 10,
                "source_raw_sha256": raw_sha,
                "server_acceptance_available": False,
            }
        ]
        pattern = ["neutral"]
        for label in replay.HELD_LABELS:
            pattern.extend((label, "neutral"))
        trace_index = 0
        pose_segments = []
        for label in pattern:
            count = 2 if label == "neutral" else 3
            start_tick = 100 + trace_index * 10
            for _ in range(count):
                angle = 0.01 * trace_index
                records.append(
                    {
                        "event": "trace_sample",
                        "trace_index": trace_index,
                        "client_fixed_tick": 100 + trace_index * 10,
                        "time_from_trace_start_seconds": trace_index / 50.0,
                        "held_condition": label,
                        "fighter_0_root": {
                            "world_position_xyz": [
                                0.01 * trace_index,
                                0.8,
                                0.02 * trace_index,
                            ],
                            "world_rotation_xyzw": [
                                0.0,
                                math.sin(angle / 2.0),
                                0.0,
                                math.cos(angle / 2.0),
                            ],
                        },
                        "fighter_1_root": {
                            "world_position_xyz": [0.0, 0.8, 0.0],
                            "world_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                )
                trace_index += 1
            if label != "neutral":
                ordinal = replay.HELD_LABELS.index(label)
                pose_segments.append(
                    {
                        "ordinal": ordinal,
                        "label": label,
                        "recorder_tick_start": start_tick,
                        "recorder_tick_stop": 100 + trace_index * 10,
                        "schedule_tick_start_inclusive": ordinal * 10,
                        "schedule_tick_stop_exclusive": ordinal * 10 + count,
                    }
                )
        trace.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        trace_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
        _write_json(
            pose,
            {
                "schema": replay.POSE_SCHEMA,
                "scope": {"local_fighter_index": 0, "exact_g1_vs_g1": True},
                "held_input_trajectories": pose_segments,
                "analysis_gate": {"suitable_for_held_root_trajectory_measurement": True},
                "sources": {
                    "recorder": {"sha256": raw_sha},
                    "transcript": {
                        "sha256": "b" * 64,
                        "schedule_sha256": "c" * 64,
                        "schedule_id": "test",
                        "schedule_run_id": "run",
                    },
                },
            },
        )
        _write_json(
            coverage,
            {
                "schema": replay.COVERAGE_SCHEMA,
                "trace_artifact": {"schema": replay.TRACE_SCHEMA, "sha256": trace_sha},
                "scope": {"local_fighter_index": 0, "exact_g1_vs_g1": True},
                "request_authority": {"server_acceptance_available": False},
            },
        )
        return trace, pose, coverage

    def test_extracts_exact_ordered_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace, pose, coverage = self._fixture(Path(directory))
            contract = replay.extract_real_contract(trace, pose, coverage)
        self.assertEqual(contract["schema"], replay.REAL_CONTRACT_SCHEMA)
        self.assertEqual(len(contract["real_segments"]), 14)
        self.assertEqual(
            [segment["label"] for segment in contract["real_segments"]],
            list(replay.HELD_LABELS),
        )
        categories = replay._expand_categories(contract["action_contract"])
        self.assertEqual(set(categories), set(range(1, 16)))
        self.assertFalse(contract["gates"]["parity_acceptance_evaluable"])
        self.assertFalse(contract["rek_parity_claim"])

    def test_rejects_coverage_for_another_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace, pose, coverage = self._fixture(Path(directory))
            value = json.loads(coverage.read_text(encoding="utf-8"))
            value["trace_artifact"]["sha256"] = "0" * 64
            _write_json(coverage, value)
            with self.assertRaisesRegex(replay.ReplayFailure, "coverage trace hash"):
                replay.extract_real_contract(trace, pose, coverage)

    def test_coordinate_mapping_matches_unity_and_mujoco_conventions(self) -> None:
        real = [
            {"tick": 0, "world_position_xyz_m": [0.0, 1.0, 0.0], "world_yaw_radians": 0.0},
            {"tick": 1, "world_position_xyz_m": [0.0, 1.1, 1.0], "world_yaw_radians": -0.2},
        ]
        candidate = [
            {"tick": 0, "world_position_xyz_m": [0.0, 0.0, 0.8], "world_yaw_radians": 0.0},
            {"tick": 1, "world_position_xyz_m": [1.0, 0.0, 0.9], "world_yaw_radians": 0.2},
        ]
        real_frame = replay._segment_frame(real, "real")
        candidate_frame = replay._segment_frame(candidate, "candidate")
        for real_point, candidate_point in zip(real_frame, candidate_frame):
            self.assertAlmostEqual(real_point["right_m"], candidate_point["right_m"])
            self.assertAlmostEqual(real_point["forward_m"], candidate_point["forward_m"])
            self.assertAlmostEqual(real_point["height_delta_m"], candidate_point["height_delta_m"])
            self.assertAlmostEqual(
                real_point["yaw_delta_radians"], candidate_point["yaw_delta_radians"]
            )

    def test_identical_frame_trajectory_has_zero_diagnostic_error(self) -> None:
        real_segment = {
            "ordinal": 0,
            "label": "W",
            "category": 2,
            "tick_start_inclusive": 3,
            "tick_stop_exclusive": 5,
            "nominal_schedule": {},
            "samples": [
                {"tick": 3, "world_position_xyz_m": [0.0, 1.0, 0.0], "world_yaw_radians": 0.0},
                {"tick": 4, "world_position_xyz_m": [0.0, 1.1, 1.0], "world_yaw_radians": -0.2},
            ],
        }
        candidate = {
            3: {"tick": 3, "world_position_xyz_m": [0.0, 0.0, 0.8], "world_yaw_radians": 0.0},
            4: {"tick": 4, "world_position_xyz_m": [1.0, 0.0, 0.9], "world_yaw_radians": 0.2},
        }
        report = replay.compare_segments([real_segment], candidate)[0]
        self.assertAlmostEqual(report["diagnostics"]["planar_max_m"], 0.0)
        self.assertAlmostEqual(report["diagnostics"]["height_delta_max_m"], 0.0)
        self.assertAlmostEqual(report["diagnostics"]["yaw_max_radians"], 0.0)


if __name__ == "__main__":
    unittest.main()
