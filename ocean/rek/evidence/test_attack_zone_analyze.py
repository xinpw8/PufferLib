#!/usr/bin/env python3

from __future__ import annotations

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attack_zone_analyze as az


RUNNER_HASH = "1" * 64
CONTROLLER_HASH = "2" * 64
SCHEDULE_HASH = "3" * 64
ASSET_HASH = "4" * 64
SESSION_HASH = "5" * 64
ROUND_HASH = "6" * 64


DISTANCE_BIN = {
    "id": "d01",
    "lower_m": 0.35,
    "upper_m": 0.45,
    "lower_inclusive": True,
    "upper_inclusive": False,
    "center_m": 0.40,
}
BEARING_BIN = {
    "id": "b05",
    "lower_deg": -5.0,
    "upper_deg": 5.0,
    "lower_inclusive": True,
    "upper_inclusive": True,
    "center_deg": 0.0,
}


def make_clock(tick: int) -> dict:
    return {
        "stopwatch_timestamp_ticks": tick * 20,
        "stopwatch_frequency_hz": 1000,
        "utc": f"2026-09-03T00:00:{tick % 60:02d}.0000000Z",
        "unity_frame": tick,
        "unity_time": tick * 0.02,
        "unity_fixed_time": tick * 0.02,
        "client_control_tick": tick,
        "client_fixed_substep": tick * 10,
    }


def identity_state(move_name: str = "left_light_attack", phase: str = "idle") -> dict:
    if phase == "start":
        local_motion = {
            "action_playing": True,
            "busy": True,
            "active_action_clip": move_name,
            "current_move_id": 4,
            "action_clip_frame": 1.0,
            "action_clip_fps": 60.0,
        }
    else:
        local_motion = {
            "action_playing": False,
            "busy": False,
            "active_action_clip": None,
            "current_move_id": 4,
            "action_clip_frame": 0.0,
            "action_clip_fps": 60.0,
        }
    return {
        "local_identity": {
            "semantic_robot_id": "t800",
            "runtime_object_name": "T800",
            "runtime_bone_count": 26,
            "runtime_bone_signature_sha256": az.T800_BONE_SIGNATURE_SHA256,
            "exact_local_t800_proven": True,
        },
        "opponent_identity": {
            "semantic_robot_id_untrusted_for_runtime_acceptance": "stale-id",
            "runtime_object_name": "T800",
            "runtime_bone_count": 26,
            "runtime_bone_signature_sha256": az.T800_BONE_SIGNATURE_SHA256,
            "runtime_identity_sha256": "9" * 64,
            "semantic_runtime_mismatch": True,
            "semantic_runtime_consistency": "opponent_semantic_non_t800_runtime_t800_exact",
            "semantic_robot_id_used_for_acceptance": False,
        },
        "geometry": {
            "planar_distance_m": 0.4,
            "local_bearing_to_opponent_deg": 0.0,
            "opponent_bearing_to_local_deg": 0.0,
        },
        "local_motion": local_motion,
        "input_state": {
            "punching": phase == "start",
            "recovering": False,
            "pending_move": False,
            "pending_special": False,
            "pending_estop": False,
        },
    }


def make_event(name: str, tick: int, detail: dict | None = None, state: dict | None = None) -> dict:
    clock = make_clock(tick)
    return {
        "event": name,
        "protocol": "rek.ui_bridge.v1",
        "attack_zone_schema": az.RUNNER_SCHEMA,
        "attack_zone_protocol_sha256": RUNNER_HASH,
        "continuous_controller_sha256": CONTROLLER_HASH,
        "authority_scope": "client_request_edges_and_local_observations_only",
        "authority_caveat": "server state unknown",
        "isolated_spark_proof": az.REQUIRED_ISOLATION_PROOF,
        "schedule_schema": az.SCHEDULE_SCHEMA,
        "schedule_sha256": SCHEDULE_HASH,
        "randomization_seed_hex": "a" * 64,
        "schedule_ordinal": 0,
        "independent_run_id": "run-1",
        "independent_run_ordinal": 0,
        "session_identity_sha256": SESSION_HASH,
        "round_identity_sha256": ROUND_HASH,
        "trial_id": "trial-1",
        "action_sequence": 1,
        "move_index": 4,
        "serialized_asset_sha256": ASSET_HASH,
        "requested_distance_bin": copy.deepcopy(DISTANCE_BIN),
        "requested_bearing_bin": copy.deepcopy(BEARING_BIN),
        "controller_phase": "fixture",
        "controller_reason": name,
        "measured_state": identity_state() if state is None else state,
        "detail": {} if detail is None else detail,
        "clocks": {},
        **clock,
        "fixed_substeps_per_control_tick": 10,
        "global_input_used": False,
        "client_request_observation_only": True,
        "server_acceptance_observed": False,
        "authoritative_execution_observed": False,
    }


def make_settle_samples(count: int = 15) -> list[dict]:
    return [{
        "clock": make_clock(index + 1),
        "distance_m": 0.4,
        "bearing_deg": 0.0,
        "local_planar_speed_m_s": 0.0,
        "local_yaw_rate_rad_s": 0.0,
        "opponent_planar_speed_m_s": 0.0,
        "opponent_yaw_rate_rad_s": 0.0,
        "opponent_motion_stratum": "stationary",
        "opponent_facing_stratum": "opponent_face_on",
        "neutral_request_method_returned": True,
        "velocity_command_exact_neutral": True,
        "local_action_ready": True,
        "no_pending_requests": True,
        "local_healthy": True,
        "opponent_healthy": True,
    } for index in range(count)]


def lifecycle_events(censored: bool = False) -> list[dict]:
    events = [
        make_event("local_command_edge_set", 20),
        make_event("client_request_method_returned", 21, {
            "send_method": "RobotInputController.SendMoveEvent",
            "send_method_returned": True,
        }),
        make_event("local_motion_start_observed", 22, state=identity_state(phase="start")),
    ]
    if censored:
        events.append(make_event("trial_censored", 30))
    else:
        events.extend([
            make_event("local_motion_completion_and_readiness_observed", 30, state=identity_state(phase="idle")),
            make_event("trial_completed", 31),
        ])
    return events


def make_bones(tie: bool = False) -> list[dict]:
    bones = []
    for index, name in enumerate(az.T800_BONE_NAMES):
        x = float(index + 1)
        if tie and index == 1:
            x = -1.0
        bones.append({"index": index, "name": name, "position_opponent_root_xyz_m": [x, 0.0, 0.0]})
    return bones


def make_protocol() -> dict:
    names = {
        2: "skill", 3: "youbiantui", 4: "left_light_attack",
        5: "right_light_attack", 9: "right_shoryuken_lm", 10: "front_kick_L",
    }
    distance_edges = [0.25, 0.35, 0.45, 0.5180000126361847, 0.60, 0.75, 0.90, 1.10, 1.50, 2.00]
    bearing_edges = [-180.0, -90.0, -60.0, -35.0, -20.0, -5.0, 5.0, 20.0, 35.0, 60.0, 90.0, 180.0]
    distance_bins = [{
        "id": f"d{index:02d}",
        "lower_m": distance_edges[index],
        "upper_m": distance_edges[index + 1],
        "lower_inclusive": index != 3,
        "upper_inclusive": index in {2, 8},
    } for index in range(9)]
    bearing_bins = [{
        "id": f"b{index:02d}",
        "lower_deg": bearing_edges[index],
        "upper_deg": bearing_edges[index + 1],
        "lower_inclusive": index <= 5,
        "upper_inclusive": index >= 5 and index != 10,
    } for index in range(11)]
    return {
        "schema": az.PROTOCOL_SCHEMA,
        "live_rek_interaction_performed": False,
        "bridge_source_modified": False,
        "claim_boundary": {
            "server_acceptance_observed": False,
            "authoritative_execution_observed": False,
            "authoritative_hit_attribution_available": False,
            "authoritative_completion_available": False,
        },
        "move_profiles": [
            {
                "move_index": move,
                "move_name": names[move],
                "serialized_asset_sha256": str(move % 10) * 64,
                "configured_impact_markers": [
                    {"ordinal": ordinal, "limb": 1, "impact_time_s": float(ordinal)}
                    for ordinal in range(1, 4 if move == 2 else 2)
                ],
            }
            for move in az.ALLOWED_MOVES
        ],
        "grid": {
            "distance_bins": distance_bins,
            "bearing_bins": bearing_bins,
            "per_move_grid_assignments": [
                {"move_index": move, "grid_id": "shared_coarse_grid_v1"}
                for move in az.ALLOWED_MOVES
            ],
        },
        "opponent_motion_confounds": {
            "motion_strata": [{"id": value} for value in az.MOTION_STRATA[:-1]],
            "opponent_facing_strata": [{"id": value} for value in (
                "opponent_face_on", "opponent_oblique", "opponent_back_turned",
            )],
        },
    }


def make_analysis_spec() -> dict:
    return {
        "schema": az.ANALYSIS_SPEC_SCHEMA,
        "input_contracts": [
            {"role": "attack_zone_runner_contract", "embedded_canonical_sha256": RUNNER_HASH},
            {"role": "continuous_lifecycle_contract", "embedded_canonical_sha256": CONTROLLER_HASH},
        ],
        "conformance_tests_required_before_real_analysis": [f"test-{index:02d}" for index in range(1, 20)],
        "final_status_rule": {"empirical_map_available": False},
    }


def make_schedule() -> dict:
    return {
        "attack_zone_trial_schema": az.RUNNER_SCHEMA,
        "protocol_sha256": RUNNER_HASH,
        "schedule_schema": az.SCHEDULE_SCHEMA,
        "randomization_algorithm": "sha256_counter_fisher_yates_rejection_v1",
        "randomization_seed_hex": "a" * 64,
        "entries": [{
            "schedule_ordinal": 0,
            "repetition_within_run": 0,
            "move_index": 4,
            "serialized_asset_sha256": ASSET_HASH,
            "distance_bin": copy.deepcopy(DISTANCE_BIN),
            "bearing_bin": copy.deepcopy(BEARING_BIN),
        }],
    }


def acquisition_event(tick: int) -> dict:
    evaluation = {
        "AcquisitionPass": True,
        "ClockValid": True,
        "RootsFinite": True,
        "GeometryValid": True,
        "AnimationValid": True,
        "NeutralRequestMethodReturned": True,
        "VelocityCommandExactNeutral": True,
        "LocalActionReady": True,
        "NoPendingRequests": True,
        "LocalHealthy": True,
        "OpponentHealthy": True,
        "DistanceCentralPass": True,
        "BearingInBinPass": True,
        "BearingErrorPass": True,
        "LocalMotionPass": True,
        "OpponentStationary": True,
        "BearingErrorDegrees": 0.0,
        "LocalPlanarSpeedMetersPerSecond": 0.0,
        "LocalYawRateRadiansPerSecond": 0.0,
        "Geometry": {
            "IsValid": True,
            "DistanceMeters": 0.4,
            "LocalBearingToOpponentDegrees": 0.0,
            "OpponentBearingToLocalDegrees": 0.0,
        },
        "Motion": {
            "MotionStratum": "stationary",
            "FacingStratum": "opponent_face_on",
            "Stationary": True,
            "OpponentPlanarSpeedMetersPerSecond": 0.0,
            "OpponentYawRateRadiansPerSecond": 0.0,
            "RadialClosingSpeedMetersPerSecond": 0.0,
            "TangentialSpeedMetersPerSecond": 0.0,
        },
    }
    return make_event("acquisition_sample", tick, {
        "target": {},
        "acquisition_decision": {"exact_neutral": True},
        "settle": {
            "Acquired": tick == 15,
            "ConsecutiveTicks": tick,
            "StreakReset": False,
            "Reason": "settle_progress",
            "digest": None,
            "current_evaluation": evaluation,
            "current_source_clock": {},
        },
    })


def complete_trial_events() -> list[dict]:
    events = [make_event("target_requested", 0)]
    events.extend(acquisition_event(tick) for tick in range(1, 16))
    events.append(make_event("target_acquired", 16, {
        "primary_stationary_stratum": True,
        "opponent_motion_stratum": "stationary",
        "opponent_facing_stratum": "opponent_face_on",
    }))
    events.extend(lifecycle_events())
    return events


def raw_hit_event(sequence_id: int = 1, tick: int = 23) -> dict:
    bones = []
    for index, name in enumerate(az.T800_BONE_NAMES):
        bones.append({
            "index": index,
            "name": name,
            "world_position_xyz_m": [float(index + 1), 0.0, 0.0],
            "world_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        })
    return make_event("raw_rek_hit_observed", tick, {
        "raw_hit_sequence": sequence_id,
        "wire_body_bytes": 29,
        "wire_body_sha256": "b" * 64,
        "decoded": {
            "world_position_xyz_m": [0.0, 0.0, 0.0],
            "world_surface_normal_xyz": [0.0, 1.0, 0.0],
            "relative_speed": 1.0,
            "is_kick_raw_byte": 0,
        },
        "contemporaneous_opponent_root_bones_and_colliders": {
            "root_position_xyz_m": [0.0, 0.0, 0.0],
            "root_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "bones": bones,
            "colliders": [],
        },
        "raw_packet_contains_fighter_identity": False,
        "raw_packet_contains_move_identity": False,
        "server_acceptance_observed": False,
        "authoritative_execution_observed": False,
    })


def score_event(sequences: list[int], tick: int = 24, local_delta: int = 1) -> dict:
    return make_event("round_score_delta_observed", tick, {
        "local_clean_hit_delta": local_delta,
        "opponent_clean_hit_delta": 0,
        "local_fall_delta": 0,
        "opponent_fall_delta": 0,
        "isolated_selected_local_action_interval": True,
        "raw_hit_sequences": sequences,
        "raw_rek_hit_alone_used_for_attribution": False,
        "body_zone_claimed": False,
        "server_acceptance_observed": False,
        "authoritative_execution_observed": False,
    })


def minimal_ledger(
    trial_id: str = "trial-1", run_id: str = "run-1", success: bool = False,
    primary: bool = True, censored: bool = False,
) -> dict:
    return {
        "trial_id": trial_id,
        "independent_run_id": run_id,
        "move_index": 4,
        "entry_distance_bin_id": "d01",
        "entry_bearing_bin_id": "b05",
        "scheduled": True,
        "target_acquired": True,
        "settle_complete": True,
        "eligible_any_motion_stratum": True,
        "request_edge_observed": True,
        "request_method_return_observed": True,
        "local_start_observed": True,
        "local_completion_observed": not censored,
        "uncensored_timing": not censored,
        "raw_hit_observed": success,
        "local_score_observed": success,
        "isolated_local_scoring_join_passed": success,
        "fall_recovery_contaminated": False,
        "primary_analysis_eligible": primary,
        "exclusion_reasons": ["censored:fixture"] if censored else [],
        "opponent_motion_path": {
            "motion_stratum": "stationary",
            "facing_stratum": "opponent_face_on",
            "time_varying_motion": False,
            "time_varying_facing": False,
            "entries": [],
        },
    }


class AttackZoneConformanceTests(unittest.TestCase):
    def assertFailure(self, code: str, callback, *args, **kwargs):
        with self.assertRaises(az.AnalysisFailure) as captured:
            callback(*args, **kwargs)
        self.assertEqual(code, captured.exception.code)

    def test_01_reject_one_byte_input_hash_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_bytes(b"abc")
            self.assertFailure("input_sha256_mismatch", az.verify_file_hash, path, "0" * 64)

    def test_02_reject_schema_build_contract_profile_and_asset_mismatch(self):
        valid = make_event("target_requested", 0)
        mutations = [
            ("runner_event_schema_mismatch", "attack_zone_schema", "wrong"),
            ("runner_contract_hash_mismatch", "attack_zone_protocol_sha256", "f" * 64),
            ("controller_contract_hash_mismatch", "continuous_controller_sha256", "e" * 64),
            ("event_move_invalid", "move_index", 8),
            ("event_asset_sha256_invalid", "serialized_asset_sha256", "bad"),
        ]
        for code, key, value in mutations:
            with self.subTest(code=code):
                event = copy.deepcopy(valid)
                event[key] = value
                self.assertFailure(code, az.validate_event_envelope, event, RUNNER_HASH, CONTROLLER_HASH)

    def test_03_stale_opponent_semantic_identity_is_recorded_not_trusted(self):
        result = az.validate_runtime_identity(identity_state())
        self.assertTrue(result["accepted"])
        self.assertTrue(result["opponent_semantic_runtime_mismatch"])
        bad = identity_state()
        bad["opponent_identity"]["semantic_robot_id_used_for_acceptance"] = True
        self.assertFailure("opponent_semantic_id_trusted", az.validate_runtime_identity, bad)

    def test_04_missing_settle_tick_and_failed_predicate_reject(self):
        self.assertFailure(
            "settle_sample_count_mismatch", az.validate_settle_samples,
            make_settle_samples(14), DISTANCE_BIN, BEARING_BIN,
        )
        samples = make_settle_samples()
        samples[8]["local_action_ready"] = False
        self.assertFailure("settle_predicate_failed", az.validate_settle_samples, samples, DISTANCE_BIN, BEARING_BIN)

    def test_05_distance_and_bearing_edges_use_exact_inclusivity(self):
        self.assertTrue(az.bin_contains(DISTANCE_BIN, 0.35, "m"))
        self.assertFalse(az.bin_contains(DISTANCE_BIN, 0.45, "m"))
        self.assertTrue(az.bin_contains(BEARING_BIN, -5.0, "deg"))
        self.assertTrue(az.bin_contains(BEARING_BIN, 5.0, "deg"))
        right = {
            "id": "b06", "lower_deg": 5.0, "upper_deg": 20.0,
            "lower_inclusive": False, "upper_inclusive": True, "center_deg": 12.5,
        }
        self.assertFalse(az.bin_contains(right, 5.0, "deg"))
        self.assertTrue(az.bin_contains(right, 20.0, "deg"))

    def test_06_signed_bearing_and_native_negative_sign_yaw(self):
        self.assertEqual(-180.0, az.wrap_to_180(180.0))
        self.assertEqual(0.0, az.native_facing_yaw(17.5))
        self.assertAlmostEqual(-1.0, az.native_facing_yaw(30.0), places=12)
        self.assertAlmostEqual(1.0, az.native_facing_yaw(-30.0), places=12)
        self.assertEqual(-1.5, az.native_facing_yaw(90.0))

    def test_07_exact_composer_start_and_completion_predicates(self):
        result = az.validate_lifecycle(lifecycle_events(), "left_light_attack")
        self.assertFalse(result["censored"])
        bad_start = lifecycle_events()
        bad_start[2]["measured_state"]["local_motion"]["active_action_clip"] = "wrong"
        self.assertFailure("start_clip_identity_mismatch", az.validate_lifecycle, bad_start, "left_light_attack")
        bad_completion = lifecycle_events()
        bad_completion[3]["measured_state"]["local_motion"]["busy"] = True
        self.assertFailure("completion_composer_still_busy", az.validate_lifecycle, bad_completion, "left_light_attack")

    def test_08_timeout_is_right_censored_not_imputed(self):
        result = az.validate_lifecycle(lifecycle_events(censored=True), "left_light_attack")
        self.assertTrue(result["censored"])
        self.assertIsNone(result["request_to_completion_readiness_s"])
        self.assertEqual("trial_censored", result["censor_reason"])
        self.assertAlmostEqual(0.16, result["censor_durations_s"]["local_composer_start_to_completion_readiness_s"])
        timing_row = {
            "trial_id": "trial-1", "independent_run_id": "run-1", "move_index": 4,
            "entry_distance_bin_id": "d01", "entry_bearing_bin_id": "b05",
            "timing_analysis_eligible": True, "opponent_motion_stratum": "stationary",
            **result,
        }
        distributions = az._timing_distributions([timing_row], make_protocol())
        completion = next(value for value in distributions if (
            value["move_index"], value["entry_distance_bin_id"], value["entry_bearing_bin_id"], value["metric"]
        ) == (4, "d01", "b05", "local_composer_start_to_completion_readiness_s"))
        self.assertEqual("estimated_with_right_censoring", completion["kaplan_meier"]["status"])
        self.assertEqual(1, completion["kaplan_meier"]["curve"][0]["censored"])

    def test_09_fall_recovery_exclusion_and_post_start_censor(self):
        pre = az.classify_fall_recovery(request_tick=10, start_tick=None, contamination_tick=9)
        self.assertTrue(pre["whole_trial_primary_excluded"])
        self.assertFalse(pre["pre_fall_evidence_table_only"])
        post = az.classify_fall_recovery(request_tick=10, start_tick=12, contamination_tick=15)
        self.assertTrue(post["completion_right_censored"])
        self.assertTrue(post["remainder_of_round_excluded"])
        self.assertTrue(post["pre_fall_evidence_table_only"])

    def test_10_known_quaternion_opponent_root_transform(self):
        half = math.sqrt(0.5)
        result = az.opponent_root_transform([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, half, 0.0, half])
        self.assertAlmostEqual(0.0, result[0], places=12)
        self.assertAlmostEqual(0.0, result[1], places=12)
        self.assertAlmostEqual(1.0, result[2], places=12)

    def test_11_root_interpolation_records_bracket_and_rejects_missing_bracket(self):
        samples = [
            {"client_fixed_substep": 10, "position_xyz_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            {"client_fixed_substep": 20, "position_xyz_m": [2.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        ]
        result = az.interpolate_root_pose(samples, 15)
        self.assertEqual("bracketed_client_fixed_substep_interpolation", result["method"])
        self.assertEqual([10, 20], result["source_substeps"])
        self.assertEqual([1.0, 0.0, 0.0], result["position_xyz_m"])
        self.assertFailure("root_interpolation_bracket_missing", az.interpolate_root_pose, samples, 25)

    def test_12_bone_sample_over_10_ms_is_unresolved(self):
        self.assertFailure(
            "bone_sample_time_offset_exceeded",
            az.select_nearest_bone_sample,
            [{"time_s": 1.020, "bones": []}],
            1.0,
        )
        selected = az.select_nearest_bone_sample([{"time_s": 1.009, "bones": []}], 1.0)
        self.assertAlmostEqual(0.009, selected["bone_sample_time_offset_s"])

    def test_13_nearest_bone_tie_is_unresolved(self):
        self.assertFailure("nearest_bone_tie", az.nearest_bone_candidate, [0.0, 0.0, 0.0], make_bones(tie=True))

    def test_14_raw_hit_without_local_score_join_stays_unattributed(self):
        candidate = az.raw_hit_location_candidate(raw_hit_event())
        self.assertEqual("unattributed_rek_hit_location_candidate", candidate["association_status"])
        self.assertTrue(candidate["candidate_only"])
        self.assertEqual(26, len(candidate["candidate_distances_to_all_opponent_bones_m"]))
        candidate.update({
            "trial_id": "trial-1", "independent_run_id": "run-1", "move_index": 4,
            "entry_distance_bin_id": "d01", "entry_bearing_bin_id": "b05",
        })
        maps = az._impact_maps([minimal_ledger()], [candidate], make_protocol())
        failures = next(value for value in maps["cell_alignment_and_join_failure_counts"] if (
            value["move_index"], value["entry_distance_bin_id"], value["entry_bearing_bin_id"]
        ) == (4, "d01", "b05"))
        self.assertEqual(1, failures["temporal_join_failure_count"])
        self.assertEqual(0, failures["root_alignment_failure_count"])
        self.assertEqual(0, failures["bone_alignment_failure_count"])

    def test_15_local_score_with_multiple_raw_hits_stays_unresolved(self):
        raws = [raw_hit_event(1), raw_hit_event(2)]
        result = az.join_local_scoring_candidate(raws, score_event([1, 2]), lifecycle_events(), [])
        self.assertFalse(result["passed"])
        self.assertEqual("raw_hit_pairing_not_unique", result["reason"])

    def test_16_one_isolated_local_action_and_explicit_raw_hit_pair_is_candidate_only(self):
        raw = raw_hit_event(1, tick=23)
        result = az.join_local_scoring_candidate([raw], score_event([1], tick=24), lifecycle_events(), [])
        self.assertTrue(result["passed"])
        self.assertEqual("client_isolated_local_scoring_zone_candidate", result["label"])
        self.assertTrue(result["still_not_authoritative"])
        candidate = az.raw_hit_location_candidate(raw)
        candidate.update({
            "trial_id": "trial-1", "independent_run_id": "run-1", "move_index": 4,
            "entry_distance_bin_id": "d01", "entry_bearing_bin_id": "b05",
            "association_status": result["label"],
        })
        maps = az._impact_maps([minimal_ledger(success=True)], [candidate], make_protocol())
        self.assertEqual(1, len(maps["region_maps"]))
        region = maps["region_maps"][0]
        self.assertEqual(1, region["nearest_opponent_bone_distance_m"]["count"])
        self.assertEqual({"LINK_BASE": 1}, region["nearest_opponent_bone_counts"])

    def test_17_configured_marker_never_becomes_contact(self):
        marker = make_event("configured_asset_marker_projected", 25, {
            "ImpactTimeSeconds": 0.39,
            "projected_stopwatch_timestamp_ticks": 500,
            "observed_contact": False,
            "observed_hit_ownership": False,
        })
        result = az.validate_configured_marker_event(marker)
        self.assertFalse(result["contact"])
        bad = copy.deepcopy(marker)
        bad["detail"]["observed_contact"] = True
        self.assertFailure("configured_marker_claims_contact", az.validate_configured_marker_event, bad)

    def test_18_fewer_than_five_independent_runs_keeps_variance_unknown(self):
        result = az.one_way_variance_components({
            "r1": [0.1, 0.2], "r2": [0.2, 0.3], "r3": [0.3, 0.4], "r4": [0.4, 0.5],
        })
        self.assertEqual("unknown_insufficient_independent_runs", result["status"])
        self.assertIsNone(result["between_run_variance_tau2"])
        binary_unknown = az.beta_binomial_run_variance({f"r{index}": [index % 2] for index in range(4)})
        self.assertEqual("unknown_insufficient_independent_runs", binary_unknown["status"])
        binary = az.beta_binomial_run_variance({
            f"r{index}": [0, 1, index % 2] for index in range(5)
        })
        self.assertEqual("estimated_beta_binomial_method_of_moments", binary["status"])
        self.assertEqual(5, len(binary["leave_one_run_out_probability"]))
        groups = {f"r{index}": [float(index), float(index + 1)] for index in range(5)}
        first = az.run_cluster_bootstrap(groups, "median", {"fixture": "bootstrap"})
        second = az.run_cluster_bootstrap(groups, "median", {"fixture": "bootstrap"})
        self.assertEqual(first, second)
        self.assertEqual(10_000, first["replicate_count"])
        self.assertEqual("estimated_run_cluster_bootstrap", first["status"])

    def test_19_duplicate_analysis_outputs_are_byte_identical(self):
        first_result = az.analyze_event_bundle(
            make_protocol(), make_analysis_spec(), [(SCHEDULE_HASH, make_schedule())],
            complete_trial_events(), "fixture",
        )
        second_result = az.analyze_event_bundle(
            make_protocol(), make_analysis_spec(), [(SCHEDULE_HASH, make_schedule())],
            complete_trial_events(), "fixture",
        )
        self.assertEqual(first_result, second_result)
        self.assertFalse(first_result["mapping_completed"])
        self.assertEqual(1, len(first_result["trial_ledger"]))
        self.assertTrue(first_result["trial_ledger"][0]["primary_analysis_eligible"])
        self.assertEqual(594, len(first_result["per_move_cell_outcomes"]))
        self.assertEqual(
            594 * len(az.TIMING_METRICS) + 2 * 9 * 11,
            len(first_result["per_move_timing_distributions"]),
        )
        empty_cell = next(value for value in first_result["per_move_cell_outcomes"] if (
            value["move_index"], value["entry_distance_bin_id"], value["entry_bearing_bin_id"]
        ) == (2, "d00", "b00"))
        self.assertEqual(0, empty_cell["scheduled_attempts"])
        self.assertEqual("unresolved", empty_cell["label"])
        stationary = first_result["opponent_motion_strata"]["motion_strata"][0]
        self.assertEqual("stationary", stationary["opponent_motion_stratum"])
        self.assertEqual(1, stationary["flow_counts"]["scheduled"])
        self.assertIn("binary_trial_success", first_result["repeated_run_variance"])
        resolved = [{"label": "transition"} for _ in range(594)]
        self.assertTrue(az.mapping_completed_from_cells("real", resolved, 594))
        self.assertFalse(az.mapping_completed_from_cells("fixture", resolved, 594))
        ledgers = [
            minimal_ledger(f"valid-{index}", f"run-{index % 5}", success=True)
            for index in range(40)
        ]
        ledgers.append(minimal_ledger("censored-extra", "run-0", primary=False, censored=True))
        cell = next(value for value in az._cell_outcomes(ledgers, make_protocol()) if (
            value["move_index"], value["entry_distance_bin_id"], value["entry_bearing_bin_id"]
        ) == (4, "d01", "b05"))
        self.assertEqual("supported_client_temporal_zone", cell["label"])
        self.assertEqual(40, cell["primary_eligible_trials"])
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            input_audit = [{
                "role": "fixture", "path": "C:\\fixture\\events.jsonl",
                "bytes": 123, "sha256": "a" * 64,
            }]
            az.write_outputs(first, first_result, input_audit)
            az.write_outputs(second, second_result, input_audit)
            first_files = {path.name: path.read_bytes() for path in first.iterdir()}
            second_files = {path.name: path.read_bytes() for path in second.iterdir()}
            self.assertEqual(first_files, second_files)
            audit = __import__("json").loads(first_files["analysis-audit.json"])
            self.assertFalse(audit["mapping_completed"])
            manifest = __import__("json").loads(first_files["sha256-manifest.json"])
            self.assertTrue(any(value["role"] == "input" for value in manifest["files"]))
            self.assertTrue(any(value["role"] == "output" for value in manifest["files"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
