#!/usr/bin/env python3
"""Build a source-pinned T800 high-level strategy contract.

This tool describes the command boundary above T800 balance and motion
execution.  It deliberately does not synthesize move trajectories, server
dynamics, rewards, or action acceptance rules.  The output is therefore an
interface contract and readiness report, not a simulator-parity claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "rek.t800_strategy_contract.v2"
BUILD_FINGERPRINT = (
    "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659"
)

SOURCE_HASHES = {
    "asset_probe": "b132eb19cb7b223a87ee3885c16e521e82d7e99006c09ed63e8cc899ad057686",
    "enum_evidence": "21b07b64f4b4360bf33224dd08d8817ae8abda3367c56d5cf5cbfb654e739b89",
    "KeyboardControlScheme.txt": (
        "16e03df273b8e55d3b805653038074c03dd4aae3e8c166e8bbcb92f653f8c553"
    ),
    "AIOpponentController.txt": (
        "330956e4c257623fb78b0d619783c94b9a1f476bae4cf4e7c554491eb69c73e2"
    ),
    "RobotInputController.txt": (
        "f248df08449e3ff0706ce15ea07e4d58517f2fc9ed3f3143473fa48c4323bc21"
    ),
    "RobotInputControllerCommand.txt": (
        "21788b81903cc2659e7245ad064b06f9ff7ad0a49a9201310d4d6159e623cdea"
    ),
    "EngineAIPolicyRunner.txt": (
        "7fb4fe854362ead4b4ef04a06094180888906c4bee49a0e6149c0228206ccf31"
    ),
}

ROBOT_CONFIG = {
    "container": "sharedassets0.assets",
    "path_id": 2739,
    "serialized_sha256": (
        "4cfc389375f3a80e778eb4ac73b7ea52ef58da0b12764a0848576894a532a405"
    ),
}

ENGINE_RUNNER = {
    "container": "sharedassets0.assets",
    "path_id": 3356,
    "serialized_sha256": (
        "5919088fb298d1dddbc563495240025307255aad8715d6af8f7df8fe3d139e66"
    ),
}

PHYSICS_STEP_POLICIES = (
    ("level1", 2883, "c8c7cfcd05061f50372ab65192f3b93c75c84c9dc881fd1f36d1ca30ceda2966"),
    ("level2", 2886, "ba9fe181951b5999044925dc5e736d977883355999fdc96bc4fb6b2300d87f01"),
    ("level3", 2883, "d5568acd9e5f3ed0a0ffac9afcb53d0ec6ea3f320e94ec743e8cd4cf4ae0e80f"),
)

KEYBOARD_SCHEMES = {
    2741: {
        "name": "T800_EngineAiKeys",
        "serialized_sha256": (
            "4ce6c5b7528f24ae9f8df08b447d0531c0fde9d28b5b266f0c650e7221d8f711"
        ),
    },
    2743: {
        "name": "T800_REKKeys",
        "serialized_sha256": (
            "1ad2e6337dc31f8b590ca65fefec0605737e8240555e132d4662f90862422b29"
        ),
    },
}

MOVE_OBJECTS = {
    2727: ("debut", "eaf4c7742a938d7b544eeb66171c01cf8d4517deb980c58c4ceacd24878c7dec"),
    2728: ("fangsong3", "8a7d70f84be23f490fe517e508f513f23e091e0bbae539994571c7a62e39f2e2"),
    2729: ("fangsong4", "be802e9ae8c3bb80c2f74a0f8d60edcd81d6bbc45b999fd9eb6ec7e0058a1b12"),
    2730: ("front_kick_L", "cd5b286f6e4f5c3003cb0f5c9de5e5690ca92ed58e5a1b789f4394e4d7911ee8"),
    2731: ("idle_moving", "70a4b6874724c23713e62d8ac6dab226a001b68e9536ca397b4c07adc675c712"),
    2732: ("left_light_attack", "32081b731a59b7553d94022ebff865764b34c83dbf274aadf26540fb17daad2e"),
    2733: ("left_side_front_kick_L", "985766c2bdfbac70466950364b0242780479aa0ef5e68f4ec1cc563682da3b9d"),
    2734: ("right_light_attack", "b1c1b2c000dd612e3eb4c33c5d90e03c2c9306e5cc194747c14248b0d77b7dea"),
    2735: ("right_shoryuken_lm", "cc298f53d04ffd56be57ce3049559d3d30c7724fe4d2839a66ea8f3008ca8deb"),
    2736: ("skill", "233f952edecb7bf8d1959c6549c0edb95e1833451fff988f57a4b14d92b14dd4"),
    2737: ("switch_to_stance_idle", "470f6a7dcbee176aa71b55e71327743a4e92aa9786737611ca04ab59237c2f6f"),
    2738: ("youbiantui", "70f36a2c7b9b53c10e47cc613d87a770eb86fb2e683ed64ee39efcccf2e75636"),
}

MOVE_SLOT_PATHS = (0, 0, 2736, 2738, 2732, 2734, 0, 0, 0, 2735, 2730, 0)
POLICY_MOVE_SLOTS = (2, 3, 4, 5, 9, 10)

SPECIAL_COMMANDS = {
    "None": 0,
    "Straighten": 1,
    "GetUpProne": 2,
    "GetUpSupine": 3,
    "Dampen": 4,
}

KEY_NAMES = {
    0: "None", 1: "Space", 6: "Semicolon", 15: "A", 16: "B",
    18: "D", 19: "E", 22: "H", 23: "I", 24: "J", 25: "K",
    26: "L", 29: "O", 30: "P", 31: "Q", 33: "S", 35: "U",
    37: "W", 39: "Y", 41: "Digit1", 42: "Digit2",
    51: "LeftShift", 60: "Escape", 61: "LeftArrow",
    62: "RightArrow", 63: "UpArrow", 64: "DownArrow",
    65: "Backspace",
}

AI_TUNING_FIELDS = (
    "engageYawSpeed", "engageForwardSpeed", "engageStopDistance",
    "facingThreshold", "maxEngageTime", "repositionChance",
    "repositionStrafeSpeed", "repositionBackSpeed", "minRepositionTime",
    "maxRepositionTime", "minFootworkTime", "settleTime", "recoveryTime",
    "initialDelay", "maxPunchDuration", "downedOpponentSpace",
    "downedBackOffSpeed", "getUpGraceTime", "kickChance",
    "difficultyAggressionPerLevel", "difficultyKickPerLevel",
    "difficultyRepositionPerLevel", "faultEStopDelay",
)


class ContractError(ValueError):
    pass


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path} does not contain a JSON object")
    return value


def require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(f"{label}: expected {expected!r}, got {actual!r}")


def exact_target(
    probe: dict[str, Any], class_name: str, container: str, path_id: int
) -> dict[str, Any]:
    matches = [
        item for item in probe.get("targets", [])
        if item.get("class") == class_name
        and item.get("container") == container
        and item.get("path_id") == path_id
    ]
    if len(matches) != 1:
        raise ContractError(
            f"expected one {class_name} at {container}:{path_id}, got {len(matches)}"
        )
    target = matches[0]
    if target.get("parsed") is not True or not isinstance(target.get("values"), dict):
        raise ContractError(f"unparsed {class_name} at {container}:{path_id}")
    return target


def pointer_path(value: Any, label: str) -> int:
    if not isinstance(value, dict):
        raise ContractError(f"{label} is not a pointer")
    require(value.get("m_FileID"), 0, f"{label} file ID")
    path_id = value.get("m_PathID")
    if not isinstance(path_id, int):
        raise ContractError(f"{label} path ID is not an integer")
    return path_id


def key_record(value: int) -> dict[str, Any]:
    if value not in KEY_NAMES:
        raise ContractError(f"unmapped Unity InputSystem Key value {value}")
    return {"value": value, "name": KEY_NAMES[value]}


def binding_record(
    binding: dict[str, Any], moves: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    path_id = pointer_path(binding.get("move"), "keyboard move")
    if path_id not in moves:
        raise ContractError(f"keyboard binding references unknown move {path_id}")
    keys = [key_record(int(binding[name])) for name in ("key1", "key2", "key3")]
    return {
        "move_path_id": path_id,
        "move_name": moves[path_id]["name"],
        "keys": keys,
        "double_tap": bool(binding.get("doubleTap")),
    }


def special_binding(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        raise ContractError("special binding is not an object")
    return [key_record(int(value[name])) for name in ("key1", "key2", "key3")]


def validate_source_paths(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    records = {}
    for source_id, path in paths.items():
        expected = SOURCE_HASHES[source_id]
        actual = sha256_path(path)
        require(actual, expected, f"{source_id} SHA-256")
        records[source_id] = {"path": str(path.resolve()), "sha256": actual}
    return records


def build_contract(
    probe: dict[str, Any], enum_evidence: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    require(probe.get("schema"), "rek.mujoco_asset_probe.v1", "asset probe schema")
    require(probe.get("build_fingerprint"), BUILD_FINGERPRINT, "build fingerprint")
    require(enum_evidence.get("schema"),
            "rek.high_level_enum_and_locomotion_constants.v1",
            "enum evidence schema")
    require(enum_evidence.get("host"), "D21", "enum evidence host")

    enum_specials = {
        item["name"]: item["value"] for item in enum_evidence.get("special_command", [])
    }
    require(enum_specials, SPECIAL_COMMANDS, "SpecialCommand constants")
    enum_keys = {item["value"]: item["name"] for item in enum_evidence.get("input_keys", [])}
    require(enum_keys, KEY_NAMES, "InputSystem Key constants")
    constants = enum_evidence.get("read_locomotion_native_constants")
    require([item.get("float32") for item in constants or []], [1.0, -1.0],
            "ReadLocomotion constants")

    config = exact_target(probe, "RobotConfig", **{
        "container": ROBOT_CONFIG["container"], "path_id": ROBOT_CONFIG["path_id"]
    })
    require(config.get("serialized_sha256"), ROBOT_CONFIG["serialized_sha256"],
            "T800 RobotConfig hash")
    config_values = config["values"]
    require(config_values.get("m_Name"), "RobotConfig_T800_EngineAiFighting",
            "T800 RobotConfig name")
    require(config_values.get("robotId"), "t800", "T800 robot ID")
    require(config_values.get("allowMoveInterrupt"), 0, "T800 move interruption")
    move_paths = tuple(
        pointer_path(value, f"RobotConfig move slot {index}")
        for index, value in enumerate(config_values.get("moves", []))
    )
    require(move_paths, MOVE_SLOT_PATHS, "T800 move slot map")
    require(
        [pointer_path(value, "keyboardControlSchemes")
         for value in config_values.get("keyboardControlSchemes", [])],
        [2743, 2741], "T800 keyboard scheme order")

    runner = exact_target(probe, "EngineAIPolicyRunner", **{
        "container": ENGINE_RUNNER["container"], "path_id": ENGINE_RUNNER["path_id"]
    })
    require(runner.get("serialized_sha256"), ENGINE_RUNNER["serialized_sha256"],
            "T800 EngineAIPolicyRunner hash")
    require(runner["values"].get("manualSwitchCooldown"), 0.5,
            "T800 manual switch cooldown")
    runner_profiles = runner["values"].get("profiles")
    if not isinstance(runner_profiles, list) or not runner_profiles:
        raise ContractError("T800 runner profiles are missing")
    for index, profile in enumerate(runner_profiles):
        for pointer_name in ("onnxBytes", "configJson", "trajectoryCsv"):
            require(pointer_path(profile.get(pointer_name),
                                 f"runner profile {index} {pointer_name}"), 0,
                    f"runner profile {index} {pointer_name} payload")

    physics_sources = []
    for container, path_id, digest in PHYSICS_STEP_POLICIES:
        target = exact_target(probe, "PhysXStepPolicy", container, path_id)
        require(target.get("serialized_sha256"), digest,
                f"PhysXStepPolicy {container}:{path_id} hash")
        require(target["values"].get("manualStepRateHz"), 50.0,
                f"PhysXStepPolicy {container}:{path_id} rate")
        physics_sources.append({
            "container": container,
            "path_id": path_id,
            "serialized_sha256": digest,
            "manual_step_rate_hz": 50.0,
        })

    moves: dict[int, dict[str, Any]] = {}
    for path_id, (name, expected_hash) in MOVE_OBJECTS.items():
        target = exact_target(probe, "MocapClipConfig", "sharedassets0.assets", path_id)
        require(target.get("serialized_sha256"), expected_hash,
                f"MocapClipConfig {path_id} hash")
        values = target["values"]
        require(values.get("m_Name"), name, f"MocapClipConfig {path_id} name")
        require(pointer_path(values.get("npzFile"), f"move {path_id} npzFile"), 0,
                f"move {path_id} embedded trajectory")
        moves[path_id] = {
            "path_id": path_id,
            "serialized_sha256": expected_hash,
            "name": name,
            "display_name": values.get("displayName"),
            "policy_profile": values.get("policyProfile"),
            "playback_speed": values.get("playbackSpeed"),
            "blend_in_s": values.get("blendInTime"),
            "blend_out_s": values.get("blendOutTime"),
            "impact_forgiveness_s": values.get("impactForgivenessDuration"),
            "impact_events": values.get("impactEvents"),
            "impact_reversal": values.get("impactReversal"),
            "trajectory_payload": {
                "state": "absent_in_client_serialized_object",
                "pointer": {"file_id": 0, "path_id": 0},
            },
        }

    keyboard = []
    for path_id, expected in KEYBOARD_SCHEMES.items():
        target = exact_target(
            probe, "KeyboardControlScheme", "sharedassets0.assets", path_id
        )
        require(target.get("serialized_sha256"), expected["serialized_sha256"],
                f"keyboard scheme {path_id} hash")
        values = target["values"]
        require(values.get("m_Name"), expected["name"], f"keyboard scheme {path_id} name")
        keyboard.append({
            "path_id": path_id,
            "name": expected["name"],
            "serialized_sha256": expected["serialized_sha256"],
            "locomotion": {
                "forward": key_record(int(values["forwardKey"])),
                "backward": key_record(int(values["backwardKey"])),
                "strafe_left": key_record(int(values["strafeLeftKey"])),
                "strafe_right": key_record(int(values["strafeRightKey"])),
                "yaw_left": key_record(int(values["yawLeftKey"])),
                "yaw_right": key_record(int(values["yawRightKey"])),
            },
            "move_bindings": [binding_record(item, moves) for item in values["bindings"]],
            "framework_bindings": {
                "straighten": special_binding(values["straighten"]),
                "get_up_prone": special_binding(values["getUpProne"]),
                "get_up_supine": special_binding(values["getUpSupine"]),
                "dampen_or_passive": special_binding(values["passiveMode"]),
                "emergency_stop": [
                    key_record(int(values["estopKey1"])),
                    key_record(int(values["estopKey2"])),
                ],
            },
        })

    ai_targets = [item for item in probe.get("targets", [])
                  if item.get("class") == "AIOpponentController"]
    require(len(ai_targets), 6, "AIOpponentController count")
    tuning = {field: ai_targets[0]["values"][field] for field in AI_TUNING_FIELDS}
    ai_sources = []
    for item in sorted(ai_targets, key=lambda row: (row["container"], row["path_id"])):
        require(item.get("parsed"), True, "AIOpponentController parsed state")
        require({field: item["values"][field] for field in AI_TUNING_FIELDS}, tuning,
                f"AIOpponentController tuning at {item['container']}:{item['path_id']}")
        ai_sources.append({
            "container": item["container"],
            "path_id": item["path_id"],
            "owner": item.get("owner"),
            "serialized_sha256": item.get("serialized_sha256"),
        })

    move_categories = [{
        "category": 0,
        "meaning": "no_move_request",
        "move_slot": None,
        "note": "a previously accepted canned move may still be executing",
    }]
    for category, slot in enumerate(POLICY_MOVE_SLOTS, start=1):
        path_id = move_paths[slot]
        move_categories.append({
            "category": category,
            "meaning": "execute_move_by_index",
            "move_slot": slot,
            "move_path_id": path_id,
            "move_name": moves[path_id]["name"],
            "display_name": moves[path_id]["display_name"],
        })

    hard_unknowns = [
        {
            "id": "t800_canned_trajectory_payloads",
            "state": "unknown",
            "obtain_by": "recover accepted move-aligned T800 trajectories or the exact server policy payloads",
        },
        {
            "id": "authoritative_move_acceptance_timing_and_execution",
            "state": "unknown",
            "obtain_by": "align requests and measured local rejection gates with authoritative accepted/executed transitions",
        },
        {
            "id": "server_transition_and_contact_dynamics",
            "state": "unknown",
            "obtain_by": "validate the local MuJoCo plant against repeated authoritative trajectories",
        },
        {
            "id": "strategy_decision_cadence",
            "state": "unknown",
            "obtain_by": "measure accepted command boundaries against authoritative simulation ticks",
        },
        {
            "id": "combat_reward_and_terminal_semantics",
            "state": "unknown",
            "obtain_by": "recover and validate clean-hit, fall, knockout, round, and match transitions",
        },
    ]

    return {
        "schema": SCHEMA,
        "build_fingerprint": BUILD_FINGERPRINT,
        "scope": {
            "layer": "T800 high-level strategy",
            "static_interface_measured": True,
            "simulator_ready": False,
            "training_ready": False,
            "parity_claim": False,
        },
        "sources": source_records,
        "measured_interface": {
            "velocity_command": {
                "type": "UnityEngine.Vector3",
                "components": [
                    {"index": 0, "name": "forward", "keyboard_positive": "W", "keyboard_negative": "S"},
                    {"index": 1, "name": "strafe", "keyboard_positive": "A", "keyboard_negative": "D"},
                    {"index": 2, "name": "yaw", "keyboard_positive": "Q", "keyboard_negative": "E"},
                ],
                "keyboard_values": [-1.0, 0.0, 1.0],
                "robot_config": {
                    field: config_values[field] for field in (
                        "deadZone", "forwardSpeed", "strafeSpeed", "yawSpeed",
                        "keyboardYawRampTime", "stopBrakeRate",
                        "locomotionTransitionSettle",
                        "transitionSettlePlanarSpeed", "transitionSettleYawRate",
                    )
                },
            },
            "move_slots": [
                {
                    "slot": index,
                    "path_id": path_id or None,
                    "move": moves.get(path_id),
                }
                for index, path_id in enumerate(move_paths)
            ],
            "special_commands": [
                {"name": name, "value": value}
                for name, value in SPECIAL_COMMANDS.items()
            ],
            "keyboard_schemes": keyboard,
            "baseline_ai": {
                "serialized_tuning": tuning,
                "source_objects": ai_sources,
                "recovery_control": {
                    "state": "static_native_semantics",
                    "owner": "AIOpponentController.DriveRecovery",
                    "guarded_transitions": [
                        "upright clears recovery state",
                        "fallen and not dampened requests Dampen",
                        "dampened and not ready requests Straighten",
                        "recovery armed requests orientation-selected GetUpProne or GetUpSupine",
                    ],
                    "citation": "AIOpponentController.txt",
                },
            },
            "execution_gate": {
                "allow_move_interrupt": False,
                "manual_switch_cooldown_s": runner["values"]["manualSwitchCooldown"],
                "pending_buffer": {
                    "move": "one integer register plus pending boolean",
                    "special": "one integer register plus pending boolean",
                    "fifo": False,
                    "same_type_request_before_send": "overwrites prior pending value",
                },
                "client_send_order": ["velocity", "move", "special", "emergency_stop"],
                "move_request_rejected_when": [
                    "recovering",
                    "punching_or_move_in_progress",
                    "manual_switch_cooldown_active",
                ],
                "engine_move_in_progress_predicate": [
                    "runner initialized and not paused",
                    "current profile config and trajectory are non-null",
                    "profile replay is false",
                    "trajectory index is before the final trajectory sample",
                ],
                "citation": [
                    "RobotInputController.txt",
                    "RobotInputControllerCommand.txt",
                    "EngineAIPolicyRunner.txt",
                ],
            },
            "timing": {
                "arena_manual_physics_step_rate_hz": 50.0,
                "physics_step_sources": physics_sources,
                "authoritative_strategy_decision_cadence": {"state": "unknown"},
            },
        },
        "strategy_design": {
            "learned_action": {
                "velocity": {
                    "shape": [3],
                    "component_order": ["forward", "strafe", "yaw"],
                    "training_range": [-1.0, 1.0],
                    "range_role": "bounded design using measured keyboard endpoints",
                },
                "attack": {
                    "type": "categorical",
                    "semantics": "one-shot request; category 0 sends no request",
                    "categories": move_categories,
                },
            },
            "action_filter": {
                "hold_velocity_between_strategy_decisions": True,
                "deduplicate_nonzero_attack_categories": True,
                "mask_nonzero_attack_during_measured_rejection_states": True,
                "never_call_null_slot_for_no_move_request": True,
            },
            "deterministic_wrapper_invariant": {
                "preserve": "AIOpponentController.DriveRecovery guarded state machine",
                "priority": "recovery gates learned locomotion and move requests",
                "reason": (
                    "automatic get-up is orchestrated at the high-level controller "
                    "layer and would disappear if a learned policy replaced that "
                    "controller without retaining DriveRecovery"
                ),
            },
            "framework_owned_not_learned": [
                "joint-level balance and stability",
                "joint target and torque generation",
                "move trajectory execution",
                "guarded dampen, straighten, and oriented get-up recovery wrapper",
                "emergency stop",
            ],
        },
        "hard_unknowns": hard_unknowns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-probe", type=Path, required=True)
    parser.add_argument("--enum-evidence", type=Path, required=True)
    parser.add_argument("--keyboard-isil", type=Path, required=True)
    parser.add_argument("--ai-isil", type=Path, required=True)
    parser.add_argument("--robot-input-isil", type=Path, required=True)
    parser.add_argument("--robot-input-command-isil", type=Path, required=True)
    parser.add_argument("--engine-runner-isil", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_paths = {
        "asset_probe": args.asset_probe,
        "enum_evidence": args.enum_evidence,
        "KeyboardControlScheme.txt": args.keyboard_isil,
        "AIOpponentController.txt": args.ai_isil,
        "RobotInputController.txt": args.robot_input_isil,
        "RobotInputControllerCommand.txt": args.robot_input_command_isil,
        "EngineAIPolicyRunner.txt": args.engine_runner_isil,
    }
    records = validate_source_paths(source_paths)
    contract = build_contract(
        load_json(args.asset_probe), load_json(args.enum_evidence), records
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"strategy_contract: static interface measured")
    print(f"simulator_ready: {str(contract['scope']['simulator_ready']).lower()}")
    print(f"training_ready: {str(contract['scope']['training_ready']).lower()}")
    print(f"move_categories: {len(contract['strategy_design']['learned_action']['attack']['categories'])}")
    print(f"output: {args.out.resolve()}")
    print(f"sha256: {sha256_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
