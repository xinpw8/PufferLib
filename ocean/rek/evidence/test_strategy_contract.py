import copy
import unittest

import strategy_contract


def target(class_name, path_id, sha256, values, container="sharedassets0.assets"):
    return {
        "class": class_name,
        "container": container,
        "path_id": path_id,
        "parsed": True,
        "serialized_sha256": sha256,
        "owner": f"owner-{path_id}",
        "values": values,
    }


def pointer(path_id):
    return {"m_FileID": 0, "m_PathID": path_id}


def valid_documents():
    targets = []
    moves = {}
    for path_id, (name, digest) in strategy_contract.MOVE_OBJECTS.items():
        values = {
            "m_Name": name,
            "displayName": name,
            "policyProfile": name,
            "npzFile": pointer(0),
            "playbackSpeed": 1.0,
            "blendInTime": 0.1,
            "blendOutTime": 0.1,
            "impactForgivenessDuration": 0.6,
            "impactEvents": [],
            "impactReversal": {"enabled": 0},
        }
        moves[path_id] = values
        targets.append(target("MocapClipConfig", path_id, digest, values))

    config_values = {
        "m_Name": "RobotConfig_T800_EngineAiFighting",
        "robotId": "t800",
        "allowMoveInterrupt": 0,
        "moves": [pointer(path_id) for path_id in strategy_contract.MOVE_SLOT_PATHS],
        "keyboardControlSchemes": [pointer(2743), pointer(2741)],
        "deadZone": 0.1,
        "forwardSpeed": 1.0,
        "strafeSpeed": 1.0,
        "yawSpeed": 1.0,
        "keyboardYawRampTime": 0.5,
        "stopBrakeRate": 3.0,
        "locomotionTransitionSettle": 1,
        "transitionSettlePlanarSpeed": 0.15,
        "transitionSettleYawRate": 0.3,
    }
    targets.append(target(
        "RobotConfig", 2739, strategy_contract.ROBOT_CONFIG["serialized_sha256"],
        config_values,
    ))

    runner_values = {
        "manualSwitchCooldown": 0.5,
        "profiles": [{
            "name": "skill",
            "onnxBytes": pointer(0),
            "configJson": pointer(0),
            "trajectoryCsv": pointer(0),
        }],
    }
    targets.append(target(
        "EngineAIPolicyRunner", strategy_contract.ENGINE_RUNNER["path_id"],
        strategy_contract.ENGINE_RUNNER["serialized_sha256"], runner_values,
    ))

    for container, path_id, digest in strategy_contract.PHYSICS_STEP_POLICIES:
        targets.append(target(
            "PhysXStepPolicy", path_id, digest,
            {"manualStepRateHz": 50.0}, container=container,
        ))

    for path_id, expected in strategy_contract.KEYBOARD_SCHEMES.items():
        values = {
            "m_Name": expected["name"],
            "forwardKey": 37, "backwardKey": 33,
            "strafeLeftKey": 15, "strafeRightKey": 18,
            "yawLeftKey": 31, "yawRightKey": 19,
            "bindings": [{
                "move": pointer(2732), "key1": 39, "key2": 0,
                "key3": 0, "doubleTap": 0,
            }],
            "straighten": {"key1": 51, "key2": 25, "key3": 0},
            "getUpProne": {"key1": 51, "key2": 30, "key3": 0},
            "getUpSupine": {"key1": 51, "key2": 16, "key3": 0},
            "passiveMode": {"key1": 51, "key2": 26, "key3": 0},
            "estopKey1": 65, "estopKey2": 60,
        }
        targets.append(target(
            "KeyboardControlScheme", path_id, expected["serialized_sha256"], values
        ))

    tuning = {field: float(index) for index, field in
              enumerate(strategy_contract.AI_TUNING_FIELDS)}
    for level in ("level1", "level2", "level3"):
        for fighter in (0, 1):
            path_id = 3000 + int(level[-1]) * 10 + fighter
            targets.append(target(
                "AIOpponentController", path_id, str(path_id) * 16,
                dict(tuning), container=level,
            ))

    probe = {
        "schema": "rek.mujoco_asset_probe.v1",
        "build_fingerprint": strategy_contract.BUILD_FINGERPRINT,
        "targets": targets,
    }
    enums = {
        "schema": "rek.high_level_enum_and_locomotion_constants.v1",
        "host": "D21",
        "special_command": [
            {"name": name, "value": value}
            for name, value in strategy_contract.SPECIAL_COMMANDS.items()
        ],
        "input_keys": [
            {"value": value, "name": name}
            for value, name in strategy_contract.KEY_NAMES.items()
        ],
        "read_locomotion_native_constants": [
            {"float32": 1.0}, {"float32": -1.0}
        ],
    }
    return probe, enums


class StrategyContractTests(unittest.TestCase):
    def test_builds_high_level_action_without_joint_actions(self):
        probe, enums = valid_documents()
        contract = strategy_contract.build_contract(probe, enums, {})
        action = contract["strategy_design"]["learned_action"]
        self.assertEqual(action["velocity"]["component_order"],
                         ["forward", "strafe", "yaw"])
        self.assertEqual(
            [row["move_slot"] for row in action["attack"]["categories"]],
            [None, 2, 3, 4, 5, 9, 10],
        )
        self.assertNotIn("joint", action)
        self.assertEqual(
            contract["measured_interface"]["execution_gate"][
                "manual_switch_cooldown_s"
            ], 0.5)
        self.assertTrue(
            contract["strategy_design"]["action_filter"][
                "never_call_null_slot_for_no_move_request"
            ])
        self.assertEqual(
            action["attack"]["categories"][0]["meaning"],
            "no_move_request",
        )
        self.assertIn(
            "DriveRecovery",
            contract["strategy_design"]["deterministic_wrapper_invariant"][
                "preserve"
            ],
        )
        self.assertFalse(contract["scope"]["simulator_ready"])
        self.assertFalse(contract["scope"]["training_ready"])

    def test_rejects_move_slot_drift(self):
        probe, enums = valid_documents()
        changed = copy.deepcopy(probe)
        config = next(row for row in changed["targets"]
                      if row["class"] == "RobotConfig")
        config["values"]["moves"][2] = pointer(2732)
        with self.assertRaisesRegex(strategy_contract.ContractError, "move slot map"):
            strategy_contract.build_contract(changed, enums, {})

    def test_rejects_fabricated_move_payload(self):
        probe, enums = valid_documents()
        changed = copy.deepcopy(probe)
        move = next(row for row in changed["targets"]
                    if row["class"] == "MocapClipConfig" and row["path_id"] == 2736)
        move["values"]["npzFile"] = pointer(999)
        with self.assertRaisesRegex(strategy_contract.ContractError,
                                    "embedded trajectory"):
            strategy_contract.build_contract(changed, enums, {})

    def test_rejects_special_enum_drift(self):
        probe, enums = valid_documents()
        changed = copy.deepcopy(enums)
        changed["special_command"][1]["value"] = 99
        with self.assertRaisesRegex(strategy_contract.ContractError,
                                    "SpecialCommand constants"):
            strategy_contract.build_contract(probe, changed, {})

    def test_rejects_inconsistent_ai_tuning(self):
        probe, enums = valid_documents()
        changed = copy.deepcopy(probe)
        ai = next(row for row in changed["targets"]
                  if row["class"] == "AIOpponentController")
        ai["values"]["kickChance"] = 999.0
        with self.assertRaisesRegex(strategy_contract.ContractError,
                                    "AIOpponentController tuning"):
            strategy_contract.build_contract(changed, enums, {})


if __name__ == "__main__":
    unittest.main()
