#!/usr/bin/env python3
"""Produce one build-pinned static controller-path evidence artifact.

This tool records audited semantics from exact native GameAssembly.dll method
extents and bounded serialized T800 objects.  It does not execute REK and does
not claim that these client methods run in the server-authoritative private-AI
mode.

Cpp2IL dummy assemblies are useful for recovered names, signatures, and Unity
type trees.  Their restored method bodies are deliberately never read here.
Every semantic claim cites an exact native method extent instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any


SCHEMA = "rek.controller_path.v2"
BUILD_FINGERPRINT = "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659"

SOURCE_HASHES = {
    "inventory": "ea932824c7f1fa9781ab816716d4bfca9ec22b14e754466941c8c157910eff79",
    "recovery": "1fd542cd9421f95b668ba5acf5599ae1fe6435d64137292f9e3d1aae698c3372",
    "probe": "b132eb19cb7b223a87ee3885c16e521e82d7e99006c09ed63e8cc899ad057686",
    "game_assembly": "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412",
    "global_metadata": "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd",
    "EngineAIPolicyRunner.txt": "7fb4fe854362ead4b4ef04a06094180888906c4bee49a0e6149c0228206ccf31",
    "Robot.txt": "4f61233092542b15773e49d8404790a8ed89352d3b656fa41b75bab9c8283ded",
    "RobotInputController.txt": "f248df08449e3ff0706ce15ea07e4d58517f2fc9ed3f3143473fa48c4323bc21",
    "SonicPolicyRunner.txt": "5c7668aa79591cd84dfd120856ecdf96554309c85a2d5a425e8f42636381ab58",
}

# Sizes are the audited native method extents, in bytes.  This table is tied to
# SOURCE_HASHES["game_assembly"] and must be re-audited for any other build.
METHODS = (
    {
        "id": "engine_ai.fixed_update",
        "type": "REKApp.EngineAIPolicyRunner",
        "name": "FixedUpdate",
        "signature": "System.Void FixedUpdate()",
        "token": "0x0600091D",
        "rva": 0x236D000,
        "size": 0x72,
        "isil_file": "EngineAIPolicyRunner.txt",
        "native_sha256": "7e8425cf31350195c13c2660152b2802bd36ba7da01535b734e2a415b30646b9",
        "isil_sha256": "2c580cb2cf0c45ee89e459a1bb0b871b621b9ff9a97b7fd1f90138fcce40a0fe",
    },
    {
        "id": "engine_ai.control_tick",
        "type": "REKApp.EngineAIPolicyRunner",
        "name": "ControlTick",
        "signature": "System.Void ControlTick()",
        "token": "0x0600091E",
        "rva": 0x236B280,
        "size": 0x1648,
        "isil_file": "EngineAIPolicyRunner.txt",
        "native_sha256": "c2b700b929618949727d6ce7776d01abfd88b9aab03c9028381b9493a9918ab4",
        "isil_sha256": "39d373f118bf993c0ea48a1b0751ca9f9b787612950a381e18830bfe11d6c4ac",
    },
    {
        "id": "engine_ai.apply_mimic_torque_limits",
        "type": "REKApp.EngineAIPolicyRunner",
        "name": "ApplyMimicTorqueLimits",
        "signature": (
            "System.Void ApplyMimicTorqueLimits("
            "REKApp.EngineAIPolicyRunner+ProfileConfig cfg)"
        ),
        "token": "0x06000920",
        "rva": 0x236A0B0,
        "size": 0x301,
        "isil_file": "EngineAIPolicyRunner.txt",
        "native_sha256": "d79a381296ad94a8bbce01e5ed58de4326b8465c6eed47d69a95c76065a497ba",
        "isil_sha256": "cda35e7a9699fa6115a9d0b119a33b869ac130213c6492207c155c4ff6b52609",
    },
    {
        "id": "engine_ai.build_joint_mapping",
        "type": "REKApp.EngineAIPolicyRunner",
        "name": "BuildJointMapping",
        "signature": "System.Boolean BuildJointMapping()",
        "token": "0x06000918",
        "rva": 0x236AB20,
        "size": 0x497,
        "isil_file": "EngineAIPolicyRunner.txt",
        "native_sha256": "c604713da394f261dbf0e576d0d75583187978212c506dfea68728fb21131510",
        "isil_sha256": "a9ebed644868efff5c6b8e4919e56c2f52341bd0ad48fb98eeb56b46628a7e15",
    },
    {
        "id": "robot.build_mujoco_joint_motor_cache",
        "type": "REKApp.Robot",
        "name": "BuildMujocoJointMotorCache",
        "signature": "System.Void BuildMujocoJointMotorCache()",
        "token": "0x06000F9F",
        "rva": 0x23CEB70,
        "size": 0x885,
        "isil_file": "Robot.txt",
        "native_sha256": "0caa38b48f84ebd403b47c7d0cb9c0ec45f55ea0be5d6de818febe723808c932",
        "isil_sha256": "7dec392315fde10f0fdf98200d68d0deccbb699b198f6981096d1dea65a035ec",
    },
    {
        "id": "robot.configure_implicit_pd_actuators",
        "type": "REKApp.Robot",
        "name": "ConfigureImplicitPdActuators",
        "signature": (
            "System.Void ConfigureImplicitPdActuators("
            "Mujoco.MujocoLib+mjModel_* model)"
        ),
        "token": "0x06000FC1",
        "rva": 0x23D0680,
        "size": 0x1FF,
        "isil_file": "Robot.txt",
        "native_sha256": "4285c1e71b70f10eb158082a3941f610e771504df335d96db34d338f2c0ed6db",
        "isil_sha256": "3fa7ec34034bc128b41a2c002d08d4df324406e4590088246e5d81cfca4ba889",
    },
    {
        "id": "robot.apply_joint_controls",
        "type": "REKApp.Robot",
        "name": "ApplyJointControls",
        "signature": "System.Void ApplyJointControls(Mujoco.MjStepArgs args)",
        "token": "0x06000FC2",
        "rva": 0x23CD740,
        "size": 0x29E,
        "isil_file": "Robot.txt",
        "native_sha256": "4c732769dd129c3bb2314414acf50cbb8048f01a57e8e15158c42e53a49a3026",
        "isil_sha256": "a63967f9952d47c3cc693605b56d245e05647b143c5e821fd6ba9b266897a18e",
    },
    {
        "id": "robot.write_position_actuator_gains",
        "type": "REKApp.Robot",
        "name": "WritePositionActuatorGains",
        "signature": (
            "System.Void WritePositionActuatorGains("
            "Mujoco.MujocoLib+mjModel_* model, System.Int32 actuatorId, "
            "System.Single kp, System.Single kd)"
        ),
        "token": "0x06000FC3",
        "rva": 0x23DF360,
        "size": 0x82,
        "isil_file": "Robot.txt",
        "native_sha256": "064b3648806e9c11005499142636019416259190de34bda4741ca74bab4df7de",
        "isil_sha256": "c7bcefb37f6c10ecd85718ea775a2adc50e31d1519b9b230abc5798d75f49e6d",
    },
    {
        "id": "robot.write_actuator_force_limit",
        "type": "REKApp.Robot",
        "name": "WriteActuatorForceLimit",
        "signature": (
            "System.Void WriteActuatorForceLimit("
            "Mujoco.MujocoLib+mjModel_* model, System.Int32 actuatorId, "
            "System.Single limit)"
        ),
        "token": "0x06000FC4",
        "rva": 0x23DF2E0,
        "size": 0x78,
        "isil_file": "Robot.txt",
        "native_sha256": "d295f5907154916ef328d400dfecc0e4ce4c1a1862aea8d11eace129745c8866",
        "isil_sha256": "f7e7118ce105857869db6e4d1f49dabd54467f60623934fc510a51a7dccaddc5",
    },
    {
        "id": "robot_input.late_update",
        "type": "REKApp.RobotInputController",
        "name": "LateUpdate",
        "signature": "System.Void LateUpdate()",
        "token": "0x060010FC",
        "rva": 0x226D9B0,
        "size": 0xE2,
        "isil_file": "RobotInputController.txt",
        "native_sha256": "b500252de8fb4c9ebc6a6848be71600f7920a2de8c8c4e2cb4927e4c37711992",
        "isil_sha256": "3e1c26e185729b173372c49c2511ecfab43ed2d09d71c144cdb27fb8b03698e2",
    },
    {
        "id": "robot_input.send_velocity_command",
        "type": "REKApp.RobotInputController",
        "name": "SendVelocityCommand",
        "signature": (
            "System.Void SendVelocityCommand("
            "Unity.Netcode.CustomMessagingManager cmm)"
        ),
        "token": "0x06001102",
        "rva": 0x226F110,
        "size": 0x25A,
        "isil_file": "RobotInputController.txt",
        "native_sha256": "245c5591d89d313f829cae3862a3041868bb5a71c89ae79a6c88320881b00974",
        "isil_sha256": "907e532b7b54e016e8fee9ce6a66c51f4ab6f51f76d86a124aae762957da1cde",
    },
    {
        "id": "robot_input.execute_move_by_index",
        "type": "REKApp.RobotInputController",
        "name": "ExecuteMoveByIndex",
        "signature": "System.Boolean ExecuteMoveByIndex(System.Int32 index)",
        "token": "0x0600111B",
        "rva": 0x226CB10,
        "size": 0x19B,
        "isil_file": "RobotInputController.txt",
        "native_sha256": "515eef982712dcc66ccef3e4b2f4e8d4da349232dfd6bbe74b5e8ea6ac0571d4",
        "isil_sha256": "3e831efba10dfd76cbbd7005fb545930077c519dff8d7ccfa60a25038b54797e",
    },
)

RUNNER_SOURCE = {
    "container": "sharedassets0.assets",
    "path_id": 3356,
    "serialized_sha256": "5919088fb298d1dddbc563495240025307255aad8715d6af8f7df8fe3d139e66",
}
ROBOT_SOURCE = {
    "container": "sharedassets0.assets",
    "path_id": 2976,
    "serialized_sha256": "db13d1bdf99e79957c07338f8b71a5470754531186b7db9dd80b9fd5fec3830a",
}
CONFIG_SOURCE = {
    "container": "sharedassets0.assets",
    "path_id": 2739,
    "serialized_sha256": "4cfc389375f3a80e778eb4ac73b7ea52ef58da0b12764a0848576894a532a405",
}

EXPECTED_MOVE_PATH_IDS = (0, 0, 2736, 2738, 2732, 2734, 0, 0, 0, 2735, 2730, 0)
MOVE_OBJECT_SOURCES = {
    2730: {
        "container": "sharedassets0.assets",
        "path_id": 2730,
        "serialized_sha256": "cd5b286f6e4f5c3003cb0f5c9de5e5690ca92ed58e5a1b789f4394e4d7911ee8",
    },
    2732: {
        "container": "sharedassets0.assets",
        "path_id": 2732,
        "serialized_sha256": "32081b731a59b7553d94022ebff865764b34c83dbf274aadf26540fb17daad2e",
    },
    2734: {
        "container": "sharedassets0.assets",
        "path_id": 2734,
        "serialized_sha256": "b1c1b2c000dd612e3eb4c33c5d90e03c2c9306e5cc194747c14248b0d77b7dea",
    },
    2735: {
        "container": "sharedassets0.assets",
        "path_id": 2735,
        "serialized_sha256": "cc298f53d04ffd56be57ce3049559d3d30c7724fe4d2839a66ea8f3008ca8deb",
    },
    2736: {
        "container": "sharedassets0.assets",
        "path_id": 2736,
        "serialized_sha256": "233f952edecb7bf8d1959c6549c0edb95e1833451fff988f57a4b14d92b14dd4",
    },
    2738: {
        "container": "sharedassets0.assets",
        "path_id": 2738,
        "serialized_sha256": "70f36a2c7b9b53c10e47cc613d87a770eb86fb2e683ed64ee39efcccf2e75636",
    },
}
EXPECTED_MOVE_OBJECT_FIELDS = {
    2730: {
        "m_Name": "front_kick_L",
        "displayName": "Left Kick",
        "policyProfile": "front_kick_L",
        "impactEvents": [{
            "gainBoost": 1.0,
            "impactTime": 1.100000023841858,
            "leadTime": 0.20000000298023224,
            "limb": 3,
            "releaseTime": 0.15000000596046448,
        }],
        "impactReversal": {
            "emaAlpha": 0.05000000074505806,
            "enabled": 1,
            "holdTime": 0.5,
            "spikeFloorNm": 30.0,
            "spikeRatio": 1.25,
        },
    },
    2732: {
        "m_Name": "left_light_attack",
        "displayName": "Left Punch",
        "policyProfile": "left_light_attack",
        "impactEvents": [{
            "gainBoost": 1.0,
            "impactTime": 0.38999998569488525,
            "leadTime": 0.11999999731779099,
            "limb": 1,
            "releaseTime": 0.07999999821186066,
        }],
        "impactReversal": {
            "emaAlpha": 0.05000000074505806,
            "enabled": 0,
            "holdTime": 0.5,
            "spikeFloorNm": 30.0,
            "spikeRatio": 1.25,
        },
    },
    2734: {
        "m_Name": "right_light_attack",
        "displayName": "Right Punch",
        "policyProfile": "right_light_attack",
        "impactEvents": [{
            "gainBoost": 1.0,
            "impactTime": 0.2199999988079071,
            "leadTime": 0.11999999731779099,
            "limb": 2,
            "releaseTime": 0.20000000298023224,
        }],
        "impactReversal": {
            "emaAlpha": 0.05000000074505806,
            "enabled": 0,
            "holdTime": 0.5,
            "spikeFloorNm": 30.0,
            "spikeRatio": 1.25,
        },
    },
    2735: {
        "m_Name": "right_shoryuken_lm",
        "displayName": "Dragon Punch",
        "policyProfile": "right_shoryuken_lm",
        "impactEvents": [{
            "gainBoost": 1.0,
            "impactTime": 0.0,
            "leadTime": 0.0,
            "limb": 2,
            "releaseTime": 0.0,
        }],
        "impactReversal": {
            "emaAlpha": 0.05000000074505806,
            "enabled": 0,
            "holdTime": 0.5,
            "spikeFloorNm": 30.0,
            "spikeRatio": 1.25,
        },
    },
    2736: {
        "m_Name": "skill",
        "displayName": "Punch Combo",
        "policyProfile": "skill",
        "impactEvents": [
            {
                "gainBoost": 1.0,
                "impactTime": 0.7599999904632568,
                "leadTime": 0.10000000149011612,
                "limb": 1,
                "releaseTime": 0.1899999976158142,
            },
            {
                "gainBoost": 1.0,
                "impactTime": 1.149999976158142,
                "leadTime": 0.10000000149011612,
                "limb": 1,
                "releaseTime": 0.10000000149011612,
            },
            {
                "gainBoost": 1.0,
                "impactTime": 1.809999942779541,
                "leadTime": 0.10000000149011612,
                "limb": 1,
                "releaseTime": 0.10000000149011612,
            },
        ],
        "impactReversal": {
            "emaAlpha": 0.05000000074505806,
            "enabled": 0,
            "holdTime": 0.5,
            "spikeFloorNm": 30.0,
            "spikeRatio": 1.25,
        },
    },
    2738: {
        "m_Name": "youbiantui",
        "displayName": "Right Kick",
        "policyProfile": "youbiantui",
        "impactEvents": [{
            "gainBoost": 1.0,
            "impactTime": 1.1100000143051147,
            "leadTime": 0.25,
            "limb": 4,
            "releaseTime": 0.30000001192092896,
        }],
        "impactReversal": {
            "emaAlpha": 0.05000000074505806,
            "enabled": 0,
            "holdTime": 0.5,
            "spikeFloorNm": 30.0,
            "spikeRatio": 1.25,
        },
    },
}
EXPECTED_MOVE_COMMON_FIELDS = {
    "m_Enabled": 1,
    "npzFile": {"m_FileID": 0, "m_PathID": 0},
    "startFrame": 0,
    "endFrame": -1,
    "playbackSpeed": 1.0,
    "loop": 0,
    "mirror": 0,
    "blendInTime": 0.10000000149011612,
    "blendOutTime": 0.10000000149011612,
    "yawBlend": 0.0,
    "impactForgivenessDuration": 0.6000000238418579,
    "impactYawForgiveness": 0.0,
    "yawForgiveness": 0.0,
}
EXPECTED_FORCE_LIMITS = (
    415.0, 370.0, 222.0, 415.0, 160.0, 160.0, 415.0, 370.0, 222.0,
    415.0, 160.0, 160.0, 222.0, 160.0, 160.0, 160.0, 160.0, 52.0,
    160.0, 160.0, 160.0, 160.0, 52.0, 52.0, 52.0,
)

UNKNOWN_IDS = {
    "authoritative_active_runner",
    "server_implementation_and_build",
    "active_profile_assets_and_parameters",
    "observation_model_and_hidden_state",
    "authoritative_joint_cache_and_limits",
    "command_to_server_tick_alignment",
    "move_profile_execution_semantics",
}


class EvidenceError(ValueError):
    pass


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceError(f"{path} does not contain a JSON object")
    return value


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise EvidenceError(f"{label} mismatch: {actual!r} != {expected!r}")


def _verify_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise EvidenceError(f"{label} is not a file: {path}")
    actual = _sha256_path(path)
    _require(actual, expected, f"{label} SHA-256")
    return actual


def _inventory_file(inventory: dict[str, Any], relative: str) -> dict[str, Any]:
    matches = [
        item for item in inventory.get("files", [])
        if str(item.get("path", "")).replace("\\", "/") == relative
    ]
    if len(matches) != 1:
        raise EvidenceError(f"inventory has {len(matches)} records for {relative}")
    return matches[0]


def _pe_sections(image: bytes) -> list[dict[str, Any]]:
    if len(image) < 0x40 or image[:2] != b"MZ":
        raise EvidenceError("GameAssembly input is not an MZ image")
    pe_offset = struct.unpack_from("<I", image, 0x3C)[0]
    if pe_offset + 24 > len(image) or image[pe_offset:pe_offset + 4] != b"PE\0\0":
        raise EvidenceError("GameAssembly input has no valid PE header")
    section_count = struct.unpack_from("<H", image, pe_offset + 6)[0]
    optional_size = struct.unpack_from("<H", image, pe_offset + 20)[0]
    table = pe_offset + 24 + optional_size
    if table + 40 * section_count > len(image):
        raise EvidenceError("PE section table extends beyond the input")
    sections = []
    for index in range(section_count):
        offset = table + 40 * index
        name = image[offset:offset + 8].split(b"\0", 1)[0].decode("ascii", "replace")
        virtual_size, rva, raw_size, raw_offset = struct.unpack_from(
            "<IIII", image, offset + 8
        )
        characteristics = struct.unpack_from("<I", image, offset + 36)[0]
        sections.append({
            "name": name,
            "rva": rva,
            "virtual_size": virtual_size,
            "raw_size": raw_size,
            "raw_offset": raw_offset,
            "executable": bool(characteristics & 0x20000000),
        })
    return sections


def _native_extent(
    image: bytes, sections: list[dict[str, Any]], rva: int, size: int
) -> tuple[dict[str, Any], int, bytes]:
    if size <= 0:
        raise EvidenceError("native method extent is empty")
    for section in sections:
        start = int(section["rva"])
        raw_size = int(section["raw_size"])
        if start <= rva and rva + size <= start + raw_size:
            if not section["executable"]:
                raise EvidenceError(f"RVA {rva:#x} is not in an executable section")
            file_offset = int(section["raw_offset"]) + rva - start
            if file_offset + size > len(image):
                raise EvidenceError(f"RVA {rva:#x} extends beyond GameAssembly")
            return section, file_offset, image[file_offset:file_offset + size]
    raise EvidenceError(f"RVA extent {rva:#x}+{size:#x} is not file-backed")


def _normalise_method_blocks(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    starts = [match.start() for match in re.finditer(r"(?m)^Method: ", text)]
    result: dict[str, str] = {}
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(text)
        block = text[start:stop].rstrip() + "\n"
        header = block.splitlines()[0][len("Method: "):]
        if header in result:
            raise EvidenceError(f"duplicate ISIL method block: {header}")
        if "\nDisassembly:\n" not in block or "\nISIL:\n" not in block:
            raise EvidenceError(f"incomplete ISIL method block: {header}")
        result[header] = block
    return result


def derive_force_limit(
    force_range: tuple[float, float],
    control_range: tuple[float, float],
    joint_fingerprint_effort: float | None,
) -> dict[str, Any]:
    force = max(abs(float(force_range[0])), abs(float(force_range[1])))
    if force > 0:
        return {"state": "measured", "source": "force_range", "value": force}
    control = max(abs(float(control_range[0])), abs(float(control_range[1])))
    if control > 0:
        return {"state": "measured", "source": "control_range", "value": control}
    if joint_fingerprint_effort is not None and float(joint_fingerprint_effort) > 0:
        return {
            "state": "measured",
            "source": "joint_fingerprint_effort",
            "value": float(joint_fingerprint_effort),
        }
    return {
        "state": "unknown",
        "because": "force range, control range, and positive JointFingerprint.effort are absent",
    }


def _exact_target(
    probe: dict[str, Any], class_name: str, source: dict[str, Any]
) -> dict[str, Any]:
    matches = [
        target for target in probe.get("targets", [])
        if target.get("class") == class_name
        and target.get("container") == source["container"]
        and target.get("path_id") == source["path_id"]
    ]
    if len(matches) != 1:
        raise EvidenceError(
            f"probe has {len(matches)} {class_name} objects at "
            f"{source['container']}:{source['path_id']}"
        )
    target = matches[0]
    _require(target.get("serialized_sha256"), source["serialized_sha256"],
             f"{class_name} serialized hash")
    if not target.get("parsed") or not isinstance(target.get("values"), dict):
        raise EvidenceError(f"{class_name} object is not parsed")
    return target


def _pointer(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise EvidenceError(f"serialized pointer is not an object: {value!r}")
    try:
        return {"file_id": int(value["m_FileID"]), "path_id": int(value["m_PathID"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"invalid serialized pointer: {value!r}") from exc


def _expected_resolved_move_object(path_id: int) -> dict[str, Any]:
    source = MOVE_OBJECT_SOURCES[path_id]
    fields = dict(EXPECTED_MOVE_COMMON_FIELDS)
    fields.update(EXPECTED_MOVE_OBJECT_FIELDS[path_id])
    return {
        "state": "measured",
        "evidence_role": "serialized_object_fields",
        "source": {"kind": "serialized_object", **source},
        "fields": fields,
        "motion_payload_state": "npz_pointer_null_in_serialized_client_object",
        "execution_state": "not_established_by_static_object",
    }


def _resolved_move_object(probe: dict[str, Any], path_id: int) -> dict[str, Any]:
    source = MOVE_OBJECT_SOURCES[path_id]
    target = _exact_target(probe, "MocapClipConfig", source)
    values = target["values"]
    expected = _expected_resolved_move_object(path_id)
    expected_fields = expected["fields"]
    actual_fields = {field: values.get(field) for field in expected_fields}
    _require(actual_fields, expected_fields, f"T800 move object {path_id} fields")
    return expected


def _serialized_t800(probe: dict[str, Any]) -> dict[str, Any]:
    runner = _exact_target(probe, "EngineAIPolicyRunner", RUNNER_SOURCE)
    robot = _exact_target(probe, "Robot", ROBOT_SOURCE)
    config = _exact_target(probe, "RobotConfig", CONFIG_SOURCE)
    runner_values = runner["values"]
    robot_values = robot["values"]
    config_values = config["values"]

    _require(config_values.get("m_Name"), "RobotConfig_T800_EngineAiFighting",
             "T800 RobotConfig name")
    _require(config_values.get("robotId"), "t800", "T800 robotId")
    _require(_pointer(runner_values.get("robot")), {"file_id": 0, "path_id": 2976},
             "T800 runner robot pointer")
    _require(robot_values.get("bypassTorqueLimits"), 0, "T800 bypassTorqueLimits")

    profiles = runner_values.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 45:
        raise EvidenceError("T800 runner does not contain the expected 45 profiles")
    for profile in profiles:
        for field in ("onnxBytes", "configJson", "trajectoryCsv"):
            _require(_pointer(profile.get(field)), {"file_id": 0, "path_id": 0},
                     f"T800 profile {profile.get('name')!r} {field}")

    moves = config_values.get("moves")
    if not isinstance(moves, list) or len(moves) != 12:
        raise EvidenceError("T800 RobotConfig move table is not 12 slots")
    move_map = []
    for index, raw_pointer in enumerate(moves):
        pointer = _pointer(raw_pointer)
        expected_path_id = EXPECTED_MOVE_PATH_IDS[index]
        _require(pointer, {"file_id": 0, "path_id": expected_path_id},
                 f"T800 move slot {index} pointer")
        row: dict[str, Any] = {
            "index": index,
            "pointer": pointer,
            "referenced_object": (
                _resolved_move_object(probe, expected_path_id)
                if expected_path_id else None
            ),
        }
        move_map.append(row)

    actuator_targets = []
    for target in probe.get("targets", []):
        path = ((target.get("hierarchy") or {}).get("game_object_path") or "")
        if target.get("class") != "MjActuator" or not path.startswith(
            "engineai_t800_FactoryPolicy/actuators/"
        ):
            continue
        match = re.fullmatch(r"motor_J(\d{2})_.+", str(target.get("owner", "")))
        if not match:
            raise EvidenceError(f"unindexed T800 actuator: {target.get('owner')!r}")
        if not target.get("parsed") or not isinstance(target.get("values"), dict):
            raise EvidenceError(f"unparsed T800 actuator: {target.get('owner')!r}")
        common = target["values"].get("CommonParams")
        if not isinstance(common, dict):
            raise EvidenceError(f"actuator CommonParams missing: {target.get('owner')!r}")
        force = common.get("ForceRange") or {}
        control = common.get("CtrlRange") or {}
        force_range = (float(force["x"]), float(force["y"]))
        control_range = (float(control["x"]), float(control["y"]))
        derived = derive_force_limit(force_range, control_range, None)
        actuator_targets.append({
            "index": int(match.group(1)),
            "name": target["owner"],
            "source": {
                "kind": "serialized_object",
                "container": target["container"],
                "path_id": target["path_id"],
                "serialized_sha256": target["serialized_sha256"],
            },
            "serialized_values": target["values"],
            "force_range": list(force_range),
            "control_range": list(control_range),
            "joint_fingerprint_effort": {"state": "not_needed_for_this_input"},
            "derived_initial_force_limit": derived,
        })
    actuator_targets.sort(key=lambda row: row["index"])
    _require([row["index"] for row in actuator_targets], list(range(25)),
             "T800 actuator indices")
    actual_limits = tuple(
        row["derived_initial_force_limit"].get("value") for row in actuator_targets
    )
    _require(actual_limits, EXPECTED_FORCE_LIMITS, "T800 derived force limits")

    return {
        "runner": {
            "source": {"kind": "serialized_object", **RUNNER_SOURCE},
            "values": runner_values,
            "profile_count": len(profiles),
            "profile_payload_pointer_state": "all_null_in_serialized_client_object",
        },
        "robot": {
            "source": {"kind": "serialized_object", **ROBOT_SOURCE},
            "values": robot_values,
        },
        "robot_config": {
            "source": {"kind": "serialized_object", **CONFIG_SOURCE},
            "values": config_values,
            "move_map": move_map,
        },
        "actuators": actuator_targets,
        "derived_force_limit_vector": list(actual_limits),
    }


def _static_semantics() -> dict[str, Any]:
    unknown_activation = {
        "state": "unknown",
        "context": "server-authoritative private Bot1",
    }
    return {
        "engine_ai_action_to_target": {
            "evidence_role": "static_native_semantics",
            "activation": unknown_activation,
            "equations": [
                "a[i] = clamp(output[i], -action_clip, action_clip)",
                "target[i] = default_joint_q[i] + action_scale[i] * a[i]",
                "residual_target[i] = trajectory_pose[i] + action_scale[i] * a[i]",
            ],
            "conditions": [
                "active_joint_indices maps controller elements to robot motors",
                "the trajectory branch applies only when its recovered configuration selects it",
                "entry transition interpolates from the preceding target to the selected target",
            ],
            "writes": "Robot.SetJointTargetRadiansUnclamped",
            "citations": [
                "engine_ai.control_tick",
                "engine_ai.build_joint_mapping",
            ],
        },
        "engine_ai_mimic_torque_limit": {
            "evidence_role": "static_native_semantics",
            "activation": unknown_activation,
            "equations": [
                "tau[i] = joint_kp[i] * (target[i] - q[i]) - joint_kd[i] * qd[i]",
                "tau[i] = clamp(tau[i], -max_torque[i], max_torque[i]) when max_torque[i] > 0",
                "lower_count = min(lower_body_joint_count > 0 ? lower_body_joint_count : 12, joint_count)",
                "budget = max_lower_body_torque > 0 ? max_lower_body_torque : 1700",
                "scale = budget / sum(abs(tau[0:lower_count])) when that sum exceeds budget",
                "target[i] = q[i] + (tau[i] + joint_kd[i] * qd[i]) / joint_kp[i] when joint_kp[i] > 1e-6",
            ],
            "citations": ["engine_ai.apply_mimic_torque_limits"],
        },
        "engine_ai_walk_target_clamp": {
            "evidence_role": "static_native_semantics",
            "activation": unknown_activation,
            "equations": [
                "denom[i] = max(joint_kp[i], 0.001)",
                "center[i] = q[i] + joint_kd[i] * qd[i] / denom[i]",
                "radius[i] = max_torque[i] / denom[i]",
                "target[i] = clamp(target[i], center[i] - radius[i], center[i] + radius[i])",
            ],
            "citations": ["engine_ai.control_tick"],
        },
        "robot_implicit_pd_configuration": {
            "evidence_role": "static_native_semantics",
            "activation": unknown_activation,
            "writes": [
                "gainprm[0] = kp",
                "biasprm[1] = -kp",
                "biasprm[2] = -kd",
                "ctrl = target",
            ],
            "citations": [
                "robot.configure_implicit_pd_actuators",
                "robot.apply_joint_controls",
                "robot.write_position_actuator_gains",
            ],
        },
        "robot_force_limit_construction": {
            "evidence_role": "static_native_semantics",
            "activation": unknown_activation,
            "precedence": [
                "max(abs(force_range_lower), abs(force_range_upper)) when positive",
                "max(abs(control_range_lower), abs(control_range_upper)) when positive",
                "JointFingerprint.effort when positive",
                "unknown when no positive measured source exists",
            ],
            "conditional_write": (
                "a positive selected limit sets forcelimited=1 and forcerange=[-limit,+limit]; "
                "otherwise force limiting is disabled"
            ),
            "citations": [
                "robot.build_mujoco_joint_motor_cache",
                "robot.configure_implicit_pd_actuators",
                "robot.write_actuator_force_limit",
            ],
        },
        "robot_input_move_dispatch": {
            "evidence_role": "static_native_semantics",
            "activation": unknown_activation,
            "semantics": [
                "ExecuteMoveByIndex performs a bounds-checked RobotConfig.moves lookup",
                "a nonvisual local request passes the selected clip to ExecuteMove",
                "a visual-only client request queues REK_Move with networkIndex and moveIndex bytes",
            ],
            "citations": [
                "robot_input.late_update",
                "robot_input.execute_move_by_index",
            ],
        },
    }


def _hard_unknowns() -> list[dict[str, Any]]:
    return [
        {
            "id": "authoritative_active_runner",
            "state": "unknown",
            "because": "the private Bot1 client is visual-only and remotely authoritative",
            "obtain_by": "authoritative runtime instrumentation or controlled black-box replay",
        },
        {
            "id": "server_implementation_and_build",
            "state": "unknown",
            "because": "the client inventory does not pin the dedicated server binary",
            "obtain_by": "a server artifact/version identity or measured behavioral replay",
        },
        {
            "id": "active_profile_assets_and_parameters",
            "state": "unknown",
            "because": "all 45 serialized client profile payload pointers are null",
            "obtain_by": "hash and inspect the runtime-loaded profile bundle if it becomes observable",
        },
        {
            "id": "observation_model_and_hidden_state",
            "state": "unknown",
            "because": "no active ONNX/config payload or authoritative controller trace is present",
            "obtain_by": "capture the active observation, output, and recurrent-state boundary",
        },
        {
            "id": "authoritative_joint_cache_and_limits",
            "state": "unknown",
            "because": "client static construction does not establish server runtime values",
            "obtain_by": "authoritative joint-cache readback or differential system identification",
        },
        {
            "id": "command_to_server_tick_alignment",
            "state": "unknown",
            "because": "the client trace has no server tick or command acknowledgement",
            "obtain_by": "capture an acknowledged server tick or estimate delay from controlled repeats",
        },
        {
            "id": "move_profile_execution_semantics",
            "state": "unknown",
            "because": (
                "the six move configuration objects are recovered, but their npz pointers "
                "and all matching T800 ONNX/config/trajectory pointers are null and no "
                "authoritative execution trace exposes the selected profile state"
            ),
            "obtain_by": (
                "capture controlled move requests and the authoritative pose response, or "
                "inspect an authorized server-side controller package"
            ),
        },
    ]


def build_report(
    inventory_path: Path,
    recovery_path: Path,
    probe_path: Path,
    game_assembly_path: Path,
    global_metadata_path: Path,
    isil_dir: Path,
) -> dict[str, Any]:
    inventory_hash = _verify_sha(inventory_path, SOURCE_HASHES["inventory"], "inventory")
    recovery_hash = _verify_sha(recovery_path, SOURCE_HASHES["recovery"], "IL2CPP recovery")
    probe_hash = _verify_sha(probe_path, SOURCE_HASHES["probe"], "asset probe")
    game_hash = _verify_sha(game_assembly_path, SOURCE_HASHES["game_assembly"], "GameAssembly")
    metadata_hash = _verify_sha(global_metadata_path, SOURCE_HASHES["global_metadata"], "global metadata")

    inventory = _load_json(inventory_path)
    recovery = _load_json(recovery_path)
    probe = _load_json(probe_path)
    _require(inventory.get("schema"), 1, "inventory schema")
    _require(inventory.get("build_fingerprint"), BUILD_FINGERPRINT, "inventory build")
    _require((recovery.get("build") or {}).get("build_fingerprint"), BUILD_FINGERPRINT,
             "recovery build")
    _require(probe.get("schema"), "rek.mujoco_asset_probe.v1", "asset probe schema")
    _require(probe.get("build_fingerprint"), BUILD_FINGERPRINT, "asset probe build")
    _require(probe.get("inventory_sha256"), inventory_hash, "asset probe inventory hash")
    _require((recovery.get("inputs") or {}).get("game_assembly", {}).get("sha256"),
             game_hash, "recovery GameAssembly hash")
    _require((recovery.get("inputs") or {}).get("global_metadata", {}).get("sha256"),
             metadata_hash, "recovery metadata hash")
    _require(_inventory_file(inventory, "GameAssembly.dll").get("sha256"), game_hash,
             "inventory GameAssembly hash")
    _require(_inventory_file(
        inventory, "REK_Data/il2cpp_data/Metadata/global-metadata.dat"
    ).get("sha256"), metadata_hash, "inventory metadata hash")

    isil_sources: dict[str, dict[str, Any]] = {}
    block_tables: dict[str, dict[str, str]] = {}
    for filename in (
        "EngineAIPolicyRunner.txt",
        "Robot.txt",
        "RobotInputController.txt",
        "SonicPolicyRunner.txt",
    ):
        path = isil_dir / filename
        digest = _verify_sha(path, SOURCE_HASHES[filename], filename)
        isil_sources[filename] = {
            "filename": filename,
            "sha256": digest,
            "kind": "native_isil_disassembly",
            "semantic_claims_allowed": filename != "SonicPolicyRunner.txt",
            "role": (
                "alternative_controller_context_only"
                if filename == "SonicPolicyRunner.txt"
                else "native_semantic_source"
            ),
        }
        block_tables[filename] = _normalise_method_blocks(path)

    image = game_assembly_path.read_bytes()
    sections = _pe_sections(image)
    extents = sorted((method["rva"], method["rva"] + method["size"], method["id"])
                     for method in METHODS)
    for left, right in zip(extents, extents[1:]):
        if left[1] > right[0]:
            raise EvidenceError(f"native method extents overlap: {left[2]} and {right[2]}")

    method_records = []
    for method in METHODS:
        section, file_offset, body = _native_extent(
            image, sections, method["rva"], method["size"]
        )
        native_hash = hashlib.sha256(body).hexdigest()
        _require(native_hash, method["native_sha256"], f"{method['id']} native body")
        blocks = block_tables[method["isil_file"]]
        block = blocks.get(method["signature"])
        if block is None:
            raise EvidenceError(
                f"{method['id']} has no unique ISIL block named {method['signature']!r}"
            )
        block_hash = hashlib.sha256(block.encode("utf-8")).hexdigest()
        _require(block_hash, method["isil_sha256"], f"{method['id']} ISIL block")
        method_records.append({
            "id": method["id"],
            "type": method["type"],
            "name": method["name"],
            "signature": method["signature"],
            "metadata_token": method["token"],
            "rva": f"0x{method['rva']:X}",
            "size_bytes": method["size"],
            "pe_section": section["name"],
            "file_offset": f"0x{file_offset:X}",
            "native_body_sha256": native_hash,
            "isil_source": method["isil_file"],
            "isil_method_block_sha256": block_hash,
            "isil_normalization": "UTF-8; CRLF/CR to LF; rstrip; one final LF",
            "evidence_role": "static_native_semantics",
        })

    report = {
        "schema": SCHEMA,
        "build_fingerprint": BUILD_FINGERPRINT,
        "scope": {
            "robot_id": "t800",
            "evidence_role": "static_native_semantics",
            "control_equivalent": False,
            "private_ai_runtime_active": {"state": "unknown"},
            "server_equivalent": {"state": "unknown"},
            "move_object_fields": {
                "state": "measured",
                "count": len(MOVE_OBJECT_SOURCES),
                "evidence_role": "serialized_object_fields",
            },
        },
        "sources": {
            "inventory": {"filename": inventory_path.name, "sha256": inventory_hash},
            "il2cpp_recovery": {
                "filename": recovery_path.name,
                "sha256": recovery_hash,
                "kind": "identity_and_signature_catalog",
                "dummy_method_bodies_consumed": False,
                "semantic_claims_allowed": False,
                "cpp2il_commit": (recovery.get("recovery") or {}).get("cpp2il_commit"),
                "runtime_validation_completed": (
                    recovery.get("recovery") or {}
                ).get("runtime_validation_completed"),
            },
            "asset_probe": {"filename": probe_path.name, "sha256": probe_hash},
            "game_assembly": {
                "inventory_path": "GameAssembly.dll",
                "sha256": game_hash,
                "kind": "native_binary",
            },
            "global_metadata": {
                "inventory_path": "REK_Data/il2cpp_data/Metadata/global-metadata.dat",
                "sha256": metadata_hash,
                "kind": "metadata_identity_only",
                "semantic_claims_allowed": False,
            },
            "isil": isil_sources,
        },
        "native_methods": method_records,
        "static_semantics": _static_semantics(),
        "serialized_t800": _serialized_t800(probe),
        "hard_unknowns": _hard_unknowns(),
        "limits": [
            "Native client semantics do not establish private-AI runtime activation.",
            "The dedicated server implementation and build are not pinned.",
            "Serialized null payload pointers are preserved and never replaced with defaults.",
            "All six referenced move objects are bound; their serialized npzFile pointers are null.",
            "Move configuration fields do not establish authoritative motion trajectories or policy weights.",
            "No dummy or interop method body is semantic evidence.",
            "No proprietary binary payload is copied into this artifact.",
        ],
    }
    errors = validate_report(report)
    if errors:
        raise EvidenceError("internally invalid report: " + "; ".join(errors))
    return report


def validate_report(report: dict[str, Any]) -> list[str]:
    """Offline structural and fixed-build validation used by check_artifacts."""
    errors: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    expect(report.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    expect(report.get("build_fingerprint") == BUILD_FINGERPRINT,
           "unsupported or absent build fingerprint")
    scope = report.get("scope") or {}
    expect(scope.get("evidence_role") == "static_native_semantics",
           "scope must be static_native_semantics")
    expect(scope.get("control_equivalent") is False, "control_equivalent must be false")
    expect((scope.get("private_ai_runtime_active") or {}).get("state") == "unknown",
           "private-AI runtime activation must remain unknown")
    expect((scope.get("server_equivalent") or {}).get("state") == "unknown",
           "server equivalence must remain unknown")
    move_scope = scope.get("move_object_fields") or {}
    expect(move_scope.get("state") == "measured",
           "move-object fields must be measured")
    expect(move_scope.get("count") == len(MOVE_OBJECT_SOURCES),
           "move-object field count mismatch")
    expect(move_scope.get("evidence_role") == "serialized_object_fields",
           "move-object evidence role mismatch")

    sources = report.get("sources") or {}
    expected_source_hashes = {
        "inventory": SOURCE_HASHES["inventory"],
        "il2cpp_recovery": SOURCE_HASHES["recovery"],
        "asset_probe": SOURCE_HASHES["probe"],
        "game_assembly": SOURCE_HASHES["game_assembly"],
        "global_metadata": SOURCE_HASHES["global_metadata"],
    }
    for key, expected in expected_source_hashes.items():
        expect((sources.get(key) or {}).get("sha256") == expected,
               f"{key} source hash mismatch")
    recovery_source = sources.get("il2cpp_recovery") or {}
    expect(recovery_source.get("dummy_method_bodies_consumed") is False,
           "dummy method bodies must not be consumed")
    expect(recovery_source.get("semantic_claims_allowed") is False,
           "recovery projection must not be a semantic source")
    for filename in SOURCE_HASHES:
        if not filename.endswith(".txt"):
            continue
        expect(((sources.get("isil") or {}).get(filename) or {}).get("sha256")
               == SOURCE_HASHES[filename], f"{filename} source hash mismatch")

    methods = report.get("native_methods")
    if not isinstance(methods, list):
        errors.append("native_methods must be a list")
        methods = []
    by_id = {method.get("id"): method for method in methods if isinstance(method, dict)}
    expect(len(by_id) == len(METHODS) == len(methods), "native method IDs must be unique")
    for expected in METHODS:
        method = by_id.get(expected["id"]) or {}
        expect(method.get("metadata_token") == expected["token"],
               f"{expected['id']} token mismatch")
        expect(method.get("rva") == f"0x{expected['rva']:X}",
               f"{expected['id']} RVA mismatch")
        expect(method.get("size_bytes") == expected["size"],
               f"{expected['id']} extent mismatch")
        expect(method.get("native_body_sha256") == expected["native_sha256"],
               f"{expected['id']} native hash mismatch")
        expect(method.get("isil_method_block_sha256") == expected["isil_sha256"],
               f"{expected['id']} ISIL block hash mismatch")
        expect(method.get("evidence_role") == "static_native_semantics",
               f"{expected['id']} evidence role mismatch")

    facts = report.get("static_semantics")
    if not isinstance(facts, dict) or not facts:
        errors.append("static_semantics is empty")
        facts = {}
    for fact_id, fact in facts.items():
        expect(fact.get("evidence_role") == "static_native_semantics",
               f"{fact_id} evidence role mismatch")
        expect((fact.get("activation") or {}).get("state") == "unknown",
               f"{fact_id} activation must remain unknown")
        for citation in fact.get("citations", []):
            expect(citation in by_id, f"{fact_id} cites unknown method {citation!r}")

    serialized = report.get("serialized_t800") or {}
    moves = ((serialized.get("robot_config") or {}).get("move_map"))
    if not isinstance(moves, list):
        errors.append("T800 move map must be a list")
        moves = []
    expect([row.get("index") for row in moves if isinstance(row, dict)] == list(range(12)),
           "T800 move map must preserve slots 0 through 11")
    for index, row in enumerate(moves):
        expected_path_id = EXPECTED_MOVE_PATH_IDS[index]
        pointer = row.get("pointer") if isinstance(row, dict) else {}
        referenced_object = (
            row.get("referenced_object") if isinstance(row, dict) else None
        )
        expect(pointer == {"file_id": 0, "path_id": expected_path_id},
               f"T800 move slot {index} pointer mismatch")
        if expected_path_id == 0:
            expect(referenced_object is None,
                   f"T800 move slot {index} null target mismatch")
        else:
            expect(referenced_object == _expected_resolved_move_object(expected_path_id),
                   f"T800 move slot {index} referenced object mismatch")

    expect(tuple(serialized.get("derived_force_limit_vector") or ())
           == EXPECTED_FORCE_LIMITS, "T800 derived force-limit vector mismatch")
    unknowns = report.get("hard_unknowns")
    if not isinstance(unknowns, list):
        errors.append("hard_unknowns must be a list")
        unknowns = []
    unknown_by_id = {
        row.get("id"): row for row in unknowns if isinstance(row, dict)
    }
    expect(set(unknown_by_id) == UNKNOWN_IDS, "hard unknown set mismatch")
    for unknown_id, row in unknown_by_id.items():
        expect(row.get("state") == "unknown", f"{unknown_id} must remain unknown")
    return errors


def _render(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--game-assembly", type=Path, required=True)
    parser.add_argument("--global-metadata", type=Path, required=True)
    parser.add_argument("--isil-dir", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path)
    action.add_argument("--verify", type=Path)
    args = parser.parse_args()

    report = build_report(
        args.inventory.resolve(),
        args.recovery.resolve(),
        args.probe.resolve(),
        args.game_assembly.resolve(),
        args.global_metadata.resolve(),
        args.isil_dir.resolve(),
    )
    encoded = _render(report)
    if args.verify:
        existing = args.verify.read_bytes()
        if existing != encoded:
            raise EvidenceError(f"{args.verify} does not reproduce byte-for-byte")
        print(json.dumps({
            "verified": str(args.verify),
            "sha256": hashlib.sha256(existing).hexdigest(),
            "build_fingerprint": report["build_fingerprint"],
        }, indent=2))
        return 0

    assert args.out is not None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(encoded)
    print(json.dumps({
        "out": str(args.out),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "build_fingerprint": report["build_fingerprint"],
        "native_methods": len(report["native_methods"]),
        "t800_moves": len(report["serialized_t800"]["robot_config"]["move_map"]),
        "t800_non_null_move_pointers": sum(
            row["pointer"]["path_id"] != 0
            for row in report["serialized_t800"]["robot_config"]["move_map"]
        ),
        "t800_resolved_move_objects": sum(
            isinstance(row["referenced_object"], dict)
            and row["referenced_object"].get("state") == "measured"
            for row in report["serialized_t800"]["robot_config"]["move_map"]
        ),
        "private_ai_runtime_active": "unknown",
        "server_equivalent": "unknown",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
