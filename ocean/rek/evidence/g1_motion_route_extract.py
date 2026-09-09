"""Extract the complete build-pinned G1 motion-route contract.

The route table joins the G1 RobotConfig references in the byte-pinned static
probe to the byte-pinned NPZ runtime-asset manifest.  Each of the seven idle or
locomotion routes is selected through its exact RobotConfig field.  Each of the
four supported kicks is selected through its exact position in RobotConfig's
``moves`` array.  Every route then resolves one MocapClipConfig path ID and one
NPZ TextAsset identity.  Missing values, substituted identities, and ambiguous
matches fail instead of receiving defaults.

This is static routing evidence for one client build.  It does not establish
runtime selection, timing, policy-weight identity, or trajectory parity.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    from . import g1_asset_extract as asset_extract
    from . import g1_motion_clip_extract as clip_extract
except ImportError:
    import g1_asset_extract as asset_extract
    import g1_motion_clip_extract as clip_extract


HERE = Path(__file__).resolve().parent
DEFAULT_PROBE_PATH = clip_extract.DEFAULT_PROBE_PATH
DEFAULT_CONTRACT_PATH = HERE / "g1_motion_route_contract.v1.json"

CONTRACT_SCHEMA = "rek.g1_motion_route_contract.v1"
ROBOT_CONFIG_PATH_ID = 2722
ROBOT_CONFIG_NAME = "RobotConfig_G1_UnitreeFighting"
ROBOT_CONFIG_SERIALIZED_BYTES = 472
ROBOT_CONFIG_SERIALIZED_SHA256 = (
    "3b0f5dfa78591b384ab022d85c4f3027182c8c4cf48f0c325a3b5170c5ebf1eb"
)
ROBOT_CONFIG_SCRIPT_REFERENCE = (16_777_216, 5_905_580_032)
ROBOT_CONFIG_TARGET_KEYS = clip_extract.MOCAP_TARGET_KEYS
ROBOT_CONFIG_VALUE_KEYS = frozenset(
    {
        "allowMoveInterrupt",
        "deadZone",
        "forwardSpeed",
        "gamepadControlSchemes",
        "hasEStop",
        "hasSpecialCommands",
        "idle",
        "keyboardControlSchemes",
        "keyboardYawRampTime",
        "locomotionTransitionSettle",
        "m_Enabled",
        "m_GameObject",
        "m_Name",
        "m_Script",
        "moves",
        "robotId",
        "stopBrakeRate",
        "strafeLeft",
        "strafeRight",
        "strafeSpeed",
        "transitionSettlePlanarSpeed",
        "transitionSettleYawRate",
        "turnLeft",
        "turnRight",
        "vrControlSchemes",
        "walkBackward",
        "walkForward",
        "yawSpeed",
    }
)


class ExtractionError(RuntimeError):
    """A route binding, source pin, identity, or schema check failed."""


@dataclass(frozen=True)
class RouteSpec:
    route_id: int
    name: str
    kind: str
    robot_config_field: str
    runtime_move_index: int | None
    mocap_clip_config_path_id: int
    mocap_config_name: str
    npz_role: str
    npz_path_id: int
    npz_name: str


ROUTE_SPECS = (
    RouteSpec(
        0, "idle", "idle", "idle", None, 2702,
        "idle_processed", "idle", 377, "idle_processed",
    ),
    RouteSpec(
        1, "forward", "translation", "walkForward", None, 2720,
        "walking_processed", "walk", 370, "walking_processed",
    ),
    RouteSpec(
        2, "backward", "translation", "walkBackward", None, 2721,
        "walking_processed_rev", "walk", 370, "walking_processed",
    ),
    RouteSpec(
        3, "strafe_left", "translation", "strafeLeft", None, 2711,
        "left_strafe_processed", "strafe_left", 388,
        "left_strafe_processed",
    ),
    RouteSpec(
        4, "strafe_right", "translation", "strafeRight", None, 2716,
        "right_strafe_processed", "strafe_left", 388,
        "left_strafe_processed",
    ),
    RouteSpec(
        5, "turn_left", "turn", "turnLeft", None, 2719,
        "turn_in_place_right_processed_rev", "turn_right", 375,
        "turn_in_place_right_processed",
    ),
    RouteSpec(
        6, "turn_right", "turn", "turnRight", None, 2718,
        "turn_in_place_right_processed", "turn_right", 375,
        "turn_in_place_right_processed",
    ),
    RouteSpec(
        7, "kick_move_6_left_side", "kick", "moves", 6, 2710,
        "left_side_kick_processed", "kick_left_side", 392,
        "left_side_kick_processed",
    ),
    RouteSpec(
        8, "kick_move_7_left_front", "kick", "moves", 7, 2703,
        "left_front_kick_processed", "kick_left_front", 371,
        "left_front_kick_processed",
    ),
    RouteSpec(
        9, "kick_move_8_right_side", "kick", "moves", 8, 2715,
        "right_side_kick_processed", "kick_right_side", 372,
        "right_side_kick_processed",
    ),
    RouteSpec(
        10, "kick_move_9_right_knee", "kick", "moves", 9, 2714,
        "right_knee_processed", "kick_right_knee", 380,
        "right_knee_processed",
    ),
)


def _validate_route_specs() -> None:
    if tuple(spec.route_id for spec in ROUTE_SPECS) != tuple(range(11)):
        raise ExtractionError("route IDs must be the exact contiguous range 0..10")
    if len({spec.name for spec in ROUTE_SPECS}) != len(ROUTE_SPECS):
        raise ExtractionError("route names must be unique")
    if len({spec.mocap_clip_config_path_id for spec in ROUTE_SPECS}) != len(
        ROUTE_SPECS
    ):
        raise ExtractionError("route MocapClipConfig path IDs must be unique")
    if tuple(
        spec.runtime_move_index
        for spec in ROUTE_SPECS
        if spec.runtime_move_index is not None
    ) != (6, 7, 8, 9):
        raise ExtractionError("kick runtime move indices must be exactly 6..9")
    for spec in ROUTE_SPECS:
        if spec.kind == "kick":
            if spec.robot_config_field != "moves" or spec.runtime_move_index is None:
                raise ExtractionError(f"kick route {spec.name!r} binding is invalid")
        elif spec.kind not in {"idle", "translation", "turn"}:
            raise ExtractionError(f"route {spec.name!r} kind is invalid")
        elif spec.runtime_move_index is not None:
            raise ExtractionError(
                f"non-kick route {spec.name!r} has a runtime move index"
            )


def _references(value: object, label: str) -> list[int]:
    result: list[int] = []
    for index, raw_reference in enumerate(clip_extract._sequence(value, label)):
        file_id, path_id = clip_extract._reference(
            raw_reference, f"{label}[{index}]"
        )
        if file_id != 0 or path_id <= 0:
            raise ExtractionError(f"{label}[{index}] must be a local non-null reference")
        result.append(path_id)
    return result


def _validate_robot_config(
    raw_target: object,
    manifest: asset_extract.Manifest,
) -> tuple[Mapping[str, object], Mapping[str, object], list[int]]:
    label = "G1 RobotConfig"
    target = clip_extract._mapping(raw_target, label)
    clip_extract._exact_keys(target, ROBOT_CONFIG_TARGET_KEYS, label)
    expected_metadata = {
        "assembly": clip_extract.EXPECTED_MOCAP_ASSEMBLY,
        "class": "RobotConfig",
        "container": manifest.source_name,
        "enabled": True,
        "hierarchy": None,
        "namespace": clip_extract.EXPECTED_MOCAP_NAMESPACE,
        "owner": None,
        "parse_mode": "generated_typetree",
        "parsed": True,
        "path_id": ROBOT_CONFIG_PATH_ID,
        "serialized_bytes": ROBOT_CONFIG_SERIALIZED_BYTES,
        "serialized_sha256": ROBOT_CONFIG_SERIALIZED_SHA256,
    }
    for key, expected in expected_metadata.items():
        if target.get(key) != expected:
            raise ExtractionError(f"{label}.{key} mismatch")

    values = clip_extract._mapping(target.get("values"), f"{label}.values")
    clip_extract._exact_keys(values, ROBOT_CONFIG_VALUE_KEYS, f"{label}.values")
    if clip_extract._flag(values.get("m_Enabled"), f"{label}.values.m_Enabled") != 1:
        raise ExtractionError("G1 RobotConfig must be enabled")
    if clip_extract._string(values.get("m_Name"), f"{label}.values.m_Name") != ROBOT_CONFIG_NAME:
        raise ExtractionError("G1 RobotConfig name mismatch")
    if clip_extract._string(values.get("robotId"), f"{label}.values.robotId") != "g1":
        raise ExtractionError("G1 RobotConfig robotId mismatch")
    if clip_extract._reference(
        values.get("m_GameObject"), f"{label}.values.m_GameObject"
    ) != (0, 0):
        raise ExtractionError("G1 RobotConfig GameObject reference mismatch")
    if clip_extract._reference(
        values.get("m_Script"), f"{label}.values.m_Script"
    ) != ROBOT_CONFIG_SCRIPT_REFERENCE:
        raise ExtractionError("G1 RobotConfig script reference mismatch")

    for field in (
        "allowMoveInterrupt",
        "hasEStop",
        "hasSpecialCommands",
        "locomotionTransitionSettle",
    ):
        clip_extract._flag(values.get(field), f"{label}.values.{field}")
    for field in (
        "deadZone",
        "forwardSpeed",
        "keyboardYawRampTime",
        "stopBrakeRate",
        "strafeSpeed",
        "transitionSettlePlanarSpeed",
        "transitionSettleYawRate",
        "yawSpeed",
    ):
        clip_extract._number(values.get(field), f"{label}.values.{field}", minimum=0)

    for field in (
        "gamepadControlSchemes",
        "keyboardControlSchemes",
        "vrControlSchemes",
    ):
        _references(values.get(field), f"{label}.values.{field}")
    moves = _references(values.get("moves"), f"{label}.values.moves")
    if len(moves) != 17:
        raise ExtractionError("G1 RobotConfig moves must contain exactly 17 entries")

    bindings: dict[str, object] = {}
    for field in (
        "idle",
        "walkForward",
        "walkBackward",
        "strafeLeft",
        "strafeRight",
        "turnLeft",
        "turnRight",
    ):
        file_id, path_id = clip_extract._reference(
            values.get(field), f"{label}.values.{field}"
        )
        if file_id != 0 or path_id <= 0:
            raise ExtractionError(
                f"G1 RobotConfig field {field!r} must be a local non-null reference"
            )
        bindings[field] = path_id
    bindings["moves"] = moves
    return target, bindings, moves


def _asset_projection(asset: asset_extract.AssetSpec) -> dict[str, object]:
    return {
        "bytes": asset.size,
        "dof": asset.dof,
        "fps": asset.fps,
        "frames": asset.frames,
        "name": asset.name,
        "output": asset.output,
        "path_id": asset.path_id,
        "role": asset.role,
        "sha256": asset.sha256,
    }


def _config_projection(
    target: Mapping[str, object], values: Mapping[str, object]
) -> dict[str, object]:
    return {
        "blendInTime": values["blendInTime"],
        "blendOutTime": values["blendOutTime"],
        "displayName": values["displayName"],
        "endFrame": values["endFrame"],
        "impactEvents": values["impactEvents"],
        "impactForgivenessDuration": values["impactForgivenessDuration"],
        "impactReversal": values["impactReversal"],
        "impactYawForgiveness": values["impactYawForgiveness"],
        "loop": values["loop"],
        "m_Name": values["m_Name"],
        "mirror": values["mirror"],
        "path_id": target["path_id"],
        "playbackSpeed": values["playbackSpeed"],
        "policyProfile": values["policyProfile"],
        "serialized_bytes": target["serialized_bytes"],
        "serialized_sha256": target["serialized_sha256"],
        "startFrame": values["startFrame"],
        "yawBlend": values["yawBlend"],
        "yawForgiveness": values["yawForgiveness"],
    }


def build_contract(
    probe: object,
    manifest: asset_extract.Manifest,
    manifest_sha256: str,
    *,
    probe_bytes: int,
    probe_sha256: str,
) -> dict[str, object]:
    _validate_route_specs()
    try:
        clip_extract._validate_manifest(manifest)
        clip_extract._sha256(manifest_sha256, "manifest canonical SHA-256")
        clip_extract._integer(probe_bytes, "probe byte count", minimum=1)
        clip_extract._sha256(probe_sha256, "probe SHA-256")
    except clip_extract.ExtractionError as exc:
        raise ExtractionError(str(exc)) from exc
    if manifest_sha256 != asset_extract.PINNED_MANIFEST_SHA256:
        raise ExtractionError("runtime-assets manifest canonical SHA-256 mismatch")
    if probe_bytes != clip_extract.PINNED_PROBE_BYTES:
        raise ExtractionError("probe byte count is not pinned")
    if probe_sha256 != clip_extract.PINNED_PROBE_SHA256:
        raise ExtractionError("probe SHA-256 is not pinned")

    try:
        root, targets = clip_extract._validate_probe_root(probe, manifest)
    except clip_extract.ExtractionError as exc:
        raise ExtractionError(str(exc)) from exc
    robot_matches = [
        target
        for target in targets
        if isinstance(target, dict)
        and target.get("class") == "RobotConfig"
        and target.get("path_id") == ROBOT_CONFIG_PATH_ID
    ]
    if len(robot_matches) != 1:
        raise ExtractionError(
            f"expected one G1 RobotConfig path {ROBOT_CONFIG_PATH_ID}, "
            f"got {len(robot_matches)}"
        )
    try:
        robot_target, bindings, moves = _validate_robot_config(
            robot_matches[0], manifest
        )
    except clip_extract.ExtractionError as exc:
        raise ExtractionError(str(exc)) from exc

    configs: dict[int, tuple[Mapping[str, object], dict[str, object]]] = {}
    for index, raw_target in enumerate(targets):
        if (
            not isinstance(raw_target, dict)
            or raw_target.get("class") != clip_extract.EXPECTED_MOCAP_CLASS
        ):
            continue
        try:
            target, values = clip_extract._validate_mocap_target(
                raw_target, index, manifest
            )
        except clip_extract.ExtractionError as exc:
            raise ExtractionError(str(exc)) from exc
        path_id = int(target["path_id"])
        if path_id in configs:
            raise ExtractionError(f"duplicate MocapClipConfig path ID {path_id}")
        configs[path_id] = (target, values)

    assets = {asset.role: asset for asset in manifest.assets}
    routes: list[dict[str, object]] = []
    for spec in ROUTE_SPECS:
        if spec.robot_config_field == "moves":
            move_index = spec.runtime_move_index
            if move_index is None:
                raise ExtractionError(
                    f"route {spec.name!r} omits its runtime move index"
                )
            bound_path_id = moves[move_index]
            binding: dict[str, object] = {
                "field": "moves",
                "index": move_index,
            }
        else:
            bound_path_id = int(bindings[spec.robot_config_field])
            binding = {"field": spec.robot_config_field, "index": None}
        if bound_path_id != spec.mocap_clip_config_path_id:
            raise ExtractionError(
                f"route {spec.name!r} RobotConfig binding mismatch: "
                f"expected {spec.mocap_clip_config_path_id}, got {bound_path_id}"
            )
        if bound_path_id not in configs:
            raise ExtractionError(
                f"route {spec.name!r} is missing MocapClipConfig {bound_path_id}"
            )
        target, values = configs[bound_path_id]
        if values["m_Name"] != spec.mocap_config_name:
            raise ExtractionError(
                f"route {spec.name!r} MocapClipConfig name mismatch"
            )
        if values["npz_path_id"] != spec.npz_path_id:
            raise ExtractionError(f"route {spec.name!r} NPZ path ID mismatch")
        asset = assets.get(spec.npz_role)
        if asset is None:
            raise ExtractionError(
                f"route {spec.name!r} has no runtime asset role {spec.npz_role!r}"
            )
        if (asset.path_id, asset.name) != (spec.npz_path_id, spec.npz_name):
            raise ExtractionError(f"route {spec.name!r} NPZ identity mismatch")
        routes.append(
            {
                "kind": spec.kind,
                "mocap_clip_config": _config_projection(target, values),
                "name": spec.name,
                "npz": _asset_projection(asset),
                "robot_config_binding": binding,
                "route_id": spec.route_id,
                "runtime_move_index": spec.runtime_move_index,
            }
        )

    return {
        "build_fingerprint": manifest.build_fingerprint,
        "classification": "build-pinned static motion-route identity",
        "parity_validated": False,
        "routes": routes,
        "runtime_selection_observed": False,
        "schema": CONTRACT_SCHEMA,
        "source": {
            "probe": {
                "bytes": probe_bytes,
                "inventory_sha256": root["inventory_sha256"],
                "name": clip_extract.PROBE_NAME,
                "schema": clip_extract.PROBE_SCHEMA,
                "sha256": probe_sha256,
                "unity_version": root["unity_version"],
                "unitypy_version": root["unitypy_version"],
            },
            "robot_config": {
                "name": ROBOT_CONFIG_NAME,
                "path_id": robot_target["path_id"],
                "robot_id": "g1",
                "serialized_bytes": robot_target["serialized_bytes"],
                "serialized_sha256": robot_target["serialized_sha256"],
            },
            "runtime_assets_manifest": {
                "canonical_sha256": manifest_sha256,
                "name": asset_extract.MANIFEST_PATH.name,
                "schema": asset_extract.MANIFEST_SCHEMA,
            },
            "unity_container": {
                "bytes": manifest.source_size,
                "name": manifest.source_name,
                "sha256": manifest.source_sha256,
            },
        },
    }


def render_contract(contract: object) -> bytes:
    try:
        return (
            json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExtractionError(f"contract cannot be serialized: {exc}") from exc


def extract_contract(probe_path: Path = DEFAULT_PROBE_PATH) -> bytes:
    try:
        manifest, manifest_sha256 = asset_extract.load_pinned_manifest()
        probe, probe_bytes, probe_sha256 = clip_extract.validate_pinned_probe(
            probe_path
        )
    except (asset_extract.ExtractionError, clip_extract.ExtractionError) as exc:
        raise ExtractionError(str(exc)) from exc
    return render_contract(
        build_contract(
            probe,
            manifest,
            manifest_sha256,
            probe_bytes=probe_bytes,
            probe_sha256=probe_sha256,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE_PATH)
    parser.add_argument(
        "--check",
        type=Path,
        help="require this existing contract to match the generated bytes",
    )
    arguments = parser.parse_args(argv)
    try:
        rendered = extract_contract(arguments.probe)
        if arguments.check is None:
            sys.stdout.buffer.write(rendered)
        else:
            if arguments.check.is_symlink() or not arguments.check.is_file():
                raise ExtractionError("checked contract must be a regular file")
            if arguments.check.read_bytes() != rendered:
                raise ExtractionError("checked contract differs from pinned extraction")
    except (ExtractionError, OSError) as exc:
        print(f"G1 motion route extraction failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
