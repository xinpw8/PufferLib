"""Extract the pinned G1 MocapClipConfig contract from static REK evidence.

The extractor reads one byte-pinned ``mujoco_asset_probe_v8.json`` and the
already pinned G1 runtime-asset manifest. Each required role is joined to one
MocapClipConfig by the exact pair of NPZ TextAsset path ID and asset name.
No clip is selected by display name, ordering, or a partial identity match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

try:
    from . import g1_asset_extract as asset_extract
except ImportError:
    import g1_asset_extract as asset_extract


HERE = Path(__file__).resolve().parent
DEFAULT_PROBE_PATH = HERE / "evidence_out" / "mujoco_asset_probe_v8.json"
DEFAULT_CONTRACT_PATH = HERE / "g1_motion_clip_contract.v1.json"

PROBE_NAME = "mujoco_asset_probe_v8.json"
PINNED_PROBE_BYTES = 4_568_314
PINNED_PROBE_SHA256 = (
    "b132eb19cb7b223a87ee3885c16e521e82d7e99006c09ed63e8cc899ad057686"
)
PROBE_SCHEMA = "rek.mujoco_asset_probe.v1"
CONTRACT_SCHEMA = "rek.g1_motion_clip_contract.v1"
EXPECTED_UNITY_VERSION = "6000.5.8f1"
EXPECTED_MOCAP_CLASS = "MocapClipConfig"
EXPECTED_MOCAP_ASSEMBLY = "REKApp"
EXPECTED_MOCAP_NAMESPACE = "REKApp"

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
PROBE_KEYS = frozenset(
    {
        "build_fingerprint",
        "container_count",
        "dummy_dir",
        "game_root",
        "header_failures",
        "header_resolved_count",
        "inventory_sha256",
        "limits",
        "mono_behaviour_count",
        "parsed_target_count",
        "schema",
        "target_count",
        "targets",
        "unity_version",
        "unitypy_version",
        "unparsed_target_count",
    }
)
MOCAP_TARGET_KEYS = frozenset(
    {
        "assembly",
        "class",
        "container",
        "enabled",
        "hierarchy",
        "namespace",
        "owner",
        "parse_mode",
        "parsed",
        "path_id",
        "serialized_bytes",
        "serialized_sha256",
        "values",
    }
)
MOCAP_VALUE_KEYS = frozenset(
    {
        "blendInTime",
        "blendOutTime",
        "displayName",
        "endFrame",
        "impactEvents",
        "impactForgivenessDuration",
        "impactReversal",
        "impactYawForgiveness",
        "loop",
        "m_Enabled",
        "m_GameObject",
        "m_Name",
        "m_Script",
        "mirror",
        "npzFile",
        "playbackSpeed",
        "policyProfile",
        "startFrame",
        "yawBlend",
        "yawForgiveness",
    }
)
REFERENCE_KEYS = frozenset({"m_FileID", "m_PathID"})
IMPACT_EVENT_KEYS = frozenset(
    {"gainBoost", "impactTime", "leadTime", "limb", "releaseTime"}
)
IMPACT_REVERSAL_KEYS = frozenset(
    {"emaAlpha", "enabled", "holdTime", "spikeFloorNm", "spikeRatio"}
)


class ExtractionError(RuntimeError):
    """A fail-closed probe, identity, schema, or contract error."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_constant(token: str) -> None:
    raise ExtractionError(f"JSON contains nonfinite constant {token!r}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExtractionError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def load_json_bytes(raw: bytes, label: str) -> object:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"{label} is not UTF-8: {exc}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ExtractionError:
        raise
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"{label} is invalid JSON: {exc}") from exc


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ExtractionError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ExtractionError(f"{label} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ExtractionError(
            f"{label} keys mismatch: missing={missing}, unknown={unknown}"
        )


def _string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        suffix = "a string" if allow_empty else "a nonempty string"
        raise ExtractionError(f"{label} must be {suffix}")
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExtractionError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ExtractionError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ExtractionError(f"{label} must be <= {maximum}")
    return value


def _number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    nonzero: bool = False,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtractionError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise ExtractionError(f"{label} must be finite")
    if minimum is not None and float(value) < minimum:
        raise ExtractionError(f"{label} must be >= {minimum}")
    if nonzero and float(value) == 0.0:
        raise ExtractionError(f"{label} must be nonzero")
    return value


def _flag(value: object, label: str) -> int:
    return _integer(value, label, minimum=0, maximum=1)


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label)
    if SHA256_RE.fullmatch(digest) is None:
        raise ExtractionError(f"{label} must be a lowercase SHA-256")
    return digest


def _reference(value: object, label: str) -> tuple[int, int]:
    reference = _mapping(value, label)
    _exact_keys(reference, REFERENCE_KEYS, label)
    file_id = _integer(reference.get("m_FileID"), f"{label}.m_FileID", minimum=0)
    path_id = _integer(reference.get("m_PathID"), f"{label}.m_PathID", minimum=0)
    return file_id, path_id


def validate_pinned_probe(path: Path) -> tuple[Mapping[str, object], int, str]:
    if path.name != PROBE_NAME:
        raise ExtractionError(f"probe filename must be exactly {PROBE_NAME!r}")
    if path.is_symlink() or not path.is_file():
        raise ExtractionError("probe must be a regular, non-symlink file")
    raw = path.read_bytes()
    if len(raw) != PINNED_PROBE_BYTES:
        raise ExtractionError(
            f"probe byte count mismatch: expected {PINNED_PROBE_BYTES}, got {len(raw)}"
        )
    digest = sha256_bytes(raw)
    if digest != PINNED_PROBE_SHA256:
        raise ExtractionError(
            f"probe SHA-256 mismatch: expected {PINNED_PROBE_SHA256}, got {digest}"
        )
    return _mapping(load_json_bytes(raw, "probe"), "probe"), len(raw), digest


def _validate_probe_root(
    probe: object, manifest: asset_extract.Manifest
) -> tuple[Mapping[str, object], Sequence[object]]:
    root = _mapping(probe, "probe")
    _exact_keys(root, PROBE_KEYS, "probe")
    if root.get("schema") != PROBE_SCHEMA:
        raise ExtractionError("probe schema mismatch")
    if root.get("unity_version") != EXPECTED_UNITY_VERSION:
        raise ExtractionError("probe Unity version mismatch")
    if root.get("unitypy_version") != manifest.parser_version:
        raise ExtractionError("probe UnityPy version mismatch")
    build_fingerprint = _sha256(
        root.get("build_fingerprint"), "probe.build_fingerprint"
    )
    if build_fingerprint != manifest.build_fingerprint:
        raise ExtractionError("probe and runtime-assets build fingerprints mismatch")
    _sha256(root.get("inventory_sha256"), "probe.inventory_sha256")

    targets = _sequence(root.get("targets"), "probe.targets")
    target_count = _integer(root.get("target_count"), "probe.target_count", minimum=0)
    parsed_count = _integer(
        root.get("parsed_target_count"), "probe.parsed_target_count", minimum=0
    )
    unparsed_count = _integer(
        root.get("unparsed_target_count"), "probe.unparsed_target_count", minimum=0
    )
    if target_count != len(targets):
        raise ExtractionError("probe target_count does not equal targets length")
    if parsed_count + unparsed_count != target_count:
        raise ExtractionError("probe parsed and unparsed target counts mismatch")
    if unparsed_count != 0:
        raise ExtractionError("probe contains unparsed targets")

    container_count = _integer(
        root.get("container_count"), "probe.container_count", minimum=1
    )
    header_resolved_count = _integer(
        root.get("header_resolved_count"),
        "probe.header_resolved_count",
        minimum=0,
    )
    mono_behaviour_count = _integer(
        root.get("mono_behaviour_count"),
        "probe.mono_behaviour_count",
        minimum=0,
    )
    if container_count <= 0 or mono_behaviour_count < target_count:
        raise ExtractionError("probe aggregate counts are inconsistent")
    _string(root.get("dummy_dir"), "probe.dummy_dir")
    _string(root.get("game_root"), "probe.game_root")
    header_failures = _sequence(
        root.get("header_failures"), "probe.header_failures"
    )
    if header_resolved_count + len(header_failures) != mono_behaviour_count:
        raise ExtractionError("probe header resolution counts are inconsistent")
    limits = _sequence(root.get("limits"), "probe.limits")
    for index, value in enumerate(limits):
        _string(value, f"probe.limits[{index}]")
    for index, target in enumerate(targets):
        item = _mapping(target, f"probe.targets[{index}]")
        if item.get("parsed") is not True:
            raise ExtractionError(f"probe.targets[{index}] is not parsed")
    return root, targets


def _validate_impact_reversal(value: object, label: str) -> dict[str, object]:
    reversal = _mapping(value, label)
    _exact_keys(reversal, IMPACT_REVERSAL_KEYS, label)
    return {
        "emaAlpha": _number(reversal.get("emaAlpha"), f"{label}.emaAlpha", minimum=0),
        "enabled": _flag(reversal.get("enabled"), f"{label}.enabled"),
        "holdTime": _number(reversal.get("holdTime"), f"{label}.holdTime", minimum=0),
        "spikeFloorNm": _number(
            reversal.get("spikeFloorNm"), f"{label}.spikeFloorNm", minimum=0
        ),
        "spikeRatio": _number(
            reversal.get("spikeRatio"), f"{label}.spikeRatio", minimum=0
        ),
    }


def _validate_impact_events(value: object, label: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for index, raw_event in enumerate(_sequence(value, label)):
        event_label = f"{label}[{index}]"
        event = _mapping(raw_event, event_label)
        _exact_keys(event, IMPACT_EVENT_KEYS, event_label)
        events.append(
            {
                "gainBoost": _number(
                    event.get("gainBoost"), f"{event_label}.gainBoost", minimum=0
                ),
                "impactTime": _number(
                    event.get("impactTime"), f"{event_label}.impactTime", minimum=0
                ),
                "leadTime": _number(
                    event.get("leadTime"), f"{event_label}.leadTime", minimum=0
                ),
                "limb": _integer(
                    event.get("limb"), f"{event_label}.limb", minimum=1, maximum=4
                ),
                "releaseTime": _number(
                    event.get("releaseTime"),
                    f"{event_label}.releaseTime",
                    minimum=0,
                ),
            }
        )
    return events


def _validate_mocap_target(
    raw_target: object,
    index: int,
    manifest: asset_extract.Manifest,
) -> tuple[Mapping[str, object], dict[str, object]]:
    label = f"probe.targets[{index}]"
    target = _mapping(raw_target, label)
    _exact_keys(target, MOCAP_TARGET_KEYS, label)
    if target.get("assembly") != EXPECTED_MOCAP_ASSEMBLY:
        raise ExtractionError(f"{label}.assembly mismatch")
    if target.get("namespace") != EXPECTED_MOCAP_NAMESPACE:
        raise ExtractionError(f"{label}.namespace mismatch")
    if target.get("class") != EXPECTED_MOCAP_CLASS:
        raise ExtractionError(f"{label}.class mismatch")
    if target.get("container") != manifest.source_name:
        raise ExtractionError(f"{label}.container mismatch")
    if target.get("enabled") is not True or target.get("parsed") is not True:
        raise ExtractionError(f"{label} must be enabled and parsed")
    if target.get("parse_mode") != "generated_typetree":
        raise ExtractionError(f"{label}.parse_mode mismatch")
    if target.get("hierarchy") is not None or target.get("owner") is not None:
        raise ExtractionError(f"{label} hierarchy and owner must be null")
    _integer(target.get("path_id"), f"{label}.path_id", minimum=1)
    _integer(target.get("serialized_bytes"), f"{label}.serialized_bytes", minimum=1)
    _sha256(target.get("serialized_sha256"), f"{label}.serialized_sha256")

    values_label = f"{label}.values"
    values = _mapping(target.get("values"), values_label)
    _exact_keys(values, MOCAP_VALUE_KEYS, values_label)
    game_file, game_path = _reference(
        values.get("m_GameObject"), f"{values_label}.m_GameObject"
    )
    if (game_file, game_path) != (0, 0):
        raise ExtractionError(f"{values_label}.m_GameObject mismatch")
    script_file, script_path = _reference(
        values.get("m_Script"), f"{values_label}.m_Script"
    )
    if (script_file, script_path) != (16_777_216, 83_869_302_784):
        raise ExtractionError(f"{values_label}.m_Script mismatch")
    npz_file, npz_path = _reference(values.get("npzFile"), f"{values_label}.npzFile")
    if npz_file != 0:
        raise ExtractionError(f"{values_label}.npzFile.m_FileID must be zero")
    if _flag(values.get("m_Enabled"), f"{values_label}.m_Enabled") != 1:
        raise ExtractionError(f"{values_label}.m_Enabled must be one")

    start_frame = _integer(
        values.get("startFrame"), f"{values_label}.startFrame", minimum=0
    )
    end_frame = _integer(
        values.get("endFrame"), f"{values_label}.endFrame", minimum=-1
    )
    if end_frame != -1 and end_frame < start_frame:
        raise ExtractionError(f"{values_label} frame range is invalid")

    normalized = {
        "blendInTime": _number(
            values.get("blendInTime"), f"{values_label}.blendInTime", minimum=0
        ),
        "blendOutTime": _number(
            values.get("blendOutTime"), f"{values_label}.blendOutTime", minimum=0
        ),
        "displayName": _string(
            values.get("displayName"), f"{values_label}.displayName", allow_empty=True
        ),
        "endFrame": end_frame,
        "impactEvents": _validate_impact_events(
            values.get("impactEvents"), f"{values_label}.impactEvents"
        ),
        "impactForgivenessDuration": _number(
            values.get("impactForgivenessDuration"),
            f"{values_label}.impactForgivenessDuration",
            minimum=0,
        ),
        "impactReversal": _validate_impact_reversal(
            values.get("impactReversal"), f"{values_label}.impactReversal"
        ),
        "impactYawForgiveness": _number(
            values.get("impactYawForgiveness"),
            f"{values_label}.impactYawForgiveness",
            minimum=0,
        ),
        "loop": _flag(values.get("loop"), f"{values_label}.loop"),
        "mirror": _flag(values.get("mirror"), f"{values_label}.mirror"),
        "m_Name": _string(values.get("m_Name"), f"{values_label}.m_Name"),
        "npz_path_id": npz_path,
        "playbackSpeed": _number(
            values.get("playbackSpeed"),
            f"{values_label}.playbackSpeed",
            nonzero=True,
        ),
        "policyProfile": _string(
            values.get("policyProfile"),
            f"{values_label}.policyProfile",
            allow_empty=True,
        ),
        "startFrame": start_frame,
        "yawBlend": _number(
            values.get("yawBlend"), f"{values_label}.yawBlend", minimum=0
        ),
        "yawForgiveness": _number(
            values.get("yawForgiveness"),
            f"{values_label}.yawForgiveness",
            minimum=0,
        ),
    }
    return target, normalized


def _validate_manifest(manifest: asset_extract.Manifest) -> None:
    roles = [asset.role for asset in manifest.assets]
    if set(roles) != asset_extract.EXPECTED_ROLES or len(roles) != len(
        asset_extract.EXPECTED_ROLES
    ):
        raise ExtractionError("runtime-assets manifest roles mismatch")
    identities = [(asset.path_id, asset.name) for asset in manifest.assets]
    if len(identities) != len(set(identities)):
        raise ExtractionError("runtime-assets manifest has duplicate NPZ identities")
    _sha256(manifest.build_fingerprint, "manifest.build_fingerprint")
    _sha256(manifest.source_sha256, "manifest.source_sha256")


def build_contract(
    probe: object,
    manifest: asset_extract.Manifest,
    manifest_sha256: str,
    *,
    probe_bytes: int,
    probe_sha256: str,
) -> dict[str, object]:
    _validate_manifest(manifest)
    _sha256(manifest_sha256, "manifest canonical SHA-256")
    if manifest_sha256 != asset_extract.PINNED_MANIFEST_SHA256:
        raise ExtractionError("runtime-assets manifest canonical SHA-256 mismatch")
    _integer(probe_bytes, "probe byte count", minimum=1)
    if probe_bytes != PINNED_PROBE_BYTES:
        raise ExtractionError("probe byte count is not pinned")
    _sha256(probe_sha256, "probe SHA-256")
    if probe_sha256 != PINNED_PROBE_SHA256:
        raise ExtractionError("probe SHA-256 is not pinned")
    root, targets = _validate_probe_root(probe, manifest)

    mocap: list[tuple[Mapping[str, object], dict[str, object]]] = []
    config_path_ids: set[int] = set()
    for index, raw_target in enumerate(targets):
        target = _mapping(raw_target, f"probe.targets[{index}]")
        if target.get("class") != EXPECTED_MOCAP_CLASS:
            continue
        validated = _validate_mocap_target(target, index, manifest)
        config_path_id = int(validated[0]["path_id"])
        if config_path_id in config_path_ids:
            raise ExtractionError(
                f"duplicate MocapClipConfig path ID {config_path_id}"
            )
        config_path_ids.add(config_path_id)
        mocap.append(validated)

    clips: list[dict[str, object]] = []
    for asset in sorted(manifest.assets, key=lambda item: item.role):
        matches = [
            item
            for item in mocap
            if item[1]["npz_path_id"] == asset.path_id
            and item[1]["m_Name"] == asset.name
        ]
        if not matches:
            raise ExtractionError(
                f"missing MocapClipConfig for role {asset.role!r} and NPZ identity "
                f"({asset.path_id}, {asset.name!r})"
            )
        if len(matches) != 1:
            raise ExtractionError(
                f"duplicate MocapClipConfig matches for role {asset.role!r}"
            )
        target, values = matches[0]
        clips.append(
            {
                "role": asset.role,
                "npz": {
                    "bytes": asset.size,
                    "dof": asset.dof,
                    "fps": asset.fps,
                    "frames": asset.frames,
                    "name": asset.name,
                    "output": asset.output,
                    "path_id": asset.path_id,
                    "sha256": asset.sha256,
                },
                "mocap_clip_config": {
                    "path_id": target["path_id"],
                    "serialized_bytes": target["serialized_bytes"],
                    "serialized_sha256": target["serialized_sha256"],
                    "displayName": values["displayName"],
                    "loop": values["loop"],
                    "mirror": values["mirror"],
                    "playbackSpeed": values["playbackSpeed"],
                    "startFrame": values["startFrame"],
                    "endFrame": values["endFrame"],
                    "blendInTime": values["blendInTime"],
                    "blendOutTime": values["blendOutTime"],
                    "yawBlend": values["yawBlend"],
                    "yawForgiveness": values["yawForgiveness"],
                    "impactForgivenessDuration": values[
                        "impactForgivenessDuration"
                    ],
                    "impactYawForgiveness": values["impactYawForgiveness"],
                    "impactReversal": values["impactReversal"],
                    "impactEvents": values["impactEvents"],
                },
            }
        )

    return {
        "schema": CONTRACT_SCHEMA,
        "build_fingerprint": manifest.build_fingerprint,
        "source": {
            "probe": {
                "bytes": probe_bytes,
                "inventory_sha256": root["inventory_sha256"],
                "name": PROBE_NAME,
                "schema": PROBE_SCHEMA,
                "sha256": probe_sha256,
                "unity_version": root["unity_version"],
                "unitypy_version": root["unitypy_version"],
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
        "clips": clips,
    }


def render_contract(contract: object) -> bytes:
    try:
        return (json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ExtractionError(f"contract cannot be serialized: {exc}") from exc


def extract_contract(probe_path: Path = DEFAULT_PROBE_PATH) -> bytes:
    manifest, manifest_sha256 = asset_extract.load_pinned_manifest()
    probe, probe_bytes, probe_sha256 = validate_pinned_probe(probe_path)
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
        if arguments.check is not None:
            if arguments.check.is_symlink() or not arguments.check.is_file():
                raise ExtractionError("checked contract must be a regular file")
            if arguments.check.read_bytes() != rendered:
                raise ExtractionError("checked contract differs from pinned extraction")
        else:
            sys.stdout.buffer.write(rendered)
    except (ExtractionError, asset_extract.ExtractionError, OSError) as exc:
        print(f"G1 motion clip extraction failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
