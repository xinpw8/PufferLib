"""Verify and inspect the build-pinned non-Steam REK G1 SONIC staging assets.

The current Steam build contains none of these three TextAssets.  The tracked
contract therefore labels this staging configuration as comparative evidence,
not current-Steam authority.  It contains decoded JSON and only hashes, sizes,
names, and path IDs for the ONNX TextAssets.  This command never writes the
source TextAssets or model bytes.

Example:

    python g1_sonic_asset_extract.py --game-root C:\\path\\to\\REK
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT_PATH = Path(__file__).with_name("g1_sonic_staging_policy_contract.v1.json")
CONTRACT_SCHEMA = "rek.g1_sonic_policy_contract.v1"
SUMMARY_SCHEMA = "rek.g1_sonic_asset_verification.v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class SonicAssetError(RuntimeError):
    """A fail-closed contract, build, or TextAsset validation error."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SonicAssetError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise SonicAssetError(f"{label} must be an array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SonicAssetError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SonicAssetError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if SHA256_RE.fullmatch(result) is None:
        raise SonicAssetError(f"{label} must be a lowercase SHA-256")
    return result


def _script_bytes(value: object) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8", "surrogateescape")
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise SonicAssetError(
        f"TextAsset m_Script has unsupported type {type(value).__name__}"
    )


def load_contract(path: Path = CONTRACT_PATH) -> Mapping[str, Any]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SonicAssetError(f"could not load contract {path}: {exc}") from exc
    validate_contract(root)
    return root


def validate_contract(root: object) -> None:
    contract = _object(root, "contract")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise SonicAssetError("contract schema mismatch")
    parser = _object(contract.get("parser"), "parser")
    if parser.get("name") != "UnityPy":
        raise SonicAssetError("only the pinned UnityPy parser is supported")
    _string(parser.get("version"), "parser.version")

    files = _object(contract.get("files"), "files")
    expected_paths = {
        "rek_exe": "REK.exe",
        "game_assembly": "GameAssembly.dll",
        "global_metadata": "REK_Data/il2cpp_data/Metadata/global-metadata.dat",
        "source_container": "REK_Data/sharedassets1.assets",
    }
    if set(files) != set(expected_paths):
        raise SonicAssetError(f"files must be exactly {sorted(expected_paths)}")
    for key, expected_path in expected_paths.items():
        item = _object(files[key], f"files.{key}")
        if item.get("path") != expected_path:
            raise SonicAssetError(f"files.{key}.path mismatch")
        _integer(item.get("bytes"), f"files.{key}.bytes", 1)
        _sha256(item.get("sha256"), f"files.{key}.sha256")

    assets = _array(contract.get("text_assets"), "text_assets")
    expected_names = {"sonic_config", "model_encoder.onnx", "model_decoder.onnx"}
    names: set[str] = set()
    path_ids: set[int] = set()
    for index, value in enumerate(assets):
        item = _object(value, f"text_assets[{index}]")
        name = _string(item.get("name"), f"text_assets[{index}].name")
        path_id = _integer(item.get("path_id"), f"text_assets[{index}].path_id", 1)
        _integer(item.get("bytes"), f"text_assets[{index}].bytes", 1)
        _sha256(item.get("sha256"), f"text_assets[{index}].sha256")
        if name in names or path_id in path_ids:
            raise SonicAssetError("TextAsset names and path IDs must be unique")
        names.add(name)
        path_ids.add(path_id)
    if names != expected_names:
        raise SonicAssetError(f"TextAssets must be exactly {sorted(expected_names)}")

    config = _object(contract.get("config"), "config")
    joints = _array(config.get("joints"), "config.joints")
    if len(joints) != 29:
        raise SonicAssetError("config must contain exactly 29 joints")
    if sorted(config.get("isaaclab_to_mujoco", [])) != list(range(29)):
        raise SonicAssetError("isaaclab_to_mujoco must be a permutation of 0..28")
    if sorted(config.get("mujoco_to_isaaclab", [])) != list(range(29)):
        raise SonicAssetError("mujoco_to_isaaclab must be a permutation of 0..28")


def validate_build_files(game_root: Path, contract: Mapping[str, Any]) -> None:
    root = game_root.resolve(strict=True)
    for label, raw in _object(contract["files"], "files").items():
        item = _object(raw, f"files.{label}")
        path = root.joinpath(*_string(item["path"], "path").split("/"))
        if not path.is_file() or path.is_symlink():
            raise SonicAssetError(f"{label} is not an exact regular file: {path}")
        expected_size = _integer(item["bytes"], f"files.{label}.bytes", 1)
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise SonicAssetError(
                f"{label} byte count mismatch: expected={expected_size} actual={actual_size}"
            )
        expected_hash = _sha256(item["sha256"], f"files.{label}.sha256")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise SonicAssetError(
                f"{label} SHA-256 mismatch: expected={expected_hash} actual={actual_hash}"
            )


def load_text_assets(path: Path, parser_version: str) -> Mapping[int, tuple[str, bytes]]:
    try:
        import UnityPy
    except ImportError as exc:
        raise SonicAssetError(f"UnityPy {parser_version} is required") from exc
    actual_version = str(getattr(UnityPy, "__version__", ""))
    if actual_version != parser_version:
        raise SonicAssetError(
            f"UnityPy version mismatch: expected={parser_version} actual={actual_version!r}"
        )
    try:
        environment = UnityPy.load(str(path))
    except Exception as exc:
        raise SonicAssetError(f"UnityPy could not load {path}: {exc}") from exc
    loaded: dict[int, tuple[str, bytes]] = {}
    for obj in environment.objects:
        if obj.type.name != "TextAsset":
            continue
        try:
            value = obj.read()
            loaded[int(obj.path_id)] = (
                str(value.m_Name),
                _script_bytes(value.m_Script),
            )
        except Exception as exc:
            raise SonicAssetError(f"could not read TextAsset path ID {obj.path_id}: {exc}") from exc
    return loaded


def validate_text_assets(
    loaded: Mapping[int, tuple[str, bytes]], contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    observed: dict[str, Any] = {}
    config_payload: bytes | None = None
    for index, raw in enumerate(_array(contract["text_assets"], "text_assets")):
        spec = _object(raw, f"text_assets[{index}]")
        path_id = int(spec["path_id"])
        if path_id not in loaded:
            raise SonicAssetError(f"missing required TextAsset path ID {path_id}")
        actual_name, payload = loaded[path_id]
        expected_name = str(spec["name"])
        if actual_name != expected_name:
            raise SonicAssetError(
                f"TextAsset {path_id} name mismatch: expected={expected_name!r} actual={actual_name!r}"
            )
        if len(payload) != int(spec["bytes"]):
            raise SonicAssetError(f"TextAsset {path_id} byte count mismatch")
        digest = sha256_bytes(payload)
        if digest != spec["sha256"]:
            raise SonicAssetError(f"TextAsset {path_id} SHA-256 mismatch")
        observed[expected_name] = {
            "path_id": path_id,
            "bytes": len(payload),
            "sha256": digest,
        }
        if expected_name == "sonic_config":
            config_payload = payload

    if config_payload is None:
        raise SonicAssetError("sonic_config payload was not selected")
    try:
        parsed_config = json.loads(config_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SonicAssetError(f"sonic_config is not UTF-8 JSON: {exc}") from exc
    if parsed_config != contract["config"]:
        raise SonicAssetError("decoded sonic_config does not equal the tracked contract")
    return observed


def _plain_sequence(value: object) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise SonicAssetError(f"candidate value is not a sequence: {type(value).__name__}")
    return value


def _offsets(config: Mapping[str, Any], section: str) -> dict[str, tuple[int, int]]:
    cursor = 0
    result: dict[str, tuple[int, int]] = {}
    observations = _array(_object(config[section], section)["observations"], "observations")
    for raw in observations:
        item = _object(raw, "observation")
        if "dim" in item:
            width = int(item["dim"])
        else:
            width = int(item["dim_per_frame"]) * int(item["num_frames"])
        name = str(item["name"])
        result[name] = (cursor, cursor + width)
        cursor += width
    return result


def compare_candidate_constants(
    contract: Mapping[str, Any], candidate: object
) -> Mapping[str, Any]:
    """Compare every public semantic constant in gear_sonic_candidate.py.

    Array comparisons are exact and also report maximum absolute error.  This
    exposes six-decimal TextAsset rounding instead of silently hiding it behind
    a tolerance.
    """
    config = _object(contract["config"], "config")
    joints = _array(config["joints"], "config.joints")
    encoder_offsets = _offsets(config, "encoder")
    decoder_offsets = _offsets(config, "decoder")
    decoder_history = [
        item
        for item in _array(config["decoder"]["observations"], "decoder.observations")
        if item.get("type") == "history"
    ]
    history_frames = {int(item["num_frames"]) for item in decoder_history}
    if len(history_frames) != 1:
        raise SonicAssetError("decoder history observations disagree on frame count")
    motion_position = next(
        item
        for item in config["encoder"]["observations"]
        if item["name"].startswith("motion_joint_positions_10frame_step")
        and item.get("active_modes") == [0]
    )
    step_match = re.search(r"_step([0-9]+)\Z", motion_position["name"])
    if step_match is None:
        raise SonicAssetError("could not derive future step from encoder observation")

    native_defaults = contract["scope"]["native_isil_reference"]["defaults"]
    scalars = {
        "CONTROL_HZ": int(config["timing"]["controller_rate_hz"]),
        "CONTROL_DT": float(config["timing"]["physics_dt"]),
        "HISTORY_FRAMES": history_frames.pop(),
        "ENCODER_DIM": sum(end - start for start, end in encoder_offsets.values()),
        "DECODER_DIM": sum(end - start for start, end in decoder_offsets.values()),
        "TOKEN_DIM": int(config["encoder"]["token_dimension"]),
        "ACTION_DIM": len(joints),
        "ACTION_CLIP": float(config["action"]["clip_actions"]),
        "FUTURE_STEP": int(step_match.group(1)),
        "COMMAND_LPF_CUTOFF_HZ": float(native_defaults["command_lpf_cutoff_hz"]),
    }
    vectors = {
        "ISAACLAB_TO_MUJOCO": list(config["isaaclab_to_mujoco"]),
        "MUJOCO_TO_ISAACLAB": list(config["mujoco_to_isaaclab"]),
        "DEFAULT_ANGLES_MUJOCO": [float(item["default_pos"]) for item in joints],
        "PUBLIC_EFFORT_LIMIT_MUJOCO": [
            float(item["effort_limit"]) for item in joints
        ],
        "ACTION_SCALE_MUJOCO": [float(item["action_scale"]) for item in joints],
        "KP_MUJOCO": [float(item["kp"]) for item in joints],
        "KD_MUJOCO": [float(item["kd"]) for item in joints],
    }

    scalar_report: dict[str, Any] = {}
    for name, expected in scalars.items():
        actual = getattr(candidate, name)
        scalar_report[name] = {
            "exact": actual == expected,
            "contract": expected,
            "candidate": actual,
        }

    vector_report: dict[str, Any] = {}
    for name, expected in vectors.items():
        actual = _plain_sequence(getattr(candidate, name))
        if len(actual) != len(expected):
            raise SonicAssetError(
                f"candidate {name} length mismatch: expected={len(expected)} actual={len(actual)}"
            )
        mismatches = []
        max_abs = 0.0
        for index, (candidate_value, contract_value) in enumerate(zip(actual, expected)):
            equal = candidate_value == contract_value
            difference = None
            if isinstance(candidate_value, (int, float)) and isinstance(
                contract_value, (int, float)
            ):
                difference = float(candidate_value) - float(contract_value)
                if math.isfinite(difference):
                    max_abs = max(max_abs, abs(difference))
            if not equal:
                mismatch = {
                    "index": index,
                    "contract": contract_value,
                    "candidate": candidate_value,
                }
                if difference is not None:
                    mismatch["difference"] = difference
                if index < len(joints):
                    mismatch["joint"] = joints[index]["name"]
                mismatches.append(mismatch)
        vector_report[name] = {
            "exact": not mismatches,
            "mismatch_count": len(mismatches),
            "max_abs_difference": max_abs,
            "mismatches": mismatches,
        }

    expected_offset_maps = {
        "ENCODER_OFFSETS": encoder_offsets,
        "DECODER_OFFSETS": decoder_offsets,
    }
    offset_report = {}
    for name, expected in expected_offset_maps.items():
        actual = {
            str(key): tuple(value)
            for key, value in dict(getattr(candidate, name)).items()
        }
        offset_report[name] = {
            "exact": actual == expected,
            "contract": expected,
            "candidate": actual,
        }

    return {
        "scalars": scalar_report,
        "vectors": vector_report,
        "offsets": offset_report,
        "internal_only": ["RUN_SCHEMA", "TRACE_SCHEMA", "CLASSIFICATION"],
    }


def verify(game_root: Path, contract_path: Path = CONTRACT_PATH) -> Mapping[str, Any]:
    contract = load_contract(contract_path)
    validate_build_files(game_root, contract)
    source_rel = contract["files"]["source_container"]["path"]
    source = game_root.resolve().joinpath(*source_rel.split("/"))
    parser_version = str(contract["parser"]["version"])
    assets = validate_text_assets(load_text_assets(source, parser_version), contract)
    return {
        "schema": SUMMARY_SCHEMA,
        "verified": True,
        "contract": str(contract_path.resolve()),
        "game_root": str(game_root.resolve()),
        "text_assets": assets,
        "model_payloads_written": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    args = parser.parse_args(argv)
    try:
        result = verify(args.game_root, args.contract)
    except SonicAssetError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
