"""Validate and inspect the pinned public NVIDIA GEAR-SONIC G1 bundle.

This module identifies the public controller family recovered from the pinned
REK build.  It does not claim that the public model bytes are identical to the
model bytes packaged by REK; that comparison requires a successful live asset
capture from the installed build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CLASSIFICATION = "public_family_candidate"
SOURCE_REPOSITORY = "https://github.com/NVlabs/GR00T-WholeBodyControl"
SOURCE_COMMIT = "087f9ac01d46f6d8e4d0b73c01ae64799f292a38"
MODEL_REPOSITORY = "https://huggingface.co/nvidia/GEAR-SONIC"
MODEL_REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"

EXPECTED_FILES = {
    "config.json": "326dcc509ad93509cf262871256a74f508f43a8430f61ec6abe8034e985ce948",
    "LICENSE": "24ab66be50d1aca4fc5e029ef76ce4ceaac6557ea21665caf4b140695a76ffee",
    "model_decoder.onnx": "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed",
    "model_encoder.onnx": "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3",
    "observation_config.yaml": "466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b",
}

DECODER_OBSERVATIONS = (
    "token_state",
    "his_base_angular_velocity_10frame_step1",
    "his_body_joint_positions_10frame_step1",
    "his_body_joint_velocities_10frame_step1",
    "his_last_actions_10frame_step1",
    "his_gravity_dir_10frame_step1",
)

ENCODER_OBSERVATIONS = (
    "encoder_mode_4",
    "motion_joint_positions_10frame_step5",
    "motion_joint_velocities_10frame_step5",
    "motion_root_z_position_10frame_step5",
    "motion_root_z_position",
    "motion_anchor_orientation",
    "motion_anchor_orientation_10frame_step5",
    "motion_joint_positions_lowerbody_10frame_step5",
    "motion_joint_velocities_lowerbody_10frame_step5",
    "vr_3point_local_target",
    "vr_3point_local_orn_target",
    "smpl_joints_10frame_step1",
    "smpl_anchor_orientation_10frame_step1",
    "motion_joint_positions_wrists_10frame_step1",
)

G1_ENCODER_OBSERVATIONS = (
    "encoder_mode_4",
    "motion_joint_positions_10frame_step5",
    "motion_joint_velocities_10frame_step5",
    "motion_anchor_orientation_10frame_step5",
)

EXPECTED_ENCODER_GRAPH = {
    "inputs": [{"name": "obs_dict", "shape": [1, 1762], "type": "tensor(float)"}],
    "outputs": [
        {"name": "encoded_tokens", "shape": [1, 64], "type": "tensor(float)"}
    ],
}

EXPECTED_DECODER_GRAPH = {
    "inputs": [{"name": "obs_dict", "shape": [1, 994], "type": "tensor(float)"}],
    "outputs": [{"name": "action", "shape": [1, 29], "type": "tensor(float)"}],
}


class BundleError(RuntimeError):
    """The supplied public controller bundle does not match the pin."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(path))
    if path.is_symlink() or not path.is_file():
        raise BundleError(f"{label} must be a regular, non-symlink file")
    return path


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise BundleError(f"{label} must be an array")
    return value


def _enabled_names(value: Any, label: str) -> tuple[str, ...]:
    names: list[str] = []
    for index, raw in enumerate(_sequence(value, label)):
        entry = _mapping(raw, f"{label}[{index}]")
        if entry.get("enabled") is True:
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                raise BundleError(f"{label}[{index}].name must be a nonempty string")
            names.append(name)
    return tuple(names)


def validate_observation_config(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise BundleError("PyYAML is required to validate observation_config.yaml") from exc

    with path.open("r", encoding="utf-8") as stream:
        root = _mapping(yaml.safe_load(stream), "observation config")

    decoder_names = _enabled_names(root.get("observations"), "observations")
    if decoder_names != DECODER_OBSERVATIONS:
        raise BundleError("decoder observation order does not match the pinned default release")

    encoder = _mapping(root.get("encoder"), "encoder")
    if encoder.get("dimension") != 64:
        raise BundleError("encoder dimension must be 64")
    encoder_names = _enabled_names(
        encoder.get("encoder_observations"), "encoder.encoder_observations"
    )
    if encoder_names != ENCODER_OBSERVATIONS:
        raise BundleError("encoder observation order does not match the pinned default release")

    modes = _sequence(encoder.get("encoder_modes"), "encoder.encoder_modes")
    g1_mode = None
    for index, raw in enumerate(modes):
        entry = _mapping(raw, f"encoder.encoder_modes[{index}]")
        if entry.get("name") == "g1":
            g1_mode = entry
            break
    if g1_mode is None:
        raise BundleError("g1 encoder mode is missing")
    if g1_mode.get("mode_id") != 0:
        raise BundleError("g1 encoder mode id must be zero")
    required = tuple(_sequence(g1_mode.get("required_observations"), "g1.required_observations"))
    if required != G1_ENCODER_OBSERVATIONS:
        raise BundleError("g1 encoder observation selection does not match the pinned release")

    return {
        "decoder_observations": list(decoder_names),
        "encoder_dimension": 64,
        "encoder_observations": list(encoder_names),
        "g1_mode_id": 0,
        "g1_encoder_observations": list(required),
    }


def _ort_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_onnx(path: Path) -> tuple[Any, Mapping[str, Any]]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise BundleError("onnxruntime is required to inspect the controller graphs") from exc

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def pack(values: Sequence[Any]) -> list[Mapping[str, Any]]:
        return [
            {
                "name": value.name,
                "shape": [_ort_value(dimension) for dimension in value.shape],
                "type": value.type,
            }
            for value in values
        ]

    return session, {
        "providers": session.get_providers(),
        "inputs": pack(session.get_inputs()),
        "outputs": pack(session.get_outputs()),
    }


def _validate_graph(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for field in ("inputs", "outputs"):
        if actual.get(field) != expected[field]:
            raise BundleError(f"{label} {field} do not match the pinned public graph")


def _tensor_summary(value: np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(value, dtype=np.float32)
    return {
        "shape": list(array.shape),
        "finite": bool(np.isfinite(array).all()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "sha256_float32_le": hashlib.sha256(array.astype("<f4", copy=False).tobytes()).hexdigest(),
    }


def smoke_inference(encoder_session: Any, decoder_session: Any) -> Mapping[str, Any]:
    encoder_input = np.zeros((1, 1762), dtype=np.float32)
    encoded_tokens = encoder_session.run(
        ["encoded_tokens"], {"obs_dict": encoder_input}
    )[0]
    if encoded_tokens.shape != (1, 64) or not np.isfinite(encoded_tokens).all():
        raise BundleError("encoder zero-input smoke output is invalid")

    decoder_input = np.zeros((1, 994), dtype=np.float32)
    decoder_input[:, :64] = encoded_tokens
    action = decoder_session.run(["action"], {"obs_dict": decoder_input})[0]
    if action.shape != (1, 29) or not np.isfinite(action).all():
        raise BundleError("decoder zero-history smoke output is invalid")

    return {
        "input_contract": "zero G1 encoder input and zero decoder history",
        "encoded_tokens": _tensor_summary(encoded_tokens),
        "action": _tensor_summary(action),
    }


def inspect_bundle(bundle: Path, *, run_smoke: bool = False) -> Mapping[str, Any]:
    bundle = Path(os.path.abspath(bundle))
    if bundle.is_symlink() or not bundle.is_dir():
        raise BundleError("bundle must be a regular directory")

    files: dict[str, Mapping[str, Any]] = {}
    resolved: dict[str, Path] = {}
    for name, expected_hash in EXPECTED_FILES.items():
        path = _regular_file(bundle / name, name)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise BundleError(f"{name} SHA-256 mismatch")
        resolved[name] = path
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_hash,
        }

    encoder_session, encoder_graph = load_onnx(resolved["model_encoder.onnx"])
    decoder_session, decoder_graph = load_onnx(resolved["model_decoder.onnx"])
    _validate_graph(encoder_graph, EXPECTED_ENCODER_GRAPH, "encoder")
    _validate_graph(decoder_graph, EXPECTED_DECODER_GRAPH, "decoder")

    result = {
        "schema": "rek.g1_gear_sonic_public_bundle.v1",
        "classification": CLASSIFICATION,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "rek_weight_identity": {
            "status": "unknown",
            "reason": "live REK encoder and decoder payload hashes have not been captured",
        },
        "files": files,
        "observation_config": validate_observation_config(
            resolved["observation_config.yaml"]
        ),
        "encoder": encoder_graph,
        "decoder": decoder_graph,
    }
    if run_smoke:
        result["smoke_inference"] = smoke_inference(encoder_session, decoder_session)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        result = inspect_bundle(Path(arguments.bundle), run_smoke=arguments.smoke)
    except (BundleError, OSError, ValueError) as exc:
        print(f"GEAR-SONIC bundle validation failed: {exc}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
