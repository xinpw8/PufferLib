"""Bind one validated public-family exact-batch model manifest into a C build.

The generated header contains hashes, byte counts, classification, and false
authority flags only. Model payloads and machine-local paths remain external.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

from gear_sonic_batch_model import (
    BatchRewriteError,
    BUILD_GATE_EQUIVALENCE_ATOL,
    BUILD_GATE_EQUIVALENCE_PROVIDERS,
    BUILD_GATE_EQUIVALENCE_SEED,
    MANIFEST_FILENAME,
    MODEL_SPECS,
    SCHEMA,
    inspect_explicit_batch_bundle,
    rewrite_model,
    validate_exact_equivalence_record,
    verify_equivalence,
)


CLASSIFICATION = "public_family_candidate"


class ModelIdentityBuildError(RuntimeError):
    """The supplied batch manifest cannot authorize a native model build."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ModelIdentityBuildError(f"{label} must be an object")
    return value


def _plain_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise ModelIdentityBuildError(f"{label} is unavailable") from exc
    if absolute.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ModelIdentityBuildError(f"{label} must be a non-symlink directory")
    return absolute.resolve(strict=True)


def _plain_file(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise ModelIdentityBuildError(f"{label} is unavailable") from exc
    if absolute.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ModelIdentityBuildError(f"{label} must be a regular, non-symlink file")
    return absolute.resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_record(
    inspection: Mapping[str, Any],
    source_bundle: Path,
    output_dir: Path,
    filename: str,
    batch_size: int,
) -> Mapping[str, Any]:
    models = _mapping(inspection.get("models"), "validated models")
    validated = _mapping(models.get(filename), f"validated {filename}")
    source = _plain_file(source_bundle / filename, f"source {filename}")
    output_name = f"{Path(filename).stem}.batch{batch_size}.onnx"
    output = _plain_file(output_dir / output_name, f"explicit-batch {filename}")
    source_sha256 = _sha256(source)
    output_sha256 = _sha256(output)
    if source_sha256 != MODEL_SPECS[filename]["sha256"]:
        raise ModelIdentityBuildError(f"{filename} differs from the pinned source model")
    if validated.get("source_sha256") != source_sha256:
        raise ModelIdentityBuildError(f"{filename} validated source hash changed")
    if validated.get("output_sha256") != output_sha256:
        raise ModelIdentityBuildError(f"{filename} validated output hash changed")
    output_bytes = output.stat().st_size
    if output_bytes < 1:
        raise ModelIdentityBuildError(f"{filename} validated output is empty")
    return {
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
        "output_bytes": output_bytes,
    }


def _independent_model_gate(
    source_bundle: Path,
    output_dir: Path,
    filename: str,
    batch_size: int,
) -> Mapping[str, Any]:
    source = _plain_file(source_bundle / filename, f"source {filename}")
    output_name = f"{Path(filename).stem}.batch{batch_size}.onnx"
    output = _plain_file(output_dir / output_name, f"explicit-batch {filename}")
    source_sha256 = _sha256(source)
    output_sha256 = _sha256(output)

    try:
        with tempfile.TemporaryDirectory(prefix="rek-g1-model-gate-") as temporary:
            regenerated = Path(temporary) / output_name
            rewrite = rewrite_model(source, regenerated, batch_size)
            regenerated = _plain_file(
                regenerated,
                f"independently rewritten {filename}",
            )
            regenerated_sha256 = _sha256(regenerated)
            if rewrite.get("output_sha256") != regenerated_sha256:
                raise ModelIdentityBuildError(
                    f"{filename} independent rewrite reported a different hash"
                )
            if regenerated_sha256 != output_sha256:
                raise ModelIdentityBuildError(
                    f"{filename} differs from the independent dimension-only rewrite"
                )
            equivalence = verify_equivalence(
                source,
                output,
                batch_size,
                providers=BUILD_GATE_EQUIVALENCE_PROVIDERS,
                seed=BUILD_GATE_EQUIVALENCE_SEED,
                atol=BUILD_GATE_EQUIVALENCE_ATOL,
            )
            equivalence = validate_exact_equivalence_record(
                filename,
                equivalence,
                batch_size,
                expected_providers=BUILD_GATE_EQUIVALENCE_PROVIDERS,
                expected_seed=BUILD_GATE_EQUIVALENCE_SEED,
            )
    except BatchRewriteError as exc:
        raise ModelIdentityBuildError(
            f"{filename} independent model gate failed: {exc}"
        ) from exc

    if _sha256(source) != source_sha256 or _sha256(output) != output_sha256:
        raise ModelIdentityBuildError(
            f"{filename} changed during independent model validation"
        )
    return {
        "independent_rewrite_sha256": regenerated_sha256,
        "equivalence": equivalence,
    }


def _render_header(record: Mapping[str, Any]) -> str:
    encoder = _mapping(record["encoder"], "encoder")
    decoder = _mapping(record["decoder"], "decoder")
    return f"""/* Generated after independent rewrite and inference validation. Do not edit. */
#ifndef REK_G1_MODEL_IDENTITY_GENERATED_H
#define REK_G1_MODEL_IDENTITY_GENERATED_H

#include <stddef.h>
#include <stdint.h>

#define REK_G1_GENERATED_MODEL_SCHEMA \"{SCHEMA}\"
#define REK_G1_GENERATED_MODEL_CLASSIFICATION \"{CLASSIFICATION}\"
#define REK_G1_GENERATED_MODEL_MANIFEST_SHA256 \"{record['manifest_sha256']}\"
#define REK_G1_GENERATED_MODEL_ROBOT_BATCH ((size_t){record['batch_size']}u)
#define REK_G1_GENERATED_MODEL_REK_PARITY_CLAIM 0u
#define REK_G1_GENERATED_MODEL_CURRENT_STEAM_AUTHORITY 0u
#define REK_G1_GENERATED_MODEL_TRAINING_ENABLED 0u
#define REK_G1_GENERATED_ENCODER_SOURCE_SHA256 \"{encoder['source_sha256']}\"
#define REK_G1_GENERATED_ENCODER_OUTPUT_SHA256 \"{encoder['output_sha256']}\"
#define REK_G1_GENERATED_ENCODER_OUTPUT_BYTES UINT64_C({encoder['output_bytes']})
#define REK_G1_GENERATED_DECODER_SOURCE_SHA256 \"{decoder['source_sha256']}\"
#define REK_G1_GENERATED_DECODER_OUTPUT_SHA256 \"{decoder['output_sha256']}\"
#define REK_G1_GENERATED_DECODER_OUTPUT_BYTES UINT64_C({decoder['output_bytes']})

#endif
"""


def generate_header(
    source_bundle: Path,
    manifest_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    source_bundle = _plain_directory(source_bundle, "public model source bundle")
    manifest_path = _plain_file(manifest_path, "explicit-batch manifest")
    if manifest_path.name != MANIFEST_FILENAME:
        raise ModelIdentityBuildError(
            f"manifest filename must be {MANIFEST_FILENAME}"
        )
    output_dir = _plain_directory(manifest_path.parent, "explicit-batch directory")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = _mapping(
            json.loads(manifest_bytes.decode("utf-8")),
            "explicit-batch manifest",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelIdentityBuildError("manifest must be valid UTF-8 JSON") from exc
    batch_size = manifest.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ModelIdentityBuildError("manifest batch_size must be an integer")
    if batch_size < 2 or batch_size % 2 != 0:
        raise ModelIdentityBuildError("manifest batch_size must be a positive fighter pair count")
    if manifest.get("schema") != SCHEMA:
        raise ModelIdentityBuildError("manifest schema mismatch")
    if manifest.get("classification") != CLASSIFICATION:
        raise ModelIdentityBuildError("manifest classification mismatch")
    if manifest.get("rek_parity_claim") is not False:
        raise ModelIdentityBuildError("manifest must not claim REK parity")

    try:
        inspection = inspect_explicit_batch_bundle(
            source_bundle,
            output_dir,
            batch_size,
        )
    except BatchRewriteError as exc:
        raise ModelIdentityBuildError(str(exc)) from exc
    if Path(str(inspection.get("manifest_path"))).resolve(strict=True) != manifest_path:
        raise ModelIdentityBuildError("validator inspected a different manifest")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if inspection.get("manifest_sha256") != manifest_sha256:
        raise ModelIdentityBuildError("manifest changed while it was being validated")

    encoder = dict(
        _model_record(
            inspection,
            source_bundle,
            output_dir,
            "model_encoder.onnx",
            batch_size,
        )
    )
    encoder.update(
        _independent_model_gate(
            source_bundle,
            output_dir,
            "model_encoder.onnx",
            batch_size,
        )
    )
    decoder = dict(
        _model_record(
            inspection,
            source_bundle,
            output_dir,
            "model_decoder.onnx",
            batch_size,
        )
    )
    decoder.update(
        _independent_model_gate(
            source_bundle,
            output_dir,
            "model_decoder.onnx",
            batch_size,
        )
    )

    record = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "rek_parity_claim": False,
        "current_steam_authority": False,
        "training_enabled": False,
        "manifest_sha256": manifest_sha256,
        "batch_size": batch_size,
        "encoder": encoder,
        "decoder": decoder,
    }

    output_path = Path(os.path.abspath(os.fspath(output_path)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            stream.write(_render_header(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = generate_header(args.source_bundle, args.manifest, args.out)
    except ModelIdentityBuildError as exc:
        raise SystemExit(f"model identity build rejected: {exc}") from exc
    print(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
