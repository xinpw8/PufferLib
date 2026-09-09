"""Rewrite the pinned GEAR-SONIC graphs for one explicit vector batch size.

The published encoder bakes batch size one into Constant-fed Reshape nodes.
The decoder operations are already batch-safe, but its graph input and output
metadata are also fixed to one.  This tool changes only those batch axes,
checks the resulting graph, and compares every output row with an invocation
of the original batch-one model.

Generated ONNX files contain model weights and must remain outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "rek.g1_gear_sonic_explicit_batch.v1"
MANIFEST_FILENAME = "explicit_batch_manifest.json"
BUILD_GATE_EQUIVALENCE_PROVIDERS = ("CPUExecutionProvider",)
BUILD_GATE_EQUIVALENCE_SEED = 20260831
BUILD_GATE_EQUIVALENCE_ATOL = 0.0

MODEL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "model_encoder.onnx": {
        "sha256": "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3",
        "input": "obs_dict",
        "input_width": 1762,
        "output": "encoded_tokens",
        "output_width": 64,
    },
    "model_decoder.onnx": {
        "sha256": "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed",
        "input": "obs_dict",
        "input_width": 994,
        "output": "action",
        "output_width": 29,
    },
}

# Every fixed leading batch axis observed in the pinned encoder.  Requiring
# this exact table prevents a different graph revision from being rewritten by
# a filename-based heuristic.
ENCODER_RESHAPES: Mapping[str, tuple[int, ...]] = {
    "/Reshape_1": (1, 1, 10, 58),
    "/Reshape_2": (1, 1, 6),
    "/Reshape_3": (1, 1, 10, 6),
    "/Reshape_4": (1, 1, 240),
    "/Reshape_5": (1, 1, 9),
    "/Reshape_6": (1, 1, 12),
    "/Reshape_7": (1, 1, 10, 72),
    "/Reshape_8": (1, 1, 10, 6),
    "/Reshape_9": (1, 1, 10, 6),
    "/g1/Reshape": (1, 640),
    "/g1/Reshape_1": (1, 2, 32),
    "/quantizer/Reshape": (1, 2, 1, 32),
    "/quantizer/Reshape_1": (1, 2, 32),
    "/teleop/Reshape": (1, 2, 32),
    "/quantizer_1/Reshape": (1, 2, 1, 32),
    "/quantizer_1/Reshape_1": (1, 2, 32),
    "/smpl/Reshape": (1, 840),
    "/smpl/Reshape_1": (1, 2, 32),
    "/quantizer_2/Reshape": (1, 2, 1, 32),
    "/quantizer_2/Reshape_1": (1, 2, 32),
}


class BatchRewriteError(RuntimeError):
    """The source graph or equivalence check violated the pinned contract."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BatchRewriteError(f"{label} must be an object")
    return value


def _lowercase_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise BatchRewriteError(f"{label} is unavailable") from exc
    if absolute.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise BatchRewriteError(f"{label} must be a regular, non-symlink file")
    return absolute.resolve(strict=True)


def _plain_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise BatchRewriteError(f"{label} is unavailable") from exc
    if absolute.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise BatchRewriteError(f"{label} must be a regular, non-symlink directory")
    return absolute.resolve(strict=True)


def _write_new_text(path: Path, text: str, label: str) -> Path:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            resolved,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as exc:
        raise BatchRewriteError(f"refusing to overwrite {label} {resolved}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    except Exception:
        try:
            resolved.unlink()
        except OSError:
            pass
        raise
    return resolved


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def validate_exact_equivalence_record(
    filename: str,
    equivalence: Mapping[str, Any],
    batch_size: int,
    *,
    expected_providers: Sequence[str] | None = None,
    expected_seed: int | None = None,
) -> Mapping[str, Any]:
    """Require the exact binary32 equivalence contract used by native builds."""

    if filename not in MODEL_SPECS:
        raise BatchRewriteError(f"unsupported model filename {filename!r}")
    equivalence = _mapping(equivalence, f"models.{filename}.equivalence")
    expected_probe = (
        "g1_mode_zero_with_only_active_encoder_channels_nonzero"
        if filename == "model_encoder.onnx"
        else "finite_distinct_decoder_rows"
    )
    if equivalence.get("probe_contract") != expected_probe:
        raise BatchRewriteError(f"{filename} equivalence probe mismatch")

    providers = equivalence.get("providers")
    if (
        not isinstance(providers, list)
        or not providers
        or any(not isinstance(provider, str) or not provider for provider in providers)
    ):
        raise BatchRewriteError(f"{filename} equivalence providers are invalid")
    seed = equivalence.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BatchRewriteError(f"{filename} equivalence seed is invalid")
    if expected_providers is not None and tuple(providers) != tuple(expected_providers):
        raise BatchRewriteError(f"{filename} equivalence providers differ")
    if expected_seed is not None and seed != expected_seed:
        raise BatchRewriteError(f"{filename} equivalence seed differs")

    tolerance = equivalence.get("absolute_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance != 0.0
    ):
        raise BatchRewriteError(f"{filename} equivalence tolerance must be zero")
    difference = equivalence.get("max_absolute_difference")
    if (
        isinstance(difference, bool)
        or not isinstance(difference, (int, float))
        or not math.isfinite(difference)
        or difference != 0.0
    ):
        raise BatchRewriteError(f"{filename} equivalence difference must be zero")
    if equivalence.get("exact_float32_equal") is not True:
        raise BatchRewriteError(f"{filename} equivalence is not exact float32")

    output_diversity = equivalence.get("minimum_pairwise_output_linf")
    if batch_size > 1 and (
        isinstance(output_diversity, bool)
        or not isinstance(output_diversity, (int, float))
        or not math.isfinite(output_diversity)
        or output_diversity <= 0.0
    ):
        raise BatchRewriteError(
            f"{filename} equivalence output rows are not distinct"
        )
    expected_hash = equivalence.get("expected_sha256_float32_le")
    actual_hash = equivalence.get("actual_sha256_float32_le")
    if not _lowercase_sha256(expected_hash) or not _lowercase_sha256(actual_hash):
        raise BatchRewriteError(f"{filename} equivalence tensor hash is invalid")
    if expected_hash != actual_hash:
        raise BatchRewriteError(f"{filename} equivalence tensor hashes differ")
    return dict(equivalence)


def _tensor_shape(value: Any) -> tuple[int | str, ...]:
    result: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(str(dimension.dim_param))
        else:
            result.append("")
    return tuple(result)


def _set_first_dimension(value: Any, batch_size: int) -> None:
    dimensions = value.type.tensor_type.shape.dim
    if len(dimensions) != 2:
        raise BatchRewriteError(f"tensor {value.name!r} must be rank two")
    dimensions[0].ClearField("dim_param")
    dimensions[0].dim_value = batch_size


def rewrite_model(source: Path, target: Path, batch_size: int) -> Mapping[str, Any]:
    """Write one exact-batch graph after validating its pinned source identity."""

    if batch_size < 1:
        raise BatchRewriteError("batch size must be positive")
    source = _regular_file(source, "source ONNX model")
    if source.name not in MODEL_SPECS:
        raise BatchRewriteError(f"unsupported model filename {source.name!r}")
    spec = MODEL_SPECS[source.name]
    source_hash = sha256_file(source)
    if source_hash != spec["sha256"]:
        raise BatchRewriteError(f"{source.name} SHA-256 mismatch")
    target = target.resolve()
    if target.exists():
        raise BatchRewriteError(f"refusing to overwrite output model {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise BatchRewriteError("onnx is required") from exc

    model = onnx.load(source)
    graph_inputs = {value.name: value for value in model.graph.input}
    graph_outputs = {value.name: value for value in model.graph.output}
    input_value = graph_inputs.get(str(spec["input"]))
    output_value = graph_outputs.get(str(spec["output"]))
    if input_value is None or output_value is None:
        raise BatchRewriteError("pinned graph input or output is missing")
    if _tensor_shape(input_value) != (1, int(spec["input_width"])):
        raise BatchRewriteError("pinned graph input shape mismatch")
    if _tensor_shape(output_value) != (1, int(spec["output_width"])):
        raise BatchRewriteError("pinned graph output shape mismatch")

    producers = {
        output: node for node in model.graph.node for output in node.output
    }
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    observed_fixed: dict[str, tuple[int, ...]] = {}
    mutations: list[dict[str, Any]] = []
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape_name = node.input[1]
        tensor = initializers.get(shape_name)
        owner = "initializer"
        if tensor is None:
            producer = producers.get(shape_name)
            if producer is None or producer.op_type != "Constant":
                continue
            tensor_attributes = [
                attribute
                for attribute in producer.attribute
                if attribute.type == onnx.AttributeProto.TENSOR
            ]
            if len(tensor_attributes) != 1:
                raise BatchRewriteError(
                    f"Reshape {node.name!r} does not have one tensor Constant"
                )
            tensor = tensor_attributes[0].t
            owner = "constant"
        array = numpy_helper.to_array(tensor)
        if array.ndim != 1 or array.size == 0 or int(array[0]) != 1:
            continue
        old_shape = tuple(int(value) for value in array.tolist())
        observed_fixed[node.name] = old_shape
        rewritten = array.copy()
        rewritten[0] = batch_size
        replacement = numpy_helper.from_array(
            rewritten.astype(array.dtype, copy=False), name=tensor.name
        )
        tensor.CopyFrom(replacement)
        mutations.append(
            {
                "reshape": node.name,
                "owner": owner,
                "old": list(old_shape),
                "new": [int(value) for value in rewritten.tolist()],
            }
        )

    expected_fixed = ENCODER_RESHAPES if source.name == "model_encoder.onnx" else {}
    if observed_fixed != expected_fixed:
        raise BatchRewriteError(
            "fixed Reshape table differs from the pinned model contract"
        )
    _set_first_dimension(input_value, batch_size)
    _set_first_dimension(output_value, batch_size)
    try:
        onnx.checker.check_model(model)
        onnx.save(model, target)
    except Exception as exc:
        if target.exists():
            target.unlink()
        raise BatchRewriteError(f"rewritten ONNX validation failed: {exc}") from exc

    return {
        "source_path": str(source),
        "source_sha256": source_hash,
        "output_path": str(target),
        "output_file": target.name,
        "output_sha256": sha256_file(target),
        "mutations": mutations,
    }


def verify_equivalence(
    source: Path,
    target: Path,
    batch_size: int,
    *,
    providers: Sequence[str],
    seed: int,
    atol: float,
) -> Mapping[str, Any]:
    """Compare each vector output row against the original batch-one graph."""

    if not providers:
        raise BatchRewriteError("at least one execution provider is required")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise BatchRewriteError("onnxruntime is required") from exc
    spec = MODEL_SPECS[source.name]
    baseline = ort.InferenceSession(str(source), providers=list(providers))
    vector = ort.InferenceSession(str(target), providers=list(providers))
    rng = np.random.default_rng(seed)
    if source.name == "model_encoder.onnx":
        # The environment uses only G1 encoder mode zero.  Other mode branches
        # have different fixed-shape scatter logic and are outside this tool's
        # contract. The four-value mode field stores scalar mode ID zero plus
        # zero padding for G1; it is not one-hot. Exercise every active G1
        # channel with distinct row data.
        observation = np.zeros(
            (batch_size, int(spec["input_width"])), dtype=np.float32
        )
        observation[:, 4:584] = rng.standard_normal(
            (batch_size, 580), dtype=np.float32
        )
        observation[:, 601:661] = rng.standard_normal(
            (batch_size, 60), dtype=np.float32
        )
        probe_contract = "g1_mode_zero_with_only_active_encoder_channels_nonzero"
    else:
        observation = rng.standard_normal(
            (batch_size, int(spec["input_width"])), dtype=np.float32
        )
        probe_contract = "finite_distinct_decoder_rows"
    expected = np.concatenate(
        [
            baseline.run(
                [str(spec["output"])],
                {str(spec["input"]): observation[index : index + 1]},
            )[0]
            for index in range(batch_size)
        ],
        axis=0,
    )
    actual = vector.run(
        [str(spec["output"])], {str(spec["input"]): observation}
    )[0]
    expected_shape = (batch_size, int(spec["output_width"]))
    if actual.shape != expected_shape or expected.shape != expected_shape:
        raise BatchRewriteError("equivalence output shape mismatch")
    if actual.dtype != np.float32 or expected.dtype != np.float32:
        raise BatchRewriteError("equivalence output dtype must be float32")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise BatchRewriteError("equivalence output contains non-finite values")
    max_abs = float(np.max(np.abs(actual.astype(np.float64) - expected)))
    if max_abs > atol:
        raise BatchRewriteError(
            f"batched output differs from batch-one baseline: {max_abs} > {atol}"
        )
    pairwise_output_linf = [
        float(
            np.max(
                np.abs(
                    actual[left].astype(np.float64)
                    - actual[right].astype(np.float64)
                )
            )
        )
        for left in range(batch_size)
        for right in range(left + 1, batch_size)
    ]
    minimum_pairwise_output_linf = (
        min(pairwise_output_linf) if pairwise_output_linf else None
    )
    if (
        minimum_pairwise_output_linf is not None
        and minimum_pairwise_output_linf <= 0.0
    ):
        raise BatchRewriteError(
            "equivalence probe produced indistinguishable output rows"
        )
    expected_sha256 = hashlib.sha256(
        np.ascontiguousarray(expected, dtype="<f4").tobytes()
    ).hexdigest()
    actual_sha256 = hashlib.sha256(
        np.ascontiguousarray(actual, dtype="<f4").tobytes()
    ).hexdigest()
    report = {
        "providers": list(providers),
        "seed": seed,
        "probe_contract": probe_contract,
        "absolute_tolerance": atol,
        "max_absolute_difference": max_abs,
        "minimum_pairwise_output_linf": minimum_pairwise_output_linf,
        "exact_float32_equal": bool(np.array_equal(actual, expected)),
        "expected_sha256_float32_le": expected_sha256,
        "actual_sha256_float32_le": actual_sha256,
    }
    if atol == 0.0:
        validate_exact_equivalence_record(
            source.name,
            report,
            batch_size,
            expected_providers=providers,
            expected_seed=seed,
        )
    return report


def inspect_explicit_batch_bundle(
    source_bundle: Path,
    output_dir: Path,
    batch_size: int,
) -> Mapping[str, Any]:
    """Validate a generated exact-N bundle without trusting stored paths."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise BatchRewriteError("batch size must be a positive integer")
    source_bundle = _plain_directory(source_bundle, "source bundle")
    output_dir = _plain_directory(output_dir, "batch output")

    manifest_path = _regular_file(
        output_dir / MANIFEST_FILENAME, "explicit-batch manifest"
    )
    try:
        report = _mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            "explicit-batch manifest",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchRewriteError("explicit-batch manifest is not valid UTF-8 JSON") from exc
    if report.get("schema") != SCHEMA:
        raise BatchRewriteError("explicit-batch manifest schema mismatch")
    if report.get("classification") != "public_family_candidate":
        raise BatchRewriteError("explicit-batch classification mismatch")
    if report.get("rek_parity_claim") is not False:
        raise BatchRewriteError("explicit-batch manifest must not claim REK parity")
    if report.get("batch_size") != batch_size:
        raise BatchRewriteError("explicit-batch size differs from the requested environment count")

    models = _mapping(report.get("models"), "explicit-batch models")
    if set(models) != set(MODEL_SPECS):
        raise BatchRewriteError("explicit-batch model set mismatch")
    validated_models: dict[str, Mapping[str, Any]] = {}
    for filename, spec in MODEL_SPECS.items():
        record = _mapping(models.get(filename), f"models.{filename}")
        source = _regular_file(source_bundle / filename, f"source {filename}")
        source_hash = sha256_file(source)
        if source_hash != spec["sha256"] or record.get("source_sha256") != source_hash:
            raise BatchRewriteError(f"{filename} source identity mismatch")

        expected_output_file = f"{Path(filename).stem}.batch{batch_size}.onnx"
        if record.get("output_file") != expected_output_file:
            raise BatchRewriteError(f"{filename} output filename mismatch")
        output = _regular_file(
            output_dir / expected_output_file, f"explicit-batch {filename}"
        )
        output_hash = sha256_file(output)
        if record.get("output_sha256") != output_hash:
            raise BatchRewriteError(f"{filename} output identity mismatch")

        equivalence = validate_exact_equivalence_record(
            filename,
            _mapping(record.get("equivalence"), f"models.{filename}.equivalence"),
            batch_size,
        )

        validated_models[filename] = {
            "source_path": str(source),
            "source_sha256": source_hash,
            "output_path": str(output),
            "output_sha256": output_hash,
            "equivalence": dict(equivalence),
        }

    return {
        "schema": SCHEMA,
        "classification": "public_family_candidate",
        "rek_parity_claim": False,
        "batch_size": batch_size,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "models": validated_models,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=_positive_int)
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="comma-separated ONNX Runtime providers used for equivalence",
    )
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--atol", type=_nonnegative_float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = _plain_directory(args.bundle, "bundle")
    providers = tuple(item for item in args.providers.split(",") if item)
    models: dict[str, Any] = {}
    for filename in MODEL_SPECS:
        source = bundle / filename
        target = args.output_dir / f"{Path(filename).stem}.batch{args.batch_size}.onnx"
        rewrite = rewrite_model(source, target, args.batch_size)
        rewrite["equivalence"] = verify_equivalence(
            source,
            target,
            args.batch_size,
            providers=providers,
            seed=args.seed,
            atol=args.atol,
        )
        models[filename] = rewrite
    report = {
        "schema": SCHEMA,
        "classification": "public_family_candidate",
        "rek_parity_claim": False,
        "batch_size": args.batch_size,
        "models": models,
        "limits": [
            "This proves graph-rewrite equivalence only for the recorded provider and probe input.",
            "It does not establish identity with the current REK server model or trajectory parity.",
        ],
    }
    text = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    _write_new_text(args.output_dir / MANIFEST_FILENAME, text + "\n", "batch manifest")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
