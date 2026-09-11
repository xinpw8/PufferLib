#!/usr/bin/env python3
"""Compare native Gear-Sonic ONNX outputs with Python ONNX Runtime exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


ENCODER_INPUT_WIDTH = 1762
ENCODER_OUTPUT_WIDTH = 64
DECODER_INPUT_WIDTH = 994
DECODER_OUTPUT_WIDTH = 29


def _probe_values(batch_size: int, start: int, stop: int, salt: int) -> np.ndarray:
    rows = np.arange(batch_size, dtype=np.uint64)[:, None]
    columns = np.arange(start, stop, dtype=np.uint64)[None, :]
    values = (rows * 131 + columns * 17 + np.uint64(salt * 29)) % 2003
    return (values.astype(np.float32) - np.float32(1001.0)) / np.float32(317.0)


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _sha256_float32_le(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def _comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual_bytes = actual.astype("<f4", copy=False).tobytes(order="C")
    expected_bytes = expected.astype("<f4", copy=False).tobytes(order="C")
    exact = actual.shape == expected.shape and actual_bytes == expected_bytes
    numerically_equal = bool(np.array_equal(actual, expected))
    maximum = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    return {
        "array_equal": numerically_equal,
        "bitwise_equal": exact,
        "max_absolute_difference": maximum,
        "actual_sha256_float32_le": hashlib.sha256(actual_bytes).hexdigest(),
        "expected_sha256_float32_le": hashlib.sha256(expected_bytes).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    encoder_input = np.zeros(
        (args.batch_size, ENCODER_INPUT_WIDTH), dtype=np.float32
    )
    encoder_input[:, 0] = np.float32(1.0)
    encoder_input[:, 4:584] = _probe_values(args.batch_size, 4, 584, 1)
    encoder_input[:, 601:661] = _probe_values(args.batch_size, 601, 661, 2)
    decoder_input = _probe_values(args.batch_size, 0, DECODER_INPUT_WIDTH, 3)

    encoder = _session(args.encoder)
    decoder = _session(args.decoder)
    expected_tokens = np.asarray(
        encoder.run(["encoded_tokens"], {"obs_dict": encoder_input})[0],
        dtype=np.float32,
    )
    expected_actions = np.asarray(
        decoder.run(["action"], {"obs_dict": decoder_input})[0], dtype=np.float32
    )
    actual_tokens = np.fromfile(args.tokens, dtype="<f4")
    actual_actions = np.fromfile(args.actions, dtype="<f4")
    expected_token_shape = (args.batch_size, ENCODER_OUTPUT_WIDTH)
    expected_action_shape = (args.batch_size, DECODER_OUTPUT_WIDTH)
    if actual_tokens.size != np.prod(expected_token_shape, dtype=np.int64):
        raise ValueError("native token file size does not match the batch contract")
    if actual_actions.size != np.prod(expected_action_shape, dtype=np.int64):
        raise ValueError("native action file size does not match the batch contract")
    actual_tokens = actual_tokens.reshape(expected_token_shape)
    actual_actions = actual_actions.reshape(expected_action_shape)

    result = {
        "schema": "rek.g1_gear_sonic_native_ort_equivalence.v1",
        "batch_size": args.batch_size,
        "onnxruntime": ort.__version__,
        "providers": ["CPUExecutionProvider"],
        "encoder_input_sha256_float32_le": _sha256_float32_le(encoder_input),
        "decoder_input_sha256_float32_le": _sha256_float32_le(decoder_input),
        "tokens": _comparison(actual_tokens, expected_tokens),
        "actions": _comparison(actual_actions, expected_actions),
        "rek_parity_claim": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return (
        0
        if result["tokens"]["bitwise_equal"]
        and result["actions"]["bitwise_equal"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
