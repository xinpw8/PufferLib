#!/usr/bin/env python3
"""Compare CUDA GEAR-SONIC inference with pinned CPU ONNX Runtime outputs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from gpu_controller import GearSonicGpuController


SCHEMA = "rek.g1_gear_sonic_gpu_controller_probe.v1"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _probe_values(batch_size: int, width: int, salt: int) -> np.ndarray:
    rows = np.arange(batch_size, dtype=np.uint64)[:, None]
    columns = np.arange(width, dtype=np.uint64)[None, :]
    values = (rows * 131 + columns * 17 + np.uint64(salt * 29)) % 2003
    return (values.astype(np.float32) - np.float32(1001.0)) / np.float32(317.0)


def _encoder_probe(batch_size: int) -> np.ndarray:
    observation = np.zeros((batch_size, 1762), dtype=np.float32)
    observation[:, 4:584] = _probe_values(batch_size, 580, 1)
    observation[:, 601:661] = _probe_values(batch_size, 60, 2)
    return observation


def _sha256_float32_le(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(values, dtype="<f4").tobytes()
    return hashlib.sha256(payload).hexdigest()


def _comparison(actual: np.ndarray, expected: np.ndarray, atol: float) -> dict[str, Any]:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    maximum = float(difference.max()) if difference.size else 0.0
    return {
        "within_absolute_tolerance": bool(maximum <= atol),
        "absolute_tolerance": atol,
        "max_absolute_difference": maximum,
        "mean_absolute_difference": float(difference.mean()) if difference.size else 0.0,
        "actual_sha256_float32_le": _sha256_float32_le(actual),
        "expected_sha256_float32_le": _sha256_float32_le(expected),
    }


def _cpu_session(ort: Any, path: Path) -> Any:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _benchmark_ms(
    torch: Any,
    function: Callable[[Any], Any],
    value: Any,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function(value)
    torch.cuda.synchronize(value.device)
    start = time.perf_counter_ns()
    for _ in range(iterations):
        function(value)
    torch.cuda.synchronize(value.device)
    elapsed_ns = time.perf_counter_ns() - start
    return elapsed_ns / 1_000_000.0 / iterations


def _benchmark_replay_ms(
    torch: Any,
    replay: Callable[[], Any],
    warmup: int,
    iterations: int,
    device: Any,
) -> float:
    for _ in range(warmup):
        replay()
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    for _ in range(iterations):
        replay()
    torch.cuda.synchronize(device)
    elapsed_ns = time.perf_counter_ns() - start
    return elapsed_ns / 1_000_000.0 / iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--batch-bundle", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--iterations", type=_positive_int, default=100)
    parser.add_argument("--atol", type=float, default=2e-5)
    args = parser.parse_args()
    if not np.isfinite(args.atol) or args.atol < 0.0:
        parser.error("--atol must be finite and nonnegative")

    import onnxruntime as ort
    import torch

    manifest = args.batch_bundle / "explicit_batch_manifest.json"
    controller = GearSonicGpuController.from_manifest(
        manifest, args.source_bundle, device=args.device
    )
    batch_size = controller.batch_size
    encoder_input = _encoder_probe(batch_size)
    decoder_input = _probe_values(batch_size, 994, 3)

    encoder_session = _cpu_session(ort, controller.encoder_identity.path)
    decoder_session = _cpu_session(ort, controller.decoder_identity.path)
    expected_tokens = np.asarray(
        encoder_session.run(
            [controller.encoder_identity.output_name],
            {controller.encoder_identity.input_name: encoder_input},
        )[0],
        dtype=np.float32,
    )
    expected_actions = np.asarray(
        decoder_session.run(
            [controller.decoder_identity.output_name],
            {controller.decoder_identity.input_name: decoder_input},
        )[0],
        dtype=np.float32,
    )
    del encoder_session, decoder_session
    gc.collect()

    encoder_cuda = torch.from_numpy(encoder_input).to(controller.device)
    decoder_cuda = torch.from_numpy(decoder_input).to(controller.device)
    actual_tokens_tensor = controller.encode(encoder_cuda)
    actual_actions_tensor = controller.decode(decoder_cuda)
    torch.cuda.synchronize(controller.device)
    actual_tokens = actual_tokens_tensor.cpu().numpy()
    actual_actions = actual_actions_tensor.cpu().numpy()

    captured = controller.capture_pair(
        encoder_cuda, decoder_cuda, warmup=args.warmup
    )
    graph_tokens_tensor, graph_actions_tensor = captured.replay()
    torch.cuda.synchronize(controller.device)
    graph_tokens = graph_tokens_tensor.cpu().numpy()
    graph_actions = graph_actions_tensor.cpu().numpy()

    encoder_ms = _benchmark_ms(
        torch,
        controller.encode,
        encoder_cuda,
        args.warmup,
        args.iterations,
    )
    decoder_ms = _benchmark_ms(
        torch,
        controller.decode,
        decoder_cuda,
        args.warmup,
        args.iterations,
    )
    graph_ms = _benchmark_replay_ms(
        torch,
        captured.replay,
        args.warmup,
        args.iterations,
        controller.device,
    )
    token_comparison = _comparison(actual_tokens, expected_tokens, args.atol)
    action_comparison = _comparison(actual_actions, expected_actions, args.atol)
    graph_token_comparison = _comparison(
        graph_tokens, expected_tokens, args.atol
    )
    graph_action_comparison = _comparison(
        graph_actions, expected_actions, args.atol
    )
    report = {
        "schema": SCHEMA,
        "batch_size": batch_size,
        "device": {
            "name": torch.cuda.get_device_name(controller.device),
            "capability": list(torch.cuda.get_device_capability(controller.device)),
        },
        "versions": {
            "torch": torch.__version__,
            "onnx": __import__("onnx").__version__,
            "onnxruntime": ort.__version__,
            "providers": ["CPUExecutionProvider"],
        },
        "models": {
            "encoder_sha256": controller.encoder_identity.sha256,
            "decoder_sha256": controller.decoder_identity.sha256,
            "cuda_resident_bytes": controller.resident_bytes(),
        },
        "comparison": {
            "eager": {
                "encoder_tokens": token_comparison,
                "decoder_actions": action_comparison,
            },
            "cuda_graph": {
                "encoder_tokens": graph_token_comparison,
                "decoder_actions": graph_action_comparison,
            },
        },
        "benchmark": {
            "warmup_calls": args.warmup,
            "measured_calls": args.iterations,
            "encoder_ms_per_batch": encoder_ms,
            "decoder_ms_per_batch": decoder_ms,
            "combined_ms_per_batch": encoder_ms + decoder_ms,
            "combined_robot_inferences_per_second": (
                batch_size * 1000.0 / (encoder_ms + decoder_ms)
            ),
            "cuda_graph_ms_per_batch": graph_ms,
            "cuda_graph_robot_inferences_per_second": (
                batch_size * 1000.0 / graph_ms
            ),
        },
        "backend": "pinned_torch_graph_reconstruction",
        "encoder_execution_contract": "g1_mode_zero",
        "onnx2torch_used": False,
        "cuda_graph_captured": True,
        "per_replay_host_tensor_copies": 0,
        "controller_inference_cuda_only": True,
        "cpu_use": "one-thread ONNX Runtime reference generation and orchestration only",
        "rek_parity_claim": False,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return (
        0
        if token_comparison["within_absolute_tolerance"]
        and action_comparison["within_absolute_tolerance"]
        and graph_token_comparison["within_absolute_tolerance"]
        and graph_action_comparison["within_absolute_tolerance"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
