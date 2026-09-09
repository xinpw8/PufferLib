#!/usr/bin/env python3
"""Verify the native vector controller batch against the Python reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

import gear_sonic_candidate as candidate
import sonic_candidate as plant


FRAMES = 67


def _probe_float(rows: np.ndarray, columns: np.ndarray, salt: int) -> np.ndarray:
    value = (rows * 131 + columns * 17 + np.uint64(salt * 29)) % 2003
    return (value.astype(np.float32) - np.float32(1001.0)) / np.float32(317.0)


def _probe_double(rows: np.ndarray, columns: np.ndarray, salt: int) -> np.ndarray:
    value = (rows * 137 + columns * 19 + np.uint64(salt * 31)) % 2011
    return (value.astype(np.float64) - 1005.0) / 911.0


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def _read(path: Path, shape: tuple[int, int]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape, dtype=np.int64))
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} values; expected {expected}")
    return values.reshape(shape)


def _compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    return {
        "array_equal": bool(np.array_equal(actual, expected)),
        "max_absolute_difference": float(np.max(np.abs(actual - expected))),
        "actual_sha256_float32_le": _sha256(actual),
        "expected_sha256_float32_le": _sha256(expected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--encoder-output", type=Path, required=True)
    parser.add_argument("--token-output", type=Path, required=True)
    parser.add_argument("--decoder-output", type=Path, required=True)
    parser.add_argument("--action-output", type=Path, required=True)
    parser.add_argument("--target-output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.iterations <= 0:
        parser.error("batch size and iterations must be positive")

    frame_rows = np.arange(FRAMES, dtype=np.uint64)[:, None]
    joint_columns = np.arange(candidate.ACTION_DIM, dtype=np.uint64)[None, :]
    dof_position = _probe_float(frame_rows, joint_columns, 5)
    root_rotation = np.zeros((FRAMES, 4), dtype=np.float32)
    root_rotation[:, 3] = np.float32(1.0)
    motion = plant.MotionData(
        role="synthetic_native_batch_probe",
        filename="none",
        size=0,
        sha256="0" * 64,
        fps=50.0,
        dof_pos=dof_position,
        dof_vel=np.zeros_like(dof_position),
        root_pos=np.zeros((FRAMES, 3), dtype=np.float32),
        root_rot_xyzw=root_rotation,
        manifest_sha256="0" * 64,
        inventory_sha256="0" * 64,
    )
    histories = [candidate.StateHistory() for _ in range(args.batch_size)]
    last_actions = np.zeros((args.batch_size, candidate.ACTION_DIM), dtype=np.float32)
    encoder = _session(args.encoder)
    decoder = _session(args.decoder)
    encoder_observations = np.empty(
        (args.batch_size, candidate.ENCODER_DIM), dtype=np.float32
    )
    tokens = np.empty((args.batch_size, candidate.TOKEN_DIM), dtype=np.float32)
    decoder_observations = np.empty(
        (args.batch_size, candidate.DECODER_DIM), dtype=np.float32
    )
    actions = np.empty((args.batch_size, candidate.ACTION_DIM), dtype=np.float32)
    targets = np.empty_like(actions)
    base_quaternion = np.zeros((args.batch_size, 4), dtype=np.float64)
    base_quaternion[:, 0] = 1.0
    heading_delta = base_quaternion.copy()
    row_base = np.arange(args.batch_size, dtype=np.uint64)[:, None]

    for iteration in range(args.iterations):
        state_rows = row_base + np.uint64(iteration)
        angular = _probe_double(
            state_rows, np.arange(3, dtype=np.uint64)[None, :], 7
        )
        joint_position = _probe_double(state_rows, joint_columns, 11)
        joint_velocity = _probe_double(state_rows, joint_columns, 13)
        reference_frames = (
            iteration + np.arange(args.batch_size, dtype=np.int64) * 3
        ) % FRAMES
        for row in range(args.batch_size):
            encoder_observations[row], _ = candidate.build_encoder_observation(
                motion,
                int(reference_frames[row]),
                base_quaternion[row],
                heading_delta[row],
                loop=True,
            )
            histories[row].append(
                candidate.HistoryEntry(
                    base_quat_wxyz=base_quaternion[row].astype(np.float32),
                    base_ang_vel=angular[row].astype(np.float32),
                    body_q_policy=np.asarray(
                        (joint_position[row] - candidate.DEFAULT_ANGLES_MUJOCO)[
                            candidate.MUJOCO_TO_ISAACLAB
                        ],
                        dtype=np.float32,
                    ),
                    body_dq_policy=np.asarray(
                        joint_velocity[row][candidate.MUJOCO_TO_ISAACLAB],
                        dtype=np.float32,
                    ),
                    last_action_policy=last_actions[row].copy(),
                )
            )
        tokens = np.asarray(
            encoder.run(["encoded_tokens"], {"obs_dict": encoder_observations})[0],
            dtype=np.float32,
        )
        for row in range(args.batch_size):
            decoder_observations[row] = candidate.build_decoder_observation(
                tokens[row], histories[row]
            )
        actions = np.asarray(
            decoder.run(["action"], {"obs_dict": decoder_observations})[0],
            dtype=np.float32,
        )
        for row in range(args.batch_size):
            last_actions[row], targets[row] = candidate.action_to_targets(actions[row])

    comparisons = {
        "encoder_observations": _compare(
            _read(
                args.encoder_output,
                (args.batch_size, candidate.ENCODER_DIM),
            ),
            encoder_observations,
        ),
        "tokens": _compare(
            _read(args.token_output, (args.batch_size, candidate.TOKEN_DIM)), tokens
        ),
        "decoder_observations": _compare(
            _read(
                args.decoder_output,
                (args.batch_size, candidate.DECODER_DIM),
            ),
            decoder_observations,
        ),
        "actions": _compare(
            _read(args.action_output, (args.batch_size, candidate.ACTION_DIM)),
            actions,
        ),
        "targets": _compare(
            _read(args.target_output, (args.batch_size, candidate.ACTION_DIM)),
            targets,
        ),
    }
    result = {
        "schema": "rek.g1_gear_sonic_native_batch_equivalence.v1",
        "classification": "public_family_candidate",
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "onnxruntime": ort.__version__,
        "comparisons": comparisons,
        "rek_parity_claim": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if all(item["array_equal"] for item in comparisons.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
