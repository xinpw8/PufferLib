#!/usr/bin/env python3
"""Loopback-only human evaluation for the native REK G1 semantic candidate.

This process drives the same eight-row Puffer vector used by the native smoke
test.  It does not send input to REK, X11, Windows, or another application.
Browser events are converted to semantic categories inside this process.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import ipaddress
import json
import math
import os
from pathlib import Path
import struct
import threading
import time
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence
import urllib.parse
import zlib

import numpy as np


LOOPBACK_HOST = "127.0.0.1"
CONTROL_RATE_HZ = 50
CONTROL_PERIOD_SECONDS = 1.0 / CONTROL_RATE_HZ
ROBOT_ROWS = 8
OBSERVATION_FLOATS = 223
ENTITY_FLOATS = 86
ACTION_HEADS = 1
ACTION_CATEGORIES = 20
CONTINUE_CATEGORY = 0
NEUTRAL_CATEGORY = 1
YAW_LEFT_CATEGORY = 6
YAW_RIGHT_CATEGORY = 7
KICK_MOVE_9_CATEGORY = 19
KICK_DURATION_TICKS = (157, 145, 158, 139)
KICK_MOVE_TO_CATEGORY = {6: 16, 7: 17, 8: 18, 9: 19}
KICK_METADATA = (
    {"move": 6, "category": 16, "identity": "left_side", "keys": ["6"]},
    {"move": 7, "category": 17, "identity": "left_front", "keys": ["7", "U"]},
    {"move": 8, "category": 18, "identity": "right_side", "keys": ["8", "I"]},
    {"move": 9, "category": 19, "identity": "right_knee", "keys": ["9"]},
)

SEMANTIC_OFFSET = 2 * ENTITY_FLOATS
EFFECTIVE_FORWARD_INDEX = SEMANTIC_OFFSET + 4
EFFECTIVE_STRAFE_INDEX = SEMANTIC_OFFSET + 5
EFFECTIVE_YAW_INDEX = SEMANTIC_OFFSET + 6
ACTIVE_ROUTE_INDEX = SEMANTIC_OFFSET + 7
LOCOMOTION_ACTIVE_INDEX = SEMANTIC_OFFSET + 8
TRANSITION_SETTLING_INDEX = SEMANTIC_OFFSET + 9
ACTION_PLAYING_INDEX = SEMANTIC_OFFSET + 10
COMPOSER_BUSY_INDEX = SEMANTIC_OFFSET + 11
FIGHT_OFFSET = SEMANTIC_OFFSET + 12
SELF_FALL_OFFSET = 71
FALL_PHASE_OFFSET = SELF_FALL_OFFSET + 8

HELD_CATEGORY_BY_SYMBOLS: Mapping[frozenset[str], int] = {
    frozenset(): 1,
    frozenset({"W"}): 2,
    frozenset({"S"}): 3,
    frozenset({"A"}): 4,
    frozenset({"D"}): 5,
    frozenset({"Q"}): 6,
    frozenset({"E"}): 7,
    frozenset({"W", "Q"}): 8,
    frozenset({"W", "E"}): 9,
    frozenset({"S", "Q"}): 10,
    frozenset({"S", "E"}): 11,
    frozenset({"A", "Q"}): 12,
    frozenset({"A", "E"}): 13,
    frozenset({"D", "Q"}): 14,
    frozenset({"D", "E"}): 15,
}
TRANSLATION_SYMBOLS = frozenset({"W", "S", "A", "D"})
YAW_SYMBOLS = frozenset({"Q", "E"})
VALID_HELD_SYMBOLS = TRANSLATION_SYMBOLS | YAW_SYMBOLS

JOINT_NAME_SUFFIXES = (
    "joint__left_hip_pitch_joint_3047",
    "joint__left_hip_roll_joint_3248",
    "joint__left_hip_yaw_joint_3267",
    "joint__left_knee_joint_3137",
    "joint__left_ankle_pitch_joint_2982",
    "joint__left_ankle_roll_joint_2905",
    "joint__right_hip_pitch_joint_3298",
    "joint__right_hip_roll_joint_3059",
    "joint__right_hip_yaw_joint_3071",
    "joint__right_knee_joint_3412",
    "joint__right_ankle_pitch_joint_3312",
    "joint__right_ankle_roll_joint_3474",
    "joint__waist_yaw_joint_3441",
    "joint__waist_roll_joint_3341",
    "joint__waist_pitch_joint_3233",
    "joint__left_shoulder_pitch_joint_3340",
    "joint__left_shoulder_roll_joint_3184",
    "joint__left_shoulder_yaw_joint_2923",
    "joint__left_elbow_joint_3144",
    "joint__left_wrist_roll_joint_3260",
    "joint__left_wrist_pitch_joint_3007",
    "joint__left_wrist_yaw_joint_3398",
    "joint__right_shoulder_pitch_joint_3242",
    "joint__right_shoulder_roll_joint_3044",
    "joint__right_shoulder_yaw_joint_3176",
    "joint__right_elbow_joint_3407",
    "joint__right_wrist_roll_joint_3378",
    "joint__right_wrist_pitch_joint_3437",
    "joint__right_wrist_yaw_joint_3226",
)


class HumanEvalFailure(RuntimeError):
    """A required identity, ABI, state, or action invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanEvalFailure(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sha256(value: str, description: str) -> str:
    normalized = value.lower()
    _require(
        len(normalized) == 64 and all(character in "0123456789abcdef" for character in normalized),
        f"{description} expected SHA-256 is invalid",
    )
    return normalized


def _regular_file(path: Path, description: str) -> Path:
    _require(not path.is_symlink(), f"{description} must not be a symlink")
    resolved = path.resolve()
    _require(resolved.is_file(), f"{description} is not a regular file")
    return resolved


def _regular_directory(path: Path, description: str) -> Path:
    _require(not path.is_symlink(), f"{description} must not be a symlink")
    resolved = path.resolve()
    _require(resolved.is_dir(), f"{description} is not a directory")
    return resolved


@dataclass(frozen=True)
class RuntimeIdentity:
    extension: Path
    extension_sha256: str
    semantic_assets: Path
    asset_manifest_sha256: str
    model_sha256: str
    encoder: Path
    encoder_sha256: str
    decoder: Path
    decoder_sha256: str


@dataclass(frozen=True)
class VerifiedRuntimeIdentity:
    extension: Path
    extension_sha256: str
    semantic_assets: Path
    asset_manifest: Path
    asset_manifest_sha256: str
    model: Path
    model_sha256: str
    encoder: Path
    encoder_sha256: str
    decoder: Path
    decoder_sha256: str

    def report(self) -> dict[str, Any]:
        return {
            "extension": {"path": str(self.extension), "sha256": self.extension_sha256},
            "semantic_assets": str(self.semantic_assets),
            "asset_manifest": {
                "path": str(self.asset_manifest),
                "sha256": self.asset_manifest_sha256,
            },
            "model": {"path": str(self.model), "sha256": self.model_sha256},
            "encoder": {"path": str(self.encoder), "sha256": self.encoder_sha256},
            "decoder": {"path": str(self.decoder), "sha256": self.decoder_sha256},
        }


def _observation_f32_le_b64(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f4")
    _require(array.shape == (OBSERVATION_FLOATS,), "trace observation shape mismatch")
    _require(bool(np.isfinite(array).all()), "trace observation is nonfinite")
    return base64.b64encode(array.tobytes(order="C")).decode("ascii")


class JsonlTraceWriter:
    """Append-only control-tick trace for paired replay and trajectory comparison."""

    SCHEMA = "rek.g1_human_eval_trace.v1"

    def __init__(self, path: Path, identity_report: Mapping[str, Any]) -> None:
        _require(not path.is_symlink(), "trace output must not be a symlink")
        parent = path.parent.resolve()
        parent.mkdir(parents=True, exist_ok=True)
        _require(parent.is_dir(), "trace output parent is not a directory")
        self.path = (parent / path.name).resolve()
        _require(self.path.parent == parent, "trace output escaped its parent directory")
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        self._sequence = 0
        self._closed = False
        self.write(
            {
                "event": "trace_start",
                "wall_clock_unix_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "control_rate_hz": CONTROL_RATE_HZ,
                "observation_encoding": "223_binary32_little_endian_base64",
                "runtime_identity": dict(identity_report),
                "rek_parity_claim": False,
            }
        )

    def write(self, record: Mapping[str, Any]) -> None:
        _require(not self._closed, "trace writer is closed")
        value = dict(record)
        value["schema"] = self.SCHEMA
        value["trace_sequence"] = self._sequence
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._stream.write(payload + "\n")
        self._stream.flush()
        self._sequence += 1

    def close(self) -> None:
        if self._closed:
            return
        self.write(
            {
                "event": "trace_end",
                "wall_clock_unix_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
            }
        )
        self._closed = True
        self._stream.close()


def verify_runtime_identity(identity: RuntimeIdentity) -> VerifiedRuntimeIdentity:
    extension = _regular_file(identity.extension, "native extension")
    semantic_assets = _regular_directory(identity.semantic_assets, "semantic asset directory")
    asset_manifest = _regular_file(
        semantic_assets / "semantic_duel_assets_manifest.json", "semantic asset manifest"
    )
    model = _regular_file(
        semantic_assets / "model.two_fighter_arena.xml", "two-fighter render model"
    )
    encoder = _regular_file(identity.encoder, "encoder ONNX")
    decoder = _regular_file(identity.decoder, "decoder ONNX")

    expected = {
        "extension": _expected_sha256(identity.extension_sha256, "native extension"),
        "asset_manifest": _expected_sha256(
            identity.asset_manifest_sha256, "semantic asset manifest"
        ),
        "model": _expected_sha256(identity.model_sha256, "two-fighter render model"),
        "encoder": _expected_sha256(identity.encoder_sha256, "encoder ONNX"),
        "decoder": _expected_sha256(identity.decoder_sha256, "decoder ONNX"),
    }
    actual = {
        "extension": _sha256(extension),
        "asset_manifest": _sha256(asset_manifest),
        "model": _sha256(model),
        "encoder": _sha256(encoder),
        "decoder": _sha256(decoder),
    }
    for name in expected:
        _require(actual[name] == expected[name], f"{name} SHA-256 mismatch")

    try:
        manifest = json.loads(asset_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HumanEvalFailure(f"semantic asset manifest is unreadable: {error}") from error
    _require(isinstance(manifest, dict), "semantic asset manifest root is not an object")
    _require(
        manifest.get("schema") == "rek.g1_semantic_duel_assets.v1",
        "semantic asset manifest schema mismatch",
    )
    files = manifest.get("files")
    _require(isinstance(files, dict), "semantic asset manifest files are absent")
    model_record = files.get("model.two_fighter_arena.xml")
    _require(isinstance(model_record, dict), "semantic asset manifest model record is absent")
    _require(model_record.get("sha256") == actual["model"], "manifest model SHA-256 mismatch")
    if "bytes" in model_record:
        _require(
            isinstance(model_record["bytes"], int)
            and not isinstance(model_record["bytes"], bool)
            and model_record["bytes"] == model.stat().st_size,
            "manifest model byte count mismatch",
        )

    return VerifiedRuntimeIdentity(
        extension=extension,
        extension_sha256=actual["extension"],
        semantic_assets=semantic_assets,
        asset_manifest=asset_manifest,
        asset_manifest_sha256=actual["asset_manifest"],
        model=model,
        model_sha256=actual["model"],
        encoder=encoder,
        encoder_sha256=actual["encoder"],
        decoder=decoder,
        decoder_sha256=actual["decoder"],
    )


@contextlib.contextmanager
def _native_environment(identity: VerifiedRuntimeIdentity) -> Iterator[None]:
    values = {
        "REK_G1_SEMANTIC_ASSETS_DIR": str(identity.semantic_assets),
        "REK_G1_ENCODER_ONNX": str(identity.encoder),
        "REK_G1_DECODER_ONNX": str(identity.decoder),
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_extension(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location("_C", path)
    _require(
        specification is not None and specification.loader is not None,
        "failed to construct native extension specification",
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _float_view(pointer: int, rows: int, columns: int) -> np.ndarray:
    _require(pointer != 0, "native vector exposed a null float buffer")
    storage = (ctypes.c_float * (rows * columns)).from_address(pointer)
    return np.ctypeslib.as_array(storage).reshape(rows, columns)


def _optional_action_mask(vector: Any) -> np.ndarray | None:
    has_pointer = hasattr(vector, "action_mask_ptr")
    has_size = hasattr(vector, "action_mask_size")
    _require(has_pointer == has_size, "native action-mask properties are incomplete")
    if not has_pointer:
        return None
    pointer = int(vector.action_mask_ptr)
    stride = int(vector.action_mask_size)
    _require(pointer != 0, "native action-mask pointer is null")
    _require(stride == ACTION_CATEGORIES, "native action-mask stride mismatch")
    storage = (ctypes.c_uint8 * (ROBOT_ROWS * stride)).from_address(pointer)
    return np.ctypeslib.as_array(storage).reshape(ROBOT_ROWS, stride)


def vector_arguments(max_steps: int, physics_workers: int) -> dict[str, Any]:
    _require(max_steps > 0, "max_steps must be positive")
    _require(physics_workers > 0, "physics_workers must be positive")
    return {
        "vec": {"total_agents": ROBOT_ROWS, "num_buffers": 1},
        "env": {
            "max_steps": max_steps,
            "physics_workers": physics_workers,
            "locomotion_segment_ticks": 1,
            "kick_move_6_duration_ticks": KICK_DURATION_TICKS[0],
            "kick_move_7_duration_ticks": KICK_DURATION_TICKS[1],
            "kick_move_8_duration_ticks": KICK_DURATION_TICKS[2],
            "kick_move_9_duration_ticks": KICK_DURATION_TICKS[3],
        },
    }


class NativeVectorBoundary:
    def __init__(
        self,
        vector: Any,
        observations: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        action_masks: np.ndarray | None,
        identity: VerifiedRuntimeIdentity,
    ) -> None:
        self.vector = vector
        self.observations = observations
        self.rewards = rewards
        self.terminals = terminals
        self.action_masks = action_masks
        self.identity = identity

    @classmethod
    def open(
        cls,
        identity: RuntimeIdentity,
        *,
        max_steps: int,
        physics_workers: int,
        extension_loader: Callable[[Path], ModuleType] = _load_extension,
    ) -> "NativeVectorBoundary":
        verified = verify_runtime_identity(identity)
        native: ModuleType
        vector: Any | None = None
        with _native_environment(verified):
            native = extension_loader(verified.extension)
            _require(getattr(native, "env_name", None) == "rek_g1", "extension environment mismatch")
            vector = native.create_vec(vector_arguments(max_steps, physics_workers), 0)
        try:
            _require(vector.total_agents == ROBOT_ROWS, "extension row-count mismatch")
            _require(vector.obs_size == OBSERVATION_FLOATS, "extension observation ABI mismatch")
            _require(vector.num_atns == ACTION_HEADS, "extension action-head ABI mismatch")
            _require(vector.act_sizes == [ACTION_CATEGORIES], "extension action-category ABI mismatch")
            _require(vector.obs_elem_size == 4, "extension observation element width mismatch")
            _require(vector.obs_dtype == "FloatTensor", "extension observation dtype mismatch")
            observations = _float_view(int(vector.obs_ptr), ROBOT_ROWS, OBSERVATION_FLOATS)
            rewards = _float_view(int(vector.rewards_ptr), ROBOT_ROWS, 1).reshape(ROBOT_ROWS)
            terminals = _float_view(int(vector.terminals_ptr), ROBOT_ROWS, 1).reshape(ROBOT_ROWS)
            action_masks = _optional_action_mask(vector)
            _require(
                action_masks is not None,
                "native extension does not expose the exact action mask",
            )
            boundary = cls(vector, observations, rewards, terminals, action_masks, verified)
            boundary.reset()
            return boundary
        except BaseException:
            if vector is not None:
                vector.close()
            raise

    def reset(self) -> None:
        self.vector.reset()
        _require(bool(np.isfinite(self.observations).all()), "reset observations are nonfinite")
        _require(bool(np.equal(self.rewards, 0.0).all()), "reset rewards are nonzero")
        _require(bool(np.equal(self.terminals, 0.0).all()), "reset terminals are nonzero")

    def step(self, actions: np.ndarray) -> None:
        _require(actions.shape == (ROBOT_ROWS, ACTION_HEADS), "action buffer shape mismatch")
        _require(actions.dtype == np.float32, "action buffer dtype mismatch")
        categories = actions[:, 0].astype(np.int64)
        _require(
            bool(np.equal(categories.astype(np.float32), actions[:, 0]).all()),
            "action buffer contains non-integral categories",
        )
        _require(
            bool(np.logical_and(categories >= 0, categories < ACTION_CATEGORIES).all()),
            "action buffer category is out of range",
        )
        if self.action_masks is not None:
            for row, category in enumerate(categories.tolist()):
                _require(
                    bool(self.action_masks[row, category]),
                    f"row {row} category {category} is masked",
                )
        self.vector.cpu_step(int(actions.ctypes.data))
        _require(bool(np.isfinite(self.observations).all()), "step observations are nonfinite")
        _require(bool(np.isfinite(self.rewards).all()), "step rewards are nonfinite")
        _require(
            bool(np.logical_or(self.terminals == 0.0, self.terminals == 1.0).all()),
            "step terminals are not Boolean",
        )

    def close(self) -> None:
        self.vector.close()


class BrowserInputState:
    def __init__(self) -> None:
        self.held: frozenset[str] = frozenset()
        self.sequence = 0
        self.pending_kick_move: int | None = None

    def reset(self) -> None:
        self.held = frozenset()
        self.sequence = 0
        self.pending_kick_move = None

    def update(self, payload: Mapping[str, Any]) -> bool:
        sequence = payload.get("sequence")
        _require(
            isinstance(sequence, int) and not isinstance(sequence, bool) and sequence >= 0,
            "input sequence must be a nonnegative integer",
        )
        if sequence <= self.sequence:
            return False
        raw_held = payload.get("held")
        _require(isinstance(raw_held, list), "held must be an array")
        _require(
            all(isinstance(symbol, str) and symbol in VALID_HELD_SYMBOLS for symbol in raw_held),
            "held contains an unsupported key",
        )
        held = frozenset(raw_held)
        _require(len(held) == len(raw_held), "held contains duplicate keys")
        _require(
            len(held & TRANSLATION_SYMBOLS) <= 1,
            "held contains multiple translation directions",
        )
        _require(len(held & YAW_SYMBOLS) <= 1, "held contains opposite yaw directions")
        _require(held in HELD_CATEGORY_BY_SYMBOLS, "held combination has no semantic category")
        kick_move = payload.get("kick_move")
        if kick_move is not None:
            _require(
                isinstance(kick_move, int)
                and not isinstance(kick_move, bool)
                and kick_move in KICK_MOVE_TO_CATEGORY,
                "kick_move must be one of 6, 7, 8, or 9",
            )
            self.pending_kick_move = kick_move
        self.held = held
        self.sequence = sequence
        return True

    def take_kick_edge(self) -> int | None:
        move = self.pending_kick_move
        self.pending_kick_move = None
        return move


@dataclass(frozen=True)
class ActionChoice:
    category: int
    reason: str


def _binary_observation_flag(row: np.ndarray, index: int, description: str) -> bool:
    value = float(row[index])
    _require(value in (0.0, 1.0), f"{description} is not Boolean")
    return value == 1.0


class ConservativeActionPlanner:
    """Mirror mask behavior without guessing a hidden native fact."""

    STABLE_TICKS_BEFORE_KICK = 5

    def __init__(self) -> None:
        self.continuation_ticks = 0
        self.stable_ticks = 0

    def reset(self) -> None:
        self.continuation_ticks = 0
        self.stable_ticks = 0

    def _observably_settled(self, row: np.ndarray) -> bool:
        _require(row.shape == (OBSERVATION_FLOATS,), "observation row shape mismatch")
        _require(bool(np.isfinite(row).all()), "observation row is nonfinite")
        upright = float(row[FALL_PHASE_OFFSET]) == 0.0
        return (
            upright
            and abs(float(row[EFFECTIVE_FORWARD_INDEX])) < 1e-6
            and abs(float(row[EFFECTIVE_STRAFE_INDEX])) < 1e-6
            and not _binary_observation_flag(row, LOCOMOTION_ACTIVE_INDEX, "locomotion_active")
            and not _binary_observation_flag(
                row, TRANSITION_SETTLING_INDEX, "transition_settling"
            )
            and not _binary_observation_flag(row, ACTION_PLAYING_INDEX, "action_playing")
            and not _binary_observation_flag(row, COMPOSER_BUSY_INDEX, "composer_busy")
        )

    def select(
        self,
        preferred: int,
        fallback_locomotion: int,
        row: np.ndarray,
        mask: np.ndarray | None,
    ) -> ActionChoice:
        _require(0 <= preferred < ACTION_CATEGORIES, "preferred category is out of range")
        _require(
            1 <= fallback_locomotion <= 15,
            "fallback category is not a locomotion category",
        )
        settled = self._observably_settled(row)
        self.stable_ticks = self.stable_ticks + 1 if settled else 0

        if mask is not None:
            _require(mask.shape == (ACTION_CATEGORIES,), "action-mask row shape mismatch")
            _require(
                bool(np.logical_or(mask == 0, mask == 1).all()),
                "action-mask row is not Boolean",
            )
            if bool(mask[preferred]):
                return ActionChoice(preferred, "native_mask_preferred")
            if bool(mask[fallback_locomotion]):
                return ActionChoice(fallback_locomotion, "native_mask_fallback")
            if bool(mask[0]):
                return ActionChoice(0, "native_mask_continue")
            if bool(mask[1]):
                return ActionChoice(1, "native_mask_neutral")
            raise HumanEvalFailure("native mask exposes no conservative action")

        if self.continuation_ticks > 0:
            self.continuation_ticks -= 1
            return ActionChoice(0, "tracked_finite_kick_continue")
        if preferred >= 16:
            if self.stable_ticks < self.STABLE_TICKS_BEFORE_KICK:
                return ActionChoice(fallback_locomotion, "observed_settle_gate")
            duration = KICK_DURATION_TICKS[preferred - 16]
            self.continuation_ticks = duration - 1
            self.stable_ticks = 0
            return ActionChoice(preferred, "observed_settle_kick_start")
        return ActionChoice(preferred, "locomotion_always_legal_when_idle")


class CandidateApproachDummy:
    LABEL = "deterministic_state_based_approach_facing_kick_candidate_dummy"

    def __init__(self) -> None:
        self.next_kick_offset = 0

    def reset(self) -> None:
        self.next_kick_offset = 0

    @staticmethod
    def _yaw(quaternion_wxyz: Sequence[float]) -> float:
        w, x, y, z = (float(value) for value in quaternion_wxyz)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        _require(math.isfinite(norm) and norm > 1e-9, "dummy quaternion is invalid")
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def preferred_action(self, row: np.ndarray) -> int:
        _require(row.shape == (OBSERVATION_FLOATS,), "dummy observation shape mismatch")
        _require(bool(np.isfinite(row).all()), "dummy observation is nonfinite")
        if float(row[FALL_PHASE_OFFSET]) != 0.0:
            return 1
        self_position = row[0:3].astype(np.float64)
        opponent_position = row[ENTITY_FLOATS : ENTITY_FLOATS + 3].astype(np.float64)
        delta = opponent_position[:2] - self_position[:2]
        distance = float(np.linalg.norm(delta))
        yaw = self._yaw(row[3:7])
        forward = math.cos(yaw) * delta[0] + math.sin(yaw) * delta[1]
        lateral = -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1]
        bearing = math.atan2(lateral, forward)
        if abs(bearing) > 0.16:
            return 6 if bearing > 0.0 else 7
        if distance > 1.25:
            return 2
        if distance < 0.72:
            return 3
        return 16 + self.next_kick_offset

    def note_selected(self, category: int) -> None:
        if 16 <= category <= 19:
            self.next_kick_offset = (category - 16 + 1) % 4


class SemanticHumanEvalCore:
    def __init__(
        self,
        boundary: Any,
        trace_writer: JsonlTraceWriter | None = None,
    ) -> None:
        self.boundary = boundary
        self.trace_writer = trace_writer
        self.browser_input = BrowserInputState()
        self.planners = [ConservativeActionPlanner() for _ in range(ROBOT_ROWS)]
        self.dummy = CandidateApproachDummy()
        self.tick = 0
        self.last_actions = [1] * ROBOT_ROWS
        self.last_action_reasons = ["reset"] * ROBOT_ROWS
        self.last_kick_disposition = "none"
        self.reset()

    def reset(self) -> None:
        self.boundary.reset()
        _require(
            self.boundary.observations.shape == (ROBOT_ROWS, OBSERVATION_FLOATS),
            "boundary observation shape mismatch",
        )
        for planner in self.planners:
            planner.reset()
        self.browser_input.reset()
        self.dummy.reset()
        self.tick = 0
        self.last_actions = [1] * ROBOT_ROWS
        self.last_action_reasons = ["reset"] * ROBOT_ROWS
        self.last_kick_disposition = "none"
        if self.trace_writer is not None:
            self.trace_writer.write(
                {
                    "event": "environment_reset",
                    "wall_clock_unix_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                    "tick": self.tick,
                    "arena_0_observation_f32_le_b64": _observation_f32_le_b64(
                        np.asarray(self.boundary.observations[0], dtype=np.float32)
                    ),
                }
            )

    def update_input(self, payload: Mapping[str, Any]) -> bool:
        accepted = self.browser_input.update(payload)
        if accepted and self.trace_writer is not None:
            self.trace_writer.write(
                {
                    "event": "browser_input_received",
                    "wall_clock_unix_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                    "after_completed_tick": self.tick,
                    "applies_no_earlier_than_tick": self.tick + 1,
                    "input_sequence": self.browser_input.sequence,
                    "held": sorted(self.browser_input.held),
                    "kick_move": payload.get("kick_move"),
                }
            )
        return accepted

    def step_once(self) -> np.ndarray:
        observations = np.asarray(self.boundary.observations, dtype=np.float32)
        _require(
            observations.shape == (ROBOT_ROWS, OBSERVATION_FLOATS),
            "boundary observation shape mismatch",
        )
        masks_value = getattr(self.boundary, "action_masks", None)
        masks = None if masks_value is None else np.asarray(masks_value, dtype=np.uint8)
        if masks is not None:
            _require(
                masks.shape == (ROBOT_ROWS, ACTION_CATEGORIES),
                "boundary action-mask shape mismatch",
            )

        held_category = HELD_CATEGORY_BY_SYMBOLS[self.browser_input.held]
        kick_move = self.browser_input.take_kick_edge()
        human_preferred = held_category
        if kick_move is not None:
            if self.browser_input.held & TRANSLATION_SYMBOLS:
                self.last_kick_disposition = "discarded_translation_held"
            else:
                human_preferred = KICK_MOVE_TO_CATEGORY[kick_move]

        actions = np.full((ROBOT_ROWS, ACTION_HEADS), 1.0, dtype=np.float32)
        choices: list[ActionChoice] = []
        human_choice = self.planners[0].select(
            human_preferred,
            held_category,
            observations[0],
            None if masks is None else masks[0],
        )
        choices.append(human_choice)
        if kick_move is not None and not (self.browser_input.held & TRANSLATION_SYMBOLS):
            self.last_kick_disposition = (
                "accepted" if human_choice.category == human_preferred else human_choice.reason
            )

        dummy_preferred = self.dummy.preferred_action(observations[1])
        dummy_choice = self.planners[1].select(
            dummy_preferred,
            1,
            observations[1],
            None if masks is None else masks[1],
        )
        choices.append(dummy_choice)
        self.dummy.note_selected(dummy_choice.category)

        for row in range(2, ROBOT_ROWS):
            choices.append(
                self.planners[row].select(
                    1,
                    1,
                    observations[row],
                    None if masks is None else masks[row],
                )
            )
        for row, choice in enumerate(choices):
            actions[row, 0] = float(choice.category)

        self.boundary.step(actions)
        self.tick += 1
        self.last_actions = [choice.category for choice in choices]
        self.last_action_reasons = [choice.reason for choice in choices]
        terminals = np.asarray(self.boundary.terminals, dtype=np.float32)
        _require(terminals.shape == (ROBOT_ROWS,), "boundary terminal shape mismatch")
        for row, terminal in enumerate(terminals.tolist()):
            _require(terminal in (0.0, 1.0), "boundary terminal is not Boolean")
            if terminal == 1.0:
                self.planners[row].reset()
                if row == 1:
                    self.dummy.reset()
        if self.trace_writer is not None:
            self.trace_writer.write(
                {
                    "event": "control_step",
                    "wall_clock_unix_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                    "tick": self.tick,
                    "input_sequence": self.browser_input.sequence,
                    "held": sorted(self.browser_input.held),
                    "kick_move_edge": kick_move,
                    "kick_disposition": self.last_kick_disposition,
                    "actions": self.last_actions.copy(),
                    "action_reasons": self.last_action_reasons.copy(),
                    "arena_0_observation_f32_le_b64": _observation_f32_le_b64(
                        np.asarray(self.boundary.observations[0], dtype=np.float32)
                    ),
                    "arena_0_terminals": terminals[:2].astype(float).tolist(),
                }
            )
        return actions.copy()

    def close_trace(self) -> None:
        if self.trace_writer is not None:
            self.trace_writer.close()

    def state(self) -> dict[str, Any]:
        row = np.asarray(self.boundary.observations[0], dtype=np.float32)
        return {
            "schema": "rek.g1_human_eval_state.v1",
            "tick": self.tick,
            "control_rate_hz": CONTROL_RATE_HZ,
            "classification": "public_family_semantic_candidate_human_evaluation",
            "rek_parity_claim": False,
            "training_enabled": False,
            "paired_replay_trace_enabled": self.trace_writer is not None,
            "paired_replay_trace": (
                None
                if self.trace_writer is None
                else {
                    "schema": JsonlTraceWriter.SCHEMA,
                    "path": str(self.trace_writer.path),
                }
            ),
            "controller": "native_rek_g1_semantic_candidate",
            "opponent": CandidateApproachDummy.LABEL,
            "opponent_is_bot_1": False,
            "automatic_getup_enabled": False,
            "runtime_get_up_authority": "unknown",
            "action_mask_source": (
                "native_pointer"
                if getattr(self.boundary, "action_masks", None) is not None
                else "conservative_observation_and_local_segment_tracker"
            ),
            "input_sequence": self.browser_input.sequence,
            "held": sorted(self.browser_input.held),
            "last_kick_disposition": self.last_kick_disposition,
            "last_actions": self.last_actions.copy(),
            "last_action_reasons": self.last_action_reasons.copy(),
            "player": {
                "position_m": row[0:3].astype(float).tolist(),
                "quaternion_wxyz": row[3:7].astype(float).tolist(),
                "fall_phase": int(row[FALL_PHASE_OFFSET]),
                "active_route_id": int(row[ACTIVE_ROUTE_INDEX]),
                "effective_command": row[
                    EFFECTIVE_FORWARD_INDEX : EFFECTIVE_YAW_INDEX + 1
                ].astype(float).tolist(),
                "action_playing": bool(row[ACTION_PLAYING_INDEX]),
                "composer_busy": bool(row[COMPOSER_BUSY_INDEX]),
            },
            "opponent_state": {
                "position_m": row[ENTITY_FLOATS : ENTITY_FLOATS + 3].astype(float).tolist(),
                "quaternion_wxyz": row[
                    ENTITY_FLOATS + 3 : ENTITY_FLOATS + 7
                ].astype(float).tolist(),
                "fall_phase": int(row[ENTITY_FLOATS + FALL_PHASE_OFFSET]),
            },
            "fight": {
                "round": int(row[FIGHT_OFFSET + 2]),
                "time_remaining_seconds": float(row[FIGHT_OFFSET + 5]),
                "player_clean_hits": int(row[FIGHT_OFFSET + 6]),
                "opponent_clean_hits": int(row[FIGHT_OFFSET + 7]),
                "player_falls": int(row[FIGHT_OFFSET + 8]),
                "opponent_falls": int(row[FIGHT_OFFSET + 9]),
            },
            "rewards": np.asarray(self.boundary.rewards, dtype=np.float32).astype(float).tolist(),
            "terminals": np.asarray(self.boundary.terminals, dtype=np.float32).astype(float).tolist(),
            "kick_controls": [dict(value) for value in KICK_METADATA],
        }


@dataclass(frozen=True)
class FighterQposMap:
    root_qpos_address: int
    joint_qpos_addresses: np.ndarray


class ObservationQposProjector:
    EXPECTED_DIMENSIONS = {
        "nbody": 63,
        "njnt": 60,
        "nq": 72,
        "nv": 70,
        "nu": 58,
        "ngeom": 91,
    }

    def __init__(self, model: Any) -> None:
        for field, expected in self.EXPECTED_DIMENSIONS.items():
            _require(int(getattr(model, field)) == expected, f"render model {field} mismatch")
        self.model = model
        self.maps: list[FighterQposMap] = []
        used: set[int] = set()
        for fighter, prefix in enumerate(("player__", "opponent__")):
            free_name = prefix + "joint__floating_base_joint_3081"
            try:
                free_joint_id = int(model.joint(free_name).id)
                joint_ids = [int(model.joint(prefix + suffix).id) for suffix in JOINT_NAME_SUFFIXES]
            except Exception as error:
                raise HumanEvalFailure(f"render model named joint mapping failed: {error}") from error
            root = int(model.jnt_qposadr[free_joint_id])
            _require(root == fighter * 36, "render model free-joint qpos address mismatch")
            addresses = np.asarray(
                [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids], dtype=np.int32
            )
            complete = set(range(root, root + 7)) | set(addresses.tolist())
            _require(len(addresses) == 29 and len(set(addresses.tolist())) == 29, "render joint map mismatch")
            _require(len(complete) == 36, "render fighter qpos map is incomplete")
            _require(not used.intersection(complete), "render fighter qpos maps overlap")
            used.update(complete)
            self.maps.append(FighterQposMap(root, addresses))
        _require(used == set(range(72)), "render qpos maps do not cover the exact model")

    def project(self, player_observation: np.ndarray) -> np.ndarray:
        row = np.asarray(player_observation, dtype=np.float32)
        _require(row.shape == (OBSERVATION_FLOATS,), "render observation row shape mismatch")
        _require(bool(np.isfinite(row).all()), "render observation row is nonfinite")
        qpos = np.asarray(self.model.qpos0, dtype=np.float64).copy()
        _require(qpos.shape == (72,), "render model qpos0 shape mismatch")
        for fighter, mapping in enumerate(self.maps):
            entity = row[fighter * ENTITY_FLOATS : (fighter + 1) * ENTITY_FLOATS]
            quaternion = entity[3:7].astype(np.float64)
            norm = float(np.linalg.norm(quaternion))
            _require(abs(norm - 1.0) <= 1e-3, "render quaternion is not unit length")
            qpos[mapping.root_qpos_address : mapping.root_qpos_address + 3] = entity[0:3]
            qpos[mapping.root_qpos_address + 3 : mapping.root_qpos_address + 7] = quaternion
            qpos[mapping.joint_qpos_addresses] = entity[13:42]
        return qpos


def _png_bytes(rgb: np.ndarray) -> bytes:
    height, width, channels = rgb.shape
    _require(channels == 3 and rgb.dtype == np.uint8, "render output is not uint8 RGB")
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 3))
        + chunk(b"IEND", b"")
    )


class PassiveObservationRenderer:
    def __init__(self, model_path: Path) -> None:
        try:
            import mujoco
        except ImportError as error:
            raise HumanEvalFailure("mujoco Python package is required for rendering") from error
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.projector = ObservationQposProjector(self.model)
        # The pinned arena XML declares MuJoCo's default 640-pixel offscreen
        # framebuffer width. Render within that immutable asset contract and
        # let the browser scale the image.
        self.renderer = mujoco.Renderer(self.model, height=360, width=640)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = 5.4
        self.camera.azimuth = 90.0
        self.camera.elevation = -30.0

    def frame(self, player_observation: np.ndarray) -> bytes:
        self.data.qpos[:] = self.projector.project(player_observation)
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        roots = np.vstack(
            [
                self.data.qpos[
                    mapping.root_qpos_address : mapping.root_qpos_address + 3
                ]
                for mapping in self.projector.maps
            ]
        )
        self.camera.lookat[:] = roots.mean(axis=0)
        self.camera.lookat[2] = max(0.75, float(self.camera.lookat[2]))
        self.renderer.update_scene(self.data, camera=self.camera)
        return _png_bytes(self.renderer.render())

    def close(self) -> None:
        self.renderer.close()


INDEX_HTML = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>REK G1 Semantic Candidate Human Eval</title>
<style>
  :root { color-scheme: dark; font-family: system-ui, sans-serif; }
  body { margin: 0; background: #0d1117; color: #e6edf3; }
  main { max-width: 1100px; margin: 0 auto; padding: 14px; }
  h1 { font-size: 20px; margin: 0 0 8px; }
  .warning { color: #f0c36a; line-height: 1.4; }
  .viewport { background: #05070a; border: 1px solid #30363d; }
  #frame { display: block; width: 100%; aspect-ratio: 16/9; object-fit: contain; }
  .hud { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 8px; margin: 10px 0; }
  .hud div { background: #161b22; padding: 8px; border-radius: 5px; }
  .label { display: block; color: #8b949e; font-size: 12px; }
  kbd,button { border: 1px solid #484f58; border-radius: 4px; padding: 3px 7px; }
  button { background: #21262d; color: #e6edf3; margin: 3px; cursor: pointer; }
  .controls { line-height: 1.8; }
  @media (max-width: 650px) { .hud { grid-template-columns: repeat(2,minmax(0,1fr)); } }
</style>
<main>
  <h1>REK G1 Semantic Candidate Human Eval</h1>
  <p class="warning">This is the public-family candidate, not accepted REK parity. Orange is a deterministic state-based approach/facing/kick candidate dummy, not Bot 1. Automatic get-up is disabled because current runtime get-up authority is unknown.</p>
  <div class="viewport"><img id="frame" alt="Passive rendering of candidate observations"></div>
  <div class="hud">
    <div><span class="label">Tick</span><span id="tick">0</span></div>
    <div><span class="label">Held</span><span id="held">neutral</span></div>
    <div><span class="label">Player action</span><span id="playerAction">1</span></div>
    <div><span class="label">Dummy action</span><span id="dummyAction">1</span></div>
    <div><span class="label">Player hits</span><span id="playerHits">0</span></div>
    <div><span class="label">Dummy hits</span><span id="dummyHits">0</span></div>
    <div><span class="label">Round time</span><span id="roundTime">0</span></div>
    <div><span class="label">Kick edge</span><span id="kickDisposition">none</span></div>
  </div>
  <p class="controls"><kbd>W</kbd>/<kbd>S</kbd> forward/back, <kbd>A</kbd>/<kbd>D</kbd> strafe, <kbd>Q</kbd>/<kbd>E</kbd> yaw. Movement keys are held inputs.</p>
  <div>
    <button data-move="6">6: move 6, left-side</button>
    <button data-move="7">7 or U: move 7, left-front</button>
    <button data-move="8">8 or I: move 8, right-side</button>
    <button data-move="9">9: move 9, right-knee</button>
    <button id="reset">Reset all four candidate arenas</button>
  </div>
  <p>The U and I aliases are user-confirmed convenience aliases. They are not a claim about the original REK keybind map.</p>
</main>
<script>
(() => {
  const held = new Set();
  const translations = new Set(['W','S','A','D']);
  const yaws = new Set(['Q','E']);
  const movement = new Map([
    ['KeyW','W'],['KeyS','S'],['KeyA','A'],['KeyD','D'],['KeyQ','Q'],['KeyE','E']
  ]);
  const kicks = new Map([
    ['Digit6',6],['Numpad6',6],['Digit7',7],['Numpad7',7],['KeyU',7],
    ['Digit8',8],['Numpad8',8],['KeyI',8],['Digit9',9],['Numpad9',9]
  ]);
  let sequence = 0;
  async function sendInput(kickMove = null) {
    const body = {sequence: ++sequence, held: Array.from(held).sort(), kick_move: kickMove};
    await fetch('/input', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(body)});
  }
  addEventListener('keydown', event => {
    const symbol = movement.get(event.code);
    if (symbol) {
      event.preventDefault();
      if (translations.has(symbol)) for (const key of translations) held.delete(key);
      if (yaws.has(symbol)) for (const key of yaws) held.delete(key);
      held.add(symbol);
      sendInput().catch(() => {});
      return;
    }
    const kick = kicks.get(event.code);
    if (kick && !event.repeat) {
      event.preventDefault();
      sendInput(kick).catch(() => {});
    }
  });
  addEventListener('keyup', event => {
    const symbol = movement.get(event.code);
    if (!symbol) return;
    event.preventDefault();
    held.delete(symbol);
    sendInput().catch(() => {});
  });
  addEventListener('blur', () => { held.clear(); sendInput().catch(() => {}); });
  for (const button of document.querySelectorAll('[data-move]')) {
    button.addEventListener('click', () => sendInput(Number(button.dataset.move)).catch(() => {}));
  }
  document.getElementById('reset').addEventListener('click', async () => {
    held.clear();
    await fetch('/reset', {
      method:'POST',
      headers:{'content-type':'application/json'},
      body:'{}'
    });
    sequence = 0;
  });
  const image = document.getElementById('frame');
  function nextFrame() { image.src = '/frame.png?t=' + Date.now(); }
  image.addEventListener('load', () => setTimeout(nextFrame, 30));
  image.addEventListener('error', () => setTimeout(nextFrame, 500));
  nextFrame();
  async function poll() {
    try {
      const state = await (await fetch('/state', {cache:'no-store'})).json();
      document.getElementById('tick').textContent = state.tick;
      document.getElementById('held').textContent = state.held.join('+') || 'neutral';
      document.getElementById('playerAction').textContent = state.last_actions[0];
      document.getElementById('dummyAction').textContent = state.last_actions[1];
      document.getElementById('playerHits').textContent = state.fight.player_clean_hits;
      document.getElementById('dummyHits').textContent = state.fight.opponent_clean_hits;
      document.getElementById('roundTime').textContent = state.fight.time_remaining_seconds.toFixed(2);
      document.getElementById('kickDisposition').textContent = state.last_kick_disposition;
    } catch (_) {}
    setTimeout(poll, 100);
  }
  poll();
})();
</script>
"""


def _require_http_request_boundary(
    headers: Mapping[str, str],
    port: int,
    *,
    require_json: bool,
) -> None:
    authority = f"{LOOPBACK_HOST}:{port}"
    _require(headers.get("Host") == authority, "HTTP Host is not the loopback authority")
    origin = headers.get("Origin")
    if origin is not None:
        _require(origin == f"http://{authority}", "HTTP Origin is not the loopback evaluator")
    if require_json:
        content_type = headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        _require(content_type == "application/json", "POST content type is not application/json")


class SemanticHumanEvaluator:
    def __init__(
        self,
        core: SemanticHumanEvalCore,
        renderer: Any,
        identity_report: Mapping[str, Any],
    ) -> None:
        self.core = core
        self.renderer = renderer
        self.identity_report = dict(identity_report)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.failure: str | None = None

    def start(self) -> None:
        _require(self.thread is None, "evaluation loop already started")
        self.thread = threading.Thread(target=self._run, name="rek-g1-human-eval", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        deadline = time.monotonic()
        try:
            while not self.stop_event.is_set():
                deadline += CONTROL_PERIOD_SECONDS
                with self.lock:
                    self.core.step_once()
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    self.stop_event.wait(remaining)
                else:
                    deadline = time.monotonic()
        except BaseException as error:
            with self.lock:
                self.failure = f"{type(error).__name__}: {error}"
            self.stop_event.set()

    def update_input(self, payload: Mapping[str, Any]) -> bool:
        with self.lock:
            return self.core.update_input(payload)

    def reset(self) -> None:
        with self.lock:
            self.core.reset()
            self.failure = None

    def state(self) -> dict[str, Any]:
        with self.lock:
            state = self.core.state()
            state["runtime_identity"] = self.identity_report
            state["ok"] = self.failure is None
            state["failure"] = self.failure
            return state

    def frame(self) -> bytes:
        with self.lock:
            observation = np.asarray(
                self.core.boundary.observations[0], dtype=np.float32
            ).copy()
        return self.renderer.frame(observation)

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        with self.lock:
            self.renderer.close()
            self.core.close_trace()
            self.core.boundary.close()


class HumanEvalHandler(BaseHTTPRequestHandler):
    evaluator: SemanticHumanEvaluator

    def _send(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: int, value: Mapping[str, Any]) -> None:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        self._send(status, "application/json", payload)

    def do_GET(self) -> None:
        try:
            _require_http_request_boundary(
                self.headers,
                int(self.server.server_port),
                require_json=False,
            )
            path = urllib.parse.urlsplit(self.path).path
            if path == "/":
                self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
            elif path == "/health":
                state = self.evaluator.state()
                self._json(200 if state["ok"] else 503, {"ok": state["ok"]})
            elif path == "/state":
                self._json(200, self.evaluator.state())
            elif path == "/frame.png":
                self._send(200, "image/png", self.evaluator.frame())
            else:
                self._json(404, {"error": "not found"})
        except HumanEvalFailure as error:
            self._json(503, {"error": str(error)})

    def do_POST(self) -> None:
        try:
            _require_http_request_boundary(
                self.headers,
                int(self.server.server_port),
                require_json=True,
            )
            path = urllib.parse.urlsplit(self.path).path
            if path == "/input":
                length = int(self.headers.get("Content-Length", "0"))
                _require(0 < length <= 8192, "input body length is invalid")
                payload = json.loads(self.rfile.read(length))
                _require(isinstance(payload, dict), "input body must be an object")
                accepted = self.evaluator.update_input(payload)
                self._json(200, {"ok": True, "accepted": accepted})
            elif path == "/reset":
                self.evaluator.reset()
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})
        except (HumanEvalFailure, UnicodeError, json.JSONDecodeError, ValueError) as error:
            self._json(400, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


def make_http_server(
    evaluator: SemanticHumanEvaluator,
    port: int,
    server_factory: Callable[..., HTTPServer] = HTTPServer,
) -> HTTPServer:
    _require(1 <= port <= 65535, "port is out of range")
    _require(ipaddress.ip_address(LOOPBACK_HOST).is_loopback, "server host is not loopback")

    # MuJoCo's EGL context belongs to the process main thread. Keep HTTP
    # handling on that thread; the 50 Hz simulation loop remains separate and
    # both paths serialize through the evaluator lock.

    class BoundHandler(HumanEvalHandler):
        pass

    BoundHandler.evaluator = evaluator
    return server_factory((LOOPBACK_HOST, port), BoundHandler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--semantic-assets", type=Path, required=True)
    parser.add_argument("--asset-manifest-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--encoder-sha256", required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--decoder-sha256", required=True)
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--max-steps", type=int, default=2_000_000_000)
    parser.add_argument("--physics-workers", type=int, default=4)
    parser.add_argument("--trace-out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    identity = RuntimeIdentity(
        extension=args.extension,
        extension_sha256=args.extension_sha256,
        semantic_assets=args.semantic_assets,
        asset_manifest_sha256=args.asset_manifest_sha256,
        model_sha256=args.model_sha256,
        encoder=args.encoder,
        encoder_sha256=args.encoder_sha256,
        decoder=args.decoder,
        decoder_sha256=args.decoder_sha256,
    )
    boundary = NativeVectorBoundary.open(
        identity,
        max_steps=args.max_steps,
        physics_workers=args.physics_workers,
    )
    renderer: PassiveObservationRenderer | None = None
    trace_writer: JsonlTraceWriter | None = None
    evaluator: SemanticHumanEvaluator | None = None
    server: HTTPServer | None = None
    try:
        renderer = PassiveObservationRenderer(boundary.identity.model)
        trace_writer = JsonlTraceWriter(args.trace_out, boundary.identity.report())
        evaluator = SemanticHumanEvaluator(
            SemanticHumanEvalCore(boundary, trace_writer),
            renderer,
            boundary.identity.report(),
        )
        server = make_http_server(evaluator, args.port)
        evaluator.start()
        print(
            json.dumps(
                {
                    "ready": True,
                    "host": LOOPBACK_HOST,
                    "port": args.port,
                    "rows": ROBOT_ROWS,
                    "opponent": CandidateApproachDummy.LABEL,
                    "opponent_is_bot_1": False,
                    "rek_parity_claim": False,
                    "trace": str(trace_writer.path),
                    "runtime_identity": boundary.identity.report(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            pass
    finally:
        if server is not None:
            server.server_close()
        if evaluator is not None:
            evaluator.close()
        else:
            if trace_writer is not None:
                trace_writer.close()
            if renderer is not None:
                renderer.close()
            boundary.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
