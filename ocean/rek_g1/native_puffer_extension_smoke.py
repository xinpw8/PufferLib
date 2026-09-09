"""Exercise the native REK G1 Puffer candidate through its Python ABI.

This is a diagnostic smoke test. It exercises build-pinned fall geometry,
keeps training disabled, and makes no REK parity claim. Runtime G1 recovery
availability is unknown, so this candidate provisionally selects no automatic
get-up. Its fallen path retains 10 percent of the live joint drive, counts, and
performs the recovered two-phase spawn reset. Heading forgiveness is the exact
zero recovered for this client build.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from types import ModuleType
from typing import Sequence

import numpy as np


ROBOT_ROWS = 8
OBSERVATION_FLOATS = 223
ENTITY_FLOATS = 86
BASE_ENTITY_FLOATS = 71
FALL_FLOATS = 15
ACTION_HEADS = 1
ACTION_CATEGORIES = 20
KICK_DURATION_TICKS = (157, 145, 158, 139)
CONTINUE_CATEGORY = 0
NEUTRAL_CATEGORY = 1
YAW_LEFT_CATEGORY = 6
YAW_RIGHT_CATEGORY = 7
KICK_MOVE_9_CATEGORY = 19
KICK_MOVE_9_ROUTE_ID = 10
SEMANTIC_OFFSET = 2 * ENTITY_FLOATS
EFFECTIVE_YAW_INDEX = SEMANTIC_OFFSET + 6
ACTIVE_ROUTE_INDEX = SEMANTIC_OFFSET + 7
ACTION_PLAYING_INDEX = SEMANTIC_OFFSET + 10
COMPOSER_BUSY_INDEX = SEMANTIC_OFFSET + 11


class SmokeFailure(RuntimeError):
    """The extension violated its diagnostic runtime contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise SmokeFailure(f"{variable} is required")
    unresolved = Path(value)
    if unresolved.is_symlink():
        raise SmokeFailure(f"{variable} must identify a regular non-symlink file")
    path = unresolved.resolve()
    if not path.is_file():
        raise SmokeFailure(f"{variable} must identify a regular non-symlink file")
    return path


def _required_directory(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise SmokeFailure(f"{variable} is required")
    unresolved = Path(value)
    if unresolved.is_symlink():
        raise SmokeFailure(f"{variable} must identify a regular non-symlink directory")
    path = unresolved.resolve()
    if not path.is_dir():
        raise SmokeFailure(f"{variable} must identify a regular non-symlink directory")
    return path


def _load_extension(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location("_C", path)
    if specification is None or specification.loader is None:
        raise SmokeFailure("failed to construct an extension module specification")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _float_view(pointer: int, rows: int, columns: int = 1) -> np.ndarray:
    if pointer == 0:
        raise SmokeFailure("native vector exposed a null buffer")
    values = rows * columns
    storage = (ctypes.c_float * values).from_address(pointer)
    return np.ctypeslib.as_array(storage).reshape(rows, columns)


def _byte_view(pointer: int, rows: int, columns: int) -> np.ndarray:
    if pointer == 0:
        raise SmokeFailure("native vector exposed a null byte buffer")
    storage = (ctypes.c_uint8 * (rows * columns)).from_address(pointer)
    return np.ctypeslib.as_array(storage).reshape(rows, columns)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _require_finite(observations: np.ndarray, phase: str) -> None:
    _require(
        bool(np.isfinite(observations).all()),
        f"{phase} observations contain a nonfinite value",
    )


def _require_shared_pair_views(observations: np.ndarray, phase: str) -> None:
    for arena_base in range(0, ROBOT_ROWS, 2):
        player = observations[arena_base]
        opponent = observations[arena_base + 1]
        _require(
            np.array_equal(
                player[:ENTITY_FLOATS],
                opponent[ENTITY_FLOATS : 2 * ENTITY_FLOATS],
            ),
            f"{phase} player self state differs from opponent paired view",
        )
        _require(
            np.array_equal(
                opponent[:ENTITY_FLOATS],
                player[ENTITY_FLOATS : 2 * ENTITY_FLOATS],
            ),
            f"{phase} opponent self state differs from player paired view",
        )


def _require_reset_fall_contract(observations: np.ndarray) -> None:
    self_fall = observations[
        :, BASE_ENTITY_FLOATS : BASE_ENTITY_FLOATS + FALL_FLOATS
    ]
    _require(
        bool(np.equal(self_fall[:, 0], 1.0).all()),
        "reset fall tracking is not active",
    )
    _require(
        bool(np.less_equal(np.abs(self_fall[:, 1]), 1e-4).all()),
        "reset tilt is not zero within tolerance",
    )
    _require(
        bool(np.less_equal(np.abs(self_fall[:, 2] - 1.0), 1e-6).all()),
        "reset pelvis height ratio is not one within tolerance",
    )
    _require(
        bool(np.equal(self_fall[:, 3], 1.0).all()),
        "reset both-feet-off-floor fact changed",
    )
    _require(
        bool(np.equal(self_fall[:, 4:7], 0.0).all()),
        "reset floor-contact facts changed",
    )
    _require(
        bool(np.equal(self_fall[:, 7], 0.0).all()),
        "provisional candidate CanGetUp setting is not false",
    )
    _require(
        bool(np.equal(self_fall[:, 8:11], 0.0).all()),
        "reset fall phase or elapsed timers changed",
    )
    _require(
        bool(np.equal(self_fall[:, 11], 3.0).all()),
        "reset fall timeout is not 3 seconds",
    )
    _require(
        bool(np.equal(self_fall[:, 12:15], 0.0).all()),
        "reset fall grace, recovery, or event state changed",
    )


def _step(vector: object, actions: np.ndarray) -> None:
    _require(actions.shape == (ROBOT_ROWS, ACTION_HEADS), "action shape mismatch")
    _require(actions.dtype == np.float32, "actions must be binary32")
    vector.cpu_step(int(actions.ctypes.data))


def _filled_action(category: int) -> np.ndarray:
    return np.full(
        (ROBOT_ROWS, ACTION_HEADS), float(category), dtype=np.float32
    )


def _row_actions(categories: Sequence[int]) -> np.ndarray:
    _require(len(categories) == ROBOT_ROWS, "row action count mismatch")
    _require(
        all(0 <= category < ACTION_CATEGORIES for category in categories),
        "row action category is out of range",
    )
    return np.asarray(categories, dtype=np.float32).reshape(
        ROBOT_ROWS, ACTION_HEADS
    )


def run_smoke(locomotion_steps: int, throughput_steps: int) -> dict[str, object]:
    extension_path = _required_file("REK_G1_PUFFER_EXTENSION")
    asset_root = _required_directory("REK_G1_SEMANTIC_ASSETS_DIR")
    encoder_path = _required_file("REK_G1_ENCODER_ONNX")
    decoder_path = _required_file("REK_G1_DECODER_ONNX")
    native = _load_extension(extension_path)

    arguments = {
        "vec": {"total_agents": ROBOT_ROWS, "num_buffers": 1},
        "env": {
            "max_steps": 500,
            "physics_workers": 4,
            "locomotion_segment_ticks": 1,
            "kick_move_6_duration_ticks": KICK_DURATION_TICKS[0],
            "kick_move_7_duration_ticks": KICK_DURATION_TICKS[1],
            "kick_move_8_duration_ticks": KICK_DURATION_TICKS[2],
            "kick_move_9_duration_ticks": KICK_DURATION_TICKS[3],
        },
    }
    vector = native.create_vec(arguments, 0)
    try:
        _require(native.env_name == "rek_g1", "extension environment mismatch")
        _require(vector.total_agents == ROBOT_ROWS, "agent row count mismatch")
        _require(vector.obs_size == OBSERVATION_FLOATS, "observation ABI mismatch")
        _require(vector.num_atns == ACTION_HEADS, "action head count mismatch")
        _require(vector.act_sizes == [ACTION_CATEGORIES], "action category ABI mismatch")
        _require(vector.obs_elem_size == 4, "observation element width mismatch")
        _require(vector.obs_dtype == "FloatTensor", "observation dtype mismatch")

        observations = _float_view(
            int(vector.obs_ptr), ROBOT_ROWS, OBSERVATION_FLOATS
        )
        rewards = _float_view(int(vector.rewards_ptr), ROBOT_ROWS)
        terminals = _float_view(int(vector.terminals_ptr), ROBOT_ROWS)
        _require(
            hasattr(vector, "action_mask_ptr") and hasattr(vector, "action_mask_size"),
            "Python vector ABI does not expose the native action mask",
        )
        _require(
            int(vector.action_mask_size) == ACTION_CATEGORIES,
            "native action-mask stride mismatch",
        )
        action_masks = _byte_view(
            int(vector.action_mask_ptr), ROBOT_ROWS, ACTION_CATEGORIES
        )

        vector.reset()
        reset_first = observations.copy()
        _require_finite(reset_first, "first reset")
        _require_shared_pair_views(reset_first, "first reset")
        _require_reset_fall_contract(reset_first)
        _require(bool(np.equal(rewards, 0.0).all()), "reset rewards are not zero")
        _require(bool(np.equal(terminals, 0.0).all()), "reset terminals are not zero")

        neutral = _filled_action(1)
        _step(vector, neutral)
        _require_finite(observations, "neutral step")
        vector.reset()
        reset_second = observations.copy()
        _require(
            np.array_equal(reset_first, reset_second),
            "native reset is not binary32 deterministic",
        )

        vector.reset()
        locomotion_reset = observations.copy()
        for _ in range(locomotion_steps):
            _step(vector, neutral)
            _require_finite(observations, "neutral locomotion baseline")
        neutral_locomotion_final = observations.copy()
        _require_shared_pair_views(
            neutral_locomotion_final, "neutral locomotion baseline"
        )
        neutral_root_displacement = np.linalg.norm(
            neutral_locomotion_final[:, :3].astype(np.float64)
            - locomotion_reset[:, :3].astype(np.float64),
            axis=1,
        )

        vector.reset()
        _require(
            np.array_equal(locomotion_reset, observations),
            "locomotion comparison did not start from the same reset state",
        )
        held_forward = _filled_action(2)
        for _ in range(locomotion_steps):
            _step(vector, held_forward)
            _require_finite(observations, "held-forward progression")
        locomotion_final = observations.copy()
        held_distinct_from_neutral = np.any(
            locomotion_final[:, :3] != neutral_locomotion_final[:, :3],
            axis=1,
        )
        _require_shared_pair_views(locomotion_final, "held-forward progression")
        held_root_displacement = np.linalg.norm(
            locomotion_final[:, :3].astype(np.float64)
            - locomotion_reset[:, :3].astype(np.float64),
            axis=1,
        )
        held_vs_neutral_root_endpoint = np.linalg.norm(
            locomotion_final[:, :3].astype(np.float64)
            - neutral_locomotion_final[:, :3].astype(np.float64),
            axis=1,
        )

        for category in range(1, 16):
            vector.reset()
            _step(vector, _filled_action(category))
            _require_finite(observations, f"locomotion category {category}")
            _require_shared_pair_views(
                observations, f"locomotion category {category}"
            )

        continuation = np.zeros((ROBOT_ROWS, ACTION_HEADS), dtype=np.float32)
        for kick_index, duration in enumerate(KICK_DURATION_TICKS):
            category = 16 + kick_index
            vector.reset()
            _step(vector, _filled_action(category))
            for _ in range(duration - 1):
                _step(vector, continuation)
            _require_finite(observations, f"kick category {category} traversal")
            _step(vector, neutral)
            _require_finite(observations, f"post-kick category {category}")
            _require_shared_pair_views(
                observations, f"post-kick category {category}"
            )

        # Exercise row-local held-yaw updates against the exact native mask.
        # The adapter's remaining-tick counter is not part of the Python ABI,
        # so the configured duration boundary is checked through observations.
        vector.reset()
        pre_kick_categories = (
            YAW_LEFT_CATEGORY,
            YAW_RIGHT_CATEGORY,
            YAW_LEFT_CATEGORY,
            YAW_RIGHT_CATEGORY,
            NEUTRAL_CATEGORY,
            NEUTRAL_CATEGORY,
            YAW_LEFT_CATEGORY,
            YAW_RIGHT_CATEGORY,
        )
        _step(vector, _row_actions(pre_kick_categories))
        pre_kick_yaw = observations[:, EFFECTIVE_YAW_INDEX].copy()
        _require(
            bool(np.greater(pre_kick_yaw[[0, 2, 6]], 0.0).all()),
            "pre-kick Q rows did not expose positive effective yaw",
        )
        _require(
            bool(np.less(pre_kick_yaw[[1, 3, 7]], 0.0).all()),
            "pre-kick E rows did not expose negative effective yaw",
        )
        _require(
            bool(np.equal(pre_kick_yaw[[4, 5]], 0.0).all()),
            "pre-kick neutral rows exposed yaw",
        )

        kick_duration = KICK_DURATION_TICKS[3]
        _step(vector, _filled_action(KICK_MOVE_9_CATEGORY))
        _require_finite(observations, "held-yaw kick start")
        _require(
            bool(np.equal(terminals, 0.0).all()),
            "held-yaw kick start terminated an arena",
        )
        _require(
            bool(np.equal(observations[:, EFFECTIVE_YAW_INDEX], 0.0).all()),
            "kick start did not suppress effective yaw",
        )
        _require(
            bool(
                np.equal(
                    observations[:, ACTIVE_ROUTE_INDEX],
                    float(KICK_MOVE_9_ROUTE_ID),
                ).all()
            ),
            "kick start did not select the move-9 route",
        )
        _require(
            bool(np.equal(observations[:, ACTION_PLAYING_INDEX], 1.0).all()),
            "move-9 action was not playing after its start tick",
        )
        expected_active_kick_mask = np.zeros(ACTION_CATEGORIES, dtype=np.uint8)
        expected_active_kick_mask[
            [CONTINUE_CATEGORY, NEUTRAL_CATEGORY, YAW_LEFT_CATEGORY, YAW_RIGHT_CATEGORY]
        ] = 1
        _require(
            bool(np.equal(action_masks, expected_active_kick_mask).all()),
            "active-kick native action mask is not exactly continue/neutral/Q/E",
        )

        active_kick_categories = (
            CONTINUE_CATEGORY,
            CONTINUE_CATEGORY,
            NEUTRAL_CATEGORY,
            NEUTRAL_CATEGORY,
            YAW_LEFT_CATEGORY,
            YAW_RIGHT_CATEGORY,
            YAW_RIGHT_CATEGORY,
            YAW_LEFT_CATEGORY,
        )
        active_kick_actions = _row_actions(active_kick_categories)
        for elapsed_tick in range(1, kick_duration):
            _step(vector, active_kick_actions)
            _require_finite(observations, "held-yaw active kick")
            _require(
                bool(np.equal(terminals, 0.0).all()),
                "held-yaw active kick terminated an arena",
            )
            _require(
                bool(
                    np.equal(
                        observations[:, EFFECTIVE_YAW_INDEX], 0.0
                    ).all()
                ),
                "active kick exposed effective yaw",
            )
            _require(
                bool(
                    np.equal(
                        observations[:, ACTIVE_ROUTE_INDEX],
                        float(KICK_MOVE_9_ROUTE_ID),
                    ).all()
                ),
                "active-kick yaw update changed route identity",
            )
            expected_playing = 1.0 if elapsed_tick + 1 < kick_duration else 0.0
            _require(
                bool(
                    np.equal(
                        observations[:, ACTION_PLAYING_INDEX],
                        expected_playing,
                    ).all()
                ),
                "move-9 traversal did not complete at its configured boundary",
            )
            if elapsed_tick + 1 < kick_duration:
                _require(
                    bool(np.equal(action_masks, expected_active_kick_mask).all()),
                    "active-kick action mask changed before the duration boundary",
                )

        _require(
            bool(np.equal(observations[:, COMPOSER_BUSY_INDEX], 1.0).all()),
            "completed non-loop kick layer stopped being busy before replacement",
        )
        post_kick_categories = (
            YAW_LEFT_CATEGORY,
            YAW_RIGHT_CATEGORY,
            NEUTRAL_CATEGORY,
            NEUTRAL_CATEGORY,
            YAW_LEFT_CATEGORY,
            YAW_RIGHT_CATEGORY,
            YAW_RIGHT_CATEGORY,
            YAW_LEFT_CATEGORY,
        )
        _step(vector, _row_actions(post_kick_categories))
        _require_finite(observations, "held-yaw post-kick resume")
        post_kick_yaw = observations[:, EFFECTIVE_YAW_INDEX].copy()
        expected_post_kick_yaw = np.asarray(
            (1.0, -1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0),
            dtype=np.float32,
        )
        _require(
            bool(np.array_equal(post_kick_yaw, expected_post_kick_yaw)),
            "post-kick desired yaw did not resume row-locally",
        )
        _require_shared_pair_views(observations, "held-yaw post-kick resume")

        vector.reset()
        long_neutral_reset = observations.copy()
        start = time.perf_counter()
        for _ in range(throughput_steps):
            _step(vector, neutral)
        elapsed = time.perf_counter() - start
        long_neutral_final = observations.copy()
        _require(math.isfinite(elapsed) and elapsed > 0.0, "invalid timing result")
        _require_finite(observations, "throughput run")
        _require(bool(np.equal(rewards, 0.0).all()), "diagnostic rewards changed")
        _require(bool(np.equal(terminals, 0.0).all()), "diagnostic terminals changed")

        return {
            "schema": "rek.g1_native_puffer_extension_smoke.v1",
            "classification": (
                "public_family_candidate_with_pinned_fall_geometry_"
                "and_provisional_recovery"
            ),
            "rek_parity_claim": False,
            "training_enabled": False,
            "abi": {
                "robot_rows": ROBOT_ROWS,
                "observation_float32": OBSERVATION_FLOATS,
                "action_heads": ACTION_HEADS,
                "action_categories": ACTION_CATEGORIES,
            },
            "runtime_configuration": {
                "fall_measurement": "build_pinned_mujoco_root_floor_contacts",
                "runtime_get_up_authority": "unknown",
                "candidate_can_get_up": False,
                "candidate_no_recovery_count_seconds": 3.0,
                "fallen_live_drive_retention": 0.1,
                "fight_spawn_reset": (
                    "root_teleport_then_joint_and_controller_reset_"
                    "after_next_2ms_boundary"
                ),
                "locomotion_segment_ticks": 1,
                "reconstructed_compositor_kick_duration_ticks": list(
                    KICK_DURATION_TICKS
                ),
            },
            "checks": {
                "finite_observations": True,
                "binary32_deterministic_reset": True,
                "shared_arena_pair_views": True,
                "candidate_reset_fall_contract": True,
                "held_forward_distinct_from_neutral_baseline": bool(
                    held_distinct_from_neutral.all()
                ),
                "all_20_action_categories_exercised": True,
                "active_kick_allowed_categories_exercised": True,
                "active_kick_effective_yaw_suppressed": True,
                "active_kick_route_identity_preserved": True,
                "active_kick_configured_duration_boundary_observed": True,
                "post_kick_candidate_yaw_state_row_local": True,
                "python_extension_action_mask_exact": True,
                "zero_diagnostic_rewards": True,
                "zero_diagnostic_terminals": True,
            },
            "active_kick_adapter_probe": {
                "kick_category": KICK_MOVE_9_CATEGORY,
                "kick_route_id": KICK_MOVE_9_ROUTE_ID,
                "configured_duration_ticks": kick_duration,
                "accepted_categories_exercised": sorted(
                    set(active_kick_categories)
                ),
                "accepted_category_meanings": [
                    "continue",
                    "neutral",
                    "Q",
                    "E",
                ],
                "pre_kick_effective_yaw": pre_kick_yaw.tolist(),
                "post_kick_effective_yaw": post_kick_yaw.tolist(),
                "python_extension_action_mask": {
                    "status": "exposed_and_checked",
                    "exact_mask_claim": True,
                    "stride": ACTION_CATEGORIES,
                    "active_kick_allowed_categories": [
                        CONTINUE_CATEGORY,
                        NEUTRAL_CATEGORY,
                        YAW_LEFT_CATEGORY,
                        YAW_RIGHT_CATEGORY,
                    ],
                },
                "remaining_tick_counter": {
                    "status": "not_exposed",
                    "exact_counter_claim": False,
                    "observed_boundary": (
                        "kick action-playing stayed active before the configured "
                        "boundary, cleared on it, and yaw resumed on the next tick"
                    ),
                },
                "authority": (
                    "provisional candidate adapter behavior; not recovered REK parity"
                ),
            },
            "motion": {
                "locomotion_steps": locomotion_steps,
                "neutral_root_displacement": neutral_root_displacement.tolist(),
                "held_forward_root_displacement": held_root_displacement.tolist(),
                "held_vs_neutral_root_endpoint": (
                    held_vs_neutral_root_endpoint.tolist()
                ),
                "held_distinct_from_neutral_by_row": (
                    held_distinct_from_neutral.tolist()
                ),
                "neutral_semantic_tail": (
                    neutral_locomotion_final[
                        :, 2 * ENTITY_FLOATS + 4 : OBSERVATION_FLOATS
                    ].tolist()
                ),
                "held_forward_semantic_tail": (
                    locomotion_final[
                        :, 2 * ENTITY_FLOATS + 4 : OBSERVATION_FLOATS
                    ].tolist()
                ),
                "held_vs_neutral_observation_linf": np.max(
                    np.abs(
                        locomotion_final.astype(np.float64)
                        - neutral_locomotion_final.astype(np.float64)
                    ),
                    axis=1,
                ).tolist(),
                "held_vs_neutral_joint_position_linf": np.max(
                    np.abs(
                        locomotion_final[:, 13:42].astype(np.float64)
                        - neutral_locomotion_final[:, 13:42].astype(np.float64)
                    ),
                    axis=1,
                ).tolist(),
            },
            "long_neutral_state": {
                "steps": throughput_steps,
                "reset_root_z": long_neutral_reset[:, 2].tolist(),
                "end_root_z": long_neutral_final[:, 2].tolist(),
                "reset_root_quaternion_wxyz": (
                    long_neutral_reset[:, 3:7].tolist()
                ),
                "end_root_quaternion_wxyz": (
                    long_neutral_final[:, 3:7].tolist()
                ),
            },
            "throughput": {
                "vector_steps": throughput_steps,
                "elapsed_seconds": elapsed,
                "vector_steps_per_second": throughput_steps / elapsed,
                "robot_steps_per_second": ROBOT_ROWS * throughput_steps / elapsed,
            },
            "input_sha256": {
                "extension": _sha256(extension_path),
                "encoder": _sha256(encoder_path),
                "decoder": _sha256(decoder_path),
                "asset_manifest": _sha256(
                    asset_root / "semantic_duel_assets_manifest.json"
                ),
            },
        }
    finally:
        vector.close()


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locomotion-steps", type=_positive_integer, default=64)
    parser.add_argument("--throughput-steps", type=_positive_integer, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_smoke(arguments.locomotion_steps, arguments.throughput_steps)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
