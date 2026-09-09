"""Compare the native two-G1 rollout with an independent Python rollout.

Both paths run the same public-family candidate controller and the same
build-pinned initial-spawn arena.  This is an implementation-equivalence test,
not a comparison against a recorded REK trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import g1_two_fighter_arena as duel
import gear_sonic_candidate as candidate
from gear_sonic_vector_env import GearSonicController
import sonic_candidate as plant


@dataclass
class _RobotState:
    heading_delta_wxyz: np.ndarray
    history: candidate.StateHistory
    last_action_policy: np.ndarray
    policy_tick: int = 0
    command_lpf_state_mujoco: np.ndarray | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--spawn-contract", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--motion-role", required=True)
    parser.add_argument("--frame-mode", choices=("clamp", "loop"), required=True)
    parser.add_argument("--arenas", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--native-qpos", type=Path, required=True)
    parser.add_argument("--native-qvel", type=Path, required=True)
    parser.add_argument("--native-action", type=Path, required=True)
    parser.add_argument("--native-target", type=Path, required=True)
    parser.add_argument("--state-atol", type=float, default=1e-9)
    return parser


def _read_exact(path: Path, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path} has {actual_bytes} bytes; expected exactly {expected_bytes}"
        )
    value = np.fromfile(path, dtype=dtype).reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f"{path} contains non-finite values")
    return value


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _comparison(
    reference: np.ndarray,
    native: np.ndarray,
    *,
    absolute_tolerance: float,
    require_exact: bool,
) -> dict[str, Any]:
    if reference.shape != native.shape or reference.dtype != native.dtype:
        raise ValueError(
            "comparison arrays differ in shape or dtype: "
            f"{reference.shape} {reference.dtype} versus {native.shape} {native.dtype}"
        )
    exact = bool(np.array_equal(reference, native))
    difference = np.abs(reference.astype(np.float64) - native.astype(np.float64))
    maximum = float(difference.max(initial=0.0))
    equivalent = exact if require_exact else maximum <= absolute_tolerance
    mismatch = np.argwhere(reference != native)
    first_mismatch = None
    if mismatch.size:
        location = tuple(int(index) for index in mismatch[0])
        first_mismatch = {
            "index": list(location),
            "python": float(reference[location]),
            "native": float(native[location]),
            "absolute_error": float(difference[location]),
        }
    return {
        "absolute_error_max": maximum,
        "absolute_tolerance": absolute_tolerance,
        "equivalent": equivalent,
        "exact": exact,
        "exact_required": require_exact,
        "first_mismatch": first_mismatch,
        "mismatch_count": int(mismatch.shape[0]),
        "native_sha256": _sha256(native),
        "python_sha256": _sha256(reference),
    }


def _frame(tick: int, frame_count: int, loop: bool) -> int:
    return tick % frame_count if loop else min(tick, frame_count - 1)


def _initialize(
    reference: duel.TwoFighterArenaReference,
    motion: plant.MotionData,
    arenas: int,
) -> tuple[list[Any], list[_RobotState]]:
    model = reference.model
    mujoco = reference.mujoco
    data_rows = [mujoco.MjData(model) for _ in range(arenas)]
    states: list[_RobotState] = []
    reference_quaternion = plant._xyzw_to_wxyz(motion.root_rot_xyzw[0]).astype(
        np.float64
    )
    for data in data_rows:
        mujoco.mj_resetData(model, data)
        for role in duel.ROLES:
            runtime_map = reference.runtime_maps[role]
            expected = np.asarray(
                reference.contract.spawn_rebase(role).free_joint_qpos_prefix,
                dtype=np.float64,
            )
            root = runtime_map.root_qpos_address
            if not np.allclose(
                np.asarray(data.qpos[root : root + 7]),
                expected,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"{role} qpos0 spawn prefix changed before reset")
            joint_position, _ = candidate.clip_targets_to_joint_ranges(
                model, runtime_map, motion.dof_pos[0]
            )
            data.qpos[runtime_map.qpos_addresses] = joint_position
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        for role in duel.ROLES:
            runtime_map = reference.runtime_maps[role]
            root = runtime_map.root_qpos_address
            expected = np.asarray(
                reference.contract.spawn_rebase(role).free_joint_qpos_prefix,
                dtype=np.float64,
            )
            if not np.allclose(
                np.asarray(data.qpos[root : root + 7]),
                expected,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"{role} qpos0 spawn prefix changed during initialization")
            states.append(
                _RobotState(
                    heading_delta_wxyz=candidate.reference_heading_delta(
                        np.asarray(data.qpos[root + 3 : root + 7], dtype=np.float64),
                        reference_quaternion,
                    ),
                    history=candidate.StateHistory(),
                    last_action_policy=np.zeros(candidate.ACTION_DIM, dtype=np.float32),
                )
            )
    return data_rows, states


def _rollout(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference = duel.create_two_fighter_arena_reference(
        xml=args.xml,
        arena=args.arena,
        spawn_contract=args.spawn_contract,
        runtime_manifest=args.manifest,
    )
    duel.assert_spawn_rebase_applied(reference)
    motion = plant.load_motion(args.assets_dir, args.motion_role, args.manifest)
    model = reference.model
    mujoco = reference.mujoco
    model.opt.timestep = candidate.PHYSICS_DT
    for role in duel.ROLES:
        runtime_map = reference.runtime_maps[role]
        limits = candidate.resolve_force_limits(runtime_map, "public-model-config")
        candidate.configure_native_position_actuators(
            mujoco, model, runtime_map, limits
        )
    encoder, encoder_version = candidate._create_session(args.encoder)
    decoder, decoder_version = candidate._create_session(args.decoder)
    if decoder_version != encoder_version:
        raise ValueError("encoder and decoder ONNX Runtime versions differ")
    controller = GearSonicController(encoder, decoder)
    data_rows, states = _initialize(reference, motion, args.arenas)
    loop = args.frame_mode == "loop"
    frame_count = int(motion.dof_pos.shape[0])
    final_actions: np.ndarray | None = None
    final_targets: np.ndarray | None = None
    interval = candidate.native_scheduler_interval(
        candidate.PHYSICS_DT, candidate.NATIVE_WORK_RATE_HZ
    )
    alpha = candidate.command_lpf_alpha(
        candidate.COMMAND_LPF_CUTOFF_HZ, interval * candidate.PHYSICS_DT
    )
    if interval != 2:
        raise ValueError(f"unexpected command LPF interval {interval}")

    for _ in range(args.steps):
        encoder_rows: list[np.ndarray] = []
        for arena_index, data in enumerate(data_rows):
            for fighter_index, role in enumerate(duel.ROLES):
                row = arena_index * 2 + fighter_index
                state = states[row]
                runtime_map = reference.runtime_maps[role]
                state.history.append(
                    candidate.state_to_history_entry(
                        mujoco,
                        model,
                        data,
                        runtime_map,
                        state.last_action_policy,
                    )
                )
                root = runtime_map.root_qpos_address
                observation, _indices = candidate.build_encoder_observation(
                    motion,
                    _frame(state.policy_tick, frame_count, loop),
                    np.asarray(data.qpos[root + 3 : root + 7], dtype=np.float64),
                    state.heading_delta_wxyz,
                    loop=loop,
                )
                encoder_rows.append(observation)
        tokens = controller.encode(
            np.ascontiguousarray(np.stack(encoder_rows), dtype=np.float32)
        )
        decoder_rows = np.ascontiguousarray(
            np.stack(
                [
                    candidate.build_decoder_observation(tokens[row], states[row].history)
                    for row in range(len(states))
                ]
            ),
            dtype=np.float32,
        )
        raw_actions = controller.decode(decoder_rows)
        clipped_actions = np.empty_like(raw_actions)
        targets = np.empty_like(raw_actions)
        for row in range(len(states)):
            clipped_actions[row], targets[row] = candidate.action_to_targets(
                raw_actions[row]
            )

        for physics_substep in range(10):
            for row, state in enumerate(states):
                if physics_substep % interval == 0:
                    if state.command_lpf_state_mujoco is None:
                        state.command_lpf_state_mujoco = targets[row].copy()
                    else:
                        candidate.update_command_lpf(
                            state.command_lpf_state_mujoco, targets[row], alpha
                        )
            for arena_index, data in enumerate(data_rows):
                data.ctrl[:] = 0.0
                for fighter_index, role in enumerate(duel.ROLES):
                    row = arena_index * 2 + fighter_index
                    filtered = states[row].command_lpf_state_mujoco
                    if filtered is None:
                        raise ValueError("command LPF was not initialized")
                    applied, _ = candidate.clip_targets_to_joint_ranges(
                        model, reference.runtime_maps[role], filtered
                    )
                    data.ctrl[reference.runtime_maps[role].actuator_ids] = applied
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("Python duel state became non-finite")
        for row, state in enumerate(states):
            state.last_action_policy = clipped_actions[row].copy()
            state.policy_tick += 1
        final_actions = raw_actions.copy()
        final_targets = targets.copy()

    if final_actions is None or final_targets is None:
        raise ValueError("rollout did not execute")
    qpos = np.ascontiguousarray(
        np.stack([np.asarray(data.qpos).copy() for data in data_rows]),
        dtype=np.dtype("<f8"),
    )
    qvel = np.ascontiguousarray(
        np.stack([np.asarray(data.qvel).copy() for data in data_rows]),
        dtype=np.dtype("<f8"),
    )
    return (
        qpos,
        qvel,
        np.ascontiguousarray(final_actions, dtype=np.dtype("<f4")),
        np.ascontiguousarray(final_targets, dtype=np.dtype("<f4")),
    )


def main() -> int:
    args = _parser().parse_args()
    if args.arenas < 1 or args.steps < 1:
        raise ValueError("arenas and steps must be positive")
    if not np.isfinite(args.state_atol) or args.state_atol < 0.0:
        raise ValueError("state-atol must be finite and nonnegative")
    robot_count = args.arenas * 2
    python_qpos, python_qvel, python_action, python_target = _rollout(args)
    native_qpos = _read_exact(
        args.native_qpos, np.dtype("<f8"), (args.arenas, 72)
    )
    native_qvel = _read_exact(
        args.native_qvel, np.dtype("<f8"), (args.arenas, 70)
    )
    native_action = _read_exact(
        args.native_action, np.dtype("<f4"), (robot_count, 29)
    )
    native_target = _read_exact(
        args.native_target, np.dtype("<f4"), (robot_count, 29)
    )
    comparisons = {
        "action": _comparison(
            python_action, native_action, absolute_tolerance=0.0, require_exact=True
        ),
        "qpos": _comparison(
            python_qpos,
            native_qpos,
            absolute_tolerance=args.state_atol,
            require_exact=False,
        ),
        "qvel": _comparison(
            python_qvel,
            native_qvel,
            absolute_tolerance=args.state_atol,
            require_exact=False,
        ),
        "target": _comparison(
            python_target, native_target, absolute_tolerance=0.0, require_exact=True
        ),
    }
    equivalent = all(value["equivalent"] for value in comparisons.values())
    report = {
        "arena_count": args.arenas,
        "classification": "public_family_candidate",
        "comparisons": comparisons,
        "equivalent": equivalent,
        "frame_mode": args.frame_mode,
        "motion_role": args.motion_role,
        "physical_robot_count": robot_count,
        "policy_steps": args.steps,
        "shared_contact_physics_per_arena": True,
        "spawn_prefixes_verified": True,
        "rek_parity_claim": False,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
