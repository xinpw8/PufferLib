#!/usr/bin/env python3
"""Export raw simulator root motion for one verified locomotion command.

The runner is the deterministic, non-realtime equivalent of
``PolicyHumanEval`` in ``human_eval_server.py``.  It uses the same official
EngineAI walking and supine-recovery controllers at 100 Hz and recomputes PD
torque before every 500 Hz MuJoCo step.  It records the root transform after
every 2 ms physics step without interpolation.

Only the six pinned locomotion selectors are supported.  Every selector runs
as an isolated same-reset trial with one press and one release.  This program
does not synthesize combat trajectories and its output makes no parity claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from engineai_t800_policy import (
    CONTROL_DT,
    DEFAULT_Q,
    OFFICIAL_DEFAULT_ROOT_CLEARANCE_M,
    RECOVERY_POLICY_SHA256,
    RECOVERY_TRAJECTORY_SHA256,
    T800MuJoCoBinding,
    T800SupineRecoveryController,
    T800WalkingController,
    world_box_top_height,
)


TRACE_SCHEMA = "rek.paired_motion_trace.v1"
CALIBRATION_SCHEMA = "rek.paired_motion_calibration.v1"
TRIAL_SCHEMA = "rek.clone.single_press_trial.v1"
RESET_SCHEMA = "rek.clone.composite_reset.v1"
RUNTIME_SCHEMA = "rek.clone.runtime_provenance.v1"
SOURCE = "clone:rek_fight_engineai"
PHYSICS_DT = 0.002
EXPECTED_MJCF_SHA256 = (
    "01caa6ed90277a90fc71b4c16d7959fb70f7702fa15bea7c68d3eaa9d5f27b2c"
)
EXPECTED_WALKING_POLICY_SHA256 = (
    "cbcb90f86dbb2fde39bdc5a25c8d0530d5c79c7a8f84b1f90863d8c9065b6427"
)
EXPECTED_CALIBRATION_SHA256 = (
    "a1c1d2321b34035b1ad7fb60770ac578c6cfa1b45b60b4e5619e9aa9615ed129"
)
ENGINEAI_SDK_COMMIT = "335c60e88772c26c7852d0abd6b3c7439037dd8f"
TRIAL_PROTOCOL_ID = "rek.clone.isolated_same_reset_single_press.50hz.v1"
TRIAL_RATE_HZ = 50.0
TRIAL_FIXED_SUBSTEPS = 10
TRIAL_NEUTRAL_START_TICK = 0
TRIAL_PRESS_TICK = 50
TRIAL_RELEASE_TICK = 150
TRIAL_STOP_TICK = 200
RESET_KEYFRAME_ID = 0
RESET_KEYFRAME_NAME = "client_observed_round2_first_active"
ARENA_SUPPORT_GEOM_NAME = "arena_Collider_Floor_Rektagon"
NEUTRAL_COMMAND = (0.0, 0.0, 0.0)
TRIAL_PROTOCOL_DEFINITION = {
    "schema": TRIAL_SCHEMA,
    "identity": TRIAL_PROTOCOL_ID,
    "trial_rate_hz": TRIAL_RATE_HZ,
    "neutral_start_trial_tick": TRIAL_NEUTRAL_START_TICK,
    "press_trial_tick": TRIAL_PRESS_TICK,
    "release_trial_tick": TRIAL_RELEASE_TICK,
    "stop_trial_tick": TRIAL_STOP_TICK,
    "controlled_fighter_initialization": "same_composite_reset_for_every_selector",
    "controlled_fighter_command_contract": (
        "neutral_pre_roll_then_one_non_neutral_press_then_neutral_release"
    ),
    "attack_command_count": 0,
}
RESET_PROCEDURE_STEPS = [
    {
        "sequence": 1,
        "operation": "mj_resetDataKeyframe",
        "keyframe_id": RESET_KEYFRAME_ID,
        "keyframe_name": RESET_KEYFRAME_NAME,
    },
    {
        "sequence": 2,
        "operation": "T800MuJoCoBinding.set_sdk_standing_state",
        "fighter_index": 0,
        "arena_placement": "retain_keyframe_xy_and_heading",
        "state_authority": "engineai_sdk_model",
    },
    {
        "sequence": 3,
        "operation": "T800WalkingController.reset",
        "fighter_index": 0,
    },
    {
        "sequence": 4,
        "operation": "T800MuJoCoBinding.set_sdk_standing_state",
        "fighter_index": 1,
        "arena_placement": "retain_keyframe_xy_and_heading",
        "state_authority": "engineai_sdk_model",
    },
    {
        "sequence": 5,
        "operation": "T800WalkingController.reset",
        "fighter_index": 1,
    },
    {"sequence": 6, "operation": "set_recovering_flags_false"},
    {"sequence": 7, "operation": "mj_forward"},
    {"sequence": 8, "operation": "set_trial_counters_zero"},
]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    identity: str
    normalized_command: tuple[float, float, float]


# These signs deliberately match MeasuredSchedule in RekUiBridgeAgent and
# V6_VELOCITY_IDENTITIES in client_fixed_import.py.  They are not inferred from
# screen-space motion or fitted to a reference trace.
COMMANDS = {
    "forward": CommandSpec("forward", "walk_forward:press:v1", (1.0, 0.0, 0.0)),
    "backward": CommandSpec(
        "backward", "walk_backward:press:v1", (-1.0, 0.0, 0.0)
    ),
    "strafe-left": CommandSpec(
        "strafe-left", "strafe_left:press:v1", (0.0, -1.0, 0.0)
    ),
    "strafe-right": CommandSpec(
        "strafe-right", "strafe_right:press:v1", (0.0, 1.0, 0.0)
    ),
    "yaw-left": CommandSpec(
        "yaw-left", "yaw_left:press:v1", (0.0, 0.0, -1.0)
    ),
    "yaw-right": CommandSpec(
        "yaw-right", "yaw_right:press:v1", (0.0, 0.0, 1.0)
    ),
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


TRIAL_PROTOCOL_SHA256 = _canonical_sha256(TRIAL_PROTOCOL_DEFINITION)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_sha256(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"unexpected {label} SHA-256: expected {expected}, got {actual}"
        )
    return actual


def _quaternion_xyzw(quaternion_wxyz: Sequence[float]) -> list[float]:
    w, x, y, z = quaternion_wxyz
    return [float(x), float(y), float(z), float(w)]


def _array_sha256(values: Any) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _named_array_state_sha256(fields: Sequence[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, values in fields:
        array = np.asarray(values, dtype="<f8").reshape(-1)
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(array.size).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"runtime component is not a file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_identity(
    module: Any,
    *,
    distribution: str,
    runtime_version: str | None = None,
    extra_native_prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    name = str(getattr(module, "__name__", ""))
    module_file_raw = getattr(module, "__file__", None)
    if not name or not isinstance(module_file_raw, str) or not module_file_raw:
        raise RuntimeError(f"cannot identify loaded {distribution} module")
    versions: dict[str, str] = {}
    api_version = getattr(module, "__version__", None)
    if isinstance(api_version, str) and api_version:
        versions["module_api"] = api_version
    distribution_version = _distribution_version(distribution)
    if distribution_version:
        versions["python_distribution"] = distribution_version
    if runtime_version:
        versions["native_runtime_api"] = runtime_version
    if not versions:
        raise RuntimeError(f"cannot establish loaded {distribution} version")

    native_suffixes = (".so", ".pyd", ".dll", ".dylib")
    prefixes = (name, *extra_native_prefixes)
    native_paths: dict[str, Path] = {}
    for loaded_name, loaded_module in tuple(sys.modules.items()):
        if not any(
            loaded_name == prefix or loaded_name.startswith(prefix + ".")
            for prefix in prefixes
        ):
            continue
        loaded_path_raw = getattr(loaded_module, "__file__", None)
        if not isinstance(loaded_path_raw, str) or not loaded_path_raw:
            continue
        loaded_path = Path(loaded_path_raw)
        if not loaded_path.name.lower().endswith(native_suffixes):
            continue
        try:
            resolved = loaded_path.resolve(strict=True)
        except OSError:
            continue
        native_paths[str(resolved)] = resolved
    module_path = Path(module_file_raw).resolve(strict=True)
    if module_path.name.lower().endswith(native_suffixes):
        native_paths[str(module_path)] = module_path
    if not native_paths:
        raise RuntimeError(f"cannot identify a loaded native {distribution} binary")
    return {
        "module": name,
        "versions": versions,
        "module_file": _file_identity(module_path),
        "loaded_native_components": [
            _file_identity(path)
            for _, path in sorted(native_paths.items())
        ],
    }


def collect_runtime_provenance(mujoco_module: Any, mnn_module: Any) -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    record = {
        "schema": RUNTIME_SCHEMA,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": _file_identity(executable),
        },
        "numpy": _module_identity(np, distribution="numpy"),
        "mujoco": _module_identity(
            mujoco_module,
            distribution="mujoco",
            runtime_version=str(mujoco_module.mj_versionString()),
        ),
        "mnn": _module_identity(
            mnn_module,
            distribution="MNN",
            extra_native_prefixes=("_mnncengine",),
        ),
    }
    record["identity_sha256"] = _canonical_sha256(record)
    return record


class EngineAIMotionRunner:
    """Deterministic two-fighter EngineAI controller and MuJoCo runner."""

    def __init__(
        self,
        model_path: Path,
        walking_policy_path: Path,
        recovery_policy_path: Path,
        recovery_trajectory_path: Path,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model_path = model_path.resolve(strict=True)
        self.model_sha256 = _checked_sha256(
            self.model_path,
            EXPECTED_MJCF_SHA256,
            "two-fighter diagnostic MJCF",
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if self.model.nkey != 1:
            raise ValueError(f"expected exactly one reset keyframe, got {self.model.nkey}")
        self.keyframe_name = mujoco.mj_id2name(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            RESET_KEYFRAME_ID,
        )
        if self.keyframe_name != RESET_KEYFRAME_NAME:
            raise ValueError(
                f"expected reset keyframe {RESET_KEYFRAME_NAME!r}, "
                f"got {self.keyframe_name!r}"
            )
        keyframe_qpos_sha256 = _array_sha256(
            self.model.key_qpos[RESET_KEYFRAME_ID]
        )
        self.keyframe_qpos_sha256 = keyframe_qpos_sha256
        self.source_timestep = float(self.model.opt.timestep)
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)
        self.support_height_m = world_box_top_height(
            self.model,
            mujoco,
            ARENA_SUPPORT_GEOM_NAME,
        )
        self.bindings = [
            T800MuJoCoBinding(mujoco, self.model, fighter=fighter)
            for fighter in range(2)
        ]
        self.controllers = [T800WalkingController(walking_policy_path) for _ in range(2)]
        self.recovery_controllers = [
            T800SupineRecoveryController(
                recovery_policy_path, recovery_trajectory_path
            )
            for _ in range(2)
        ]
        self.recovering = [False, False]
        ratio = CONTROL_DT / float(self.model.opt.timestep)
        self.substeps = int(round(ratio))
        if abs(ratio - self.substeps) > 1e-9:
            raise ValueError(
                "walking control period is not divisible by the model timestep"
            )
        if self.substeps != 5:
            raise ValueError(
                f"expected five physics steps per control step, got {self.substeps}"
            )
        self.runtime_provenance = collect_runtime_provenance(
            mujoco,
            self.controllers[0].policy.MNN,
        )
        self._reset_record: dict[str, Any] | None = None
        self.reset()

    def reset(self) -> None:
        self.mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            RESET_KEYFRAME_ID,
        )
        for binding, controller in zip(self.bindings, self.controllers):
            binding.set_sdk_standing_state(self.data, self.support_height_m)
            controller.reset()
        self.recovering[:] = [False, False]
        self.mujoco.mj_forward(self.model, self.data)
        self.physics_tick = 0
        self.control_tick = 0
        qpos_sha256 = _array_sha256(self.data.qpos)
        controller_states = []
        for fighter, controller in enumerate(self.controllers):
            state = {
                "fighter_index": fighter,
                "history_sha256": _array_sha256(controller.history),
                "previous_action_sha256": _array_sha256(
                    controller.previous_action
                ),
                "first_observation": bool(controller.first_observation),
                "command_filter_output": (
                    None
                    if controller.command_filter.output is None
                    else [
                        float(value)
                        for value in controller.command_filter.output
                    ]
                ),
            }
            state["state_sha256"] = _canonical_sha256(state)
            controller_states.append(state)
        measured_state = {
            "time_s": float(self.data.time),
            "qpos_count": int(self.data.qpos.size),
            "qpos_sha256": qpos_sha256,
            "qvel_count": int(self.data.qvel.size),
            "qvel_sha256": _array_sha256(self.data.qvel),
            "act_count": int(self.data.act.size),
            "act_sha256": _array_sha256(self.data.act),
            "ctrl_count": int(self.data.ctrl.size),
            "ctrl_sha256": _array_sha256(self.data.ctrl),
            "walking_controller_states": controller_states,
            "recovering_flags": list(self.recovering),
        }
        measured_state["aggregate_state_sha256"] = _named_array_state_sha256(
            (
                ("time_s", [self.data.time]),
                ("qpos", self.data.qpos),
                ("qvel", self.data.qvel),
                ("act", self.data.act),
                ("ctrl", self.data.ctrl),
            )
        )
        canonical_state_authority = {
            "kind": "engineai_native_sdk_model_and_walking_configuration",
            "sdk_commit": ENGINEAI_SDK_COMMIT,
            "model_path": "assets/resource/t800.xml",
            "walking_configuration_path": (
                "assets/config/t800/rl_walking_example/default.yaml"
            ),
            "default_joint_q_sha256": _array_sha256(DEFAULT_Q),
            "root_clearance_above_support_m": (
                OFFICIAL_DEFAULT_ROOT_CLEARANCE_M
            ),
            "root_orientation": (
                "sdk_upright_roll_pitch_with_retained_arena_heading"
            ),
            "root_and_joint_velocities": "zero",
            "support_geometry": ARENA_SUPPORT_GEOM_NAME,
            "support_height_m": self.support_height_m,
        }
        identity_material = {
            "model_sha256": self.model_sha256,
            "keyframe_id": RESET_KEYFRAME_ID,
            "keyframe_name": self.keyframe_name,
            "keyframe_qpos_sha256_before_joint_override": (
                self.keyframe_qpos_sha256
            ),
            "composite_qpos_sha256_after_joint_override": qpos_sha256,
            "physics_timestep_s": PHYSICS_DT,
            "procedure": (
                "mj_resetDataKeyframe_then_retain_arena_xy_heading_and_apply_"
                "per_fighter_engineai_sdk_standing_state_"
                "and_walking_controller_reset_then_mj_forward_v1"
            ),
            "aggregate_state_sha256": measured_state[
                "aggregate_state_sha256"
            ],
            "walking_controller_state_sha256": [
                state["state_sha256"] for state in controller_states
            ],
            "canonical_state_authority": canonical_state_authority,
        }
        self._reset_record = {
            "schema": RESET_SCHEMA,
            "identity_sha256": _canonical_sha256(identity_material),
            "identity_material": identity_material,
            "procedure_steps": json.loads(json.dumps(RESET_PROCEDURE_STEPS)),
            "measured_state": measured_state,
            "untouched_xml_keyframe": False,
            "keyframe_joint_pose_overridden_by_engineai_default_q": True,
            "keyframe_root_state_overridden_by_engineai_sdk_model": True,
            "keyframe_arena_xy_and_heading_retained": True,
            "canonical_state_authority": canonical_state_authority,
        }

    def reset_record(self) -> dict[str, Any]:
        if self._reset_record is None:
            raise RuntimeError("runner has no measured composite reset record")
        return json.loads(json.dumps(self._reset_record, allow_nan=False))

    def _bot_command(self) -> np.ndarray:
        bot = self.bindings[1]
        delta_world = (
            self.bindings[0].root_position(self.data)
            - bot.root_position(self.data)
        )
        quaternion = self.data.qpos[
            bot.root_qpos_address + 3 : bot.root_qpos_address + 7
        ]
        w, x, y, z = quaternion
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        cosine, sine = math.cos(yaw), math.sin(yaw)
        forward = cosine * delta_world[0] + sine * delta_world[1]
        lateral = -sine * delta_world[0] + cosine * delta_world[1]
        distance = float(np.hypot(forward, lateral))
        return np.array(
            [
                np.clip((distance - 1.2) * 0.8, -0.4, 0.7),
                np.clip(lateral * 0.5, -0.4, 0.4),
                0.0,
            ],
            dtype=np.float64,
        )

    def snapshot(self, phase: str, normalized_command: Sequence[float]) -> dict[str, Any]:
        roots = []
        for fighter, binding in enumerate(self.bindings):
            qpos_address = binding.root_qpos_address
            roots.append(
                {
                    "fighter_index": fighter,
                    "role": "controlled" if fighter == 0 else "approach_dummy",
                    "root_position": [
                        float(value)
                        for value in self.data.qpos[qpos_address : qpos_address + 3]
                    ],
                    "root_quaternion_xyzw": _quaternion_xyzw(
                        self.data.qpos[qpos_address + 3 : qpos_address + 7]
                    ),
                    "recovering": bool(self.recovering[fighter]),
                }
            )
        controlled = roots[0]
        return {
            "time_s": float(self.data.time),
            "physics_tick": self.physics_tick,
            "control_tick": self.control_tick,
            "control_phase": phase,
            "normalized_command": [float(value) for value in normalized_command],
            "root_position": controlled["root_position"],
            "root_quaternion_xyzw": controlled["root_quaternion_xyzw"],
            "fighter_roots": roots,
        }

    def step_control(
        self,
        human_command: Sequence[float],
        phase: str,
        observe: Callable[[dict[str, Any]], None],
    ) -> None:
        normalized_commands = [
            np.asarray(human_command, dtype=np.float64),
            self._bot_command(),
        ]
        targets = []
        for fighter, (binding, controller, recovery, normalized_command) in enumerate(
            zip(
                self.bindings,
                self.controllers,
                self.recovery_controllers,
                normalized_commands,
            )
        ):
            joint_q, joint_qd, quaternion, angular_velocity = binding.state(self.data)
            fallen = (
                binding.root_position(self.data)[2] < 0.65
                or binding.root_up_z(self.data) < 0.45
            )
            if fallen and not self.recovering[fighter]:
                recovery.reset(joint_q)
                self.recovering[fighter] = True
            if self.recovering[fighter]:
                target, _ = recovery.step(
                    joint_q, joint_qd, quaternion, angular_velocity
                )
                targets.append((target, recovery))
            else:
                controller.observe(
                    joint_q, joint_qd, quaternion, angular_velocity
                )
                _, target = controller.act(
                    controller.scale_command(normalized_command)
                )
                targets.append((target, controller))

        for _ in range(self.substeps):
            for binding, (target, controller) in zip(self.bindings, targets):
                joint_q, joint_qd, _, _ = binding.state(self.data)
                binding.apply_torque(
                    self.data,
                    controller.pd_torque(joint_q, joint_qd, target),
                )
            self.mujoco.mj_step(self.model, self.data)
            self.physics_tick += 1
            observe(self.snapshot(phase, human_command))

        for fighter, (binding, controller, recovery) in enumerate(
            zip(self.bindings, self.controllers, self.recovery_controllers)
        ):
            if self.recovering[fighter] and recovery.finished:
                upright = (
                    binding.root_position(self.data)[2] > 0.8
                    and binding.root_up_z(self.data) > 0.9
                )
                if upright:
                    self.recovering[fighter] = False
                    controller.reset()
        self.control_tick += 1


def controlled_command_at_control_tick(
    control_tick: int,
    command: CommandSpec,
) -> tuple[tuple[float, float, float], str, int]:
    press_control_tick = TRIAL_PRESS_TICK * 2
    release_control_tick = TRIAL_RELEASE_TICK * 2
    if control_tick < press_control_tick:
        return NEUTRAL_COMMAND, "neutral_pre_roll", TRIAL_NEUTRAL_START_TICK
    if control_tick < release_control_tick:
        return command.normalized_command, "selected_command_held", TRIAL_PRESS_TICK
    return NEUTRAL_COMMAND, "neutral_release", TRIAL_RELEASE_TICK


def generate_trace(
    runner: Any,
    command: CommandSpec,
    *,
    build_fingerprint: str,
    capture_id: str,
    run_id: str,
    artifact_provenance: dict[str, Any],
    calibration: dict[str, Any],
    sample_observer: Callable[[Any, dict[str, Any], int, bool], None] | None = None,
) -> dict[str, Any]:
    if not isinstance(build_fingerprint, str) or not build_fingerprint.strip():
        raise ValueError("build_fingerprint must be a non-empty string")
    for label, value in (("capture_id", capture_id), ("run_id", run_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")
    if (
        not isinstance(calibration, dict)
        or calibration.get("schema") != CALIBRATION_SCHEMA
    ):
        raise ValueError(
            f"calibration must be a {CALIBRATION_SCHEMA} JSON object"
        )
    edge_physics_tick = TRIAL_PRESS_TICK * TRIAL_FIXED_SUBSTEPS
    release_physics_tick = TRIAL_RELEASE_TICK * TRIAL_FIXED_SUBSTEPS
    first_physics_tick = TRIAL_NEUTRAL_START_TICK * TRIAL_FIXED_SUBSTEPS
    last_physics_tick = TRIAL_STOP_TICK * TRIAL_FIXED_SUBSTEPS

    samples: list[dict[str, Any]] = []

    def observe(sample: dict[str, Any]) -> None:
        physics_tick = sample["physics_tick"]
        expected_tick = first_physics_tick + len(samples)
        if physics_tick != expected_tick:
            raise RuntimeError(
                f"selected physics samples are not contiguous: expected "
                f"{expected_tick}, got {physics_tick}"
            )
        samples.append(sample)
        if sample_observer is not None:
            sample_observer(
                runner,
                sample,
                len(samples) - 1,
                physics_tick > edge_physics_tick,
            )

    runner.reset()
    reset_record = runner.reset_record()
    if reset_record.get("schema") != RESET_SCHEMA:
        raise RuntimeError(f"runner reset record is not {RESET_SCHEMA}")
    reset_identity_material = reset_record.get("identity_material")
    if not isinstance(reset_identity_material, dict) or reset_record.get(
        "identity_sha256"
    ) != _canonical_sha256(reset_identity_material):
        raise RuntimeError("runner reset identity hash is absent or inconsistent")
    reset_state = reset_record.get("measured_state")
    if not isinstance(reset_state, dict):
        raise RuntimeError("runner reset has no measured state hashes")
    for field in (
        "aggregate_state_sha256",
        "qpos_sha256",
        "qvel_sha256",
        "ctrl_sha256",
    ):
        if not _is_sha256(reset_state.get(field)):
            raise RuntimeError(f"runner reset measured_state.{field} is not SHA-256")
    if (
        reset_identity_material.get("aggregate_state_sha256")
        != reset_state["aggregate_state_sha256"]
        or reset_identity_material.get(
            "composite_qpos_sha256_after_joint_override"
        )
        != reset_state["qpos_sha256"]
    ):
        raise RuntimeError("runner reset identity does not bind its measured state")
    if (
        reset_record.get("untouched_xml_keyframe") is not False
        or reset_record.get(
            "keyframe_joint_pose_overridden_by_engineai_default_q"
        )
        is not True
        or reset_record.get(
            "keyframe_root_state_overridden_by_engineai_sdk_model"
        )
        is not True
        or reset_record.get("keyframe_arena_xy_and_heading_retained") is not True
        or reset_record.get("canonical_state_authority")
        != reset_identity_material.get("canonical_state_authority")
    ):
        raise RuntimeError("runner reset does not declare its SDK-derived state")
    if reset_record.get("procedure_steps") != RESET_PROCEDURE_STEPS:
        raise RuntimeError("runner reset procedure steps differ from the pinned contract")
    model_artifact = artifact_provenance.get("model")
    if (
        not isinstance(model_artifact, dict)
        or reset_identity_material.get("model_sha256")
        != model_artifact.get("sha256")
    ):
        raise RuntimeError("runner reset does not bind the trace model artifact")
    runtime_provenance = getattr(runner, "runtime_provenance", None)
    if (
        not isinstance(runtime_provenance, dict)
        or runtime_provenance.get("schema") != RUNTIME_SCHEMA
    ):
        raise RuntimeError(f"runner runtime provenance is not {RUNTIME_SCHEMA}")
    runtime_provenance = json.loads(
        json.dumps(runtime_provenance, allow_nan=False)
    )
    runtime_identity_sha256 = runtime_provenance.pop("identity_sha256", None)
    if runtime_identity_sha256 != _canonical_sha256(runtime_provenance):
        raise RuntimeError("runner runtime provenance identity hash is inconsistent")
    runtime_provenance["identity_sha256"] = runtime_identity_sha256

    observe(runner.snapshot("neutral_pre_roll:trial_tick_0", NEUTRAL_COMMAND))
    final_control_tick = TRIAL_STOP_TICK * 2
    while runner.control_tick < final_control_tick:
        normalized_command, phase, phase_trial_tick = (
            controlled_command_at_control_tick(runner.control_tick, command)
        )
        runner.step_control(
            normalized_command,
            f"{phase}:trial_tick_{phase_trial_tick}",
            observe,
        )

    expected_samples = last_physics_tick - first_physics_tick + 1
    if len(samples) != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} measured samples, got {len(samples)}"
        )
    for index, sample in enumerate(samples):
        if sample["physics_tick"] != first_physics_tick + index:
            raise RuntimeError("captured physics sample sequence is not contiguous")
        if index and not sample["time_s"] > samples[index - 1]["time_s"]:
            raise RuntimeError("physics sample times are not strictly increasing")
        if len(sample.get("fighter_roots", ())) != 2:
            raise RuntimeError("a measured sample lacks one of the two fighter roots")

    edge_sample_index = edge_physics_tick - first_physics_tick
    release_sample_index = release_physics_tick - first_physics_tick
    edge_time_s = float(samples[edge_sample_index]["time_s"])
    release_time_s = float(samples[release_sample_index]["time_s"])
    if abs(edge_time_s - edge_physics_tick * PHYSICS_DT) > 1e-9:
        raise RuntimeError("MuJoCo time does not match the command-edge physics tick")
    if abs(release_time_s - release_physics_tick * PHYSICS_DT) > 1e-9:
        raise RuntimeError("MuJoCo time does not match the release physics tick")

    controlled_command_segments = [
        {
            "phase": "neutral_pre_roll",
            "start_trial_tick": TRIAL_NEUTRAL_START_TICK,
            "end_trial_tick_exclusive": TRIAL_PRESS_TICK,
            "normalized_command": list(NEUTRAL_COMMAND),
        },
        {
            "phase": "selected_command_held",
            "start_trial_tick": TRIAL_PRESS_TICK,
            "end_trial_tick_exclusive": TRIAL_RELEASE_TICK,
            "normalized_command": list(command.normalized_command),
        },
        {
            "phase": "neutral_release",
            "start_trial_tick": TRIAL_RELEASE_TICK,
            "end_trial_tick_exclusive": TRIAL_STOP_TICK,
            "normalized_command": list(NEUTRAL_COMMAND),
        },
    ]

    return {
        "schema": TRACE_SCHEMA,
        "source": SOURCE,
        "build_fingerprint": build_fingerprint.strip(),
        "capture_id": capture_id.strip(),
        "schedule_run_id": run_id.strip(),
        "trial_run_id": run_id.strip(),
        "command": {
            "identity": command.identity,
            "edge_time_s": edge_time_s,
            "execution_state": "simulated",
            "semantic_name": command.name,
            "normalized_command": list(command.normalized_command),
            "duration_s": (
                TRIAL_RELEASE_TICK - TRIAL_PRESS_TICK
            ) / TRIAL_RATE_HZ,
            "trial_scope": "isolated_same_reset_single_press",
            "edge_trial_tick": TRIAL_PRESS_TICK,
            "edge_physics_tick": edge_physics_tick,
            "edge_control_tick": TRIAL_PRESS_TICK * 2,
            "release_time_s": release_time_s,
            "release_trial_tick": TRIAL_RELEASE_TICK,
            "release_physics_tick": release_physics_tick,
            "release_control_tick": TRIAL_RELEASE_TICK * 2,
            "identity_provenance": {
                "kind": "pinned_isolated_single_press_selector",
                "trial_protocol_identity": TRIAL_PROTOCOL_ID,
                "trial_protocol_sha256": TRIAL_PROTOCOL_SHA256,
            },
        },
        "trial": {
            "schema": TRIAL_SCHEMA,
            "protocol": {
                "identity": TRIAL_PROTOCOL_ID,
                "sha256": TRIAL_PROTOCOL_SHA256,
                "definition": json.loads(json.dumps(TRIAL_PROTOCOL_DEFINITION)),
            },
            "selector": command.name,
            "command_identity": command.identity,
            "controlled_command_segments": controlled_command_segments,
            "prior_non_neutral_command_count": 0,
            "non_neutral_press_count": 1,
            "attack_command_count": 0,
        },
        "reset": reset_record,
        "calibration": calibration,
        "timing": {
            "physics_dt_s": PHYSICS_DT,
            "physics_rate_hz": 1.0 / PHYSICS_DT,
            "controller_dt_s": CONTROL_DT,
            "controller_rate_hz": 1.0 / CONTROL_DT,
            "physics_substeps_per_controller_step": runner.substeps,
            "sample_timing": "post_mujoco_step_plus_initial_state",
            "interpolation_used": False,
            "realtime_pacing_used": False,
            "trial_rate_hz": TRIAL_RATE_HZ,
            "trial_started_from_tick": TRIAL_NEUTRAL_START_TICK,
            "capture_first_physics_tick": first_physics_tick,
            "capture_last_physics_tick": last_physics_tick,
            "capture_pre_edge_s": TRIAL_PRESS_TICK / TRIAL_RATE_HZ,
            "capture_post_release_s": (
                TRIAL_STOP_TICK - TRIAL_RELEASE_TICK
            ) / TRIAL_RATE_HZ,
        },
        "controller": {
            "controlled_fighter": "official_engineai_walking_policy",
            "opponent": "official_engineai_walking_policy_approach_dummy",
            "automatic_getup": "official_engineai_supine_to_stance_policy",
            "combat_moves_enabled": False,
        },
        "artifact_provenance": artifact_provenance,
        "runtime_provenance": runtime_provenance,
        "claims": {
            "parity_demonstrated": False,
            "combat_trajectory_present": False,
            "screen_coordinates_present": False,
            "samples_are_measured_simulator_state": True,
            "sample_interpolation_used": False,
            "selected_command_started_from_shared_pinned_reset": True,
            "controlled_prior_non_neutral_command_count": 0,
            "controlled_non_neutral_press_count": 1,
            "controlled_post_release_command_is_neutral": True,
            "rek_attack_trajectories_replayed": False,
        },
        "limitations": [
            (
                "The deterministic opponent is an EngineAI walking-policy "
                "approach dummy, not the unrecovered REK Bot 1 policy."
            ),
            "No REK combat trajectory is synthesized or replayed.",
            (
                "The reset is the declared composite controller-compatible "
                "procedure, not the untouched XML keyframe."
            ),
        ],
        "samples": samples,
    }


def write_json_exclusive(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_artifact_provenance(
    model_path: Path,
    walking_path: Path,
    recovery_path: Path,
    trajectory_path: Path,
    calibration_path: Path,
) -> dict[str, Any]:
    artifact_hashes = {
        "model": _checked_sha256(
            model_path, EXPECTED_MJCF_SHA256, "two-fighter diagnostic MJCF"
        ),
        "walking_policy": _checked_sha256(
            walking_path, EXPECTED_WALKING_POLICY_SHA256, "walking policy"
        ),
        "recovery_policy": _checked_sha256(
            recovery_path, RECOVERY_POLICY_SHA256, "recovery policy"
        ),
        "recovery_trajectory": _checked_sha256(
            trajectory_path, RECOVERY_TRAJECTORY_SHA256, "recovery trajectory"
        ),
        "calibration": _checked_sha256(
            calibration_path,
            EXPECTED_CALIBRATION_SHA256,
            "MuJoCo arena identity calibration",
        ),
    }
    adapter_path = Path(__file__).with_name("engineai_t800_policy.py").resolve()
    exporter_path = Path(__file__).resolve()
    return {
        "engineai_sdk_commit": ENGINEAI_SDK_COMMIT,
        "model": {"path": str(model_path), "sha256": artifact_hashes["model"]},
        "walking_policy": {
            "path": str(walking_path),
            "sha256": artifact_hashes["walking_policy"],
        },
        "recovery_policy": {
            "path": str(recovery_path),
            "sha256": artifact_hashes["recovery_policy"],
        },
        "recovery_trajectory": {
            "path": str(trajectory_path),
            "sha256": artifact_hashes["recovery_trajectory"],
        },
        "calibration": {
            "path": str(calibration_path),
            "sha256": artifact_hashes["calibration"],
        },
        "controller_adapter": {
            "path": str(adapter_path),
            "sha256": sha256_file(adapter_path),
        },
        "exporter": {
            "path": str(exporter_path),
            "sha256": sha256_file(exporter_path),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--walking-policy", type=Path, required=True)
    parser.add_argument("--recovery-policy", type=Path, required=True)
    parser.add_argument("--recovery-trajectory", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--command", choices=tuple(COMMANDS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--rek-build-fingerprint",
        required=True,
        help="exact build_fingerprint from the measured REK inventory",
    )
    parser.add_argument("--capture-id")
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model_path = args.model.resolve()
    walking_path = args.walking_policy.resolve()
    recovery_path = args.recovery_policy.resolve()
    trajectory_path = args.recovery_trajectory.resolve()
    calibration_path = args.calibration.resolve()
    artifact_provenance = build_artifact_provenance(
        model_path,
        walking_path,
        recovery_path,
        trajectory_path,
        calibration_path,
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    runner = EngineAIMotionRunner(
        model_path,
        walking_path,
        recovery_path,
        trajectory_path,
    )
    capture_id = args.capture_id or f"clone-capture-{uuid.uuid4()}"
    run_id = args.run_id or f"clone-run-{uuid.uuid4()}"
    document = generate_trace(
        runner,
        COMMANDS[args.command],
        build_fingerprint=args.rek_build_fingerprint,
        capture_id=capture_id,
        run_id=run_id,
        artifact_provenance=artifact_provenance,
        calibration=calibration,
    )
    output = args.out.resolve()
    write_json_exclusive(output, document)
    print(f"command: {document['command']['identity']}")
    print(f"capture_id: {capture_id}")
    print(f"run_id: {run_id}")
    print(f"samples: {len(document['samples'])}")
    print(f"output: {output}")
    print(f"sha256: {sha256_file(output)}")
    print("parity_demonstrated: false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
