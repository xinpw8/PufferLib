"""Correctness-first vector reference runtime for the GEAR-SONIC G1 candidate.

The runtime shares one encoder session, one decoder session, and one MuJoCo
model while keeping one distinct ``MjData`` and controller-state record per
environment.  It is a public-family diagnostic candidate, not a REK parity
implementation.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import gear_sonic_batch_model as batch_model_contract
import gear_sonic_bundle as bundle_contract
import gear_sonic_candidate as candidate
import sonic_candidate as plant_contract


ReferenceResolver = Callable[[int, int], tuple[int, Mapping[str, Any]]]


@dataclass
class _EnvironmentState:
    heading_delta_wxyz: np.ndarray
    history: candidate.StateHistory
    last_action_policy: np.ndarray
    policy_tick: int = 0
    command_lpf_state_mujoco: np.ndarray | None = None


@dataclass(frozen=True)
class _PreparedTick:
    current_frame: int
    reference_metadata: Mapping[str, Any]
    encoder_observation: np.ndarray
    future_indices: np.ndarray


@dataclass(frozen=True)
class GearSonicStep:
    """One policy decision and its ten completed MuJoCo physics steps."""

    env_index: int
    tick: int
    current_frame: int
    reference_metadata: Mapping[str, Any]
    future_indices: np.ndarray
    encoder_observation: np.ndarray
    token: np.ndarray
    decoder_observation: np.ndarray
    raw_action_policy: np.ndarray
    clipped_action_policy: np.ndarray
    raw_targets_mujoco: np.ndarray
    command_lpf_state_mujoco: np.ndarray
    applied_targets_mujoco: np.ndarray
    joint_target_clipped: np.ndarray
    q_before_mujoco: np.ndarray
    dq_before_mujoco: np.ndarray
    torque_raw_mujoco: np.ndarray
    torque_predicted_mujoco: np.ndarray
    torque_saturated: np.ndarray
    actuator_force_after_mujoco: np.ndarray
    q_after_mujoco: np.ndarray
    projected_gravity_after: np.ndarray
    physics_steps: int
    command_lpf_updates: int
    actuator_force_squared_sum: float
    torque_saturation_count: int
    torque_value_count: int


class GearSonicController:
    """One shared ONNX controller with explicit graph batch handling."""

    def __init__(self, encoder_session: Any, decoder_session: Any):
        self.encoder_session = encoder_session
        self.decoder_session = decoder_session
        self.encoder_batch_mode: str | None = None
        self.decoder_batch_mode: str | None = None

    @staticmethod
    def _declared_batch(session: Any, feature_count: int, label: str) -> int | None:
        try:
            inputs = list(session.get_inputs())
        except Exception as exc:
            raise candidate.GearSonicError(
                f"{label} session input metadata is unavailable"
            ) from exc
        if len(inputs) != 1 or getattr(inputs[0], "name", None) != "obs_dict":
            raise candidate.GearSonicError(
                f"{label} session must expose one obs_dict input"
            )
        shape = tuple(getattr(inputs[0], "shape", ()))
        if len(shape) != 2:
            raise candidate.GearSonicError(
                f"{label} obs_dict must be rank two, got {shape}"
            )
        width = shape[1]
        if not isinstance(width, (int, np.integer)) or int(width) != feature_count:
            raise candidate.GearSonicError(
                f"{label} obs_dict width must be {feature_count}, got {width!r}"
            )
        declared = shape[0]
        if declared is None or isinstance(declared, str):
            return None
        if not isinstance(declared, (int, np.integer)) or int(declared) < 1:
            raise candidate.GearSonicError(
                f"{label} obs_dict batch dimension is invalid: {declared!r}"
            )
        return int(declared)

    @staticmethod
    def _validate_output(
        value: Any, batch_size: int, output_width: int, output_name: str
    ) -> np.ndarray:
        array = np.asarray(value)
        expected = (batch_size, output_width)
        if (
            array.shape != expected
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise candidate.GearSonicError(
                f"{output_name} has invalid dtype, shape, or finite values: "
                f"{array.dtype} {array.shape}; expected float32 {expected}"
            )
        return np.ascontiguousarray(array)

    def _run(
        self,
        session: Any,
        output_name: str,
        observations: np.ndarray,
        input_width: int,
        output_width: int,
        label: str,
    ) -> tuple[np.ndarray, str]:
        batch = np.ascontiguousarray(observations, dtype=np.float32)
        if batch.ndim != 2 or batch.shape[1] != input_width or batch.shape[0] < 1:
            raise candidate.GearSonicError(
                f"{label} observations must have shape [N, {input_width}]"
            )
        if not np.isfinite(batch).all():
            raise candidate.GearSonicError(f"{label} observations contain non-finite values")

        batch_size = int(batch.shape[0])
        declared_batch = self._declared_batch(session, input_width, label)
        if batch_size == 1:
            if declared_batch not in (None, 1):
                raise candidate.GearSonicError(
                    f"{label} graph requires batch {declared_batch}, not batch 1"
                )
            raw = session.run([output_name], {"obs_dict": batch})[0]
            return self._validate_output(raw, 1, output_width, output_name), "single"

        if declared_batch is None or declared_batch == batch_size:
            raw = session.run([output_name], {"obs_dict": batch})[0]
            return (
                self._validate_output(raw, batch_size, output_width, output_name),
                "batched",
            )

        if declared_batch != 1:
            raise candidate.GearSonicError(
                f"{label} graph requires batch {declared_batch}, not batch {batch_size}"
            )

        rows: list[np.ndarray] = []
        for index in range(batch_size):
            raw = session.run(
                [output_name], {"obs_dict": np.ascontiguousarray(batch[index : index + 1])}
            )[0]
            rows.append(self._validate_output(raw, 1, output_width, output_name)[0])
        return np.ascontiguousarray(np.stack(rows)), "per_env_fixed_batch_1"

    def encode(self, observations: np.ndarray) -> np.ndarray:
        result, self.encoder_batch_mode = self._run(
            self.encoder_session,
            "encoded_tokens",
            observations,
            candidate.ENCODER_DIM,
            candidate.TOKEN_DIM,
            "encoder",
        )
        return result

    def decode(self, observations: np.ndarray) -> np.ndarray:
        result, self.decoder_batch_mode = self._run(
            self.decoder_session,
            "action",
            observations,
            candidate.DECODER_DIM,
            candidate.ACTION_DIM,
            "decoder",
        )
        return result


class GearSonicVectorEnv:
    """Advance N independent MuJoCo data instances through one controller."""

    def __init__(
        self,
        *,
        mujoco: Any,
        model: Any,
        runtime_map: plant_contract.RuntimeMap,
        motion: plant_contract.MotionData,
        controller: GearSonicController,
        data: Sequence[Any],
        force_limits: np.ndarray,
        frame_mode: str,
        physics_workers: int = 1,
        reference_resolver: ReferenceResolver | None = None,
        initializations: Sequence[Mapping[str, Any]] | None = None,
        bundle_report: Mapping[str, Any] | None = None,
        batch_model_report: Mapping[str, Any] | None = None,
        xml_contract: Any | None = None,
        arena_contract: Any | None = None,
        serialized_timestep: float | None = None,
        mujoco_version: str = "unknown",
        onnxruntime_version: str = "unknown",
    ):
        self.mujoco = mujoco
        self.model = model
        self.runtime_map = runtime_map
        self.motion = motion
        self.controller = controller
        self.data = tuple(data)
        if not self.data:
            raise candidate.GearSonicError("vector environment count must be positive")
        if len({id(value) for value in self.data}) != len(self.data):
            raise candidate.GearSonicError("each vector environment requires a distinct MjData")
        if frame_mode not in ("clamp", "loop"):
            raise candidate.GearSonicError(f"unsupported frame mode {frame_mode!r}")
        if (
            isinstance(physics_workers, bool)
            or not isinstance(physics_workers, int)
            or physics_workers < 1
        ):
            raise candidate.GearSonicError("physics_workers must be a positive integer")
        self.frame_mode = frame_mode
        self.physics_workers = physics_workers
        self.reference_resolver = reference_resolver

        self.force_limits = np.asarray(force_limits, dtype=np.float64).copy()
        if (
            self.force_limits.shape != (candidate.ACTION_DIM,)
            or not np.isfinite(self.force_limits).all()
            or np.any(self.force_limits <= 0.0)
        ):
            raise candidate.GearSonicError("vector force limits must be 29 finite positives")

        timestep = float(model.opt.timestep)
        if not math.isclose(
            timestep, candidate.PHYSICS_DT, rel_tol=0.0, abs_tol=1e-12
        ):
            raise candidate.GearSonicError(
                "vector environment requires the corrected 500 Hz physics timestep"
            )
        self.physics_steps_per_control = candidate.native_scheduler_interval(
            timestep, float(candidate.CONTROL_HZ)
        )
        if self.physics_steps_per_control != candidate.PHYSICS_STEPS_PER_CONTROL:
            raise candidate.GearSonicError(
                "vector controller scheduler did not resolve to 10 physics steps"
            )
        self.command_lpf_interval = candidate.native_scheduler_interval(
            timestep, candidate.NATIVE_WORK_RATE_HZ
        )
        self.command_lpf_dt = self.command_lpf_interval * timestep
        self.command_lpf_alpha = candidate.command_lpf_alpha(
            candidate.COMMAND_LPF_CUTOFF_HZ, self.command_lpf_dt
        )

        if initializations is None:
            self.initializations: list[Mapping[str, Any]] = [
                {} for _ in self.data
            ]
        else:
            if len(initializations) != len(self.data):
                raise candidate.GearSonicError(
                    "initialization records must match the vector environment count"
                )
            self.initializations = [dict(value) for value in initializations]
        self._states = [self._fresh_state(value) for value in self.data]
        self._step_failure: BaseException | None = None
        self._closed = False
        self._executor = self._create_executor()

        self.bundle_report = bundle_report
        self.batch_model_report = batch_model_report
        self.xml_contract = xml_contract
        self.arena_contract = arena_contract
        self.serialized_timestep = serialized_timestep
        self.mujoco_version = mujoco_version
        self.onnxruntime_version = onnxruntime_version

    @classmethod
    def from_artifacts(
        cls,
        *,
        bundle: Path,
        assets_dir: Path,
        motion_role: str,
        manifest: Path,
        xml: Path,
        arena: Path,
        num_envs: int,
        control_boundary: str,
        frame_mode: str,
        force_limit_source: str,
        physics_workers: int = 1,
        root_z_offsets: Sequence[float] | None = None,
        reference_resolver: ReferenceResolver | None = None,
        batch_models_dir: Path | None = None,
    ) -> "GearSonicVectorEnv":
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise candidate.GearSonicError("num_envs must be a positive integer")
        if control_boundary != "native-position-actuator":
            raise candidate.GearSonicError(
                f"unsupported control boundary {control_boundary!r}"
            )
        if frame_mode not in ("clamp", "loop"):
            raise candidate.GearSonicError(f"unsupported frame mode {frame_mode!r}")

        bundle_report = bundle_contract.inspect_bundle(bundle, run_smoke=False)
        motion = plant_contract.load_motion(assets_dir, motion_role, manifest)
        xml_contract = plant_contract.inspect_xml_contract(xml)
        arena_contract = plant_contract.load_arena_contract(arena)
        mujoco, model, mujoco_version = plant_contract.create_mujoco_model(
            xml_contract, None, arena_contract
        )
        runtime_map = plant_contract.build_runtime_map(model, xml_contract)
        serialized_timestep = float(model.opt.timestep)
        if not math.isclose(
            serialized_timestep,
            xml_contract.source_timestep,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise candidate.GearSonicError(
                "compiled model did not preserve the recovered timestep"
            )
        model.opt.timestep = candidate.PHYSICS_DT
        if not math.isclose(
            float(model.opt.timestep),
            candidate.PHYSICS_DT,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise candidate.GearSonicError(
                "failed to apply REK's 500 Hz runtime physics timestep"
            )

        force_limits = candidate.resolve_force_limits(runtime_map, force_limit_source)
        force_limits = candidate.configure_native_position_actuators(
            mujoco, model, runtime_map, force_limits
        )
        encoder_path = bundle / "model_encoder.onnx"
        decoder_path = bundle / "model_decoder.onnx"
        batch_model_report = None
        if batch_models_dir is not None:
            batch_model_report = batch_model_contract.inspect_explicit_batch_bundle(
                bundle,
                batch_models_dir,
                num_envs,
            )
            encoder_path = Path(
                batch_model_report["models"]["model_encoder.onnx"]["output_path"]
            )
            decoder_path = Path(
                batch_model_report["models"]["model_decoder.onnx"]["output_path"]
            )

        encoder, ort_version = candidate._create_session(encoder_path)
        decoder, decoder_ort_version = candidate._create_session(decoder_path)
        if decoder_ort_version != ort_version:
            raise candidate.GearSonicError(
                "encoder and decoder ONNX Runtime versions differ"
            )

        if root_z_offsets is None:
            offsets = [0.0] * num_envs
        else:
            if len(root_z_offsets) != num_envs:
                raise candidate.GearSonicError(
                    "root_z_offsets must match the vector environment count"
                )
            offsets = [float(value) for value in root_z_offsets]
            if not np.isfinite(offsets).all():
                raise candidate.GearSonicError("root_z_offsets must be finite")

        data = [mujoco.MjData(model) for _ in range(num_envs)]
        initializations = [
            plant_contract.initialize_simulation(
                mujoco, model, value, runtime_map, motion, offsets[index]
            )
            for index, value in enumerate(data)
        ]
        return cls(
            mujoco=mujoco,
            model=model,
            runtime_map=runtime_map,
            motion=motion,
            controller=GearSonicController(encoder, decoder),
            data=data,
            force_limits=force_limits,
            frame_mode=frame_mode,
            physics_workers=physics_workers,
            reference_resolver=reference_resolver,
            initializations=initializations,
            bundle_report=bundle_report,
            batch_model_report=batch_model_report,
            xml_contract=xml_contract,
            arena_contract=arena_contract,
            serialized_timestep=serialized_timestep,
            mujoco_version=mujoco_version,
            onnxruntime_version=ort_version,
        )

    @property
    def num_envs(self) -> int:
        return len(self.data)

    @property
    def policy_ticks(self) -> tuple[int, ...]:
        return tuple(state.policy_tick for state in self._states)

    @property
    def failed(self) -> bool:
        return self._step_failure is not None

    @property
    def closed(self) -> bool:
        return self._closed

    def _create_executor(self) -> ThreadPoolExecutor | None:
        if self.physics_workers == 1:
            return None
        return ThreadPoolExecutor(
            max_workers=self.physics_workers,
            thread_name_prefix="gear-sonic-physics",
        )

    def close(self) -> None:
        if self._closed:
            return
        executor = self._executor
        self._executor = None
        self._closed = True
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "GearSonicVectorEnv":
        if self._closed:
            raise candidate.GearSonicError("vector environment is closed")
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def _stop_executor_after_failure(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _fresh_state(self, data: Any) -> _EnvironmentState:
        root = self.runtime_map.root_qpos_address
        initial_base = np.asarray(
            data.qpos[root + 3 : root + 7], dtype=np.float64
        ).copy()
        initial_reference = plant_contract._xyzw_to_wxyz(
            self.motion.root_rot_xyzw[0]
        ).astype(np.float64)
        return _EnvironmentState(
            heading_delta_wxyz=candidate.reference_heading_delta(
                initial_base, initial_reference
            ),
            history=candidate.StateHistory(),
            last_action_policy=np.zeros(candidate.ACTION_DIM, dtype=np.float32),
        )

    def reset(
        self,
        indices: Sequence[int] | None = None,
        root_z_offsets: Sequence[float] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        if self._closed:
            raise candidate.GearSonicError("vector environment is closed")
        if indices is None:
            selected = tuple(range(self.num_envs))
        else:
            requested = tuple(indices)
            if (
                any(
                    isinstance(index, bool)
                    or not isinstance(index, (int, np.integer))
                    for index in requested
                )
                or len(set(requested)) != len(requested)
                or any(index < 0 or index >= self.num_envs for index in requested)
            ):
                raise candidate.GearSonicError(
                    "reset indices must be unique vector environment indices"
                )
            selected = tuple(int(index) for index in requested)
        full_reset = len(selected) == self.num_envs and set(selected) == set(
            range(self.num_envs)
        )
        if self._step_failure is not None and not full_reset:
            raise candidate.GearSonicError(
                "a failed vector step requires resetting every environment"
            ) from self._step_failure
        if root_z_offsets is None:
            offsets = [0.0] * len(selected)
        else:
            if len(root_z_offsets) != len(selected):
                raise candidate.GearSonicError(
                    "reset root_z_offsets must match the selected environment count"
                )
            offsets = [float(value) for value in root_z_offsets]
            if not np.isfinite(offsets).all():
                raise candidate.GearSonicError("reset root_z_offsets must be finite")

        resolver_reset = None
        if self.reference_resolver is not None:
            resolver_reset = getattr(self.reference_resolver, "reset", None)
            if not callable(resolver_reset):
                raise candidate.GearSonicError(
                    "reset with a reference resolver requires reset(indices) support"
                )

        records: list[Mapping[str, Any]] = []
        try:
            if resolver_reset is not None:
                resolver_reset(selected)
            for offset_index, env_index in enumerate(selected):
                record = plant_contract.initialize_simulation(
                    self.mujoco,
                    self.model,
                    self.data[env_index],
                    self.runtime_map,
                    self.motion,
                    offsets[offset_index],
                )
                self.initializations[env_index] = record
                self._states[env_index] = self._fresh_state(self.data[env_index])
                records.append(record)
            if full_reset and self._executor is None:
                self._executor = self._create_executor()
        except BaseException as exc:
            self._step_failure = exc
            raise
        if full_reset:
            self._step_failure = None
        return tuple(records)

    def _resolve_reference(self, env_index: int, tick: int) -> tuple[int, Mapping[str, Any]]:
        if self.reference_resolver is None:
            frame_count = int(self.motion.dof_pos.shape[0])
            frame = tick % frame_count if self.frame_mode == "loop" else min(
                tick, frame_count - 1
            )
            return frame, {}
        try:
            frame, metadata = self.reference_resolver(env_index, tick)
        except candidate.GearSonicError:
            raise
        except Exception as exc:
            raise candidate.GearSonicError(
                f"reference resolver failed for environment {env_index} tick {tick}"
            ) from exc
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
            raise candidate.GearSonicError("reference resolver frame must be an integer")
        if not isinstance(metadata, Mapping):
            raise candidate.GearSonicError("reference resolver metadata must be a mapping")
        return int(frame), dict(metadata)

    def _prepare(self, env_index: int) -> _PreparedTick:
        data = self.data[env_index]
        state = self._states[env_index]
        history_entry = candidate.state_to_history_entry(
            self.mujoco,
            self.model,
            data,
            self.runtime_map,
            state.last_action_policy,
        )
        root = self.runtime_map.root_qpos_address
        base_quat = np.asarray(
            data.qpos[root + 3 : root + 7], dtype=np.float64
        ).copy()
        current_frame, metadata = self._resolve_reference(
            env_index, state.policy_tick
        )
        if state.policy_tick == 0 and current_frame != 0:
            raise candidate.GearSonicError(
                "reference resolver must begin at frame zero for heading alignment"
            )
        encoder_observation, future_indices = candidate.build_encoder_observation(
            self.motion,
            current_frame,
            base_quat,
            state.heading_delta_wxyz,
            loop=self.frame_mode == "loop",
        )
        state.history.append(history_entry)
        return _PreparedTick(
            current_frame=current_frame,
            reference_metadata=metadata,
            encoder_observation=encoder_observation,
            future_indices=future_indices,
        )

    def step(self) -> tuple[GearSonicStep, ...]:
        if self._closed:
            raise candidate.GearSonicError("vector environment is closed")
        if self._step_failure is not None:
            raise candidate.GearSonicError(
                "vector environment is unusable after a failed step; reset all environments"
            ) from self._step_failure
        try:
            return self._step()
        except BaseException as exc:
            self._step_failure = exc
            self._stop_executor_after_failure()
            raise

    def _step(self) -> tuple[GearSonicStep, ...]:
        prepared = [self._prepare(index) for index in range(self.num_envs)]
        encoder_batch = np.stack(
            [value.encoder_observation for value in prepared]
        ).astype(np.float32, copy=False)
        tokens = self.controller.encode(encoder_batch)
        decoder_batch = np.stack(
            [
                candidate.build_decoder_observation(tokens[index], self._states[index].history)
                for index in range(self.num_envs)
            ]
        ).astype(np.float32, copy=False)
        raw_actions = self.controller.decode(decoder_batch)
        jobs = tuple(
            (index, prepared[index], decoder_batch[index], tokens[index], raw_actions[index])
            for index in range(self.num_envs)
        )
        if self._executor is None:
            return tuple(self._apply_job(job) for job in jobs)
        return tuple(self._executor.map(self._apply_job, jobs))

    def _apply_job(
        self,
        job: tuple[int, _PreparedTick, np.ndarray, np.ndarray, np.ndarray],
    ) -> GearSonicStep:
        return self._apply(*job)

    def _apply(
        self,
        env_index: int,
        prepared: _PreparedTick,
        decoder_observation: np.ndarray,
        token: np.ndarray,
        raw_action: np.ndarray,
    ) -> GearSonicStep:
        data = self.data[env_index]
        state = self._states[env_index]
        tick = state.policy_tick
        clipped_action, targets = candidate.action_to_targets(raw_action)
        q_before = np.asarray(
            data.qpos[self.runtime_map.qpos_addresses], dtype=np.float64
        ).copy()
        dq_before = np.asarray(
            data.qvel[self.runtime_map.qvel_addresses], dtype=np.float64
        ).copy()
        torque_raw = np.zeros(candidate.ACTION_DIM, dtype=np.float64)
        torque_predicted = np.zeros(candidate.ACTION_DIM, dtype=np.float64)
        torque_saturated = np.zeros(candidate.ACTION_DIM, dtype=np.bool_)
        joint_target_clipped = np.zeros(candidate.ACTION_DIM, dtype=np.bool_)
        applied_targets = targets.copy()
        command_lpf_updates = 0
        actuator_force_squared_sum = 0.0
        torque_saturation_count = 0
        torque_value_count = 0

        for physics_substep in range(self.physics_steps_per_control):
            fixed_step = tick * self.physics_steps_per_control + physics_substep
            if fixed_step % self.command_lpf_interval == 0:
                if state.command_lpf_state_mujoco is None:
                    state.command_lpf_state_mujoco = targets.copy()
                else:
                    candidate.update_command_lpf(
                        state.command_lpf_state_mujoco,
                        targets,
                        self.command_lpf_alpha,
                    )
                command_lpf_updates += 1
            if state.command_lpf_state_mujoco is None:
                raise candidate.GearSonicError(
                    "command LPF was not initialized on scheduler step zero"
                )
            applied_targets, clipped_this_substep = (
                candidate.clip_targets_to_joint_ranges(
                    self.model, self.runtime_map, state.command_lpf_state_mujoco
                )
            )
            joint_target_clipped |= clipped_this_substep
            q_substep = np.asarray(
                data.qpos[self.runtime_map.qpos_addresses], dtype=np.float64
            ).copy()
            dq_substep = np.asarray(
                data.qvel[self.runtime_map.qvel_addresses], dtype=np.float64
            ).copy()
            raw_substep, predicted_substep, saturated = candidate.pd_torque(
                q_substep,
                dq_substep,
                applied_targets,
                -self.force_limits,
                self.force_limits,
                "declared-effort",
            )
            if physics_substep == 0:
                torque_raw = raw_substep
                torque_predicted = predicted_substep
            torque_saturated |= saturated
            data.ctrl[:] = 0.0
            data.ctrl[self.runtime_map.actuator_ids] = applied_targets
            self.mujoco.mj_step(self.model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise candidate.GearSonicError(
                    f"MuJoCo state became non-finite in environment {env_index} "
                    f"at policy tick {tick}, physics substep {physics_substep}"
                )
            actuator_force = np.asarray(
                data.actuator_force[self.runtime_map.actuator_ids], dtype=np.float64
            )
            actuator_force_squared_sum += float(
                np.dot(actuator_force, actuator_force)
            )
            torque_saturation_count += int(np.count_nonzero(saturated))
            torque_value_count += candidate.ACTION_DIM

        q_after = np.asarray(
            data.qpos[self.runtime_map.qpos_addresses], dtype=np.float64
        ).copy()
        actuator_force_after = np.asarray(
            data.actuator_force[self.runtime_map.actuator_ids], dtype=np.float64
        ).copy()
        root = self.runtime_map.root_qpos_address
        gravity_after = candidate.projected_gravity(
            np.asarray(data.qpos[root + 3 : root + 7], dtype=np.float64)
        )
        if state.command_lpf_state_mujoco is None:
            raise candidate.GearSonicError("command LPF state disappeared after stepping")

        result = GearSonicStep(
            env_index=env_index,
            tick=tick,
            current_frame=prepared.current_frame,
            reference_metadata=dict(prepared.reference_metadata),
            future_indices=prepared.future_indices.copy(),
            encoder_observation=prepared.encoder_observation.copy(),
            token=np.asarray(token, dtype=np.float32).copy(),
            decoder_observation=np.asarray(decoder_observation, dtype=np.float32).copy(),
            raw_action_policy=np.asarray(raw_action, dtype=np.float32).copy(),
            clipped_action_policy=clipped_action.copy(),
            raw_targets_mujoco=targets.copy(),
            command_lpf_state_mujoco=state.command_lpf_state_mujoco.copy(),
            applied_targets_mujoco=applied_targets.copy(),
            joint_target_clipped=joint_target_clipped.copy(),
            q_before_mujoco=q_before,
            dq_before_mujoco=dq_before,
            torque_raw_mujoco=torque_raw,
            torque_predicted_mujoco=torque_predicted,
            torque_saturated=torque_saturated.copy(),
            actuator_force_after_mujoco=actuator_force_after,
            q_after_mujoco=q_after,
            projected_gravity_after=gravity_after.copy(),
            physics_steps=self.physics_steps_per_control,
            command_lpf_updates=command_lpf_updates,
            actuator_force_squared_sum=actuator_force_squared_sum,
            torque_saturation_count=torque_saturation_count,
            torque_value_count=torque_value_count,
        )
        state.last_action_policy = clipped_action.copy()
        state.policy_tick += 1
        return result
