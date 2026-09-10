"""CUDA integration of the existing G1 controller, motion, physics and combat.

This is a candidate runtime. GPU residency does not establish authentic REK
parity, and the existing round-terminal episode boundary is retained here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from gpu_actuator_drive import GpuActuatorDrive
from gpu_combat_measurement import GpuFallMeasurementBatch, RekG1GpuCombatMeasurement
from gpu_controller import GearSonicGpuController
from gpu_duel_physics import GpuDuelPhysics
from gpu_duel_reset import GpuDuelReset
from gpu_motion_assets import GpuMotionAssets
from gpu_native_motion import GpuNativeMotion
from gpu_observation import GpuObservationAssembler
from gpu_robot_state import G1GpuControllerState
from gpu_semantic_scheduler import GpuSemanticScheduler


@dataclass(frozen=True)
class GpuDuelConfig:
    model: Path
    model_sha256: str
    assets: Path
    assets_sha256: str
    controller_manifest: Path
    controller_source: Path
    motion_features: Path
    motion_library: Path
    combat_library: Path
    move_duration_ticks: tuple[int, ...]
    locomotion_segment_ticks: int = 1
    device: str = "cuda:0"


class GpuSemanticDuel:
    def __init__(self, config: GpuDuelConfig):
        from gpu_native_combat import GpuNativeCombat

        self.config = config
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        self.stream = torch.cuda.Stream(device=config.device)
        self.stream.wait_stream(torch.cuda.current_stream(config.device))
        with torch.cuda.stream(self.stream):
            self.controller = GearSonicGpuController.from_manifest(
                config.controller_manifest, config.controller_source, device=config.device,
            )
            self.rows = self.controller.batch_size
            if self.rows < 2 or self.rows % 2:
                raise ValueError("two fighters are required per arena")
            self.arenas = self.rows // 2
            self.physics = GpuDuelPhysics(
                config.model, config.model_sha256, arenas=self.arenas, device=config.device,
            )
            self.assets = GpuMotionAssets(config.assets, config.assets_sha256, config.device)
            self.physical_reset = GpuDuelReset(self.physics, self.assets)
            self.all_arenas = torch.ones(self.arenas, device=config.device, dtype=torch.bool)
            self.all_rows = torch.ones(self.rows, device=config.device, dtype=torch.bool)
            self.physical_reset.full(self.all_arenas, reset_clock=True)
            self.observer = GpuObservationAssembler(self.physics)
            self.robot_state = G1GpuControllerState(self.rows, config.device)
            self.drive = GpuActuatorDrive.from_physics(self.physics)
            self.motion = GpuNativeMotion(
                self.assets, config.motion_features, config.motion_library, self.rows,
            )
            self.scheduler = GpuSemanticScheduler(
                self.motion, config.locomotion_segment_ticks, config.move_duration_ticks,
            )
            self.measurement = RekG1GpuCombatMeasurement.from_physics(self.physics)
            self.combat = GpuNativeCombat(
                config.combat_library, self.measurement, self.motion,
                active_route_ids=self.scheduler.active_route_ids,
            )
            self.actions = torch.ones(self.rows, device=config.device)
            self.can_get_up = torch.zeros(self.rows, dtype=torch.uint8, device=config.device)
            self.local_velocity = torch.empty((self.rows, 6), device=config.device)
            self.completed_reset_in_tick = torch.zeros(self.rows, dtype=torch.uint8, device=config.device)
            self.observations = self.observer.observations
            self.rewards = torch.zeros(self.rows, device=config.device)
            self.terminals = torch.zeros(self.rows, device=config.device)
            self.action_mask = self.scheduler.action_mask
            self._graph = None
            self._warp_capture = None
            self._reset_impl()
        self.stream.synchronize()

    def _gather(self):
        values = self.observer.gather_kinematics()
        self.local_velocity[:, :3] = values[:, 10:13]
        self.local_velocity[:, 3:] = values[:, 7:10]
        return values

    def _reset_impl(self):
        self.physical_reset.full(self.all_arenas, reset_clock=True)
        self.robot_state.reset(self.all_rows)
        self.drive.complete_reset(self.all_rows)
        self.scheduler.reset_rows(
            self.all_rows.to(torch.uint8), self.physical_reset.initial_heading.float(),
        )
        output = self.combat.reset()
        self.completed_reset_in_tick.zero_()
        self.actions.fill_(1)
        self.rewards.zero_()
        self.terminals.zero_()
        self._gather()
        self.observer.pack(output.fall, self.scheduler.observation12, output.fight)

    def reset(self):
        with torch.cuda.stream(self.stream):
            self._reset_impl()
        self.stream.synchronize()

    def _step_impl(self):
        output = self.combat.begin_tick()
        episode_mask = output.episode_reset != 0
        episode_rows = episode_mask.repeat_interleave(2)
        self.physical_reset.full(episode_mask)
        self.robot_state.reset(episode_rows)
        self.drive.complete_reset(episode_rows)
        self.scheduler.reset_rows(
            episode_rows.to(torch.uint8), self.physical_reset.initial_heading.float(),
        )
        self.completed_reset_in_tick.zero_()
        kinematics = self._gather()
        suspended = (~self.drive.active).to(torch.uint8)
        reference, next_reference, roots, heading = self.scheduler.pre_step(
            self.actions, self.local_velocity, suspended,
        )
        encoder_observation = self.robot_state.prepare(
            kinematics[:, 3:7], kinematics[:, 10:13], kinematics[:, 13:42],
            kinematics[:, 42:71], heading, reference, next_reference, roots,
            self.scheduler.active_policy,
        )
        tokens = self.controller.encode(encoder_observation)
        raw_actions = self.controller.decode(self.robot_state.decoder_input(tokens))
        targets = self.robot_state.apply_actions(raw_actions, self.scheduler.active_policy)
        for substep in range(10):
            q = self.physics.qpos[:, self.physical_reset.qindices].reshape(self.rows, 29)
            dq = self.physics.qvel[:, self.physical_reset.dqindices].reshape(self.rows, 29)
            controls = self.drive.prepare(targets, q, dq, substep=substep)
            self.physics.ctrl.copy_(controls.reshape(self.arenas, 58))
            self.physics.step()
            measurement = self.measurement.sample(substep, self.can_get_up)
            output = self.combat.post_step(measurement)
            live_controls = self.physics.ctrl.reshape(self.rows, 29)
            self.drive.set_dampened(self.drive.dampened | output.dampened, live_controls)
            begin = output.begin_reset != 0
            begin_rows = begin.repeat_interleave(2)
            self.drive.begin_reset(begin_rows, live_controls)
            self.physical_reset.begin(begin)
            self.completed_reset_in_tick.bitwise_or_(begin_rows.to(torch.uint8))
            complete = output.complete_reset != 0
            complete_rows = complete.repeat_interleave(2)
            self.physical_reset.complete(complete)
            self.drive.complete_reset(complete_rows)
            self.robot_state.reset(complete_rows)
            self.scheduler.reset_rows(
                complete_rows.to(torch.uint8), self.physical_reset.initial_heading.float(),
            )
            self.completed_reset_in_tick.bitwise_or_(complete_rows.to(torch.uint8))
        self._gather()
        # Only physically reset rows need refreshed fall geometry. Unaffected
        # rows retain the measurement from their final physics substep.
        refreshed = self.measurement.sample_fall(self.can_get_up)
        reset_rows = self.completed_reset_in_tick != 0
        fall = GpuFallMeasurementBatch(
            floats=torch.where(reset_rows[:, None], refreshed.floats, measurement.fall.floats),
            integers=torch.where(reset_rows[:, None], refreshed.integers, measurement.fall.integers),
            valid=torch.where(reset_rows, refreshed.valid, measurement.fall.valid),
            floor_body_contacts=torch.where(
                reset_rows.reshape(self.arenas, 2).any(dim=1)[:, None],
                refreshed.floor_body_contacts, measurement.fall.floor_body_contacts,
            ),
        )
        output = self.combat.observe(fall)
        self.scheduler.post_step(
            self.local_velocity, output.fall[:, 8].to(torch.int32).contiguous(),
            suspended=(~self.drive.active).to(torch.uint8),
            input_reset=output.input_reset.to(torch.uint8),
            reset_event=self.completed_reset_in_tick,
            terminal=output.terminals.to(torch.uint8),
        )
        self.observer.pack(output.fall, self.scheduler.observation12, output.fight)
        self.rewards.copy_(output.rewards)
        self.terminals.copy_(output.terminals)

    def capture_step(self):
        """Compile and capture all hot-path operations before starting training."""
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self._step_impl()
            self.stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                with self.physics.wp.ScopedCapture(stream=self.physics.stream, external=True) as capture:
                    self._step_impl()
            self._graph = graph
            self._warp_capture = capture
            self._reset_impl()
        self.stream.synchronize()
        self.check_status()

    def step(self, actions):
        if actions.shape != (self.rows, 1) or actions.device != self.actions.device:
            raise ValueError("actions must be CUDA [robot_rows, 1]")
        if self._graph is None:
            raise RuntimeError("capture_step must complete before environment stepping")
        caller_stream = torch.cuda.current_stream(self.actions.device)
        self.stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.stream):
            self.actions.copy_(actions[:, 0])
            self._graph.replay()
        caller_stream.wait_stream(self.stream)

    def check_status(self):
        """Explicit reporting boundary; never called inside a simulation step."""
        self.scheduler.check_status()
        self.combat.check_status()
        if not torch.isfinite(self.observations).all().item():
            raise RuntimeError("CUDA duel produced nonfinite observations")
        if not self.action_mask.any(dim=1).all().item():
            raise RuntimeError("CUDA duel produced an empty action mask")

    def log(self):
        self.check_status()
        return {}

    def close(self):
        torch.cuda.synchronize(self.actions.device)
