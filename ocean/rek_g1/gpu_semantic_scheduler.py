"""GPU semantic scheduling using the original categorical/native motion code.

Physics, observations of bodies, fall/combat state and reset poses are supplied
by the caller. This module has no physics backend or CPU per-fighter hot path.
The categorical adapter retains no hidden keyboard buffer. Human key-edge
buffering belongs to the existing human-evaluation adapter.
"""
from __future__ import annotations

import ctypes as ct

import torch

from gpu_native_motion import (DeviceStructs, F, HeldState, I, InputDecision,
    InputTiming, LocomotionConfig, LocomotionState, P, VelocityCommand, Z, upload)

U8 = ct.c_uint8
U16 = ct.c_uint16
U32 = ct.c_uint32


class SemanticCommand(ct.Structure):
    _fields_ = [('kind', I), ('held_code', U8), ('duration_ticks', U32),
        ('move_registry_index', U16)]


class SemanticScheduler(ct.Structure):
    _fields_ = [('input_state', HeldState), ('command', SemanticCommand),
        ('remaining_ticks', U32), ('active', U8), ('first_tick', U8), ('move_accepted', U8)]


class SemanticTick(ct.Structure):
    _fields_ = [('status', I), ('input', InputDecision), ('kind', I),
        ('move_registry_index', U16), ('remaining_ticks', U32),
        ('command_started', U8), ('segment_complete', U8), ('move_start_edge', U8),
        ('move_active', U8), ('move_blocked', U8)]


class Category(ct.Structure):
    _fields_ = [('kind', I), ('command', SemanticCommand)]


class ActionTable(ct.Structure):
    _fields_ = [('categories', P), ('count', U32), ('move_indices', P),
        ('move_duration_ticks', P), ('move_registry_count', U16)]


class ActionTableStorage(ct.Structure):
    _fields_ = [('categories', Category * 33), ('move_indices', U16 * 17),
        ('move_duration_ticks', U32 * 17), ('table', ActionTable)]


class Adapter(ct.Structure):
    _fields_ = [('scheduler', SemanticScheduler), ('table', P)]


class PufferStep(ct.Structure):
    _fields_ = [('status', I), ('category', U32), ('semantic', SemanticTick)]


class CommandConfig(ct.Structure):
    _fields_ = [('locomotion_speed_scale', F), ('command_yaw_rate_scale', F),
        ('heading_yaw_rate_scale', F), ('controller_rate_hz', U32)]


class SchedulerConfig(ct.Structure):
    _fields_ = [('timing', InputTiming), ('command', CommandConfig),
        ('locomotion', LocomotionConfig)]


class SchedulerRow(ct.Structure):
    _fields_ = [('adapter', Adapter), ('locomotion', LocomotionState),
        ('effective_velocity', VelocityCommand), ('active_route_id', I),
        ('semantic', SemanticTick), ('translation_settled', U8), ('action_busy', U8),
        ('recovery_active', U8), ('status', I)]


class SchedulerBuffers(ct.Structure):
    _fields_ = [(name, P) for name in ('rows', 'composers', 'table', 'config',
        'routes', 'route_kinds', 'move_routes', 'reference_timing', 'mirror',
        'references', 'heading_wxyz', 'observation12', 'masks')]


class GpuSemanticScheduler:
    def __init__(self, motion, locomotion_segment_ticks: int, move_duration_ticks):
        if locomotion_segment_ticks <= 0 or locomotion_segment_ticks > 0xffffffff:
            raise ValueError('locomotion segment ticks must be a positive uint32')
        durations = tuple(int(value) for value in move_duration_ticks)
        if len(durations) != 17 or any(value <= 0 or value > 0xffffffff for value in durations):
            raise ValueError('17 configured move durations are required, indexed by runtime move ID')
        self.motion = motion
        self.count, self.device, self.library = motion.count, motion.device, motion.library
        # These are the pinned binding.c values, not estimates from animation.
        self.config_value = SchedulerConfig(InputTiming(0.02, 0.5),
            CommandConfig(1.0, 1.0, 1.0, 50), LocomotionConfig(0.03, 0.03, 2.0, 1))
        self.config = upload(self.config_value, self.device)
        self.table = DeviceStructs(ActionTableStorage, 1, self.device)
        self.durations = torch.tensor(durations, dtype=torch.uint32, device=self.device)
        self.table_status = torch.empty(1, dtype=torch.int32, device=self.device)
        function = self.library.rek_g1_cuda_semantic_table_init
        function.argtypes = [P, U32, P, P, P]
        function.restype = I
        with torch.cuda.device(self.device):
            motion._check_launch(function(self.table.tensor.data_ptr(), locomotion_segment_ticks,
                self.durations.data_ptr(), self.table_status.data_ptr(), motion.stream))
        if self.table_status.item():
            raise ValueError(f'native action table rejected: {self.table_status.item()}')
        kinds = {'idle': 0, 'translation': 1, 'turn': 2, 'discrete_move': 3, 'kick': 3}
        route_kinds, move_routes = [], [-1] * 17
        for index, route in enumerate(motion.assets.routes):
            if route['route_id'] != index:
                raise ValueError('route commands must have original contiguous route IDs')
            route_kinds.append(kinds[route['kind']])
            if route['runtime_move_index'] is not None:
                move_routes[route['runtime_move_index']] = index
        if len(route_kinds) != 24 or -1 in move_routes:
            raise ValueError('original complete 24-route/17-move table is required')
        self.route_kinds = torch.tensor(route_kinds, device=self.device, dtype=torch.int32)
        self.move_routes = torch.tensor(move_routes, device=self.device, dtype=torch.int32)
        self.rows = DeviceStructs(SchedulerRow, self.count, self.device, zero=True)
        self.heading = torch.zeros((self.count, 4), device=self.device)
        self.heading[:, 0] = 1.0
        self.observation12 = torch.empty((self.count, 12), device=self.device)
        self.action_mask = torch.empty((self.count, 33), device=self.device, dtype=torch.uint8)
        self.zero_flags = torch.zeros(self.count, device=self.device, dtype=torch.uint8)
        self.active_policy = torch.ones(self.count, device=self.device, dtype=torch.bool)
        buffers = SchedulerBuffers(self.rows.tensor.data_ptr(), motion.composers.tensor.data_ptr(),
            self.table.tensor.data_ptr(), self.config.data_ptr(), motion.route_commands.data_ptr(),
            self.route_kinds.data_ptr(), self.move_routes.data_ptr(), motion.timing.data_ptr(),
            motion.mirror.data_ptr(), motion.outputs.data_ptr(), self.heading.data_ptr(),
            self.observation12.data_ptr(), self.action_mask.data_ptr())
        self.buffers = upload(buffers, self.device)
        self.statuses = self.rows.field('status')
        self.active_route_ids = self.rows.field('active_route_id')
        # All field accessors below are views into the native device structs.
        self.semantic_kind = self.rows.field('semantic.kind')
        self.move_registry_index = self.rows.field('semantic.move_registry_index')
        self.move_start_edge = self.rows.field('semantic.move_start_edge')
        self.move_active = self.rows.field('semantic.move_active')
        self.reset_rows(torch.ones_like(self.zero_flags), self.heading)
        self.check_status()

    def _input(self, value, shape, dtype, name):
        if value.shape != shape or value.dtype != dtype or value.device != self.device \
                or not value.is_contiguous():
            raise ValueError(f'{name} must be contiguous {dtype} {shape} on {self.device}')
        return value

    def _flags(self, value, name):
        if value is None:
            return self.zero_flags
        return self._input(value, (self.count,), torch.uint8, name)

    def _call(self, name, *tensors):
        function = getattr(self.library, name)
        function.argtypes = [P] * (len(tensors) + 1) + [Z, P]
        function.restype = I
        with torch.cuda.device(self.device):
            self.motion._check_launch(function(self.buffers.data_ptr(),
                *(value.data_ptr() for value in tensors), self.count, self.motion.stream))

    def reset_rows(self, reset, heading_wxyz):
        """Reset composer/input state after the caller restores physical state."""
        self._call('rek_g1_cuda_semantic_reset', self._flags(reset, 'reset'),
            self._input(heading_wxyz, (self.count, 4), torch.float32, 'heading'))

    def pre_step(self, actions, local_velocity, suspended=None):
        """Select motion and build references before the controller/physics tick."""
        actions = self._input(actions, (self.count,), torch.float32, 'actions')
        velocity = self._input(local_velocity, (self.count, 6), torch.float32, 'local velocity')
        suspended = self._flags(suspended, 'suspended')
        self._call('rek_g1_cuda_semantic_pre', actions, velocity, suspended)
        torch.logical_and(suspended == 0, self.statuses == 0, out=self.active_policy)
        return self.motion.positions, self.motion.next_positions, self.motion.rotations, self.heading

    def post_step(self, local_velocity, fall_phase, suspended=None, input_reset=None,
            reset_event=None, terminal=None):
        """Advance composer once after physics and publish next masks/facts."""
        self._call('rek_g1_cuda_semantic_post',
            self._input(local_velocity, (self.count, 6), torch.float32, 'local velocity'),
            self._input(fall_phase, (self.count,), torch.int32, 'fall phase'),
            self._flags(suspended, 'suspended'), self._flags(input_reset, 'input reset'),
            self._flags(reset_event, 'reset event'), self._flags(terminal, 'terminal'))
        return self.observation12, self.action_mask

    def check_status(self):
        """Explicit synchronization for initialization/tests/reporting, not hot path."""
        statuses = self.statuses.cpu()
        if torch.any(statuses != 0):
            rows = self.rows.host()
            composers = self.motion.composers.host()
            matchers = self.motion.matchers.host()
            details = []
            for index in torch.nonzero(statuses, as_tuple=False).flatten().tolist():
                row, composer, matcher = rows[index], composers[index], matchers[index]
                details.append({
                    'fighter_row': index,
                    'status': int(row.status),
                    'loop_matcher_status': int(matcher.last_status),
                    'semantic_kind': int(row.semantic.kind),
                    'move_registry_index': int(row.semantic.move_registry_index),
                    'move_start_edge': int(row.semantic.move_start_edge),
                    'active_route_id': int(row.active_route_id),
                    'current_cursor': float(composer.current_layer.cursor),
                    'current_active': int(composer.current_layer.active),
                    'current_loop': int(composer.current_layer.config.loop),
                    'from_cursor': float(composer.from_layer.cursor),
                    'action_playing': int(composer.action_playing),
                })
            raise RuntimeError(f'native CUDA semantic scheduler failed: {details}')
