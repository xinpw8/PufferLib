"""CUDA tensor bridge to the original native motion state machines."""
from __future__ import annotations

import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

I = ct.c_int32
F = ct.c_float
P = ct.c_void_p
Z = ct.c_size_t


class Clip(ct.Structure):
    _fields_ = [('dof_position_mujoco', P), ('root_quaternion_wxyz', P),
        ('dof_position_count', Z), ('root_quaternion_count', Z), ('frame_count', Z), ('fps', F)]


class Config(ct.Structure):
    _fields_ = [('mirror', I), ('loop', I), ('playback_speed', F),
        ('start_frame', I), ('end_frame', I), ('blend_in_seconds', F),
        ('blend_out_seconds', F), ('yaw_blend', F)]


class Layer(ct.Structure):
    _fields_ = [('clip', Clip), ('config', Config), ('speed', F), ('per_tick', F),
        ('cursor', F), ('start_frame', I), ('end_frame', I), ('prev_heading', F),
        ('last_heading_delta', F), ('has_clip', I), ('has_config', I), ('active', I),
        ('heading_valid', I), ('heading_resync', I)]


class Backends(ct.Structure):
    _fields_ = [('quaternion_slerp', P), ('atan2_f', P), ('sin_cos_f', P),
        ('loop_entry_matcher', P), ('context', P)]


class Composer(ct.Structure):
    _fields_ = [('controller_rate_hz', I), ('current_layer', Layer), ('from_layer', Layer),
        ('backends', Backends), ('xt', I), ('w_in', I), ('w_out', I), ('w_total', I),
        ('action_playing', I), ('action_move_id', I), ('pending_heading_delta', F)]


class MatcherSlot(ct.Structure):
    _fields_ = [('clip_dof_position_identity', P), ('clip_root_quaternion_identity', P),
        ('clip_frame_count', Z), ('root_local_foot_xyz', P),
        ('root_local_foot_xyz_count', Z), ('registered', I)]


class MatcherDiagnostics(ct.Structure):
    _fields_ = [('outgoing_feature_cursor', F), ('transition_center_ticks', F),
        ('target_best_frame', I), ('best_squared_distance', F), ('valid', I)]


class Matcher(ct.Structure):
    _fields_ = [('slots', P), ('slot_capacity', Z), ('slot_count', Z),
        ('controller_rate_hz', I), ('last_status', I), ('diagnostics', MatcherDiagnostics)]


class MotionAsset(ct.Structure):
    _fields_ = [('clip', Clip), ('root_local_foot_xyz', P), ('root_local_foot_xyz_count', Z)]


class ComposerCommand(ct.Structure):
    _fields_ = [('operation', I), ('clip', Clip), ('config', Config), ('scale', F)]


class AdvanceResult(ct.Structure):
    _fields_ = [('current_wrapped', I), ('current_completed', I),
        ('outgoing_wrapped', I), ('outgoing_completed', I), ('weight_current', F)]


class ReferenceTiming(ct.Structure):
    _fields_ = [('current_offsets', I * 10), ('next_offsets', I * 10)]


class MirrorTable(ct.Structure):
    _fields_ = [('source_indices', P), ('negate', P), ('source_index_count', Z), ('negate_count', Z)]


class ReferenceOutput(ct.Structure):
    _fields_ = [('dof_position_mujoco', P), ('dof_next_position_mujoco', P),
        ('root_rotation_xyzw', P), ('dof_position_capacity', Z),
        ('dof_next_position_capacity', Z), ('root_rotation_capacity', Z)]


class VelocityCommand(ct.Structure):
    _fields_ = [('forward', F), ('strafe', F), ('yaw', F)]


class BaseVelocity(ct.Structure):
    _fields_ = [('angular_velocity_local', VelocityCommand),
        ('linear_velocity_local', VelocityCommand), ('available', ct.c_uint8)]


class LocomotionState(ct.Structure):
    _fields_ = [('stop_brake_command', VelocityCommand), ('last_driven_command', VelocityCommand),
        ('current_route_id', I), ('transition_from_route_id', I), ('momentum_route_id', I),
        ('locomotion_active', ct.c_uint8), ('transition_settling', ct.c_uint8),
        ('stop_braking', ct.c_uint8), ('has_momentum', ct.c_uint8)]


class LocomotionConfig(ct.Structure):
    _fields_ = [('settle_linear_speed', F), ('settle_yaw_rate', F), ('stop_brake_rate', F),
        ('transition_settle', ct.c_uint8)]


class LocomotionInput(ct.Structure):
    _fields_ = [('command', VelocityCommand), ('base_velocity', BaseVelocity), ('delta_seconds', F),
        ('restrict_yaw', ct.c_uint8), ('composer_action_playing', ct.c_uint8),
        ('composer_busy', ct.c_uint8), ('selected_route_playable', ct.c_uint8)]


class LocomotionResult(ct.Structure):
    _fields_ = [('next_state', LocomotionState), ('effective_velocity', VelocityCommand),
        ('selected_route_id', I), ('event_route_id', I), ('event', I),
        ('velocity_write', ct.c_uint8), ('transition_check_performed', ct.c_uint8),
        ('transition_settled', ct.c_uint8)]


class HeldState(ct.Structure):
    _fields_ = [('held', ct.c_uint8), ('yaw_ramp', F), ('yaw_sign', ct.c_int8)]


class InputFrame(ct.Structure):
    _fields_ = [('held', ct.c_uint8), ('attack_edge', ct.c_uint8)]


class InputTiming(ct.Structure):
    _fields_ = [('elapsed_seconds', F), ('yaw_ramp_seconds', F)]


class InputDecision(ct.Structure):
    _fields_ = [('status', I), ('attack_gate', I), ('held', ct.c_uint8),
        ('pressed_edges', ct.c_uint8), ('released_edges', ct.c_uint8),
        ('forward', ct.c_int8), ('strafe', ct.c_int8), ('desired_yaw', ct.c_int8),
        ('yaw', F), ('yaw_ramp', F), ('yaw_suppressed_for_attack', ct.c_uint8),
        ('blocked_attack_retention_unknown', ct.c_uint8)]


def upload(value, device):
    raw = np.frombuffer(ct.string_at(ct.addressof(value), ct.sizeof(value)), dtype=np.uint8).copy()
    return torch.as_tensor(raw, device=device)


class DeviceStructs:
    """Opaque C structs with typed, strided CUDA views of individual fields."""
    def __init__(self, kind, count, device, *, zero=False):
        self.kind, self.count = kind, count
        allocator = torch.zeros if zero else torch.empty
        self.tensor = allocator((count, ct.sizeof(kind)), dtype=torch.uint8, device=device)

    def field(self, name):
        offset, kind = 0, self.kind
        for part in name.split('.'):
            offset += getattr(kind, part).offset
            kind = dict(kind._fields_)[part]
        dimensions = []
        while issubclass(kind, ct.Array):
            dimensions.append(kind._length_)
            kind = kind._type_
        dtype = {I: torch.int32, F: torch.float32, ct.c_uint8: torch.uint8,
            ct.c_int8: torch.int8, ct.c_uint16: torch.uint16, ct.c_uint32: torch.uint32,
            Z: torch.uint64, P: torch.uint64}[kind]
        width = ct.sizeof(kind)
        values = self.tensor.view(dtype)
        if not dimensions:
            return values[:, offset // width]
        count = int(np.prod(dimensions))
        return values[:, offset // width:offset // width + count].reshape(self.count, *dimensions)

    def host(self):
        return (self.kind * self.count).from_buffer_copy(self.tensor.cpu().numpy().tobytes())


class GpuNativeMotion:
    def __init__(self, assets, feature_root: Path, library: Path, count: int):
        if count <= 0:
            raise ValueError('fighter count must be positive')
        self.assets = assets
        self.count = count
        self.device = next(iter(assets.arrays.values())).device
        self.library = ct.CDLL(str(library))
        self.features = {}
        feature_manifest = json.loads((feature_root / 'foot_features_manifest.json').read_text())
        if feature_manifest['asset_manifest_sha256'] != assets.manifest_sha256:
            raise ValueError('foot features belong to another asset manifest')
        records = {record['npz_path_id']: record for record in feature_manifest['clips']}
        self.clip_views = {}
        asset_views = []
        for clip_id, clip in assets.clips.items():
            positions = assets.arrays[clip['files']['mujoco_joint_order']]
            roots = assets.arrays[clip['files']['wxyz']]
            view = Clip(positions.data_ptr(), roots.data_ptr(), positions.numel(), roots.numel(),
                clip['frames'], clip['fps'])
            self.clip_views[clip_id] = view
            record = records[clip_id]
            raw = (feature_root / record['file']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record['sha256'] or len(raw) != clip['frames'] * 24:
                raise ValueError('foot feature identity/shape mismatch')
            values = np.frombuffer(raw, dtype='<f4').copy()
            if not np.isfinite(values).all():
                raise ValueError('foot features are nonfinite')
            self.features[clip_id] = torch.as_tensor(values, device=self.device)
            asset_views.append(MotionAsset(view, self.features[clip_id].data_ptr(), values.size))
        self.asset_tensor = upload((MotionAsset * len(asset_views))(*asset_views), self.device)
        route_views = []
        for route in assets.routes:
            route_views.append(ComposerCommand(1, self.clip_views[route['npz_path_id']],
                Config(**route['config']), 0))
        self.route_commands = upload((ComposerCommand * len(route_views))(*route_views),
            self.device).reshape(len(route_views), ct.sizeof(ComposerCommand))
        self.composers = DeviceStructs(Composer, count, self.device)
        self.matchers = DeviceStructs(Matcher, count, self.device)
        self.slots = DeviceStructs(MatcherSlot, count * len(asset_views), self.device)
        self.commands = DeviceStructs(ComposerCommand, count, self.device)
        self.advances = DeviceStructs(AdvanceResult, count, self.device)
        self.statuses = torch.empty(count, dtype=torch.int32, device=self.device)
        self.init_statuses = torch.empty((count, 2), dtype=torch.int32, device=self.device)
        # The native runtime allocates these with checked_calloc. Suspended
        # rows retain their references until their next active compose tick.
        self.positions = torch.zeros((count, 10, 29), device=self.device)
        self.next_positions = torch.zeros_like(self.positions)
        self.rotations = torch.zeros((count, 10, 4), device=self.device)
        outputs = (ReferenceOutput * count)(*[ReferenceOutput(
            self.positions.data_ptr() + i * 290 * 4,
            self.next_positions.data_ptr() + i * 290 * 4,
            self.rotations.data_ptr() + i * 40 * 4, 290, 290, 40) for i in range(count)])
        self.outputs = upload(outputs, self.device)
        # These tables are the existing binding.c and semantic_duel_runtime.c tables.
        indices = [6,7,8,9,10,11,0,1,2,3,4,5,12,13,14,22,23,24,25,26,27,28,15,16,17,18,19,20,21]
        negate = [0,1,1,0,0,1,0,1,1,0,0,1,1,1,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1]
        self.mirror_indices = torch.tensor(indices, dtype=torch.int32, device=self.device)
        self.mirror_negate = torch.tensor(negate, dtype=torch.uint8, device=self.device)
        self.mirror = upload(MirrorTable(self.mirror_indices.data_ptr(),
            self.mirror_negate.data_ptr(), 29, 29), self.device)
        self.timing = upload(ReferenceTiming((I * 10)(*range(0, 50, 5)),
            (I * 10)(*range(1, 51, 5))), self.device)
        self.heading_delta = torch.empty(count, device=self.device)
        self.heading_ownership = torch.empty(count, device=self.device)
        init = self.library.rek_g1_cuda_motion_init
        init.argtypes = [P, P, P, P, Z, I, P, Z, P]
        init.restype = I
        with torch.cuda.device(self.device):
            status = init(self.composers.tensor.data_ptr(), self.matchers.tensor.data_ptr(),
                self.slots.tensor.data_ptr(), self.asset_tensor.data_ptr(), len(asset_views), 50,
                self.init_statuses.data_ptr(), count, self.stream)
            self._check_launch(status)
        if torch.any(self.init_statuses != 0).item():
            raise RuntimeError(f'native CUDA motion initialization failed: {self.init_statuses.cpu().tolist()}')

    @property
    def stream(self):
        return torch.cuda.current_stream(self.device).cuda_stream

    @staticmethod
    def _check_launch(status):
        if status:
            raise RuntimeError(f'CUDA launch failed with runtime code {status}')

    def _call(self, name, *tensors):
        function = getattr(self.library, name)
        function.argtypes = [P] * len(tensors) + [Z, P]
        function.restype = I
        with torch.cuda.device(self.device):
            status = function(*(tensor.data_ptr() for tensor in tensors), self.count, self.stream)
            self._check_launch(status)
        return self.statuses

    def command(self, route_indices, operations=None, scales=None):
        if route_indices.shape != (self.count,) or route_indices.dtype != torch.int64 \
                or route_indices.device != self.device:
            raise ValueError('route indices must be one CUDA int64 per fighter')
        torch.index_select(self.route_commands, 0, route_indices, out=self.commands.tensor)
        if operations is not None:
            self.commands.field('operation').copy_(operations)
        if scales is not None:
            self.commands.field('scale').copy_(scales)
        return self._call('rek_g1_cuda_composer_command', self.composers.tensor,
            self.commands.tensor, self.statuses)

    def advance(self):
        return self._call('rek_g1_cuda_composer_advance', self.composers.tensor,
            self.advances.tensor, self.statuses)

    def reference(self):
        return self._call('rek_g1_cuda_composer_reference', self.composers.tensor,
            self.timing, self.mirror, self.outputs, self.statuses)

    def heading(self):
        return self._call('rek_g1_cuda_composer_heading', self.composers.tensor,
            self.heading_delta, self.heading_ownership, self.statuses)
