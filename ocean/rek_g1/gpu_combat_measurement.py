"""Exact CUDA-side MuJoCo facts for the REK G1 fall and hit adapters.

The host-model constructor validates the same build-pinned identities used by
``g1_fall_mujoco.c`` and ``g1_hit_mujoco.c``. After construction, reset and
sample operations use only tensors on the execution device. A sample does not
call ``cpu()``, ``item()``, or otherwise synchronize the device.

Hit candidates retain MuJoCo's raw contact-slot order. Slot ``2*i`` tests
``geom[i, 0]`` as striker and slot ``2*i + 1`` tests ``geom[i, 1]``. Invalid
slots remain allocated and are identified by ``candidate_valid``. A separate
device-only ``candidate_order`` groups valid rows by arena while retaining each
Warp contact slot and directed-striker order. The hit cooldown state machine is
order-sensitive, so geometry-identity sorting would change native semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


PHYSICS_SUBSTEPS = 10
PHYSICS_DELTA_SECONDS = 0.002
MODEL_BODY_COUNT = 63
MODEL_GEOM_COUNT = 91
MODEL_QPOS_COUNT = 72
MODEL_QVEL_COUNT = 70
MODEL_CONTROL_COUNT = 58

BODY_PART_NONE = 0
BODY_PART_HAND = 1
BODY_PART_FOOT = 2
BODY_PART_SHIN = 8

BODY_ZONE_UNKNOWN = 0
BODY_ZONE_HEAD = 1
BODY_ZONE_TORSO = 2
BODY_ZONE_PELVIS = 3
BODY_ZONE_LEFT_SHOULDER = 4
BODY_ZONE_RIGHT_SHOULDER = 5
BODY_ZONE_LEFT_ELBOW = 6
BODY_ZONE_RIGHT_ELBOW = 7
BODY_ZONE_LEFT_WRIST = 8
BODY_ZONE_RIGHT_WRIST = 9
BODY_ZONE_LEFT_HIP = 12
BODY_ZONE_RIGHT_HIP = 13
BODY_ZONE_LEFT_KNEE = 14
BODY_ZONE_RIGHT_KNEE = 15
BODY_ZONE_LEFT_ANKLE = 16
BODY_ZONE_RIGHT_ANKLE = 17

LEFT_HAND_SLOT = 0
RIGHT_HAND_SLOT = 1
LEFT_FOOT_SLOT = 2
RIGHT_FOOT_SLOT = 3
LEFT_SHIN_SLOT = 4
RIGHT_SHIN_SLOT = 5

FALL_TILT_DEGREES = 0
FALL_PELVIS_HEIGHT_RATIO = 1
FALL_FIXED_DELTA_SECONDS = 2
FALL_FLOOR_HEIGHT = 3
FALL_STANDING_PELVIS_HEIGHT = 4
FALL_FLOAT_FIELDS = 5

FALL_TRACKING_ACTIVE = 0
FALL_BOTH_FEET_OFF_FLOOR = 1
FALL_HAS_FOOT_BODY_CONTACT = 2
FALL_DISTINCT_NONFOOT_BODY_CONTACT_COUNT = 3
FALL_CAN_GET_UP = 4
FALL_LEFT_FOOT_BODY_CONTACT = 5
FALL_RIGHT_FOOT_BODY_CONTACT = 6
FALL_INTEGER_FIELDS = 7

HIT_ARENA_INDEX = 0
HIT_PHYSICS_SUBSTEP_INDEX = 1
HIT_STRIKER_GEOM_ID = 2
HIT_TARGET_GEOM_ID = 3
HIT_STRIKER_BODY_ID = 4
HIT_TARGET_BODY_ID = 5
HIT_STRIKER_FIGHTER = 6
HIT_TARGET_FIGHTER = 7
HIT_STRIKER_BODY_SLOT = 8
HIT_STRIKER_PART = 9
HIT_STRIKER_SIDE = 10
HIT_TARGET_ZONE = 11
HIT_INTEGER_FIELDS = 12

HIT_STRIKER_POSITION = slice(0, 3)
HIT_TARGET_POSITION = slice(3, 6)
HIT_STRIKER_LINEAR_VELOCITY = slice(6, 9)
HIT_TARGET_LINEAR_VELOCITY = slice(9, 12)
HIT_RELATIVE_SPEED_MPS = 12
HIT_FLOAT_FIELDS = 13

_FLOOR_GEOM = "arena_Collider_Floor_Rektagon"
_FIGHTER_PREFIXES = ("player__", "opponent__")
_HEAD_GEOM_SUFFIX = "mjgeom_3064"
_TORSO_GEOM_SUFFIX = "mjgeom_3285"

# suffix, target zone, striker part, side, slot, exact geom count
_PINNED_BODIES = (
    ("pelvis_3266", BODY_ZONE_PELVIS, BODY_PART_NONE, -1, -1, 1),
    ("left_hip_pitch_link_3457", BODY_ZONE_LEFT_HIP, BODY_PART_NONE, -1, -1, 1),
    ("left_hip_roll_link_3425", BODY_ZONE_LEFT_HIP, BODY_PART_NONE, -1, -1, 1),
    ("left_hip_yaw_link_2943", BODY_ZONE_LEFT_HIP, BODY_PART_NONE, -1, -1, 1),
    ("left_knee_link_3106", BODY_ZONE_LEFT_KNEE, BODY_PART_SHIN, 0, LEFT_SHIN_SLOT, 1),
    ("left_ankle_pitch_link_3033", BODY_ZONE_LEFT_ANKLE, BODY_PART_NONE, -1, -1, 1),
    ("left_ankle_roll_link_3045", BODY_ZONE_LEFT_ANKLE, BODY_PART_FOOT, 0, LEFT_FOOT_SLOT, 4),
    ("right_hip_pitch_link_3469", BODY_ZONE_RIGHT_HIP, BODY_PART_NONE, -1, -1, 1),
    ("right_hip_roll_link_3345", BODY_ZONE_RIGHT_HIP, BODY_PART_NONE, -1, -1, 1),
    ("right_hip_yaw_link_3191", BODY_ZONE_RIGHT_HIP, BODY_PART_NONE, -1, -1, 1),
    ("right_knee_link_3429", BODY_ZONE_RIGHT_KNEE, BODY_PART_SHIN, 1, RIGHT_SHIN_SLOT, 1),
    ("right_ankle_pitch_link_3173", BODY_ZONE_RIGHT_ANKLE, BODY_PART_NONE, -1, -1, 1),
    ("right_ankle_roll_link_3090", BODY_ZONE_RIGHT_ANKLE, BODY_PART_FOOT, 1, RIGHT_FOOT_SLOT, 4),
    ("waist_yaw_link_3359", BODY_ZONE_UNKNOWN, BODY_PART_NONE, -1, -1, 1),
    ("waist_roll_link_2894", BODY_ZONE_UNKNOWN, BODY_PART_NONE, -1, -1, 1),
    ("torso_link_3347", BODY_ZONE_TORSO, BODY_PART_NONE, -1, -1, 2),
    ("left_shoulder_pitch_link_3161", BODY_ZONE_LEFT_SHOULDER, BODY_PART_NONE, -1, -1, 1),
    ("left_shoulder_roll_link_3360", BODY_ZONE_LEFT_SHOULDER, BODY_PART_NONE, -1, -1, 1),
    ("left_shoulder_yaw_link_3168", BODY_ZONE_LEFT_SHOULDER, BODY_PART_NONE, -1, -1, 1),
    ("left_elbow_link_3284", BODY_ZONE_LEFT_ELBOW, BODY_PART_NONE, -1, -1, 1),
    ("left_wrist_roll_link_3032", BODY_ZONE_LEFT_WRIST, BODY_PART_NONE, -1, -1, 1),
    ("left_wrist_pitch_link_2914", BODY_ZONE_LEFT_WRIST, BODY_PART_NONE, -1, -1, 1),
    ("left_wrist_yaw_link_3467", BODY_ZONE_LEFT_WRIST, BODY_PART_HAND, 0, LEFT_HAND_SLOT, 1),
    ("right_shoulder_pitch_link_3391", BODY_ZONE_RIGHT_SHOULDER, BODY_PART_NONE, -1, -1, 1),
    ("right_shoulder_roll_link_3426", BODY_ZONE_RIGHT_SHOULDER, BODY_PART_NONE, -1, -1, 1),
    ("right_shoulder_yaw_link_3107", BODY_ZONE_RIGHT_SHOULDER, BODY_PART_NONE, -1, -1, 1),
    ("right_elbow_link_3322", BODY_ZONE_RIGHT_ELBOW, BODY_PART_NONE, -1, -1, 1),
    ("right_wrist_roll_link_3331", BODY_ZONE_RIGHT_WRIST, BODY_PART_NONE, -1, -1, 1),
    ("right_wrist_pitch_link_3282", BODY_ZONE_RIGHT_WRIST, BODY_PART_NONE, -1, -1, 1),
    ("right_wrist_yaw_link_3293", BODY_ZONE_RIGHT_WRIST, BODY_PART_HAND, 1, RIGHT_HAND_SLOT, 1),
)

_EXPECTED_ROLE_CONTACT_MASK_COUNTS = (
    {(5, 9): 12, (5, 10): 25},
    {(9, 5): 12, (9, 6): 25},
)


def _tensor(values: Any, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.long, device=device)


@dataclass(frozen=True)
class GpuCombatModelMap:
    """Build-pinned identities uploaded once to the execution device."""

    body_count: int
    geom_count: int
    floor_geom_id: int
    floor_half_height: float
    root_qpos_addresses: tuple[int, int]
    root_body_ids: torch.Tensor
    left_foot_body_ids: torch.Tensor
    right_foot_body_ids: torch.Tensor
    geom_body_ids: torch.Tensor
    body_root_ids: torch.Tensor
    body_owner: torch.Tensor
    body_zone: torch.Tensor
    geom_zone: torch.Tensor
    striker_part: torch.Tensor
    striker_side: torch.Tensor
    striker_slot: torch.Tensor
    geom_contype: torch.Tensor
    geom_conaffinity: torch.Tensor

    @classmethod
    def from_host_model(
        cls,
        host_model: Any,
        device: torch.device | str,
    ) -> "GpuCombatModelMap":
        """Validate exact native identities and upload their integer maps."""
        import mujoco

        expected = {
            "nbody": MODEL_BODY_COUNT,
            "ngeom": MODEL_GEOM_COUNT,
            "nq": MODEL_QPOS_COUNT,
            "nv": MODEL_QVEL_COUNT,
            "nu": MODEL_CONTROL_COUNT,
        }
        for name, value in expected.items():
            if int(getattr(host_model, name)) != value:
                raise ValueError(f"model {name} differs from the pinned G1 duel")
        if not math.isfinite(float(host_model.opt.timestep)) or not math.isclose(
            float(host_model.opt.timestep), PHYSICS_DELTA_SECONDS,
            rel_tol=0.0, abs_tol=1e-15,
        ):
            raise ValueError("model physics timestep differs from 2 ms")

        target_device = torch.device(device)
        parent = [int(value) for value in host_model.body_parentid]

        def object_id(kind: Any, name: str) -> int:
            result = int(mujoco.mj_name2id(host_model, kind, name))
            if result < 0:
                raise ValueError(f"pinned MuJoCo identity is missing: {name}")
            return result

        def descends(body_id: int, root_id: int) -> bool:
            current = body_id
            while current > 0:
                if current == root_id:
                    return True
                current = parent[current]
            return False

        roots = tuple(
            object_id(mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis_3266")
            for prefix in _FIGHTER_PREFIXES
        )
        if roots[0] == roots[1]:
            raise ValueError("fighter root bodies overlap")
        root_qpos_addresses: list[int] = []
        for fighter, root in enumerate(roots):
            joint_id = int(host_model.body_jntadr[root])
            if (
                int(host_model.body_jntnum[root]) < 1
                or joint_id < 0
                or int(host_model.jnt_type[joint_id])
                    != int(mujoco.mjtJoint.mjJNT_FREE)
            ):
                raise ValueError("fighter root is not attached by a free joint")
            qpos_address = int(host_model.jnt_qposadr[joint_id])
            qvel_address = int(host_model.jnt_dofadr[joint_id])
            if qpos_address != fighter * 36 or qvel_address != fighter * 35:
                raise ValueError("fighter root address differs from the duel map")
            root_qpos_addresses.append(qpos_address)

        owner = [-1] * MODEL_BODY_COUNT
        owner_counts = [0, 0]
        for body_id in range(1, MODEL_BODY_COUNT):
            membership = [descends(body_id, root) for root in roots]
            if all(membership):
                raise ValueError("fighter body trees overlap")
            if any(membership):
                fighter = 0 if membership[0] else 1
                owner[body_id] = fighter
                owner_counts[fighter] += 1
        if owner_counts != [30, 30]:
            raise ValueError("fighter body-tree size differs from the pinned model")

        geom_body_ids = [int(value) for value in host_model.geom_bodyid]
        body_zone = [BODY_ZONE_UNKNOWN] * MODEL_BODY_COUNT
        geom_zone = [BODY_ZONE_UNKNOWN] * MODEL_GEOM_COUNT
        striker_part = [BODY_PART_NONE] * MODEL_BODY_COUNT
        striker_side = [-1] * MODEL_BODY_COUNT
        striker_slot = [-1] * MODEL_BODY_COUNT
        body_ids: list[list[int]] = [[], []]
        descriptor_seen: set[int] = set()
        for fighter, prefix in enumerate(_FIGHTER_PREFIXES):
            seen_slots: set[int] = set()
            for suffix, zone, part, side, slot, geom_count in _PINNED_BODIES:
                body_id = object_id(mujoco.mjtObj.mjOBJ_BODY, prefix + suffix)
                if body_id in descriptor_seen or owner[body_id] != fighter:
                    raise ValueError("pinned G1 body identity or ownership differs")
                if sum(value == body_id for value in geom_body_ids) != geom_count:
                    raise ValueError("pinned G1 body geom count differs")
                descriptor_seen.add(body_id)
                body_ids[fighter].append(body_id)
                body_zone[body_id] = zone
                striker_part[body_id] = part
                striker_side[body_id] = side
                striker_slot[body_id] = slot
                if slot >= 0:
                    if slot in seen_slots:
                        raise ValueError("striker slot mapping is not one-to-one")
                    seen_slots.add(slot)
            if seen_slots != set(range(6)):
                raise ValueError("a pinned striker slot is missing")
        if descriptor_seen != {i for i in range(1, MODEL_BODY_COUNT) if owner[i] >= 0}:
            raise ValueError("fighter descendant lacks a pinned descriptor")

        for geom_id, body_id in enumerate(geom_body_ids):
            if body_id < 0 or body_id >= MODEL_BODY_COUNT:
                raise ValueError("geom resolves to an invalid body")
            geom_zone[geom_id] = body_zone[body_id]
        for fighter, prefix in enumerate(_FIGHTER_PREFIXES):
            head = object_id(mujoco.mjtObj.mjOBJ_GEOM, prefix + _HEAD_GEOM_SUFFIX)
            torso = object_id(mujoco.mjtObj.mjOBJ_GEOM, prefix + _TORSO_GEOM_SUFFIX)
            torso_body = body_ids[fighter][15]
            if head == torso or geom_body_ids[head] != torso_body \
                    or geom_body_ids[torso] != torso_body:
                raise ValueError("pinned head or torso geometry differs")
            geom_zone[head] = BODY_ZONE_HEAD

        floor = object_id(mujoco.mjtObj.mjOBJ_GEOM, _FLOOR_GEOM)
        if (
            int(host_model.geom_type[floor]) != int(mujoco.mjtGeom.mjGEOM_BOX)
            or geom_body_ids[floor] != 0
        ):
            raise ValueError("floor geometry is not the exact fixed world box")
        floor_size = tuple(float(value) for value in host_model.geom_size[floor])
        if any(not math.isfinite(value) or value <= 0.0 for value in floor_size):
            raise ValueError("floor box size is invalid")
        floor_half_height = floor_size[2]

        contype = [int(value) for value in host_model.geom_contype]
        conaffinity = [int(value) for value in host_model.geom_conaffinity]
        role_geoms: list[list[int]] = [[], []]
        for geom_id, body_id in enumerate(geom_body_ids):
            if owner[body_id] >= 0:
                role_geoms[owner[body_id]].append(geom_id)
        for fighter in range(2):
            counts: dict[tuple[int, int], int] = {}
            for geom_id in role_geoms[fighter]:
                key = (contype[geom_id], conaffinity[geom_id])
                counts[key] = counts.get(key, 0) + 1
            if counts != _EXPECTED_ROLE_CONTACT_MASK_COUNTS[fighter]:
                raise ValueError("fighter role contact masks differ from the corrected model")
        for left in role_geoms[0]:
            for right in role_geoms[1]:
                if not (
                    contype[left] & conaffinity[right]
                    or contype[right] & conaffinity[left]
                ):
                    raise ValueError("cross-fighter contact mask blocks a geom pair")

        left_feet = (body_ids[0][6], body_ids[1][6])
        right_feet = (body_ids[0][12], body_ids[1][12])
        return cls(
            body_count=MODEL_BODY_COUNT,
            geom_count=MODEL_GEOM_COUNT,
            floor_geom_id=floor,
            floor_half_height=floor_half_height,
            root_qpos_addresses=tuple(root_qpos_addresses),
            root_body_ids=_tensor(roots, device=target_device),
            left_foot_body_ids=_tensor(left_feet, device=target_device),
            right_foot_body_ids=_tensor(right_feet, device=target_device),
            geom_body_ids=_tensor(geom_body_ids, device=target_device),
            body_root_ids=_tensor(host_model.body_rootid, device=target_device),
            body_owner=_tensor(owner, device=target_device),
            body_zone=_tensor(body_zone, device=target_device),
            geom_zone=_tensor(geom_zone, device=target_device),
            striker_part=_tensor(striker_part, device=target_device),
            striker_side=_tensor(striker_side, device=target_device),
            striker_slot=_tensor(striker_slot, device=target_device),
            geom_contype=_tensor(contype, device=target_device),
            geom_conaffinity=_tensor(conaffinity, device=target_device),
        )


@dataclass(frozen=True)
class GpuCombatTensorSource:
    """Zero-copy dynamic MuJoCo Warp tensor views."""

    qpos: torch.Tensor
    xpos: torch.Tensor
    xipos: torch.Tensor
    subtree_com: torch.Tensor
    cvel: torch.Tensor
    geom_xpos: torch.Tensor
    geom_xmat: torch.Tensor
    contact_geom: torch.Tensor
    contact_worldid: torch.Tensor
    contact_dist: torch.Tensor
    contact_pos: torch.Tensor
    contact_frame: torch.Tensor
    nacon: torch.Tensor
    time: torch.Tensor

    @classmethod
    def from_physics(cls, physics: Any) -> "GpuCombatTensorSource":
        wp = physics.wp
        data = physics.data
        return cls(
            qpos=physics.qpos,
            xpos=wp.to_torch(data.xpos),
            xipos=wp.to_torch(data.xipos),
            subtree_com=wp.to_torch(data.subtree_com),
            cvel=wp.to_torch(data.cvel),
            geom_xpos=wp.to_torch(data.geom_xpos),
            geom_xmat=wp.to_torch(data.geom_xmat),
            contact_geom=wp.to_torch(data.contact.geom),
            contact_worldid=wp.to_torch(data.contact.worldid),
            contact_dist=wp.to_torch(data.contact.dist),
            contact_pos=wp.to_torch(data.contact.pos),
            contact_frame=wp.to_torch(data.contact.frame),
            nacon=wp.to_torch(data.nacon),
            time=physics.time,
        )


@dataclass(frozen=True)
class GpuFallMeasurementBatch:
    """Tensor fields corresponding to ``RekG1FallMujocoMeasurement``."""

    floats: torch.Tensor
    integers: torch.Tensor
    valid: torch.Tensor
    floor_body_contacts: torch.Tensor


@dataclass(frozen=True)
class GpuHitCandidateBatch:
    """Fixed-capacity fields corresponding to directed hit candidates."""

    integers: torch.Tensor
    floats: torch.Tensor
    candidate_valid: torch.Tensor
    candidate_processing_key: torch.Tensor
    candidate_order: torch.Tensor
    candidate_offsets: torch.Tensor
    candidate_counts: torch.Tensor
    arena_scan_valid: torch.Tensor
    contact_capacity_overflow: torch.Tensor
    arena_time_seconds: torch.Tensor


@dataclass(frozen=True)
class GpuCombatMeasurementBatch:
    fall: GpuFallMeasurementBatch
    hits: GpuHitCandidateBatch


@dataclass(frozen=True)
class _ContactFacts:
    active: torch.Tensor
    capacity_valid: torch.Tensor
    world: torch.Tensor
    world_safe: torch.Tensor
    world_valid: torch.Tensor
    geom0: torch.Tensor
    geom1: torch.Tensor
    geom0_safe: torch.Tensor
    geom1_safe: torch.Tensor
    geom_valid: torch.Tensor
    fall_arena_valid: torch.Tensor
    floor_body_contacts: torch.Tensor


def _quat_rotate_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    w = quaternion[..., 0:1]
    xyz = quaternion[..., 1:4]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + w * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


class RekG1GpuCombatMeasurement:
    """Stateful contact-enter and reset-calibrated fall measurement."""

    def __init__(
        self,
        model_map: GpuCombatModelMap,
        source: GpuCombatTensorSource,
    ) -> None:
        self.model_map = model_map
        self.source = source
        self.device = source.qpos.device
        self._validate_shapes()
        self.arena_count = int(source.qpos.shape[0])
        self.robot_count = self.arena_count * 2
        self.contact_capacity = int(source.contact_geom.shape[0])
        self.pair_span = model_map.geom_count * model_map.geom_count
        self._previous_pairs = torch.zeros(
            (self.arena_count, self.pair_span),
            dtype=torch.bool,
            device=self.device,
        )
        self._current_pair_counts = torch.zeros(
            (self.arena_count, self.pair_span),
            dtype=torch.int32,
            device=self.device,
        )
        self._first_pair_slot = torch.empty(
            (self.arena_count, self.pair_span),
            dtype=torch.int32,
            device=self.device,
        )
        self._expected_substep = torch.zeros(
            self.arena_count, dtype=torch.long, device=self.device)
        self._upright_up_local = torch.zeros(
            (self.robot_count, 3), dtype=source.qpos.dtype, device=self.device)
        self._standing_pelvis_height = torch.zeros(
            self.robot_count, dtype=source.qpos.dtype, device=self.device)
        self._reset_floor_height = torch.zeros_like(
            self._standing_pelvis_height)
        self._calibrated = torch.zeros(
            self.robot_count, dtype=torch.bool, device=self.device)
        self._contact_slots = torch.arange(
            self.contact_capacity, dtype=torch.long, device=self.device)

    @classmethod
    def from_physics(cls, physics: Any) -> "RekG1GpuCombatMeasurement":
        if not physics.qpos.is_cuda:
            raise ValueError("REK G1 GPU combat measurement requires CUDA physics")
        model_map = GpuCombatModelMap.from_host_model(
            physics.host_model, physics.qpos.device)
        return cls(model_map, GpuCombatTensorSource.from_physics(physics))

    def _validate_shapes(self) -> None:
        source = self.source
        model = self.model_map
        arena_count = int(source.qpos.shape[0]) if source.qpos.ndim == 2 else -1
        expected_shapes = (
            (source.qpos, (arena_count, None), "qpos"),
            (source.xpos, (arena_count, model.body_count, 3), "xpos"),
            (source.xipos, (arena_count, model.body_count, 3), "xipos"),
            (source.subtree_com, (arena_count, model.body_count, 3), "subtree_com"),
            (source.cvel, (arena_count, model.body_count, 6), "cvel"),
            (source.geom_xpos, (arena_count, model.geom_count, 3), "geom_xpos"),
            (source.geom_xmat, (arena_count, model.geom_count, 3, 3), "geom_xmat"),
        )
        if arena_count < 1:
            raise ValueError("qpos must contain at least one arena")
        for tensor, expected, name in expected_shapes:
            if tensor.ndim != len(expected):
                raise ValueError(f"{name} rank differs from MuJoCo Warp")
            for actual, wanted in zip(tensor.shape, expected):
                if wanted is not None and actual != wanted:
                    raise ValueError(f"{name} shape differs from MuJoCo Warp")
            if tensor.device != self.device:
                raise ValueError("all dynamic tensors must share one device")
            if not tensor.dtype.is_floating_point:
                raise TypeError(f"{name} must be floating point")
        if source.qpos.shape[1] < max(model.root_qpos_addresses) + 7:
            raise ValueError("qpos does not contain both root quaternions")
        capacity = int(source.contact_geom.shape[0])
        contact_shapes = (
            (source.contact_geom, (capacity, 2), "contact_geom"),
            (source.contact_worldid, (capacity,), "contact_worldid"),
            (source.contact_dist, (capacity,), "contact_dist"),
            (source.contact_pos, (capacity, 3), "contact_pos"),
            (source.contact_frame, (capacity, 3, 3), "contact_frame"),
        )
        if capacity < 1:
            raise ValueError("contact capacity must be positive")
        for tensor, expected, name in contact_shapes:
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} shape differs from MuJoCo Warp")
            if tensor.device != self.device:
                raise ValueError("all contact tensors must share one device")
        if source.contact_geom.dtype.is_floating_point \
                or source.contact_worldid.dtype.is_floating_point:
            raise TypeError("contact geom and world IDs must be integer tensors")
        if source.nacon.numel() != 1 or source.nacon.device != self.device \
                or source.nacon.dtype.is_floating_point:
            raise ValueError("nacon must be one device integer")
        if tuple(source.time.shape) != (arena_count,) \
                or source.time.device != self.device \
                or not source.time.dtype.is_floating_point:
            raise ValueError("time must be one floating value per arena")
        map_tensors = (
            model.root_body_ids,
            model.left_foot_body_ids,
            model.right_foot_body_ids,
            model.geom_body_ids,
            model.body_root_ids,
            model.body_owner,
            model.body_zone,
            model.geom_zone,
            model.striker_part,
            model.striker_side,
            model.striker_slot,
            model.geom_contype,
            model.geom_conaffinity,
        )
        if any(tensor.device != self.device for tensor in map_tensors):
            raise ValueError("model maps and dynamic tensors must share one device")
        if tuple(model.root_body_ids.shape) != (2,) \
                or tuple(model.left_foot_body_ids.shape) != (2,) \
                or tuple(model.right_foot_body_ids.shape) != (2,):
            raise ValueError("fighter identity maps must have two entries")
        for tensor in (
            model.body_root_ids, model.body_owner, model.body_zone,
            model.striker_part, model.striker_side, model.striker_slot,
        ):
            if tuple(tensor.shape) != (model.body_count,):
                raise ValueError("body map shape differs from body count")
        for tensor in (
            model.geom_body_ids, model.geom_zone,
            model.geom_contype, model.geom_conaffinity,
        ):
            if tuple(tensor.shape) != (model.geom_count,):
                raise ValueError("geom map shape differs from geom count")

    def _root_quaternions(self) -> torch.Tensor:
        return torch.stack(
            tuple(
                self.source.qpos[:, address + 3:address + 7]
                for address in self.model_map.root_qpos_addresses
            ),
            dim=1,
        ).reshape(self.robot_count, 4)

    def _floor_height(self) -> tuple[torch.Tensor, torch.Tensor]:
        floor = self.model_map.floor_geom_id
        position = self.source.geom_xpos[:, floor]
        matrix = self.source.geom_xmat[:, floor]
        finite = torch.isfinite(position).all(dim=1) \
            & torch.isfinite(matrix).all(dim=(1, 2))
        horizontal = matrix[:, 2, 0].abs() <= 1e-10
        horizontal &= matrix[:, 2, 1].abs() <= 1e-10
        horizontal &= (matrix[:, 2, 2].abs() - 1.0).abs() <= 1e-10
        height = position[:, 2] + matrix[:, 2, 2].abs() \
            * self.model_map.floor_half_height
        valid = finite & horizontal & torch.isfinite(height)
        return height, valid

    def calibrate_reset(self) -> torch.Tensor:
        """Calibrate every fighter row at a reset pose without synchronizing."""
        quaternion = self._root_quaternions()
        norm_squared = (quaternion * quaternion).sum(dim=1)
        quaternion_valid = torch.isfinite(quaternion).all(dim=1) \
            & torch.isfinite(norm_squared) & (norm_squared > 0.0)
        safe_norm = torch.where(
            quaternion_valid, norm_squared.sqrt(), torch.ones_like(norm_squared))
        unit = quaternion / safe_norm[:, None]
        inverse = torch.cat((unit[:, :1], -unit[:, 1:]), dim=1)
        world_up = torch.zeros(
            (self.robot_count, 3), dtype=unit.dtype, device=self.device)
        world_up[:, 2] = 1.0
        upright_local = _quat_rotate_wxyz(inverse, world_up)

        floor_height, floor_valid = self._floor_height()
        pelvis = self.source.xpos[:, self.model_map.root_body_ids].reshape(
            self.robot_count, 3)
        row_floor = floor_height.repeat_interleave(2)
        standing = pelvis[:, 2] - row_floor
        row_valid = quaternion_valid \
            & torch.isfinite(pelvis).all(dim=1) \
            & floor_valid.repeat_interleave(2) \
            & torch.isfinite(standing) \
            & (standing > torch.finfo(torch.float64).eps)
        all_rows_valid = row_valid.all()
        calibrated = row_valid & all_rows_valid
        self._upright_up_local.copy_(upright_local)
        self._standing_pelvis_height.copy_(standing)
        self._reset_floor_height.copy_(row_floor)
        self._calibrated.copy_(calibrated)
        return calibrated

    def reset(self) -> torch.Tensor:
        """Clear hit history/substep sequence and calibrate current reset pose."""
        self._previous_pairs.zero_()
        self._current_pair_counts.zero_()
        self._expected_substep.zero_()
        return self.calibrate_reset()

    def clear_arena_contacts(self, arena_mask: torch.Tensor) -> None:
        """Clear selected pair histories while preserving substep sequence."""
        if tuple(arena_mask.shape) != (self.arena_count,) \
                or arena_mask.device != self.device:
            raise ValueError("arena_mask must match arena count and device")
        selected = arena_mask != 0
        self._previous_pairs.logical_and_(~selected[:, None])

    def _contact_facts(self) -> _ContactFacts:
        source = self.source
        model = self.model_map
        nacon = source.nacon.reshape(-1)[0].to(dtype=torch.long)
        capacity_valid = (nacon >= 0) & (nacon <= self.contact_capacity)
        active = self._contact_slots < nacon
        world = source.contact_worldid.to(dtype=torch.long)
        geom = source.contact_geom.to(dtype=torch.long)
        geom0 = geom[:, 0]
        geom1 = geom[:, 1]
        world_valid = (world >= 0) & (world < self.arena_count)
        geom_valid = (geom0 >= 0) & (geom0 < model.geom_count) \
            & (geom1 >= 0) & (geom1 < model.geom_count)
        world_safe = world.clamp(0, self.arena_count - 1)
        geom0_safe = geom0.clamp(0, model.geom_count - 1)
        geom1_safe = geom1.clamp(0, model.geom_count - 1)

        invalid_world = (active & ~world_valid).any()
        invalid_geom = active & world_valid & ~geom_valid
        arena_invalid_counts = torch.zeros(
            self.arena_count, dtype=torch.int32, device=self.device)
        arena_invalid_counts.scatter_add_(
            0, world_safe, invalid_geom.to(dtype=torch.int32))
        fall_arena_valid = capacity_valid & ~invalid_world \
            & (arena_invalid_counts == 0)

        floor0 = geom0 == model.floor_geom_id
        floor1 = geom1 == model.floor_geom_id
        floor_pair = geom_valid & ((floor0 & ~floor1) | (floor1 & ~floor0))
        other_geom = torch.where(floor0, geom1_safe, geom0_safe)
        other_body = model.geom_body_ids[other_geom]
        body_safe = other_body.clamp(0, model.body_count - 1)
        floor_body_valid = active & world_valid & floor_pair \
            & (other_body > 0) & (other_body < model.body_count)
        flat_body_contacts = torch.zeros(
            self.arena_count * model.body_count,
            dtype=torch.int32,
            device=self.device,
        )
        body_key = world_safe * model.body_count + body_safe
        flat_body_contacts.scatter_add_(
            0, body_key, floor_body_valid.to(dtype=torch.int32))
        floor_body_contacts = flat_body_contacts.reshape(
            self.arena_count, model.body_count) > 0
        return _ContactFacts(
            active=active,
            capacity_valid=capacity_valid,
            world=world,
            world_safe=world_safe,
            world_valid=world_valid,
            geom0=geom0,
            geom1=geom1,
            geom0_safe=geom0_safe,
            geom1_safe=geom1_safe,
            geom_valid=geom_valid,
            fall_arena_valid=fall_arena_valid,
            floor_body_contacts=floor_body_contacts,
        )

    def _sample_fall(
        self,
        contacts: _ContactFacts,
        can_get_up: torch.Tensor,
        fixed_delta_seconds: float,
    ) -> GpuFallMeasurementBatch:
        if tuple(can_get_up.shape) != (self.robot_count,) \
                or can_get_up.device != self.device:
            raise ValueError("can_get_up must match robot rows and device")
        if not math.isfinite(fixed_delta_seconds) or fixed_delta_seconds <= 0.0:
            raise ValueError("fixed_delta_seconds must be finite and positive")
        can_get_up_integer = can_get_up.to(dtype=torch.long)
        can_get_up_valid = (can_get_up_integer == 0) | (can_get_up_integer == 1)

        quaternion = self._root_quaternions()
        norm_squared = (quaternion * quaternion).sum(dim=1)
        quaternion_valid = torch.isfinite(quaternion).all(dim=1) \
            & torch.isfinite(norm_squared) & (norm_squared > 0.0)
        safe_norm = torch.where(
            quaternion_valid, norm_squared.sqrt(), torch.ones_like(norm_squared))
        unit = quaternion / safe_norm[:, None]
        rotated_up = _quat_rotate_wxyz(unit, self._upright_up_local)
        up_norm_squared = (rotated_up * rotated_up).sum(dim=1)
        safe_up_norm = torch.where(
            up_norm_squared > 0.0,
            up_norm_squared.sqrt(),
            torch.ones_like(up_norm_squared),
        )
        up_dot = (rotated_up[:, 2] / safe_up_norm).clamp(-1.0, 1.0)
        tilt = torch.acos(up_dot) * (180.0 / math.pi)

        floor_height, floor_valid = self._floor_height()
        pelvis = self.source.xpos[:, self.model_map.root_body_ids].reshape(
            self.robot_count, 3)
        row_floor = floor_height.repeat_interleave(2)
        ratio = (pelvis[:, 2] - row_floor) / self._standing_pelvis_height

        row_indices = torch.arange(
            self.robot_count, dtype=torch.long, device=self.device)
        row_arena = torch.div(row_indices, 2, rounding_mode="floor")
        row_fighter = row_indices.remainder(2)
        left_body = self.model_map.left_foot_body_ids[row_fighter]
        right_body = self.model_map.right_foot_body_ids[row_fighter]
        left_contact = contacts.floor_body_contacts[row_arena, left_body]
        right_contact = contacts.floor_body_contacts[row_arena, right_body]
        has_foot = left_contact | right_contact

        owner_mask = self.model_map.body_owner[None, :] == row_fighter[:, None]
        foot_mask = torch.zeros_like(owner_mask)
        foot_mask.scatter_(1, left_body[:, None], True)
        foot_mask.scatter_(1, right_body[:, None], True)
        row_body_contacts = contacts.floor_body_contacts[row_arena]
        nonfoot_count = (row_body_contacts & owner_mask & ~foot_mask).sum(dim=1)

        valid = self._calibrated \
            & contacts.fall_arena_valid.repeat_interleave(2) \
            & quaternion_valid \
            & torch.isfinite(pelvis).all(dim=1) \
            & floor_valid.repeat_interleave(2) \
            & torch.isfinite(up_norm_squared) & (up_norm_squared > 0.0) \
            & torch.isfinite(self._standing_pelvis_height) \
            & (self._standing_pelvis_height > torch.finfo(torch.float64).eps) \
            & torch.isfinite(tilt) & torch.isfinite(ratio) \
            & can_get_up_valid
        delta = torch.full_like(tilt, fixed_delta_seconds)
        floats = torch.stack(
            (tilt, ratio, delta, row_floor, self._standing_pelvis_height),
            dim=1,
        )
        integers = torch.stack(
            (
                torch.ones_like(nonfoot_count),
                (~has_foot).to(dtype=torch.long),
                has_foot.to(dtype=torch.long),
                nonfoot_count.to(dtype=torch.long),
                can_get_up_integer,
                left_contact.to(dtype=torch.long),
                right_contact.to(dtype=torch.long),
            ),
            dim=1,
        )
        return GpuFallMeasurementBatch(
            floats=floats,
            integers=integers,
            valid=valid,
            floor_body_contacts=contacts.floor_body_contacts,
        )

    def _body_velocity(self, arena: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        """Match mj_objectVelocity(mjOBJ_BODY, flg_local=0) linear output."""
        spatial = self.source.cvel[arena, body]
        root = self.model_map.body_root_ids[body]
        displacement = self.source.xipos[arena, body] \
            - self.source.subtree_com[arena, root]
        return spatial[:, 3:6] + torch.cross(
            spatial[:, 0:3], displacement, dim=1)

    def _sample_hits(
        self,
        contacts: _ContactFacts,
        physics_substep_index: int,
    ) -> GpuHitCandidateBatch:
        if physics_substep_index < 0 or physics_substep_index >= PHYSICS_SUBSTEPS:
            raise ValueError("physics_substep_index must be in [0, 9]")
        source = self.source
        model = self.model_map
        contact_finite = torch.isfinite(source.contact_dist) \
            & torch.isfinite(source.contact_pos).all(dim=1) \
            & torch.isfinite(source.contact_frame).all(dim=(1, 2))
        pair_valid = contacts.active & contacts.world_valid & contacts.geom_valid \
            & (contacts.geom0 != contacts.geom1) & contact_finite
        invalid_hit_contact = contacts.active & contacts.world_valid & ~(
            contacts.geom_valid
            & (contacts.geom0 != contacts.geom1)
            & contact_finite
        )
        arena_invalid_counts = torch.zeros(
            self.arena_count, dtype=torch.int32, device=self.device)
        arena_invalid_counts.scatter_add_(
            0,
            contacts.world_safe,
            invalid_hit_contact.to(dtype=torch.int32),
        )
        invalid_world = (contacts.active & ~contacts.world_valid).any()
        sequence_valid = self._expected_substep == physics_substep_index
        base_arena_valid = contacts.capacity_valid & ~invalid_world \
            & (arena_invalid_counts == 0) & sequence_valid \
            & torch.isfinite(source.time)

        first_geom = torch.minimum(contacts.geom0_safe, contacts.geom1_safe)
        second_geom = torch.maximum(contacts.geom0_safe, contacts.geom1_safe)
        pair = first_geom * model.geom_count + second_geom
        pair_key = contacts.world_safe * self.pair_span + pair
        self._current_pair_counts.zero_()
        self._current_pair_counts.reshape(-1).scatter_add_(
            0, pair_key, pair_valid.to(dtype=torch.int32))
        current_pairs = self._current_pair_counts > 0
        self._first_pair_slot.fill_(self.contact_capacity)
        first_source = torch.where(
            pair_valid,
            self._contact_slots.to(dtype=torch.int32),
            torch.full_like(self._contact_slots, self.contact_capacity,
                            dtype=torch.int32),
        )
        self._first_pair_slot.reshape(-1).scatter_reduce_(
            0, pair_key, first_source, reduce="amin", include_self=True)
        is_first = pair_valid & (
            self._first_pair_slot.reshape(-1)[pair_key].to(dtype=torch.long)
            == self._contact_slots
        )
        was_present = self._previous_pairs.reshape(-1)[pair_key]
        enter = is_first & ~was_present \
            & base_arena_valid[contacts.world_safe]

        striker_geom = torch.stack(
            (contacts.geom0_safe, contacts.geom1_safe), dim=1).reshape(-1)
        target_geom = torch.stack(
            (contacts.geom1_safe, contacts.geom0_safe), dim=1).reshape(-1)
        arena = contacts.world_safe[:, None].expand(-1, 2).reshape(-1)
        direction_enter = enter[:, None].expand(-1, 2).reshape(-1)
        striker_body = model.geom_body_ids[striker_geom]
        target_body = model.geom_body_ids[target_geom]
        striker_body_safe = striker_body.clamp(0, model.body_count - 1)
        target_body_safe = target_body.clamp(0, model.body_count - 1)
        striker_owner = model.body_owner[striker_body_safe]
        target_owner = model.body_owner[target_body_safe]
        slot = model.striker_slot[striker_body_safe]
        part = model.striker_part[striker_body_safe]
        side = model.striker_side[striker_body_safe]
        target_zone = model.geom_zone[target_geom]
        cross_fighter = (striker_owner >= 0) & (target_owner >= 0) \
            & (striker_owner != target_owner)
        tagged_striker = slot >= 0
        mapping_valid = (slot < 6) \
            & ((part == BODY_PART_HAND) | (part == BODY_PART_FOOT)
               | (part == BODY_PART_SHIN)) \
            & ((side == 0) | (side == 1)) \
            & (target_zone >= BODY_ZONE_UNKNOWN) \
            & (target_zone <= BODY_ZONE_RIGHT_ANKLE)
        eligible = cross_fighter & tagged_striker & mapping_valid
        mapping_corrupt = direction_enter & cross_fighter & tagged_striker \
            & ~mapping_valid

        striker_position = source.xpos[arena, striker_body_safe]
        target_position = source.xpos[arena, target_body_safe]
        striker_velocity = self._body_velocity(arena, striker_body_safe)
        target_velocity = self._body_velocity(arena, target_body_safe)
        relative_speed = torch.linalg.vector_norm(
            striker_velocity - target_velocity, dim=1)
        candidate_finite = torch.isfinite(striker_position).all(dim=1) \
            & torch.isfinite(target_position).all(dim=1) \
            & torch.isfinite(striker_velocity).all(dim=1) \
            & torch.isfinite(target_velocity).all(dim=1) \
            & torch.isfinite(relative_speed)
        preliminary = direction_enter & eligible
        candidate_invalid = mapping_corrupt | (preliminary & ~candidate_finite)
        arena_candidate_invalid = torch.zeros(
            self.arena_count, dtype=torch.int32, device=self.device)
        arena_candidate_invalid.scatter_add_(
            0, arena, candidate_invalid.to(dtype=torch.int32))
        arena_scan_valid = base_arena_valid & (arena_candidate_invalid == 0)
        candidate_valid = preliminary & candidate_finite \
            & arena_scan_valid[arena]

        committed = torch.where(
            arena_scan_valid[:, None], current_pairs, self._previous_pairs)
        self._previous_pairs.copy_(committed)
        next_substep = (physics_substep_index + 1) % PHYSICS_SUBSTEPS
        self._expected_substep.copy_(torch.where(
            arena_scan_valid,
            torch.full_like(self._expected_substep, next_substep),
            self._expected_substep,
        ))

        candidate_counts = torch.zeros(
            self.arena_count, dtype=torch.long, device=self.device)
        candidate_counts.scatter_add_(
            0, arena, candidate_valid.to(dtype=torch.long))
        directed_source_index = torch.arange(
            striker_geom.numel(), dtype=torch.long, device=self.device)
        candidate_processing_key = arena * (2 * self.contact_capacity) \
            + directed_source_index
        invalid_processing_key = torch.full_like(
            candidate_processing_key, torch.iinfo(torch.long).max)
        candidate_order = torch.argsort(torch.where(
            candidate_valid,
            candidate_processing_key,
            invalid_processing_key,
        ))
        candidate_offsets = torch.cumsum(candidate_counts, dim=0) \
            - candidate_counts
        substep = torch.full_like(arena, physics_substep_index)
        integers = torch.stack(
            (
                arena,
                substep,
                striker_geom,
                target_geom,
                striker_body,
                target_body,
                striker_owner,
                target_owner,
                slot,
                part,
                side,
                target_zone,
            ),
            dim=1,
        )
        floats = torch.cat(
            (
                striker_position,
                target_position,
                striker_velocity,
                target_velocity,
                relative_speed[:, None],
            ),
            dim=1,
        )
        return GpuHitCandidateBatch(
            integers=integers,
            floats=floats,
            candidate_valid=candidate_valid,
            candidate_processing_key=candidate_processing_key,
            candidate_order=candidate_order,
            candidate_offsets=candidate_offsets,
            candidate_counts=candidate_counts,
            arena_scan_valid=arena_scan_valid,
            contact_capacity_overflow=~contacts.capacity_valid,
            arena_time_seconds=source.time,
        )

    def sample(
        self,
        physics_substep_index: int,
        can_get_up: torch.Tensor,
        *,
        fixed_delta_seconds: float = PHYSICS_DELTA_SECONDS,
    ) -> GpuCombatMeasurementBatch:
        """Measure fall rows and directed contact-enter facts for one 2 ms step."""
        contacts = self._contact_facts()
        fall = self._sample_fall(contacts, can_get_up, fixed_delta_seconds)
        hits = self._sample_hits(contacts, physics_substep_index)
        return GpuCombatMeasurementBatch(fall=fall, hits=hits)

    def sample_fall(
        self,
        can_get_up: torch.Tensor,
        *,
        fixed_delta_seconds: float = PHYSICS_DELTA_SECONDS,
    ) -> GpuFallMeasurementBatch:
        """Measure fall facts without advancing contact-enter sequence state."""
        return self._sample_fall(
            self._contact_facts(), can_get_up, fixed_delta_seconds)
