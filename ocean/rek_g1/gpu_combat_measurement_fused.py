"""Opt-in live-contact fusion; original fall math and native scoring retained.

No runtime compilation or CPU fallback. A separately built native library must
be provided explicitly. Host mode exists only for CPU differential fixtures.
Invalid candidate payload and the unused tail of order are unspecified.
"""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
from pathlib import Path

import torch

from gpu_combat_measurement import (
    GpuCombatModelMap, GpuCombatTensorSource, GpuHitCandidateBatch, RekG1GpuCombatMeasurement,
)


_POINTER_FIELDS = (
    "geom world nacon dist pos frame time xpos xipos com cvel "
    "geom_body body_root owner zone part side slot previous expected first "
    "floor_count fall_bad hit_bad candidate_bad world_bad capacity_overflow "
    "fall_valid floor_contact base_valid scan_valid integers floats "
    "candidate_valid keys counts velocity speed"
).split()


class _Descriptor(ct.Structure):
    _fields_ = [(name, ct.c_int32) for name in (
        "arenas", "bodies", "geoms", "capacity", "floor")]
    _fields_ += [(name, ct.c_void_p) for name in _POINTER_FIELDS]


@dataclass(frozen=True)
class _Facts:
    fall_arena_valid: torch.Tensor
    floor_body_contacts: torch.Tensor


class FusedGpuCombatMeasurement(RekG1GpuCombatMeasurement):
    @classmethod
    def from_physics(cls, physics, *, library):
        return cls(GpuCombatModelMap.from_host_model(physics.host_model, physics.qpos.device),
                   GpuCombatTensorSource.from_physics(physics), library)

    def __init__(self, model_map, source, library: str | Path, *, host_test=False):
        super().__init__(model_map, source)
        if self.device.type != ("cpu" if host_test else "cuda"):
            raise ValueError("CUDA is required except explicit host-test mode")
        self.host_test = host_test
        path = Path(library).resolve(strict=True)
        self.library = ct.CDLL(str(path))
        self.library.rek_measurement_descriptor_size.restype = ct.c_size_t
        if self.library.rek_measurement_descriptor_size() != ct.sizeof(_Descriptor):
            raise RuntimeError("fused measurement descriptor ABI mismatch")
        self.library.rek_measurement_facts.argtypes = [ct.POINTER(_Descriptor), ct.c_void_p]
        self.library.rek_measurement_hits_prepare.argtypes = [ct.POINTER(_Descriptor), ct.c_int, ct.c_void_p]
        self.library.rek_measurement_hits_finish.argtypes = [ct.POINTER(_Descriptor), ct.c_int, ct.c_void_p]
        self.library.rek_measurement_facts.restype = ct.c_int
        self.library.rek_measurement_hits_prepare.restype = ct.c_int
        self.library.rek_measurement_hits_finish.restype = ct.c_int
        for name in ("contact_geom", "contact_worldid", "nacon"):
            value = getattr(source, name)
            if value.dtype != torch.int32 or not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous int32")
        for name in ("contact_dist", "contact_pos", "contact_frame", "time", "xpos", "xipos", "subtree_com", "cvel"):
            value = getattr(source, name)
            if value.dtype != torch.float32 or not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous float32")
        maps = ("geom_body_ids", "body_root_ids", "body_owner", "geom_zone", "striker_part", "striker_side", "striker_slot")
        for name in maps:
            value = getattr(model_map, name)
            if value.dtype != torch.int64 or not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous int64")
        a, c, b = self.arena_count, self.contact_capacity, model_map.body_count
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=self.device)
        self._native = {
            "geom": source.contact_geom, "world": source.contact_worldid, "nacon": source.nacon,
            "dist": source.contact_dist, "pos": source.contact_pos, "frame": source.contact_frame,
            "time": source.time, "xpos": source.xpos, "xipos": source.xipos,
            "com": source.subtree_com, "cvel": source.cvel,
            "geom_body": model_map.geom_body_ids, "body_root": model_map.body_root_ids,
            "owner": model_map.body_owner, "zone": model_map.geom_zone,
            "part": model_map.striker_part, "side": model_map.striker_side, "slot": model_map.striker_slot,
            "previous": self._previous_pairs, "expected": self._expected_substep,
            "first": self._first_pair_slot,
            "floor_count": zeros((a,b), torch.int32),
            "fall_bad": zeros(a, torch.int32), "hit_bad": zeros(a, torch.int32),
            "candidate_bad": zeros(a, torch.int32), "world_bad": zeros(1, torch.int32),
            "capacity_overflow": zeros((), torch.bool), "fall_valid": zeros(a, torch.bool),
            "floor_contact": zeros((a,b), torch.bool), "base_valid": zeros(a, torch.bool),
            "scan_valid": zeros(a, torch.bool), "integers": zeros((2*c,12), torch.int64),
            "floats": zeros((2*c,13), torch.float32), "candidate_valid": zeros(2*c, torch.bool),
            "keys": zeros(2*c, torch.int64), "counts": zeros(a, torch.int64),
            "velocity": zeros((a,b,3), torch.float32), "speed": zeros(2*c, torch.float32),
        }
        self._descriptor = _Descriptor(a, b, model_map.geom_count, c, model_map.floor_geom_id,
            *(self._native[name].data_ptr() for name in _POINTER_FIELDS))
        self._order = zeros(2*c, torch.int64)
        self._offsets = zeros(a, torch.int64)
        self._relative_velocity = zeros((2*c,3), torch.float32)
        self._facts = _Facts(self._native["fall_valid"], self._native["floor_contact"])
        self._hits = GpuHitCandidateBatch(
            integers=self._native["integers"], floats=self._native["floats"],
            candidate_valid=self._native["candidate_valid"],
            candidate_processing_key=self._native["keys"], candidate_order=self._order,
            candidate_offsets=self._offsets, candidate_counts=self._native["counts"],
            arena_scan_valid=self._native["scan_valid"],
            contact_capacity_overflow=self._native["capacity_overflow"], arena_time_seconds=source.time,
        )

    def _stream(self):
        return None if self.host_test else torch.cuda.current_stream(self.device).cuda_stream

    def _contact_facts(self):
        code = self.library.rek_measurement_facts(ct.byref(self._descriptor), self._stream())
        if code:
            raise RuntimeError(f"fused measurement facts launch failed: {code}")
        return self._facts

    def _sample_hits(self, contacts, physics_substep_index):
        if not 0 <= physics_substep_index < 10:
            raise ValueError("physics_substep_index must be in [0, 9]")
        # Keep the oracle's floating kernels and operation order. Compute
        # velocity once per body instead of gathering both bodies of every
        # padded directed contact before computing their cross products.
        spatial = self.source.cvel
        displacement = self.source.xipos - self.source.subtree_com[:, self.model_map.body_root_ids]
        torch.add(spatial[:, :, 3:6], torch.cross(spatial[:, :, 0:3], displacement, dim=2),
                  out=self._native["velocity"])
        code = self.library.rek_measurement_hits_prepare(ct.byref(self._descriptor), physics_substep_index, self._stream())
        if code:
            raise RuntimeError(f"fused measurement hits launch failed: {code}")
        torch.sub(self._native["floats"][:, 6:9], self._native["floats"][:, 9:12],
                  out=self._relative_velocity)
        torch.linalg.vector_norm(self._relative_velocity, dim=1, out=self._native["speed"])
        code = self.library.rek_measurement_hits_finish(ct.byref(self._descriptor), physics_substep_index, self._stream())
        if code:
            raise RuntimeError(f"fused measurement hits finish failed: {code}")
        torch.argsort(self._native["keys"], stable=False, out=self._order)
        torch.cumsum(self._native["counts"], dim=0, out=self._offsets)
        self._offsets.sub_(self._native["counts"])
        return self._hits
