"""Experimental full semantic tensor bridge for the pinned Puffysics solver.

All measured state and contacts come from the native B3World. MuJoCo is used
once for model compilation and kinematic frame mapping, never for stepping.
The existing observation, reset, combat and controller modules consume this
facade unchanged. Engine trajectory equivalence is not claimed.
"""
from __future__ import annotations

from contextlib import nullcontext
import ctypes as ct
from types import SimpleNamespace

import numpy as np
import torch

from export_rek_compiled_model import matrix_quat, quat_matrix
from puffysics_backend import PuffysicsBackend


_POINTERS = (
    "qpos qvel base angular time xpos xquat xmat xipos ximat com cvel "
    "geom_xpos geom_xmat contact_geom contact_world nacon contact_dist "
    "contact_pos contact_frame counts offsets body_map geom_map pre_centers"
).split()


class _Descriptor(ct.Structure):
    _fields_ = [(name, ct.c_int32) for name in ("arenas", "bodies", "geoms", "capacity")]
    _fields_ += [(name, ct.c_void_p) for name in _POINTERS]


class _TensorViews:
    """Only the two Warp interface operations used by semantic consumers."""
    @staticmethod
    def to_torch(value):
        if not isinstance(value, torch.Tensor):
            raise TypeError("Puffysics semantic views must already be Torch tensors")
        return value

    @staticmethod
    def ScopedCapture(**kwargs):
        return nullcontext()


def build_frame_maps(model, export):
    """Derive source body/geometry frames once with CPU kinematics only."""
    import mujoco

    if model.nbody > 64:
        raise ValueError("native semantic export supports at most 64 source bodies")
    source = mujoco.MjData(model)
    source.qpos[:] = export["initial_qpos"]
    mujoco.mj_kinematics(model, source)
    rigid_by_source = {int(b["source_body_id"]): int(b["id"]) + 1 for b in export["bodies"]}
    owner = np.zeros(model.nbody, dtype=np.int32)
    poses = [(np.zeros(3), np.eye(3))]
    poses += [(np.asarray(b["world_pose"]["position_xyz"]),
               quat_matrix(b["world_pose"]["quaternion_xyzw"])) for b in export["bodies"]]
    body_rows = []
    for b in range(model.nbody):
        if b and not 0 <= int(model.body_parentid[b]) < b:
            raise ValueError("compiled source body parents must precede children")
        owner[b] = rigid_by_source.get(b, owner[int(model.body_parentid[b])])
        position, rotation = poses[owner[b]]
        local_pos = rotation.T @ (source.xpos[b] - position)
        local_rot = rotation.T @ source.xmat[b].reshape(3, 3)
        inertial_pos = rotation.T @ (source.xipos[b] - position)
        inertial_rot = rotation.T @ source.ximat[b].reshape(3, 3)
        body_rows.append([owner[b], model.body_parentid[b], model.body_rootid[b], model.body_mass[b],
                          *local_pos, *matrix_quat(local_rot), *inertial_pos, *matrix_quat(inertial_rot)])
    visited = {0, *(int(root["rigid_body_id"]) + 1 for root in export["free_bases"])}
    for joint in export["hinges"]:
        parent, child = int(joint["parent_rigid_body_id"]) + 1, int(joint["child_rigid_body_id"]) + 1
        if parent not in visited or child in visited:
            raise ValueError("native reset requires parent-before-child hinge order")
        visited.add(child)
    if visited != set(range(61)):
        raise ValueError("hinges and free roots must span all rigid bodies")
    geom_rows = []
    for geom in export["shapes"]:
        pose = geom["source_local_pose"]
        if geom["source_geom_id"] != len(geom_rows):
            raise ValueError("shape order must match source geometry IDs")
        geom_rows.append([geom["rigid_body_id"] + 1, *pose["position_xyz"], *pose["quaternion_xyzw"]])
    return np.asarray(body_rows, np.float32), np.asarray(geom_rows, np.float32)


class PuffysicsSemanticPhysics(PuffysicsBackend):
    def __init__(self, config, *, arenas, device="cuda:0", solver_mode=1, library, export_path,
                 contacts_per_arena=128):
        if solver_mode not in (0, 1):
            raise ValueError("solver_mode must be 0 (standard b3_step) or 1 (experimental articulated solver)")
        super().__init__(config, arenas=arenas, device=device, solver_mode=solver_mode,
                         library=library, export_path=export_path)
        self.wp = _TensorViews()
        self.stream = torch.cuda.current_stream(self.device)
        self.model = SimpleNamespace(is_sparse=False)
        model = self.host_model
        self.root_bodies = [model.body(role + "__pelvis_3266").id for role in ("player", "opponent")]
        body_map, geom_map = build_frame_maps(model, self.export)
        a, b, g, capacity = self.arenas, model.nbody, model.ngeom, self.arenas * contacts_per_arena

        def zeros(shape, dtype=torch.float32):
            return torch.zeros(shape, device=self.device, dtype=dtype)

        native = {
            "qpos": self.qpos, "qvel": self.qvel, "base": self.base,
            "angular": self.angular_local, "time": self.time,
            "xpos": zeros((a, b, 3)), "xquat": zeros((a, b, 4)), "xmat": zeros((a, b, 9)),
            "xipos": zeros((a, b, 3)), "ximat": zeros((a, b, 9)), "com": zeros((a, b, 3)),
            "cvel": zeros((a, b, 6)), "geom_xpos": zeros((a, g, 3)), "geom_xmat": zeros((a, g, 3, 3)),
            "contact_geom": zeros((capacity, 2), torch.int32), "contact_world": zeros(capacity, torch.int32),
            "nacon": zeros(1, torch.int32), "contact_dist": zeros(capacity),
            "contact_pos": zeros((capacity, 3)), "contact_frame": zeros((capacity, 3, 3)),
            "counts": zeros(a, torch.int32), "offsets": zeros(a, torch.int32),
            "body_map": torch.as_tensor(body_map, device=self.device),
            "geom_map": torch.as_tensor(geom_map, device=self.device),
            "pre_centers": zeros((a, 61, 3)),
        }
        self._fields = native
        self._descriptor = _Descriptor(a, b, g, capacity, *(native[name].data_ptr() for name in _POINTERS))
        lib = self._native
        lib.rps_descriptor_size.restype = ct.c_size_t
        if lib.rps_descriptor_size() != ct.sizeof(_Descriptor):
            raise RuntimeError("native semantic descriptor size mismatch")
        lib.rps_step_controls.argtypes = [ct.c_void_p, ct.POINTER(_Descriptor), ct.c_void_p, ct.c_void_p]
        lib.rps_forward_selected.argtypes = [ct.c_void_p, ct.POINTER(_Descriptor), ct.c_void_p, ct.c_void_p]
        lib.rps_refresh.argtypes = [ct.c_void_p, ct.POINTER(_Descriptor), ct.c_void_p]
        lib.rps_step_controls.restype = lib.rps_forward_selected.restype = lib.rps_refresh.restype = ct.c_int
        self.data = SimpleNamespace(
            **{name: native[name] for name in ("xpos", "xquat", "xmat", "xipos", "ximat", "cvel", "geom_xpos", "geom_xmat")},
            qpos=self.qpos, qvel=self.qvel, ctrl=self.ctrl, time=self.time,
            subtree_com=native["com"], nacon=native["nacon"],
            contact=SimpleNamespace(geom=native["contact_geom"], worldid=native["contact_world"],
                                    dist=native["contact_dist"], pos=native["contact_pos"], frame=native["contact_frame"]),
        )
        # GpuDuelReset clears these MuJoCo-only work arrays. This solver has no
        # corresponding arrays: its actual warm starts clear in masked forward.
        for name in ("qacc", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "act"):
            setattr(self.data, name, zeros((a, 0)))
        self._all = torch.ones(a, device=self.device, dtype=torch.bool)
        self.refresh_export()
        art_contacts = None
        if hasattr(lib, "rp_art_contacts_enabled"):
            lib.rp_art_contacts_enabled.argtypes = []
            lib.rp_art_contacts_enabled.restype = ct.c_int
            art_contacts = bool(lib.rp_art_contacts_enabled())
        self.metadata.update({
            "diagnostic_geometry_substitution": self.export.get("diagnostic_geometry_substitution"),
            "semantic_tensor_adapter": "puffysics_semantic_native.v1",
            "native_step_function": "b3_step" if solver_mode == 0 else "rp_art_step",
            "native_solver_scope": (
                "standard Puffysics maximal-coordinate solver" if solver_mode == 0
                else "experimental articulated-body prototype solver"
            ),
            "rotor_armature_in_free_integration": solver_mode == 1,
            "engine_parity_required_for_this_diagnostic": False,
            "articulated_contact_response_enabled": art_contacts,
            "contact_solver": (
                "not_reported_by_library" if art_contacts is None
                else "articulated_response" if art_contacts
                else "independent_maximal_coordinates"
            ),
            "semantic_contact_capacity": capacity,
            "semantic_contact_export": "all native manifold points, packed in arena/shape/point order",
            "semantic_contact_position_time": "collision detection before the 2 ms integration",
            "semantic_body_velocity_time": "after the 2 ms integration",
            "masked_reset": "generalized state reconstruction; selected solver warm starts cleared",
            "semantic_native_launches_per_step": 5,
            "cpu_kinematic_mapping_calls_at_setup": 1,
        })

    def _stream_pointer(self):
        return torch.cuda.current_stream(self.device).cuda_stream

    def refresh_export(self):
        code = self._native.rps_refresh(self._handle, ct.byref(self._descriptor), self._stream_pointer())
        if code:
            raise RuntimeError("semantic export failed: " + self._last_error())

    def reset(self):
        super().reset()
        if hasattr(self, "_descriptor"):
            self.refresh_export()

    def step(self):
        code = self._native.rps_step_controls(self._handle, ct.byref(self._descriptor),
                                               self.ctrl.data_ptr(), self._stream_pointer())
        if code:
            raise RuntimeError("semantic native step failed: " + self._last_error())

    def forward_selected(self, mask):
        if mask.shape != (self.arenas,) or mask.device != self.qpos.device or mask.dtype != torch.bool:
            raise ValueError("forward mask must be a CUDA Boolean per arena")
        if not mask.is_contiguous():
            raise ValueError("forward mask must be contiguous")
        code = self._native.rps_forward_selected(self._handle, ct.byref(self._descriptor),
                                                  mask.data_ptr(), self._stream_pointer())
        if code:
            raise RuntimeError("semantic masked forward failed: " + self._last_error())

    def forward(self):
        self.forward_selected(self._all)

    def check_status(self):
        status = self.stats()
        if any(status[name] for name in ("nonfinite_arenas", "articulated_solver_failure_arenas",
                                         "contact_capacity_reached_arenas")):
            raise RuntimeError("Puffysics solver invalid: " + repr(status))
        count = int(self.data.nacon.item())
        if not 0 <= count <= self._descriptor.capacity:
            raise RuntimeError(f"Puffysics semantic contact capacity exceeded: {count}")
        for name in ("qpos", "qvel", "xpos", "xipos", "ximat", "com", "cvel", "geom_xpos", "geom_xmat"):
            if not bool(torch.isfinite(self._fields[name]).all().item()):
                raise RuntimeError(f"nonfinite Puffysics semantic field: {name}")
        return status
