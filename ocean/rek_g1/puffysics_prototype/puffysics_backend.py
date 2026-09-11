"""CUDA-resident bridge for the isolated compiled-model Puffysics experiment.

Host model compilation and flat-array packing happen once. Native stepping
uses caller-owned Torch buffers and the current CUDA stream, without a state
download. This adapter makes no claim of authentic REK or MuJoCo parity.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import gear_sonic_candidate as candidate
from gpu_motion_assets import _normalize_clip_heading_wxyz


SCHEMA = "rek.puffysics.compiled_model_adapter.v1"
SHAPE_ENUM = {"sphere": 0, "capsule": 1, "box": 2, "cylinder": 3}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _pose(pose):
    position = np.asarray(pose["position_xyz"], dtype=np.float64)
    quaternion = np.asarray(pose["quaternion_xyzw"], dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("pose must contain XYZ position and XYZW quaternion")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("pose must be finite")
    if abs(float(quaternion @ quaternion) - 1) > 1e-7:
        raise ValueError("exported quaternion must be unit length")
    return [*position, *quaternion]


def _multiply_xyzw(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.r_[a[3]*b[:3] + b[3]*a[:3] + np.cross(a[:3], b[:3]),
                 a[3]*b[3] - np.dot(a[:3], b[:3])]


def pack_export(adapter):
    """Pack the exact flat C ABI on the CPU, without CUDA initialization."""
    _require_equal(adapter["schema"], SCHEMA, "adapter schema")
    _require_equal(adapter["coordinate_system"], "right_handed_z_up", "world frame")
    _require_equal(adapter["quaternion_order"], "xyzw", "pose quaternion order")
    expected = {"rigid_bodies": 60, "shapes": 91, "hinges": 58,
                "free_bases": 2, "actuators": 58, "nq": 72, "nv": 70}
    for name, value in expected.items():
        _require_equal(adapter["counts"][name], value, f"count {name}")
    for name in ("bodies", "shapes", "hinges", "actuators"):
        values = adapter[name]
        _require_equal([item["id"] for item in values], list(range(len(values))), f"{name} order")
    _require_equal(adapter["runtime_timestep_seconds"], 0.002, "physics timestep")
    np.testing.assert_array_equal(np.asarray(adapter["gravity_xyz_m_s2"], dtype=np.float32),
                                  np.asarray([0, 0, -9.81], dtype=np.float32))
    np.testing.assert_array_equal(adapter["initial_qvel"], np.zeros(70))
    for key in ("source_exclude_signature", "source_pair_geom1", "source_pair_geom2"):
        if adapter[key]:
            raise ValueError(f"native prototype does not implement nonempty {key}")
    body_array = np.asarray([
        [*_pose(body["world_pose"]), body["mass_kg"], *body["principal_inertia_kg_m2"]]
        for body in adapter["bodies"]
    ], dtype=np.float32)
    if np.any(body_array[:, 7:] <= 0):
        raise ValueError("all body masses and principal inertias must be positive")
    shape_rows = []
    for shape in adapter["shapes"]:
        if shape["kind"] not in SHAPE_ENUM:
            raise ValueError(f"unsupported shape {shape['kind']!r}")
        _require_equal(shape["condim"], 3, "prototype contact dimension")
        if shape["kind"] in ("capsule", "cylinder"):
            _require_equal(shape["target_long_axis"], "y", "native shape long axis")
        rigid = shape["rigid_body_id"]
        if not -1 <= rigid < 60:
            raise ValueError("shape rigid body ID is outside the packed world")
        for key in ("contype", "conaffinity"):
            value = shape[key]
            if not isinstance(value, int) or value < 0 or int(np.float32(value)) != value:
                raise ValueError("collision bits must round-trip through the float32 C ABI")
        dimensions = shape["dimensions"]
        if shape["kind"] == "box":
            size = dimensions["half_extents_xyz"]
        else:
            size = [dimensions["radius_m"], dimensions.get("half_length_m", 0), 0]
        # MuJoCo condim=3 has sliding friction only. Its solref/solimp are not
        # restitution parameters; the explicit prototype restitution is zero.
        shape_rows.append([
            rigid, SHAPE_ENUM[shape["kind"]], *_pose(shape["target_local_pose"]),
            *size, shape["friction"][0], shape["contype"], shape["conaffinity"], 0, 0,
        ])
    shape_array = np.asarray(shape_rows, dtype=np.float32)
    joint_rows = []
    for joint, actuator in zip(adapter["hinges"], adapter["actuators"], strict=True):
        index = joint["id"]
        _require_equal(actuator["hinge_id"], index, "actuator/hinge order")
        _require_equal(actuator["id"], index, "actuator control index")
        _require_equal(actuator["source_joint_id"], joint["source_joint_id"], "actuator joint")
        _require_equal((actuator["dyntype"], actuator["gaintype"], actuator["biastype"]),
                       (0, 0, 1), "instantaneous fixed-gain affine actuator")
        np.testing.assert_array_equal(actuator["gear"], [1, 0, 0, 0, 0, 0])
        if not actuator["forcelimited"] or actuator["ctrllimited"]:
            raise ValueError("prototype requires force-limited, control-unbounded position actuators")
        gain = np.asarray(actuator["gainprm"], dtype=np.float64)
        bias = np.asarray(actuator["biasprm"], dtype=np.float64)
        if gain[0] <= 0 or bias[2] > 0 or bias[0] != 0 or bias[1] != -gain[0]:
            raise ValueError("actuator is not the supported affine position-PD form")
        if np.any(gain[1:] != 0) or np.any(bias[3:] != 0):
            raise ValueError("unsupported extra gain/bias coefficients")
        lower_force, upper_force = actuator["forcerange"]
        if upper_force <= 0 or lower_force != -upper_force:
            raise ValueError("force limits must be positive and symmetric")
        if not joint["limited"] or joint["stiffness_Nm_per_rad"] != 0:
            raise ValueError("prototype requires limited hinges without passive springs")
        if not -1 <= joint["parent_rigid_body_id"] < 60 or not 0 <= joint["child_rigid_body_id"] < 60:
            raise ValueError("hinge body ID outside packed world")
        lower, upper = joint["limits_relative_to_initial_rad"]
        if lower > upper:
            raise ValueError("hinge interval is reversed")
        joint_rows.append([
            joint["parent_rigid_body_id"], joint["child_rigid_body_id"],
            joint["qposadr"], joint["dofadr"], *joint["anchor_parent_xyz"],
            *joint["anchor_child_xyz"], *joint["axis_parent_xyz"],
            joint["mujoco_initial_angle_rad"], lower, upper,
            gain[0], -bias[2], upper_force, joint["armature_kg_m2"],
            joint["damping_Nm_s_per_rad"], joint["frictionloss_Nm"],
        ])
    joint_array = np.asarray(joint_rows, dtype=np.float32)
    roots = sorted(adapter["free_bases"], key=lambda value: value["qposadr"])
    _require_equal([root["qposadr"] for root in roots], [0, 36], "free-root qpos addresses")
    _require_equal([root["dofadr"] for root in roots], [0, 35], "free-root qvel addresses")
    root_rows = []
    for root in roots:
        local = _pose(root["original_link_in_rigid"])
        body_quaternion = adapter["bodies"][root["rigid_body_id"]]["world_pose"]["quaternion_xyzw"]
        composed = _multiply_xyzw(body_quaternion, local[3:7])
        expected_wxyz = np.asarray(root["initial_qpos_xyz_wxyz"][3:7])
        expected_xyzw = expected_wxyz[[1, 2, 3, 0]]
        # Pose matrices do not preserve quaternion sign. Keep the source free
        # joint's signed QPOS representation at the controller/state boundary.
        if np.dot(composed, expected_xyzw) < 0:
            local[3:7] = (-np.asarray(local[3:7])).tolist()
            composed = -composed
        np.testing.assert_allclose(composed, expected_xyzw, rtol=0, atol=1e-7)
        root_rows.append([root["rigid_body_id"], root["qposadr"], root["dofadr"], *local, 0, 0])
    root_array = np.asarray(root_rows, dtype=np.float32)
    result = {"bodies": body_array, "shapes": shape_array, "joints": joint_array, "roots": root_array}
    for (name, value), shape in zip(result.items(), ((60, 11), (91, 17), (58, 22), (2, 12)), strict=True):
        if value.shape != shape or not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValueError(f"invalid finite contiguous float32 {name} layout {value.shape}")
    return result


def initial_headings(config, model, arenas):
    """Match GpuMotionAssets heading normalization without duplicate GPU storage."""
    root = Path(config["assets"])
    manifest_path = root / "semantic_duel_assets_manifest.json"
    _require_equal(_sha256(manifest_path), config["assets_sha256"], "asset manifest digest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = [clip for clip in manifest["clips"] if clip["role"] == "idle"]
    if len(clips) != 1:
        raise ValueError("exactly one pinned idle clip is required")
    name = clips[0]["files"]["wxyz"]
    if Path(name).name != name:
        raise ValueError("idle root array must be a relative bundle filename")
    entry = manifest["files"][name]
    raw = (root / name).read_bytes()
    _require_equal(hashlib.sha256(raw).hexdigest(), entry["sha256"], "idle root digest")
    _require_equal(len(raw), entry["bytes"], "idle root byte count")
    _require_equal(entry["dtype"], "float32_le", "idle root storage type")
    roots = np.frombuffer(raw, dtype="<f4").reshape(entry["shape"]).copy()
    _normalize_clip_heading_wxyz(roots)
    headings = [candidate.reference_heading_delta(model.qpos0[side*36+3:side*36+7], roots[0])
                for side in range(2)]
    return np.tile(headings, (arenas, 1))


class PuffysicsBackend:
    def __init__(self, config, *, arenas, device="cuda:0", solver_mode=1, library, export_path):
        import mujoco

        self._handle = None
        self.arenas = int(arenas)
        self.device = torch.device(device)
        if self.arenas < 1 or self.device.type != "cuda" or solver_mode not in (0, 1):
            raise ValueError("positive arena count, CUDA device, and solver mode 0 or 1 required")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.library_path = Path(library).resolve(strict=True)
        self.export_path = Path(export_path).resolve(strict=True)
        self.export = json.loads(self.export_path.read_text(encoding="utf-8"))
        _require_equal(_sha256(config["model"]), config["model_sha256"], "compiled model source digest")
        _require_equal(self.export["source"]["model_sha256"], config["model_sha256"], "export source digest")
        _require_equal(self.export["initial_pose"]["kind"], "semantic_idle_frame_zero_clipped", "initial pose")
        _require_equal(self.export["initial_pose"]["manifest_sha256"], config["assets_sha256"], "export idle manifest")
        self.host_arrays = pack_export(self.export)
        self.host_model = mujoco.MjModel.from_xml_string(Path(config["model"]).read_text())
        model = self.host_model
        _require_equal((model.nq, model.nv, model.nu), (72, 70, 58), "compiled model dimensions")
        self.actuator_ids, self.joint_qpos, self.joint_qvel = [], [], []
        for side in range(2):
            ids = np.arange(side*29, (side+1)*29)
            model.actuator_gainprm[ids] = 0
            model.actuator_biasprm[ids] = 0
            candidate.configure_native_position_actuators(
                mujoco, model, SimpleNamespace(actuator_ids=ids), candidate.PUBLIC_EFFORT_LIMIT_MUJOCO,
            )
            joints = model.actuator_trnid[ids, 0]
            self.actuator_ids.append(ids)
            self.joint_qpos.append(model.jnt_qposadr[joints].copy())
            self.joint_qvel.append(model.jnt_dofadr[joints].copy())
        model.opt.timestep = 0.002
        np.testing.assert_array_equal(self.export["gravity_xyz_m_s2"], model.opt.gravity)
        np.testing.assert_array_equal(self.export["model_qpos0"], model.qpos0)
        np.testing.assert_array_equal(self.host_arrays["joints"][:, 2], np.asarray(self.joint_qpos).ravel())
        np.testing.assert_array_equal(self.host_arrays["joints"][:, 3], np.asarray(self.joint_qvel).ravel())
        for index, actuator in enumerate(self.export["actuators"]):
            for name in ("gainprm", "biasprm", "forcerange", "gear"):
                np.testing.assert_array_equal(actuator[name], getattr(model, "actuator_"+name)[index])
        self._native = ctypes.CDLL(str(self.library_path))
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        self._native.rp_last_error.argtypes = []
        self._native.rp_last_error.restype = ctypes.c_char_p
        self._native.rp_create.argtypes = [integer, pointer, integer, pointer, integer,
                                         pointer, integer, pointer, integer, integer]
        self._native.rp_create.restype = pointer
        self._native.rp_reset.argtypes = [pointer] * 7
        self._native.rp_reset.restype = integer
        self._native.rp_step.argtypes = [pointer] * 8
        self._native.rp_step.restype = integer
        self._native.rp_get_stats.argtypes = [pointer, pointer]
        self._native.rp_get_stats.restype = integer
        if hasattr(self._native, "rp_get_failure"):
            self._native.rp_get_failure.argtypes = [pointer, integer, pointer, pointer]
            self._native.rp_get_failure.restype = integer
        self._native.rp_destroy.argtypes = [pointer]
        self._native.rp_destroy.restype = None
        cache_enabled = False
        if hasattr(self._native, "rp_cache_enabled"):
            self._native.rp_cache_enabled.argtypes = []
            self._native.rp_cache_enabled.restype = integer
            cache_enabled = bool(self._native.rp_cache_enabled())
        with torch.cuda.device(self.device):
            self.qpos = torch.zeros((arenas, 72), device=self.device, dtype=torch.float32)
            self.qvel = torch.zeros((arenas, 70), device=self.device, dtype=torch.float32)
            self.ctrl = torch.zeros((arenas, 58), device=self.device, dtype=torch.float32)
            self.base = torch.zeros((arenas*2, 4), device=self.device, dtype=torch.float32)
            self.angular_local = torch.zeros((arenas*2, 3), device=self.device, dtype=torch.float32)
            self.time = torch.zeros(arenas, device=self.device, dtype=torch.float32)
            self.initial_heading = torch.as_tensor(initial_headings(config, model, arenas), device=self.device)
            arrays = self.host_arrays
            self._handle = self._native.rp_create(
                arenas, arrays["bodies"].ctypes.data, 60, arrays["shapes"].ctypes.data, 91,
                arrays["joints"].ctypes.data, 58, arrays["roots"].ctypes.data, 2, solver_mode,
            )
            if not self._handle:
                raise RuntimeError("native create failed: " + self._last_error())
            self.reset()
        gaps = [
            "MuJoCo implicitfast integration is not reproduced by this native solver",
            "MuJoCo contact solref/solimp softness is retained in the export but not mapped",
            "joint frictionloss uses moving Coulomb friction; static friction is unimplemented",
            "Puffysics cylinder contact uses a single contact point rather than a MuJoCo manifold",
            "restitution is explicitly zero; source soft-contact damping is not mapped to restitution",
            "MuJoCo-equivalent complete collision filtering and numeric trajectories remain unverified",
        ]
        if solver_mode == 0:
            gaps.append("maximal-coordinate mode ignores rotor armature in free integration")
        else:
            gaps.extend([
                "ABA mode includes the joint-armature hook; equivalence is unverified",
                "ABA mode uses predictive hard joint limits rather than MuJoCo soft limits",
            ])
        self.metadata = {
            "backend": "puffysics_native_prototype", "solver_mode": solver_mode,
            "fixed_pose_mass_factor_cache": cache_enabled,
            "solver_name": "maximal_coordinates" if solver_mode == 0 else "articulated_body",
            "model_sha256": config["model_sha256"], "model_export_path": str(self.export_path),
            "model_export_sha256": _sha256(self.export_path), "library_path": str(self.library_path),
            "library_sha256": _sha256(self.library_path), "counts": self.export["counts"],
            "native_contact_capacity_per_arena": 128,
            "physics_timestep_seconds": 0.002, "kernel_launches_per_physics_step": 1,
            "gravity_xyz_m_s2_float32": np.asarray(model.opt.gravity, dtype=np.float32).tolist(),
            "body_frame": "principal_inertial_center_of_mass",
            "base_quaternion": "floating_root_wxyz", "angular_velocity": "pelvis_inertial_local_xyz",
            "ctrl_semantics": "position_target_after_existing_250Hz_filter",
            "flat_array_shapes": {name: list(value.shape) for name, value in self.host_arrays.items()},
            "flat_array_sha256_float32": {name: hashlib.sha256(value.tobytes()).hexdigest()
                                           for name, value in self.host_arrays.items()},
            "support_gaps": gaps, "authentic_rek_parity_established": False,
            "cpu_physics_steps": 0,
        }

    def _last_error(self):
        return (self._native.rp_last_error() or b"unknown native error").decode("utf-8", errors="replace")

    def _state_args(self):
        if not self._handle:
            raise RuntimeError("native handle is closed")
        return (self.qpos.data_ptr(), self.qvel.data_ptr(), self.base.data_ptr(),
                self.angular_local.data_ptr(), self.time.data_ptr(),
                torch.cuda.current_stream(self.device).cuda_stream)

    def reset(self):
        self.ctrl.zero_()
        if self._native.rp_reset(self._handle, *self._state_args()):
            raise RuntimeError("native reset failed: " + self._last_error())

    def step(self):
        if self._native.rp_step(self._handle, self.ctrl.data_ptr(), *self._state_args()):
            raise RuntimeError("native step failed: " + self._last_error())

    def stats(self):
        """Explicit reporting boundary. Never call inside a captured step."""
        torch.cuda.synchronize(self.device)
        values = np.empty((self.arenas, 4), dtype=np.int32)
        if self._native.rp_get_stats(self._handle, values.ctypes.data):
            raise RuntimeError("native stats failed: " + self._last_error())
        result = {
            "max_contact_count": int(values[:, 0].max()),
            "contact_capacity_reached_arenas": int(np.count_nonzero(values[:, 0] >= 128)),
            "nonfinite_arenas": int(np.count_nonzero(values[:, 1])),
            "articulated_solver_failure_arenas": int(np.count_nonzero(values[:, 2])),
            "predictive_joint_limit_impulses": int(values[:, 3].astype(np.int64).sum()),
            "per_arena_columns": ["max_contacts", "nonfinite", "solver_failure", "predictive_limit_impulses"],
            "per_arena": values.tolist(),
        }
        failed = np.flatnonzero(values[:, 2] & 0xFFFFFF)
        if failed.size and hasattr(self._native, "rp_get_failure"):
            arena = int(failed[0])
            meta = np.empty(7, dtype=np.int32)
            data = np.empty(36, dtype=np.float32)
            if self._native.rp_get_failure(self._handle, arena, meta.ctypes.data, data.ctypes.data):
                raise RuntimeError("native failure snapshot failed: " + self._last_error())
            result["first_collision_failure"] = {
                "arena": arena,
                "meta_fields": ["status", "shape_a", "shape_b", "type_a", "type_b", "gjk_iterations", "epa_iterations"],
                "meta": meta.tolist(),
                "per_shape_fields": ["body_position_xyz", "body_quaternion_xyzw", "shape_local_position_xyz",
                                     "shape_local_quaternion_xyzw", "radius", "half_xyz"],
                "shape_a": data[:18].tolist(), "shape_b": data[18:].tolist(),
            }
        return result

    def close(self):
        if self._handle:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                self._native.rp_destroy(self._handle)
            self._handle = None
