"""Private numeric REK adapter export. No physics stepping or CUDA initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


SCHEMA = "rek.puffysics.compiled_model_adapter.v1"
MODEL_SHA256 = "6cec7d81b69187bfdf2429d71b6288ebb999b5359ecaab7990721323b21722aa"


def quat_matrix(q):
    """XY Z W quaternion to a rotation matrix; no implicit coordinate swap."""
    x, y, z, w = np.asarray(q, dtype=np.float64)
    if abs(x*x + y*y + z*z + w*w - 1) > 1e-7:
        raise ValueError("quaternion must be normalized")
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def matrix_quat(matrix):
    """Stable rotation matrix to XY Z W using a symmetric eigenproblem."""
    r = np.asarray(matrix, dtype=np.float64)
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-9, rtol=0) or np.linalg.det(r) < 0:
        raise ValueError("matrix must be a proper rotation")
    k = np.array([
        [r[0,0]-r[1,1]-r[2,2], r[1,0]+r[0,1], r[2,0]+r[0,2], r[2,1]-r[1,2]],
        [r[1,0]+r[0,1], r[1,1]-r[0,0]-r[2,2], r[2,1]+r[1,2], r[0,2]-r[2,0]],
        [r[2,0]+r[0,2], r[2,1]+r[1,2], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
        [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)],
    ]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    if q[3] < 0:
        q = -q
    return q


def pose_record(position, rotation):
    return {"position_xyz": np.asarray(position).tolist(),
            "quaternion_xyzw": matrix_quat(rotation).tolist()}


def relative_pose(parent_position, parent_rotation, position, rotation):
    return parent_rotation.T @ (position-parent_position), parent_rotation.T @ rotation


def capsule_y_pose(position, source_rotation):
    # Target local +Y maps to source capsule local +Z.
    x90 = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    return position.copy(), source_rotation @ x90


def reference_tangent(axis):
    axis = np.asarray(axis, dtype=np.float64)
    basis = np.eye(3)[np.argmin(np.abs(axis))]
    tangent = np.cross(axis, basis)
    return tangent / np.linalg.norm(tangent)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def idle_qpos(model, asset_root, manifest_sha256):
    root = Path(asset_root)
    manifest_path = root / "semantic_duel_assets_manifest.json"
    if file_digest(manifest_path) != manifest_sha256:
        raise ValueError("semantic asset manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    clips = [clip for clip in manifest["clips"] if clip["role"] == "idle"]
    if len(clips) != 1:
        raise ValueError("exactly one idle clip is required")
    name = clips[0]["files"]["mujoco_joint_order"]
    if Path(name).name != name:
        raise ValueError("idle filename must be relative to its bundle")
    entry = manifest["files"][name]
    raw = (root / name).read_bytes()
    if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("idle array identity mismatch")
    if entry["dtype"] != "float32_le":
        raise ValueError("idle array must be float32_le")
    reference = np.frombuffer(raw, dtype="<f4").reshape(entry["shape"])[0].copy()
    if reference.shape != (29,) or not np.isfinite(reference).all():
        raise ValueError("idle reference must have 29 finite joints")
    pose = model.qpos0.copy()
    for side in range(2):
        ids = np.arange(side*29, (side+1)*29)
        joints = model.actuator_trnid[ids, 0]
        limited = model.jnt_limited[joints].astype(bool)
        limits = model.jnt_range[joints]
        initial = reference.copy()
        initial[limited] = np.clip(initial[limited], limits[limited, 0], limits[limited, 1])
        pose[model.jnt_qposadr[joints]] = initial
    return pose, {"kind": "semantic_idle_frame_zero_clipped",
                  "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha256,
                  "reference_file": name, "reference_sha256": entry["sha256"]}


def export_compiled(model, data, mujoco, *, source, initial_pose):
    """Export already-compiled arrays and kinematics; never mj_step/mj_forward."""
    forbidden = ("nmesh", "nhfield", "nflex", "ntendon", "neq", "nplugin", "nmocap")
    if any(int(getattr(model, key)) != 0 for key in forbidden):
        raise ValueError("this adapter does not silently approximate mesh/flex/tendon/equality/plugin/mocap")
    dynamic_ids = np.flatnonzero(model.body_mass > 0).tolist()
    if any(int(model.body_jntnum[body]) != 1 for body in dynamic_ids):
        raise ValueError("massive fixed or multi-joint bodies require an explicit adapter extension")
    rigid_of = {body: index for index, body in enumerate(dynamic_ids)}
    owner = [-1] * model.nbody
    for body in range(1, model.nbody):
        if body in rigid_of:
            owner[body] = rigid_of[body]
        else:
            if int(model.body_jntnum[body]):
                raise ValueError("massless body with a joint cannot be welded")
            owner[body] = owner[int(model.body_parentid[body])]
    world = {-1: (np.zeros(3), np.eye(3))}
    bodies = []
    for body in dynamic_ids:
        index = rigid_of[body]
        p = data.xipos[body].copy()
        r = data.ximat[body].reshape(3, 3).copy()
        world[index] = p, r
        lp, lr = relative_pose(p, r, data.xpos[body], data.xmat[body].reshape(3, 3))
        bodies.append({"id": index, "source_body_id": body,
                       "mass_kg": float(model.body_mass[body]),
                       "principal_inertia_kg_m2": model.body_inertia[body].tolist(),
                       "world_pose": pose_record(p, r),
                       "original_link_in_rigid": pose_record(lp, lr),
                       "initial_linear_velocity_xyz": [0., 0., 0.],
                       "initial_angular_velocity_xyz": [0., 0., 0.]})
    shapes = []
    kind_by_enum = {int(getattr(mujoco.mjtGeom, "mjGEOM_"+key.upper())): key
                    for key in ("box", "sphere", "capsule", "cylinder")}
    for geom in range(model.ngeom):
        kind = kind_by_enum.get(int(model.geom_type[geom]))
        if kind is None:
            raise ValueError("unsupported geom type must not be dropped")
        body = int(model.geom_bodyid[geom])
        rigid = owner[body]
        p, r = relative_pose(*world[rigid], data.geom_xpos[geom], data.geom_xmat[geom].reshape(3, 3))
        target_p, target_r = capsule_y_pose(p, r) if kind in ("capsule", "cylinder") else (p, r)
        size = model.geom_size[geom]
        dimensions = ({"half_extents_xyz": size.tolist()} if kind == "box" else
                      {"radius_m": float(size[0])} if kind == "sphere" else
                      {"radius_m": float(size[0]), "half_length_m": float(size[1])})
        shapes.append({"id": geom, "source_geom_id": geom, "source_body_id": body,
                       "rigid_body_id": rigid, "kind": kind, "dimensions": dimensions,
                       "source_local_pose": pose_record(p, r),
                       "target_local_pose": pose_record(target_p, target_r),
                       "target_long_axis": "y" if kind in ("capsule", "cylinder") else None,
                       "contype": int(model.geom_contype[geom]),
                       "conaffinity": int(model.geom_conaffinity[geom]),
                       "condim": int(model.geom_condim[geom]),
                       "friction": model.geom_friction[geom].tolist(),
                       "solref": model.geom_solref[geom].tolist(),
                       "solimp": model.geom_solimp[geom].tolist(),
                       "solmix": float(model.geom_solmix[geom]),
                       "margin_m": float(model.geom_margin[geom]),
                       "gap_m": float(model.geom_gap[geom]),
                       "priority": int(model.geom_priority[geom])})
    hinges, free_bases = [], []
    for joint in range(model.njnt):
        body = int(model.jnt_bodyid[joint])
        child = owner[body]
        qadr, dadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        kind = int(model.jnt_type[joint])
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            free_bases.append({"source_joint_id": joint, "source_body_id": body,
                               "rigid_body_id": child, "qposadr": qadr, "dofadr": dadr,
                               "initial_qpos_xyz_wxyz": data.qpos[qadr:qadr+7].tolist(),
                               "original_link_in_rigid": bodies[child]["original_link_in_rigid"]})
            continue
        if kind != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("unsupported joint type must not be dropped")
        parent = owner[int(model.body_parentid[body])]
        anchor = data.xanchor[joint]
        axis = data.xaxis[joint]
        tangent = reference_tangent(axis)
        pp, pr = world[parent]
        cp, cr = world[child]
        angle = float(data.qpos[qadr])
        hinges.append({"id": len(hinges), "source_joint_id": joint,
                       "parent_rigid_body_id": parent, "child_rigid_body_id": child,
                       "qposadr": qadr, "dofadr": dadr,
                       "anchor_parent_xyz": (pr.T @ (anchor-pp)).tolist(),
                       "anchor_child_xyz": (cr.T @ (anchor-cp)).tolist(),
                       "axis_parent_xyz": (pr.T @ axis).tolist(),
                       "axis_child_xyz": (cr.T @ axis).tolist(),
                       "reference_tangent_parent_xyz": (pr.T @ tangent).tolist(),
                       "reference_tangent_child_xyz": (cr.T @ tangent).tolist(),
                       "mujoco_initial_angle_rad": angle,
                       "mujoco_qpos0_angle_rad": float(model.qpos0[qadr]),
                       "limited": bool(model.jnt_limited[joint]),
                       "mujoco_limits_rad": model.jnt_range[joint].tolist(),
                       "limits_relative_to_initial_rad": (model.jnt_range[joint]-angle).tolist(),
                       "armature_kg_m2": float(model.dof_armature[dadr]),
                       "damping_Nm_s_per_rad": float(model.dof_damping[dadr]),
                       "frictionloss_Nm": float(model.dof_frictionloss[dadr]),
                       "stiffness_Nm_per_rad": float(model.jnt_stiffness[joint]),
                       "limit_solref": model.jnt_solref[joint].tolist(),
                       "limit_solimp": model.jnt_solimp[joint].tolist(),
                       "friction_solref": model.dof_solref[dadr].tolist(),
                       "friction_solimp": model.dof_solimp[dadr].tolist()})
    joint_to_hinge = {item["source_joint_id"]: item["id"] for item in hinges}
    actuators = []
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        if int(model.actuator_trntype[actuator]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError("unsupported actuator transmission")
        actuators.append({"id": actuator, "source_actuator_id": actuator,
                          "source_joint_id": joint, "hinge_id": joint_to_hinge[joint],
                          "gear": model.actuator_gear[actuator].tolist(),
                          "dyntype": int(model.actuator_dyntype[actuator]),
                          "gaintype": int(model.actuator_gaintype[actuator]),
                          "biastype": int(model.actuator_biastype[actuator]),
                          "gainprm": model.actuator_gainprm[actuator].tolist(),
                          "biasprm": model.actuator_biasprm[actuator].tolist(),
                          "dynprm": model.actuator_dynprm[actuator].tolist(),
                          "ctrllimited": bool(model.actuator_ctrllimited[actuator]),
                          "ctrlrange": model.actuator_ctrlrange[actuator].tolist(),
                          "forcelimited": bool(model.actuator_forcelimited[actuator]),
                          "forcerange": model.actuator_forcerange[actuator].tolist()})
    result = {"schema": SCHEMA, "classification": "private_engine_prototype_adapter_not_parity",
              "source": source, "coordinate_system": "right_handed_z_up",
              "quaternion_order": "xyzw", "static_world_rigid_body_id": -1,
              "counts": {"source_bodies": int(model.nbody), "rigid_bodies": len(bodies),
                         "shapes": len(shapes), "hinges": len(hinges), "free_bases": len(free_bases),
                         "actuators": len(actuators), "nq": int(model.nq), "nv": int(model.nv)},
              "initial_pose": initial_pose, "initial_qpos": data.qpos.tolist(),
              "model_qpos0": model.qpos0.tolist(), "initial_qvel": data.qvel.tolist(),
              "source_body_to_rigid_body": owner,
              "source_body_parent_id": model.body_parentid.tolist(),
              "source_body_weld_id": model.body_weldid.tolist(),
              "source_exclude_signature": model.exclude_signature.tolist(),
              "source_pair_geom1": model.pair_geom1.tolist(), "source_pair_geom2": model.pair_geom2.tolist(),
              "runtime_timestep_seconds": float(model.opt.timestep),
              "gravity_xyz_m_s2": model.opt.gravity.tolist(),
              "source_solver": {"integrator": int(model.opt.integrator), "solver": int(model.opt.solver),
                                "cone": int(model.opt.cone), "iterations": int(model.opt.iterations),
                                "tolerance": float(model.opt.tolerance), "disableflags": int(model.opt.disableflags)},
              "bodies": bodies, "shapes": shapes, "hinges": hinges,
              "free_bases": free_bases, "actuators": actuators,
              "unverified_engine_semantics": ["MuJoCo implicitfast integration", "elliptic soft contacts",
                  "joint frictionloss", "joint armature", "force-limited affine position PD",
                  "MuJoCo parent/weld collision filtering", "soft joint limits"],
              "execution": {"physics_steps": 0, "gpu_initializations": 0}}
    validate_export(result)
    return result


def validate_export(result):
    if result["schema"] != SCHEMA:
        raise ValueError("adapter schema mismatch")
    encoded = json.dumps(result, allow_nan=False)
    if not encoded:
        raise ValueError("empty adapter")
    poses = {-1: (np.zeros(3), np.eye(3))}
    for body in result["bodies"]:
        pose = body["world_pose"]
        poses[body["id"]] = np.asarray(pose["position_xyz"]), quat_matrix(pose["quaternion_xyzw"])
        if body["mass_kg"] <= 0 or min(body["principal_inertia_kg_m2"]) <= 0:
            raise ValueError("dynamic mass/inertia must be positive")
    for hinge in result["hinges"]:
        pp, pr = poses[hinge["parent_rigid_body_id"]]
        cp, cr = poses[hinge["child_rigid_body_id"]]
        for key in ("anchor", "axis", "reference_tangent"):
            pa = pr @ np.asarray(hinge[key+"_parent_xyz"])
            ca = cr @ np.asarray(hinge[key+"_child_xyz"])
            if key == "anchor":
                pa, ca = pa+pp, ca+cp
            if not np.allclose(pa, ca, atol=1e-9, rtol=0):
                raise ValueError("joint endpoint frame reconstruction differs")
    for key in ("bodies", "shapes", "hinges", "actuators"):
        if [item["id"] for item in result[key]] != list(range(len(result[key]))):
            raise ValueError("adapter IDs must be dense and ordered")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", default=MODEL_SHA256)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--asset-manifest-sha256")
    parser.add_argument("--initial-pose", choices=("idle", "qpos0"), default="idle")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if file_digest(args.model) != args.model_sha256:
        raise ValueError("model SHA-256 mismatch")
    if args.out.exists():
        raise FileExistsError("adapter output already exists; select a new path")
    sys.path.insert(0, str(args.candidate_source.resolve()))
    import mujoco
    import gear_sonic_candidate as candidate
    model = mujoco.MjModel.from_xml_string(args.model.read_text())
    serialized_timestep = float(model.opt.timestep)
    if (model.nq, model.nv, model.nu, model.nbody, model.ngeom, model.njnt) != (72, 70, 58, 63, 91, 60):
        raise ValueError("model dimensions differ from the pinned REK duel")
    for side in range(2):
        ids = np.arange(side*29, (side+1)*29)
        model.actuator_gainprm[ids] = 0
        model.actuator_biasprm[ids] = 0
        from types import SimpleNamespace
        candidate.configure_native_position_actuators(mujoco, model, SimpleNamespace(actuator_ids=ids), candidate.PUBLIC_EFFORT_LIMIT_MUJOCO)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    pose_metadata = {"kind": "model_qpos0"}
    if args.initial_pose == "idle":
        if args.asset_root is None or args.asset_manifest_sha256 is None:
            raise ValueError("idle reset requires a pinned asset root and manifest SHA-256")
        data.qpos[:], pose_metadata = idle_qpos(model, args.asset_root, args.asset_manifest_sha256)
    mujoco.mj_kinematics(model, data)
    source = {"model_path": str(args.model), "model_sha256": args.model_sha256,
              "serialized_timestep_seconds": serialized_timestep,
              "mujoco_version": mujoco.__version__, "candidate_source_path": candidate.__file__,
              "candidate_source_sha256": file_digest(candidate.__file__)}
    result = export_compiled(model, data, mujoco, source=source, initial_pose=pose_metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"output": str(args.out), "sha256": file_digest(args.out), "counts": result["counts"],
                      "initial_pose": pose_metadata["kind"], "physics_steps": 0, "gpu_initializations": 0}))


if __name__ == "__main__":
    main()
