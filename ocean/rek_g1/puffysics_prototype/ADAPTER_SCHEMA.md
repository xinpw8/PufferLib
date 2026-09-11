# Private compiled model adapter v1

`export_rek_compiled_model.py` writes `rek.puffysics.compiled_model_adapter.v1`.
The JSON contains private numeric model data and belongs outside the repository.
It is an engine prototype input, not a parity certificate. It contains no game
binary, controller weights or opaque model binary.

## Frames and IDs

All vectors are XYZ in metres/radians/SI, all quaternion records are XYZW, and
the world remains right-handed Z-up. The free-base `initial_qpos_xyz_wxyz` is
explicitly named for MuJoCo's native QPOS ordering and is the sole quaternion
array not represented by a named XYZW pose record.

Each `bodies` entry is a positive-mass MuJoCo body expressed in its principal
inertial frame, at its centre of mass. Its `world_pose` is the initial COM pose;
`principal_inertia_kg_m2` is diagonal in that frame. `original_link_in_rigid`
reconstructs the source link pose from an updated engine COM pose. Body IDs are
dense 0..59 for the pinned model. Static world has body ID -1. Fixed massless
helpers transfer geometry and parent ancestry to the nearest massive ancestor.
Massive fixed bodies and multiple joints on one body fail explicitly.

Each of the 91 `shapes` references a rigid body or static world. `dimensions`
contains box half-extents, sphere radius, or capsule/cylinder radius and
half-length. `source_local_pose` reproduces the compiled MuJoCo geometry frame.
`target_local_pose` rotates capsules and cylinders by +90 degrees around local X
so B3 local Y equals MuJoCo local Z. No cylinder approximation is performed by
the exporter. Both source/target poses are relative to the owning
COM frame (or world for static shapes). Collision masks, soft-contact settings,
friction and source body ancestry are retained, not replaced by engine defaults.

## Joints, initial state and actuation

`hinges` contains 58 joints with both local anchors, local axes and reference
tangents. The tangents coincide in world space at the exported initial pose.
An engine signed relative hinge angle of zero therefore maps to
`mujoco_initial_angle_rad`; MuJoCo angle equals that offset plus the engine
angle. The intended sign is atan2(axis dot (parent_tangent cross child_tangent),
parent_tangent dot child_tangent). Both `qposadr` and `dofadr` are included.
`limits_relative_to_initial_rad` are the source limits minus the initial angle.
Joint armature, damping, friction loss, soft limit and friction parameters stay
in the schema even when the target engine does not yet implement them.

`free_bases` contains the two source free joints, their QPOS/DOF addresses, rigid
IDs and COM-to-source-link transforms. All generalized and rigid-body velocities
are zero at initial reset. `initial_qpos` is the exact clipped idle frame-zero
assignment used by GpuDuelReset; `model_qpos0` is separately preserved. The
manifest and used idle array are hash checked. `--initial-pose qpos0` explicitly
selects a different diagnostic initialization.

`actuators` preserves all 58 compiled runtime joint transmissions and gain/bias,
gear and force-limit arrays after `configure_native_position_actuators`.
Thus these are affine position-PD controls, not the unit motors serialized in
the source XML. Target filtering and retained/dampened-drive logic remain the
responsibility of the existing controller bridge.

The exporter uses CPU model compilation plus `mj_kinematics` only. It never calls
`mj_step`, `mj_forward`, a renderer or CUDA. Unsupported structures cause an
explicit error. Tests reconstruct geometry and source link world frames and
validate hinge anchors/axes/tangents, massless-helper transfer and capsule axes.
