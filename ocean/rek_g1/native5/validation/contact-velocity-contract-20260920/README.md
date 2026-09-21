# Native contact-velocity contract

Source review completed 2026-09-21 UTC. This report identifies the recovered velocity quantity and proposes a separate, later kinematic treatment. No runtime, asset, build, observation, controller, or scoring implementation changed during this review. No GPU work or client interaction occurred.

## Verified native quantity

`Robot.GetBodyLinearVelocity(MjBody)` reads the linear triplet of the body's `mjData.cvel` entry. Its recovered assembly reads the pointer at `mjData + 0x279A8`, then addresses double elements `6 * body.MujocoId + {3,4,5}`. The installed generated interop layout independently identifies offset 162216 (`0x279A8`) as `cvel`, at `generated-mjdata.cs` lines 27905-27906. The adjacent fields are `actuator_velocity` at 162208 and `cdof_dot` at 162224. The `subtree_com` pointer is at 161936, lines 27800-27801.

The decisive private source sections are:

- `Robot.txt`, lines 41198-41294: method and assembly; pointer read at 41261, element addressing and conversion at 41262-41274.
- `MjEngineTool.txt`, lines 1579-1605: `UnityVector3(double*)` casts to float and maps MuJoCo `(x,y,z)` to Unity `(x,z,y)`. This permutation preserves a vector-difference norm up to rounding; it makes no reference-point shift.
- `ContactTrackingManager.txt`, lines 3782-4092: `AppendContact` obtains the two body velocities at 3902 and 3949, subtracts them and computes their Euclidean norm at 3962-3986. The speed is not selected as the maximum collider-point speed. Contact samples aggregate position, normal, impulse, and count separately.

Contact identity is a different contract: the recovered manager packs the ordered geometry IDs as `(minGeomId << 32) | maxGeomId`. Body resolution happens afterward. Same-body colliders therefore retain distinct entry histories even though their body-velocity triplets are identical. The separate contact-entry treatment must retain those geometry keys.

## Reference semantics and version boundary

I don't know the shipped native MuJoCo engine version. The installed interop layout and the recovered pointer/index calculation are verified. They do not establish that the shipped engine is MuJoCo 3.7.0.

The candidate build links the pinned `libmujoco.so.3.7.0`. In that version, `mj_comPos` constructs world-oriented spatial quantities referenced to `subtree_com[body_rootid]`; `mj_comVel` accumulates `cvel` from generalized velocity. This provides a source-backed interpretation for the proposed candidate calculation. It is not an empirical shipped-engine equivalence result. [Pinned 3.7.0 implementation](https://raw.githubusercontent.com/google-deepmind/mujoco/3.7.0/src/engine/engine_core_smooth.c).

In the same version, `mj_objectVelocity` explicitly shifts the spatial velocity from that root-subtree COM to an object's position. The recovered Robot path does not call this shift. A collider-center finite difference or a body-COM point velocity is therefore a different quantity from the raw linear `cvel` triplet. [Pinned object-velocity implementation](https://raw.githubusercontent.com/google-deepmind/mujoco/3.7.0/src/engine/engine_core_util.c).

## Smallest proposed later treatment

The current CPU asset loader has the model, complete transformed route `qpos`, joint position/velocity addresses, and `geom_bodyid`. It currently calls `mj_kinematics` to bake geometry. `FastFrame` contains neither body `cvel` nor root-subtree COM. The available clips include root quaternion, root height, and 29 joint positions; route playback, reversal, mirroring, and yaw removal are already applied during baking.

At CPU asset load, derive generalized velocity from the same route's actual incoming frame edge using `mj_differentiatePos` with the existing 0.02 s step. This handles free-joint quaternion differences as well as translation and hinges. Then run `mj_kinematics`, `mj_comPos`, and `mj_comVel`, without `mj_step`, to store body linear triplets and root-subtree COM. [Pinned position-differencing implementation](https://raw.githubusercontent.com/google-deepmind/mujoco/3.7.0/src/engine/engine_support.c).

The 12 striker geoms map to six bodies: four spheres on each foot, one on each hand, and one on each knee. Nine target geoms map to eight bodies because two torso geoms share one body. Preserve all 108 geometry-pair histories per attacking fighter. Evaluate both fighter trees during baking unless their inertial equivalence is separately demonstrated; the current geometry-symmetry checks do not establish inertial symmetry.

For canonical baked body linear triplet `L_b`, canonical root-subtree COM `C`, external yaw rotation `R`, modeled base velocity `V=(vx,vy,0)`, and external yaw rate `Omega=(0,0,omega)`, the proposed device composition is:

```text
L_world_b = R * L_b + V + Omega cross (R * C)
relative_speed = norm(L_world_striker_body - L_world_target_body)
```

This follows the current transform `world_position=(x,y,0)+R*canonical_position`, with canonical root XY fixed at zero. Clip root tilt and height already belong to the baked state. Do not substitute the collider center for `C`, rotate by clip tilt a second time, or add a second COM derivative. This composition must be tested against direct pinned-engine `cvel` before use.

Route boundaries require explicit handling: identical old/current frames at a route start or a held non-loop endpoint have zero clip rate; a loop uses its actual wrap edge. Existing instantaneous route changes and hard wall/separation corrections do not supply physical impulse velocities. The velocity sample used for swept-contact substeps must also be specified before implementation.

This remains a kinematic proxy. It cannot recover actual controller response, constraint/contact-solver motion, balance, falling, or recovery. Replacing the current collider-center speed proxy could align the represented quantity while leaving those causal dependencies missing. No observation expansion, learned policy benefit, or fighting improvement is established here.

## Validation design, not executed

Compare composed triplets with direct pinned MuJoCo `cvel` at identical `qpos/qvel` for both fighter trees. Cover arbitrary external yaw, translational velocity, signed yaw rate, tilted roots, reverse/mirrored clips, zero rates, held endpoints, route starts, and loop wrap. Verify shared-body geoms have equal velocity triplets while retaining distinct contact-entry keys. Measure numerical error before selecting a tolerance. Keep the treatment opt-in and establish that disabled behavior is unchanged. Any subsequent authentic comparison requires separately frozen training/evaluation candidates; these source findings are not that comparison.

## Reproduction and private provenance

Private evidence root: `C:\rekagent\work\consistent-fighter-20260919-r1\contact-velocity-layout-r1`.

The coordinator generated the existing layout artifact using the following executable and arguments; this review did not rerun the decompiler:

```powershell
& 'C:\rekagent\tools\ilspycmd.exe' -t 'Mujoco.MujocoLib+mjData_' 'C:\Program Files (x86)\Steam\steamapps\common\REK Alpha Test\BepInEx\interop\Mujoco.Runtime.dll'
```

The stdout artifact is `generated-mjdata.cs`, 2,124,299 bytes, SHA256 `ca2af848f66f7c33a4d401e4db833e0df23ea9cbc8d5ff9879a9582c09842879`. The installed `Mujoco.Runtime.dll` is 1,149,952 bytes, SHA256 `3a205663026e77885d5ca3b15a7c0c7ef6b6b108f27043342b8eae41980c94a7`. Existing lengths and digests were rechecked with `Get-Item` and `Get-FileHash`.

Installed build identity is recorded independently: `GameAssembly.dll` SHA256 `6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412`; `global-metadata.dat` SHA256 `e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd`.

Exact private disassembly paths are under `C:\rekagent\work\controller-audit-isil\IsilDump`: `REKApp\REKApp\Robot.txt`, `REKApp\REKApp\ContactTrackingManager.txt`, and `Mujoco.Runtime\Mujoco\MjEngineTool.txt`. Their hashes, the decompiler hash, and the reviewed `fast_assets.cpp`, `fast_assets.h`, and `native_contact_geometry.h` hashes are recorded in `contact-velocity-provenance.json` in the evidence root. That JSON has SHA256 `4af51eb2c1b9da867e364134334aa1805c620de96afbbc5f42ed81c4561e4c46`. Dumps, binaries, and generated proprietary source remain private.
