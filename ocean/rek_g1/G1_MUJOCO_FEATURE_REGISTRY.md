# G1 MuJoCo foot-feature registry

`g1_mujoco_feature_registry.c` closes the kinematic boundary required by
`sonic_motion_entry_matcher_native.c`. It uses the already-opened, build-pinned
two-fighter MuJoCo model and the eight unique decoded clip arrays owned by
`RekG1SemanticAssets`.

The open operation validates the exact player pelvis, left ankle-roll, right
ankle-roll, and all 29 player joint names. It also validates every route-to-clip
identity against the static 11-route contract. It creates a separate scratch
`mjData`; no live arena `mjData` is read or modified.

For each of the 760 clip frames, the sampler performs this sequence:

1. Reset the scratch state to the model's deterministic initial state.
2. Write all 29 MuJoCo-order joint values through the validated player map.
3. Run `mj_kinematics` without advancing time or dynamics.
4. Read the player left ankle-roll, right ankle-roll, and pelvis world poses.
5. Subtract the pelvis world position and multiply by the transpose of the
   pelvis world rotation, which is the root inverse-transform operation.
6. Convert the MuJoCo local vector `(x, y, z)` to the current runner's Unity
   local order `(x, z, y)`.
7. Store binary32 rows as left xyz followed by right xyz.

This matches the recovered `BakeFootFeatures` operation order: pose, synchronize
kinematics, read left world position, root inverse transform, read right world
position, root inverse transform, and register all six values. The source method
identity and current Steam build association are recorded in
`SONIC_MOTION_ENTRY_MATCHER_NATIVE.md`.

The registry installs an aggregate `SonicMotionComposerNativeBackends` context.
It delegates the caller's quaternion, atan2, and sin/cos callbacks through their
original context and adds the registered loop-entry matcher. The registry,
semantic assets, and duel must remain alive until the semantic runtime closes.

## Validation

The Spark ARM64 real-asset test bakes 4,560 binary32 values from eight clips and
760 frames. Its pinned feature SHA-256 is
`f744df28c39c48fc6ccc8d779582d6952359593b63dd374b71f6481c2c1381c5`.
An independent Python MuJoCo implementation produces the same bytes. Strict GCC,
Clang AddressSanitizer plus UndefinedBehaviorSanitizer with leak detection, and
Clang static analysis all pass.

This validates deterministic kinematic feature generation for the pinned model
and clip bundle. It does not establish held-out REK trajectory parity or identify
the current server policy weights.
