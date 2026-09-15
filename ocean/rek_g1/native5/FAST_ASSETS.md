# Compact semantic motion assets

`load_fast_assets()` in `fast_assets.cpp` loads and verifies the existing private
semantic bundle, then bakes compact poses and collision proxies before training.
The returned `FastAssets` is host-owned; the CUDA runtime uploads its immutable
arrays once. No asset, model, checkpoint, or baked pose payload belongs in Git.

## Inputs and outputs

Inputs are `RekNative5Config.model_path`, `assets_path`,
`motion_features_path`, and the 17 explicit `move_duration_ticks` values. Every
entry in `semantic_duel_assets_manifest.json.files` is checked against its byte
count and SHA-256. The feature manifest must reference that asset manifest;
feature payloads are not used. XML parsing uses MuJoCo's native C API.

The 24 routes preserve source clip IDs, direction of playback, clipping bounds,
mirror rules, blend metadata, and the existing 33-category mapping. The 21
actual source clips produce 1,768 frames at 50 Hz. Discrete routes include their
final endpoint, so frame count is explicit move duration plus one. Reverse
playback and loop wrapping are baked into the sample ordering. Transition
blending remains the runtime's responsibility.

Each 284-byte frame contains 29 joint poses, root quaternion, source clip yaw,
root height, six striker spheres, and three target spheres. Strikers are left
foot, right foot, left hand, right hand, left knee, right knee. Targets are
pelvis, torso, and head. The geometry is baked using `mj_kinematics()` from the
named model bodies/geometries. No `mj_step`, `mj_forward`, split physics step,
SONIC inference, or GPU initialization is called by the loader.

Positions use a fighter XY origin and the configured reference-root rotation.
Their Z values already include root height and floor height. The runtime adds
world XY translation and heading; it must not add root height twice. Initial
source heading is removed, then the route's configured `yaw_blend` removes its
fraction of clip yaw, matching the existing reference convention.

## Explicit approximations

The bundle contains root XYZ, but horizontal coordinates are effectively zero:
the largest observed span among all clips was 2.25628e-7 m. Root height is useful
and is retained. Horizontal walking speed and collision-induced displacement
cannot be recovered from those stationary XY arrays. A separate explicit root
response model is required; this loader supplies no invented measured speed.

Collision proxies conservatively enclose each selected model geometry, or the
union of geometries attached to that striker body. Sphere/capsule/cylinder/box
dimensions are read from the XML. These bounding spheres are approximations,
not the original collision shapes. They can overestimate contact volume. Swept
tests improve temporal coverage but do not remove that geometric approximation.
The pack also does not reproduce physics-induced pose deformation, instability,
or reaction forces. Its provenance explicitly marks an approximate candidate
and makes no REK parity claim.

## Executed validation

On Spark, `fast_assets_probe.sh` builds a native C++ probe linked only to MuJoCo,
OpenSSL, and standard libraries. Linker wrappers reject calls to CPU physics
entry points. The actual bundle passed 125,528 finite-value checks, all move
duration mappings, and a root-quaternion norm-error limit of 1e-5. Observed
maximum quaternion error was 1.31867e-7; striker sphere radii were
0.0450794 to 0.177540 m. Forbidden CPU physics calls: zero. The first run took
0.02 s wall time for parsing and offline baking; this is not training SPS.

The source accepts custom private paths. A reproducible probe command is:

```sh
bash ocean/rek_g1/native5/fast_assets_probe.sh NEW_OUTPUT \
  /private/model.two_fighter_arena.xml /private/semantic-assets \
  /private/motion-foot-features
```

Exact commands, executable/source hashes, identity, dependencies, and aggregate
results are recorded in the chosen output directory. The Spark validation
directory is `/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/fast-assets-probe-r2`.
