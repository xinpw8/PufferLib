# Balance dynamics producer boundary

Read-only investigation, 2026-09-19. No game interaction, deployment, policy
change, model export, or training was performed. This note is the only new
file from the investigation. Hashes and object identifiers below identify private
inputs; no proprietary dump, controller payload, or configuration content is
published.

## Conclusion and next action

The current compact `fast_runtime.cu` has no action-conditioned balance dynamics.
Adding live count/fall features or a count-based action mask cannot train the
same behavior there: the relevant states never occur. `recovered_balance.cuh`
correctly rejects missing inputs; receipt labels cannot supply its physical
contact stream. The new received-referee contract supplies measured labels with
receipt freshness and lifecycle limits, not the dynamics that caused them.

The actual native calibration and support-contact producers are recoverable from
the pinned assembly. However, the current private-AI client explicitly disables
those producers for its visual robots. Its shipped asset set also lacks the
controller TextAssets used by the inspected SONIC loading path.

The next actionable route is either current-build controller payloads/configuration
with verified provenance, or an explicitly cross-build controller candidate
validated against held-out current-build recordings. The alternate local copy
below supplies concrete candidate payloads. Its different assembly hash does
**not** prove different controller weights, nor does possession of its weights
establish current-build equivalence. Compare payload identity where a current
reference becomes available and test closed-loop trajectories, count onset,
resolution, and physical resets before claiming parity or training new features.

An actual headless physics path already exists in `native5/runtime.cu:299` and
`native5/measurement.cu:93`, with the native MuJoCo CUDA backend documented in
`native5/mujoco_gpu/README.md`. It generates support contacts and executes the
controller. It is a candidate dynamics producer, not established REK parity.
In particular, its calibration uses reset pelvis height above the floor, whereas
native `MeasureStandingHeight` uses the lowest foot geometry. Audit that boundary
and the contact/early-gate contract against the real native producer rather than
manufacturing contacts from rendered height or tilt.

## Version binding

Current install root: `C:/Program Files (x86)/Steam/steamapps/common/REK Alpha Test`.
Alternate local copy: `C:/rekagent/rek-agent-build`.

| File | Current install SHA-256 | Alternate copy SHA-256 |
| --- | --- | --- |
| `GameAssembly.dll` | `6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412` | `90ab0339baab511c33f1100deaa85d70060c0cf291e3b9dc13c71a0814e8955b` |
| `REK_Data/il2cpp_data/Metadata/global-metadata.dat` | `e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd` | `01c5999bc51b4c8c92672119419f0b8e377fcaa468deb120451134492e1d9b7b` |

`f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659`
is the immutable-file Merkle root in
`ocean/rek/evidence/evidence_out/inventory.json`, not a GameAssembly digest.
`ocean/rek/evidence/inventory.py:115` defines domain-separated SHA-256 leaves over
sorted `(relative path, file SHA-256)` pairs and paired internal nodes; line 245
uses the immutable root as `build_fingerprint`. The inventory SHA-256 is
`ea932824c7f1fa9781ab816716d4bfca9ec22b14e754466941c8c157910eff79`.
Its assembly/metadata records match the freshly hashed current files above.
This targeted check did not recompute every installed-file Merkle leaf.

All dump paths below are under
`C:/rekagent/work/controller-audit-isil/IsilDump/REKApp/REKApp/`.
Fresh `Robot.txt` and `SonicPolicyRunner.txt` hashes exactly match the pinned
`SOURCE_HASHES` in `ocean/rek/evidence/controller_path.py:26`, which also names
the current assembly and metadata digests. Other dump hashes are recorded for
reproducibility; this investigation did not re-audit every native method extent.

| Dump file | SHA-256 |
| --- | --- |
| `Robot.txt` | `4f61233092542b15773e49d8404790a8ed89352d3b656fa41b75bab9c8283ded` |
| `SonicPolicyRunner.txt` | `5c7668aa79591cd84dfd120856ecdf96554309c85a2d5a425e8f42636381ab58` |
| `ContactTrackingManager.txt` | `ae143ecfc0e2ecb1c3c47ccfbdb4c88d969d2445aa95d059b2f8608a1524aac2` |
| `Robot_NestedType__RegisterFloorContactListener_d__299.txt` | `6ef5993b6928274b24bc97d193cd0107e70daab66b6b2d92d63110acb0fd53ea` |
| `RobotSpawner.txt` | `fd6d64e7c2e3059d845ff7fa852604a0af16d456a0de97a6621aa42ae9aa57f8` |

## Actual producer and why the captured client cannot supply it

- `Robot.txt:53385`, `MeasureStandingHeight`: initializes `standingPelvisHeight`
  to zero, enumerates `footBodies`, calls `TryGetBodyLowestPoint` (method at
  16237), and stores pelvis world Y minus the lowest foot point at 53828. Missing
  readable foot geometry disables this height-based detector. This is a real
  geometry calibration, not a universal G1 height threshold.
- `Robot.txt:28781`, `PelvisHeightRatio`: uses tracked floor-contact Y when
  available, otherwise `TryRayDownMujoco` (method at 17178), then divides pelvis
  height above that floor by `standingPelvisHeight`. The getter has a fallback
  branch; a finite ratio alone does not prove a valid calibration.
- `ContactTrackingManager.txt:1236`, `Init`, subscribes to
  `MjScene.postUpdateEvent` at 1407. `OnMujocoPostStep` starts at 2057 and includes
  native `mj_contactForce` at 2960. `DispatchContacts` starts at 3503.
  `Robot.txt:51934`, `OnContactsProcessed`, maintains contact-pair membership and
  per-body reference counts, then calls `RecomputeFloorHeight` at 53012.
  `CacheFootBodies` at 52726 and `BothFeetOffFloor` at 29011 distinguish feet.
- `Robot.txt:28485` maps `IsVisualOnly` to offset `0x21`. `FixedUpdate` checks
  that byte and returns before fall processing at 32559. The floor-listener
  registration coroutine checks the same byte at line 251 and skips registration.
- `Robot.txt:34591`, `EnterVisualOnlyMode`, sets that byte, unregisters the
  contact listener at 34885, clears both contact dictionaries at 34899/34913,
  disables the policy runner at 34991, motion composer at 35039, `RobotMjSync`
  at 35087, and child MuJoCo components at 35133. Preserved Unity colliders serve
  local VFX; they are not the disabled authoritative MuJoCo contacts.

The four previously validated Windows captures are listed in
`C:/rekagent/work/balance-heldout-windows-20260919-r1/manifest.json`; their report
is `validation-r2/balance-transfer-validation.json` in the same directory,
SHA-256 `196d69f3f1c3a8a6c8eccd93aabea8e636b99ebc4fd3c2e8cf149f5af75c78fa`.
Fresh header checks found all eight initial actors visual-only, pelvis ratio 1,
floor-contact count 0, and `BothFeetOffFloor=false`, with no recorded standing
height/contact-tracking calibration fields. These numeric values cannot certify
support contacts. Initial `CanGetUp=false` was explicitly available; continuous
recovery/contact authority was not. The missing causal dependency is the
closed-loop controller/actuator/MuJoCo state and contact evolution before native
fall classification, not another received-referee field.

## Controller loading and absence scope

`SonicPolicyRunner.txt:21805`, `InitializeInternal`, reads serialized
`configJson` through `TextAsset.get_text` at 22423; a missing reference logs an
error and returns at 23299/23300. `LoadModels` begins at 25276 and reads
`encoderOnnxBytes`, `decoderOnnxBytes`, and optional `fusedOnnxBytes` with
`TextAsset.get_bytes` at 26443, 26630, and 26776. Those field-name references
occur only in `SonicPolicyRunner.txt` across the inspected REKApp dump.
`RobotSpawner.txt:142/459` loads `Resources` path `Workshop/RobotCatalog` and
instantiates its prefab at 600.

No Addressables, AssetBundle, network-download, or model-cache read path was
found in this loader. Its `persistentDataPath` use at 34224 belongs to
`LogObsDiagnostics` writing `sonic_obs_dump.txt`, not model loading. No cache
directory is identified by the loading code, so no user-home cache search was
performed. The alternate `useDDSBridge` path at 23097/23121 requires configuration
and an external controller; `runUnityDecoderFromBridge` may still call
`LoadModels` at 23135. It is not a hidden controller-payload download.

Using the existing UnityPy 1.25.2 environment, the read-only inventory enumerated
TextAssets in current `sharedassets0` through `sharedassets3`, `resources`,
`globalgamemanagers`, and `level0` through `level3`. None contained SONIC
configuration or controller ONNX TextAssets. The observed TextAssets were motion
clips, localization and unrelated configuration. There were no standalone
`.onnx` or `.xml` files under current `REK_Data`; its `StreamingAssets` contained
no controller bundles. This bounds absence to the inspected shipped loader and
containers. It does not prove absence from an uninspected external provider or
future runtime injection. Individual SONIC prefab pointer values were not
decoded: `TypeTreeGeneratorAPI` was unavailable, and nothing was installed.

Current container hashes, under the current install's `REK_Data`:

| File | SHA-256 |
| --- | --- |
| `sharedassets0.assets` | `37f7a476c56caae37f5a04d4fa1acf5954fdc2b90f20f521830369ecff05f355` |
| `sharedassets1.assets` | `9a780ffcce97bb47744494a0eee0f69bd2cf4d70c6adff7fe0344d06ba119c82` |
| `sharedassets2.assets` | `3a17d6dd040c5200d79b6f9453241fb987b253f5c83da9c5e1c0a3f9e29db615` |
| `sharedassets3.assets` | `980d49ee3b7410d77363b78ff3cebe26870c43f30dda0e20d60a8efcd4b73540` |
| `resources.assets` | `626d2babd7209d8b39af0b8a6edcc6cb9c0fa3efdff86be75ede230f5dbaef46` |
| `globalgamemanagers.assets` | `39d776b2eff4cfb5bb59e65d31a2f5dc1e891559a01c0c04c1977329521208cf` |
| `level0` | `c96011a2b54ab1f6853bc27cb8fc10d9b7296d806b71da3dd49ed18bcd6183fb` |
| `level1` | `d3aef29c86c5d072906d68cd938987b9bc84152a09653f10991f03c805b2ed13` |
| `level2` | `132605943fc30b91ad45322f7fbe60f2fd0ae1024fa7844ef8823c20bebfe3af` |
| `level3` | `620a6cd808945ebf335cdcde4b8c3e9e7ca30409e5cbd1b6ea1af6723f79ba52` |

The current native plugins do exist: `Plugins/x86_64/mujoco.dll` SHA-256
`a6f6b6fd6f0cc35923f57fdc43e18a5cfa43b451fc719bfa355c71c745b43eea`, and
`Plugins/x86_64/onnxruntime.dll` SHA-256
`b2ba7ca16e0e4fe71ad5148744ab885a2f5809e52a0c3de4d9ba3853a03977f9`.
The recovered G1 plant is already described by
`ocean/rek/evidence/evidence_out/g1_29dof.recovered.report.json` as 30 bodies,
29 hinges and 29 actuators, with `control_equivalent=false`. Native libraries
and plant geometry alone do not supply controller payloads or an active local
non-visual execution lifecycle.

## Concrete alternate-copy candidates, not current-build proof

`C:/rekagent/rek-agent-build/REK_Data/sharedassets1.assets` SHA-256:
`a47d6fc85303142975ae825bb5fb7a893d2026efdadc4600e5252fe17547cd9b`.
Read-only in-memory payload hashing found:

| TextAsset | Path ID | Payload bytes | Payload SHA-256 |
| --- | ---: | ---: | --- |
| `model_decoder.onnx` | 186 | 40900688 | `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed` |
| `sonic_config` | 187 | 11880 | `9e8e18d763adfdce094ece2061a42acf4b48e43f89b62ccea3d8b7ba25c2a355` |
| `model_encoder.onnx` | 188 | 50100513 | `013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3` |

No payloads were exported or deployed. These are concrete inputs for a separately
declared candidate, not evidence that changing visual-mode flags, merging builds,
or feeding new live features to an upright-only policy would be valid.
