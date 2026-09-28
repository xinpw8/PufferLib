# Original model boundary: passive inventory revision 2

Status: source compiled and input staging tested; no Unity/container/model/GPU experiment has run from this package. The parent owns approval, staging and any bounded execution after independent review.

Static inspection established the exact current asset chain: globalgamemanagers ResourceManager13, resource key `workshop/robotcatalog`, file5/path2695 -> sharedassets0 RobotCatalog2695 -> g1 previewPrefab GameObject893 -> Transform1650 `g1_29dof_Prefab_SONIC`. Original RobotSpawner's catalog fallback uses this previewPrefab for TryInstantiate. Scene overrides and the deployed authoritative server selection remain unverified.

The selected hierarchy includes 226 transforms, 30 MjBody, 30 MjInertial, 29 MjHingeJoint, one MjFreeJoint, 37 MjGeom and 29 MjActuator. All159 physical/primary-controller records match the preserved probe's raw serialized hashes. Level1 MjGlobalSettings is component2882 on GameObject47. The independent receipt provides source assets and every selected record hash. Custom MonoBehaviour typetrees are stripped; an earlier custom parser has malformed script-pointer values. No reconstructed JSON is used as a replacement prefab or scene.

## First measurement

The plugin calls literal `Resources.Load<RobotCatalog>("Workshop/RobotCatalog")`, reads all catalog entries, requires exactly one entry whose observed id equals `g1`, and traverses its prefab through Unity's read APIs. It does not invoke `TryGetById`. Its metadata-only attestation is retained and explicitly labeled `invoked_by_harness=false`. The selection record states `direct_observed_catalog_enumeration`; successful original TryGetById semantics are not claimed. It rejects a scene instance and an unexpected prefab name. Original-method attestation verifies the exact isolated GameAssembly path and method bytes.

It records Transform TRS, component identities, and Unity JsonUtility serialized fields for Mujoco components and the demonstrated Robot/Sonic/Command/VR types. Loaded RobotConfig and MocapClipConfig assets are emitted separately with bounded counts. Loaded MjGlobalSettings is read without its singleton getter. Level1 is hash-pinned but never loaded by the harness: zero loaded settings means unavailable, not default values. Runtime GetInstanceID values are explicitly process-local; they are not asserted to be serialized file/pathIDs. Missing/transient/unsupported serialized fields remain unknown.

No Instantiate, CreateScene, model generation, scene load, component enable, coroutine, robot Init/reset, policy, explicit network or dynamics call occurs in the harness. Resources.Load and JsonUtility may trigger Unity loading/serialization callbacks. The footer therefore claims the harness's explicit call scope only and states that whole-process physics ticks were not instrumented. This does not establish initialized model, reset state, controller inference or authoritative server parity.

## Containment and pins

The isolation files are byte-equal to the previously reviewed original-history isolation after only replacing the session directory and unique container name. Reuse the existing image `sha256:5fb2c698c065d4502b3c09f3b847efac6aa5428541c4b4f26110e3a530cae879`; do not invoke the build-image option. Existing upstream Box64 is pinned `12a50a0f629f1ddeb08524c8f7399829e0b7101f79f4e20c094376c73a1af7ae`. No AVX/native-flags overrides are added.

New session: `/home/spark-advantage/rek-training/rek-original-model-20260928-r2/unity-harness-inventory-r2`. New container: `rek-original-model-inventory-20260928-r2`. Private network-none container, UID1000, runc/no GPU, private IPC/PID namespace, private Xvfb/Wine prefix, 4 CPUs, 8 GB RAM, 180 s outer limit and 60 s plugin guard. Original user profile/plugins/host display are excluded. Asset/interop inputs are byte-verified copies; only private config/log/output locations are writable.

The fixture verifies eight original game files, including UnityPlayer, catalog/shared assets and the MuJoCo DLL. The plugin additionally pins original GameAssembly/metadata and actual loaded REKApp, Mujoco, Unity Core and JSON interop assemblies. Four existing activation requirements remain: explicit environment enable, CLI opt-in, config/game-root equality, and fresh game-root marker matching run_id/config hash.

## Reproduction and proposed staging commands

Local CPU-only checks:

```
dotnet build probe/RekOriginalModelInventory.csproj -c Release --nologo --no-restore
python -m unittest test_prepare_inputs.py -v
```

Only after parent review, upload this package to a fresh Spark source directory. From that directory, the already-reviewed staging sequence is:

```
python3 isolation/prepare_runtime.py --prepare
python3 isolation/prepare_runtime.py --install-plugin bin/RekOriginalModelInventory.dll --plugin-sha256 bc8df76dbc4664a539fd80c2bb6634c3f03f8ac0bf1bda19f6c550cb5014cb35
python3 prepare_inputs.py --fixture fixture.json --input /home/spark-advantage/rek-training/rek-original-model-20260928-r2/unity-harness-inventory-r2/input --private-game /home/spark-advantage/rek-training/rek-original-model-20260928-r2/unity-harness-inventory-r2/game --run-id original_model_inventory_20260928_r2
python3 isolation/orchestrate.py --freeze-input
```

The parent may then use the reviewed `--create` and `--start` stages after checking the image identity and final pins. These commands have not been executed by this task. Success requires `inventory_end.success=true`, expected catalog/prefab fields and complete JSONL, not container exit alone. Preserve failure output and do not widen the probe automatically.

The subsequent model/reset export is gated on this actual serialized inventory and a separate reviewed component-lifecycle plan. No default XML or assumed global settings are supplied here.

## Preserved first failure and revision boundary

Revision 1 is immutable at the sibling local `rek-original-model-20260928-r1` and NAS `2026-09-28/rek-original-model-r1`. Its single run produced six complete records, then failed with exit 5, no OOM, and no success footer. ErrorLog places the exception path through the generated TryGetById wrapper and exception stringification. Independent review found an ABI mismatch: RobotCatalogEntry is a value type while the generated wrapper allocates an 8-byte pointer slot; original code writes a 56-byte value. The precise native crash mechanism remains incompletely traced.

This revision removes that invocation entirely. It adds no ABI shim, unmanaged call or lifecycle operation. It selects only the already-enumerated original entry and retains the exact expected prefab-name, scene-invalid and asset/interop pins. Containment changes only the fresh session directory and container name. No source asset, original function, image, translator, policy or physics option changes.

Serialized JSON 64-bit instance reference identifiers and GetInstanceID 32-bit return values are recorded as separate observations. Their equivalence is not assumed. Component output only covers serialized fields; unknown transient or unsupported values remain unknown.
