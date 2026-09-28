# Passive original model inventory: first execution

The one authorized run failed incomplete. It exited with code 5 at 2026-09-28T20:10:46Z; Docker reports no OOM. No retry or model/dynamics experiment followed.

Six complete trace records survived. Literal Resources.Load loaded RobotCatalog. Its original serialized entries identify G1/L100 -> g1_29dof_Prefab_SONIC and T800/H100 -> engineai_t800_FactoryPolicy. Both returned prefabs have scene_valid=false and active_in_hierarchy=false. Exact original TryGetById entrypoint attestation passed against the pinned isolated GameAssembly.

The last durable record is the second catalog entry. The next source operation is TryGetById; no transform, component, global-settings or inventory_end record exists. The native log reports an access violation reading address 0x210 at GameAssembly.dll+0x206f371, followed by CoreCLR failfast c0000005. This brackets the failure, but does not prove the failing call or root cause.

The static serialized-asset review remains available separately. It establishes the current ResourceManager/catalog/PPtr chain and hashes for the physical prefab. The failed runtime cannot establish its serialized component values, selected scene overrides, initialized mjModel/mjData or server model.

The initial count contains one active MjScene from normal isolated application startup. The harness explicitly made no model, spawn or dynamics call; whole-process physics activity was not instrumented. JsonUtility and Resources.Load may invoke loading or serialization callbacks. No claim of whole-process inactivity is made.

Source files and the first package are preserved. All 23 closed source files were copied with matching source/readback size and SHA-256; COPY-VERIFICATION.json and SOURCE-MANIFEST.json contain the exact evidence.
