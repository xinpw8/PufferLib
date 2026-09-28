# Original history and decoder observation oracle

The isolated original-method run completed with process exit 0. Its 28 query rows cover two fresh repetitions of 14 empty, partial, full, wraparound and Clear cases. Every one of the 27,832 decoder FP32 values matched the explicit fixture contract, and both repetitions were bit-identical. `runtime-r1/VERIFIED.json` contains the checks and nine original-method entrypoint attestations. Independent comparison against the actual native implementation is maintained by the controller-comparison task separately.

The plugin calls the installed original `GameAssembly.dll` methods through the generated IL2CPP interop wrappers: StateRingBuffer `Push`, `GetLatest`, `Clear`, plus SonicPolicyRunner `BuildObsPlans`, `FillHistoryChannel`, `BuildDecoderObs`. Four small quaternion cases also call original `CalcHeadingMj`, `YawQuatMj`, and `QuatMulMj`. The trace records method RVA, loaded module, first 32 instruction bytes' SHA256, and completed wrapper-call counts. These are actual original-function results, rather than a reimplementation labeled as an oracle.

`BuildObsPlans` generated the expected 994-value decoder layout: 64 supplied tokens, then ten oldest-to-newest frames for angular velocity (3), joint positions (29), joint velocities (29), previous actions (29), and gravity direction (3). Channel offsets are 0, 64, 94, 384, 674, 964. Snapshots already contain explicitly transformed asymmetric channels. The harness applies no joint remapping or normalization.

Observed boundary details:

- Missing frames are left-padded with null snapshot fields. `BuildDecoderObs` zero-clears its destination, producing zeros there. Direct `FillHistoryChannel` leaves those missing slots untouched; sentinel probes distinguish these two behaviors.
- `GetLatest(n, step)` uses `floor(count / step)` available samples. Step 2 with one stored snapshot returns no sample. This corner is explicitly captured, including after Clear.
- Clear resets count/write index. Subsequent decoder observations contain only the new samples and zero padding. This does not assert that Clear overwrites all backing storage.
- Decoder queries include 21-slot ring wrap and retained snapshots after 27 pushes. There were 62 original Push calls, 56 explicit GetLatest calls, 2 Clear calls, 2 BuildObsPlans calls, 28 BuildDecoderObs calls, and 140 direct FillHistoryChannel calls across the two repetitions. Internal delegate calls are additional and are not represented as separately instrumented wrapper calls.

The runner is created on an inactive GameObject and remains disabled. Its required config, token buffer, state ring and read buffer are explicitly supplied. The original Unity JsonUtility parses the pinned sonic_config. Full Init/InitializeInternal, Start, FixedUpdate, model loading, DDS and physics stepping are not called by the harness. `IsReady` remains false and is reported. Final scene inventory is zero Robot, two disabled runner components, zero composer. Whole-process physics ticks were not instrumented; the claim is limited to the harness's calls. This does not prove full automatic runner initialization, motor inference, actual state gathering, current authoritative server configuration, physics or match equivalence.

The same reviewed containment as the successful composer oracle was reused: a new private game copy, private prefix and X server, CPU-only Docker runtime, no network, no host X socket, no original user profile, read-only game and interop, no GPU devices. The preserved upstream Box64 binary was used. The independent reviewer approved the plugin and confirmed the containment change was limited to new session/container identifiers. No existing game, viewer, profile, service or live input was touched. The container is closed, not running.

Key hashes:

| Artifact | SHA256 |
| --- | --- |
| Plugin.cs | b53227ea44dcecb984006c034e7b0b8025152f9c7b1207f54328ebe2b226c94e |
| RekHistoryOracle.dll | 39289ff8f2df29f791a0fe682eaa8d29c38a435d523af06a01b4b786df315804 |
| Fixture | 8d2461ca77f156ea479b72be1f10434777b3e5e500da3124d72d1c6feed9b2c0 |
| Original sonic_config | 9e8e18d763adfdce094ece2061a42acf4b48e43f89b62ccea3d8b7ba25c2a355 |
| Closed original trace | 8495728eb588d9ff51da93d1f6783fbb3c8757cc80f56a16f6892a85bd9f308c |
| Closed-source manifest | 1370fd2eea1727007faa2c0f5bc26fa5216b5ced86c9a5a6d79e18c397e7f3a8 |
| GameAssembly | 6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412 |
| Metadata | e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd |
| Actual loaded interop | faa94fb58e24fda95e2c06810e28b9eb2d6d9f9f8327541976a0dc1011f646d2 |

`runtime-r1/closed` retains all 24 closed output/log/config/isolation receipt files with source/local hash equality. The 575 copied static game files are retained remotely and described by `STAGED.json`, without unnecessarily duplicating them in this diagnostic package. The previous composer oracle package is unchanged.

Source evidence: `SonicPolicyRunner.txt` BuildObsPlans at line 24247, BuildDecoderObs at 48888, FillHistoryChannel at 55473, original JsonUtility setup in InitializeInternal at 21805; StateRingBuffer constructor/Push/Clear/GetLatest at lines 13/79/181/193 of `SonicPolicyRunner_NestedType_StateRingBuffer.txt`, both under `C:\rekagent\work\controller-audit-isil\IsilDump\REKApp\REKApp`.

Build: `dotnet build RekHistoryOracle.csproj -c Release --nologo` succeeded with one nullable-analysis warning CS8602 at the already guarded parsed-config loop, zero errors. `prepare_inputs.py` stages an absent or verified-empty private input directory. `isolation/prepare_runtime.py` and `orchestrate.py` refuse reused runtime/containers. `verify_trace.py` can recheck the closed trace without launching Unity. The JSON run schema and activation names retain the prior composer naming solely for compatibility with the unchanged isolation image; the fixture and trace use `rek.original_history.*` schemas.
