# Original Slerp boundary experiment

This package isolates the quaternion interpolation boundary in the explicit 640-row composer fixture. It does not implement a general replacement Slerp, change the native clone, run physics, train a model, or establish server parity.

The initial native capture reproduces the previous trace byte for byte. The callback records 32,800 exact FP32 input/output tuples, reduced to 308 recorded unique inputs plus six synthetic controls. All fixture assets, source, binaries and runtime inputs are pinned in the receipts.

`Plugin.cs` is preserved as the first revision. Its isolated run rejected an ambiguous reflection overload before producing any math rows. `revision-r2/Plugin.cs` selects the exact parameter types and logs them alongside the original compiled method pointer, module, RVA and initial code bytes. Both REK `SlerpWxyz` and Unity `Quaternion.Slerp` are called twice for each exact input. A closed success footer, matching repetitions, matching wrappers and all binary/fixture hashes are required before a table is accepted. A process exit of zero alone is insufficient.

`slerp_boundary.h` substitutes measured results only when all nine input words match exactly. A missing tuple returns failure without modifying the output or falling back to libm. The first replay stopped when replacing an inner interpolation changed a later blend input. The single follow-up fixture retains all old cases and queries every measured alternative for matching intermediate output words, including combinations. Seventeen native output keys have more than one measured original alternative. The added tuples are exploratory inputs; no variant is selected as the true dependency. The final replay establishes which exact inputs actually occur.

## Source and execution

The installed game and current viewer were not modified. Each original-function run uses a fresh task-owned copy, Wine prefix, HOME and private Xvfb. Containers have no network, GPU device, host display, original profile, published port or privileged access. The upstream Box64, container image, Wine/runtime inputs and original game/interop binaries are pinned. The external timeout is 180 seconds with a 10-second termination grace. The plugin also has a 90-second guard.

The host commands are preserved in `execution/command-*/command.json`, with stdout, stderr and terminal result. New sessions are required for reruns; existing output is never overwritten.

1. Compile the CPU fixture with `bash build_native.sh build-rN` and run `native/build-rN/native-trace ASSETS NEW_TRACE NEW_CALLS`.
2. Reduce closed calls with `python3 slerp_protocol.py requests CALLS NEW_FIXTURE`.
3. On the pinned Spark host, stage the private runtime using the reviewed isolation scripts, install the exact compiled plugin, freeze the fixture/config marker, then create and verify the isolated container before starting it. `revision-r2/isolation` is the initial successful version; `revision-r3/isolation` is the one follow-up session.
4. Reduce a closed successful original trace with `python3 slerp_protocol.py table FIXTURE ORACLE NEW_TABLE --plugin-sha256 EXACT_HASH`.
5. Replay the CPU fixture with the same executable and fourth data argument `NEW_TABLE`. Any miss is an incomplete result. Only a complete callback/footer and full 640-row trace qualify for `analyze_replay.py`.

`remote_task.py` delegates SSH transport to the separately pinned previous helper identified in `VALIDATION.json`; no credentials are included. `prepare_runtime.py` requires the exact existing game, runtime and image inputs. The source/compiled plugin therefore remain reusable with those dependencies, rather than being a self-contained redistributed game.

## Interpretation

The previous component comparison's small quaternion residual can be attributed to the callback boundary only to the extent that measured substitution removes it across the complete unchanged fixture. This says nothing by itself about unaligned movement/yaw comparisons, contact physics, full reset callback ordering, sparring strategy or superhuman performance. The native velocity output is a finite-difference estimate, and the native reference root-position API is unsupported. Those limits remain explicit even if every supported fixture field becomes bit-exact.

The final measured result and hashes are in `RESULTS.json` and `RESULTS.md`. Failed runs and incomplete replays remain separate evidence.
