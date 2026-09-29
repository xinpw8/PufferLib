# Saved controls and native move translation

The control defect is the browser-category to native-index boundary. The saved keys are unchanged. The unmodified r7 `HumanInput` subtracts 16 from policy categories, but the original runtime move list has a different order. This misroutes ten of the seventeen actions.

The corrected r8 source changes only that conversion. Independent tests derived expected indices from saved command names and named motion asset routes. Old r7 fails ten mappings; r8 passes all seventeen, with every edge consumed once. Eight focused app tests and nine launcher tests also pass.

| Keys | Saved command | Policy category | Native move index |
|---|---|---:|---:|
| I | left_hook_processed | 20 | 0 |
| K | left_jab_processed | 21 | 1 |
| Space + J | double_uppercut_processed | 22 | 2 |
| O | right_hook_processed | 23 | 3 |
| L | right_jab_processed | 24 | 4 |
| Space + L | left_jab_right_uppercut_processed | 25 | 5 |
| YY | left_side_kick_processed | 16 | 6 |
| HH | left_front_kick_processed | 17 | 7 |
| UU | right_side_kick_processed | 18 | 8 |
| JJ | right_knee_processed | 19 | 9 |
| Space + Y | 6_punch_processed | 26 | 10 |
| Space + U | run_and_punch_processed | 27 | 11 |
| ; | left_right_jab_processed | 28 | 12 |
| ' | left_right_hook_processed | 29 | 13 |
| Space + K | left_hook_right_jab_processed | 30 | 14 |
| Space + H | double_hook_processed | 31 | 15 |
| Space + I | butt_smack_emote_processed | 32 | 16 |

HH previously selected left_jab. L previously selected right_side_kick. UU previously selected double_uppercut. The incorrect index alone does not establish why a particular UU action appeared absent; native acceptance, current action state and observed key timing are separate facts.

## Independent evidence

`EXPECTED-MOVES.json` binds all source hashes. Its generator uses the original controls-only export, verifies its decoded binary profile, derives numeric key names from original `Keyboard.get_*Key` methods, joins saved command names to NPZ source names and runtime routes, and cross-checks the pre-existing policy table. It never reads the corrected app to derive expected answers.

- Original `RobotInputController.txt`: `ExecuteMove` begins at line 5934 and uses `robotConfig.moves.IndexOf(clip)` at line 6548. `ExecuteMoveByIndex` begins at line 7787 and reads `moves[index]` at line 8015.
- Native `g1_semantic_action_table.c`: `MOVE_REGISTRY_ORDER` gives categories 16..32 the indices `[6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16]`.
- `semantic_duel_assets_manifest.json`: each named source clip is joined through `npz_path_id` to a route's `runtime_move_index`. The verified manifest SHA is `7d4719a3ca1e9e5a8faf571bc3c5c70e2b4c9e841be34303fa0a3f47d3692a28`.
- Native direct scheduling indexes `b.move_routes[command.move_index]` directly. Consequently the caller must supply the runtime index.

## Original gesture semantics and scope

Recovered `ControlScheme.ResolveRows` polls unscaled time and frame count. A chord requires all its non-null controls held and at least one pressed edge. The chord with the most controls wins, with original row order resolving equal counts. Double taps compare the same chord within the configured window. The first tap defers a move; an optional same-chord single action can be emitted when the window expires. The four saved double-tap keys have no corresponding single-key action.

The recovered `KeyboardControlScheme` constructor sets `doubleTapWindow=0.3f` at line 2748, and `CopyTuningFrom` copies a seed's window at line 374. The saved profile contains no timing field. Therefore 300 ms is a source-backed constructor default; this audit does not prove the live official serialized seed value or complete event-loop equivalence. The mapping fix leaves gesture recognition unchanged.

`CURRENT-WINDOWS-CONTROLS.json` records a read-only check of exactly six known `rek.controls.*` value names in `HKCU\Software\REK\REK`. All six match the archived export byte-for-byte, including G1 and T800 custom profiles and custom scheme selectors. No unrelated registry values or authentication data were read or published.

## Deployment review

GO for a fresh paused port 18773 with the reviewed app and unchanged native executable, after verifying both existing viewers' identities and paused state. Preserve their processes, input state, ports and recordings. Keep a passive pause guard during startup, and stop only the newly owned process group if a preserved viewer resumes. GPU initialization can briefly compete with an active session, so a fresh process must not be started during human play.

The reviewed launcher adds only an explicit validated port parameter bound to prepared configuration and identity. Its existing unused-port, source/binary hashes, owned-group cleanup, paused tick-zero and initial PNG checks remain. Nine CPU tests pass. This review performed no game input, live application mutation or GPU workload.

Reviewed input source SHA: `a0e6b8e7b60adf035c22db20cc61990c4700bd3d91d1b39375e2222f9022414a`.

Reviewed launcher SHA: `a053b3daff7c05457ee4e20e8c7e93ae70406f7ff5c4cbe98947266e88abde0b`.
