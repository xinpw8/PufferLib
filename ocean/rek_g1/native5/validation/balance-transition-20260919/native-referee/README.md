# Native referee countout evidence

All 18 five-point awards in the ten prespecified A/B trials are corroborated by explicit received referee countout calls. Award-size inference is no longer needed for these exact captures. This is an offline evidence export, with no physics, policy, collector, or game-input changes.

## Results

The 11,989 exact 33-byte `REK_FightState` packets contain 35 observed, deduplicated calls: 15 Slip, 2 Knockdown, 1 DoubleKnockdown, 16 Knockout, and 1 DoubleKnockout. The double countout awards five points to each fighter. All 18 per-fighter count episodes contain received count seconds 0, 1, 2 and end when the count mask clears with the explicit matching countout call. There are no left- or right-censored count episodes in this selected data.

| Trial | Slip | Knockdown | DoubleKnockdown | Knockout | DoubleKnockout | +5 awards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-r1-retry1 | 3 | 0 | 0 | 3 | 0 | 3 |
| shaped-r1-retry2 | 0 | 0 | 0 | 0 | 0 | 0 |
| baseline-r2 | 3 | 0 | 0 | 3 | 0 | 3 |
| shaped-r2 | 1 | 0 | 1 | 0 | 1 | 2 |
| baseline-r3-retry2 | 2 | 0 | 0 | 2 | 0 | 2 |
| shaped-r3 | 1 | 0 | 0 | 1 | 0 | 1 |
| baseline-r4 | 1 | 1 | 0 | 2 | 0 | 2 |
| shaped-r4-retry3 | 2 | 0 | 0 | 2 | 0 | 2 |
| baseline-r5 | 1 | 0 | 0 | 1 | 0 | 1 |
| shaped-r5-retry4 | 1 | 1 | 0 | 2 | 0 | 2 |

The explicit single-faller Knockout labels identify slot 1 twelve times and slot 0 four times; DoubleKnockout identifies both through the two count-mask bits and uses faller -1 in its call. Each score receipt uniquely matches a call within a prespecified +/-0.25 s client receipt window, has the exact received scorer counter, and increments that counter by five relative to the preceding fight snapshot. Referee receipt trails score receipt by 0.000603 to 0.0700914 s. This is receipt/counter corroboration without a shared causal packet ID. It assigns no attacker, hit, move, or request cause.

Median observed count duration is 3.005388 s, range 2.993099 to 4.118543 s. The longer duration occurs for the first counted fighter in shaped-r2's double-count sequence; the second fighter's count lasts 3.019340 s. The recovered source restarts the shared deadline when a second fighter falls. These durations use client monotonic receipt clocks, with nominal 0.1 s fight-state publication, and are not exact server event timestamps.

Across 119,976 native sampled actor observations, both `falling` and `fallen` are always false, with no missing flag values. Those are visual robot flags, not received referee flags. All ten rounds ultimately report `WonByPoints`, and every received `knockout_occurred` field is zero. Thus the Knockout referee calls in these G1 rounds are countout awards followed by continuation/reset, not terminal round KOs.

## Schema and safeguards

Private outputs retain every decoded referee snapshot, deduplicated call, count episode, and associated five-point score. Call records include the preceding native paired root pose; all 35 observed calls have a qualifying pose. Raw poses are not copied into this repository. Session groups remain the unchanged six-group manifest from the original transition audit, including the two passive groups without packet exports. All ten A/B captures remain selected, including shaped-r1-retry2 with no calls.

Sequence zero is an empty snapshot even though its default enum name is Slip. Repeated nonzero sequences are one latched call. IDs include process-clock session, capture, round, reset epoch, and byte-wrap epoch. A first nonzero snapshot is historical/left-censored. Round/redo/fight changes, observed zero resets, and uncertain sequence decreases create new namespaces. Server sequence 255 to 1 is a valid wrap. Conflicting payloads under the same nonreset sequence fail closed. A snapshot stream can miss intermediate latched calls between publications; this export claims observed call coverage only.

All source files must match the preceding receipt audit's SHA-256. Each packet's body size, SHA-256, and every decoded numeric field are checked against its packed bytes. Score associations remain unknown when candidates are ambiguous or counter/recipient/time checks fail. Attacker, executed move, active clip, causal hit, and server clock remain null. Countout labels do not justify an action-conditioned fall predictor or new reward weights by themselves.

## Verified source paths for a future live encoder change

The following source evidence identifies the missing live fields. No integration is implemented here.

- `ocean/rek/evidence/windows/RekEvidenceRecorder/Plugin.cs:1045` copies the full 33-byte body at `FightCoordinator.ApplyFightStateSnapshot` prefix. Lines 1089-1095 decode `referee_count_mask`, `referee_count_seconds`, `referee_call_sequence`, `referee_call_type`, `referee_call_faller`, and `referee_call_points`. Packed offsets are 25, 26, 27, 28, signed 29, and 30 respectively. The recorder also retains Unity frame/time/unscaled time, fixed tick, and monotonic receipt time.
- `C:/rekagent/work/controller-audit-isil/IsilDump/REKApp/REKApp/FightCoordinator.txt:34794` calls `ApplyRefereeSnapshot`. Its implementation begins at line 34910. It persists client call sequence and the seen flag at lines 35075 and 35136-35137, count mask at 35279, and count seconds at 35292. `clientRefCountMask` and `clientRefCountSeconds` therefore persist on the visual client. `clientRefCallSeq` alone does not preserve call type, faller, or points.
- The same dump at line 35113 reads `OnRefereeCall`; its callback carries the received call type/faller/points. Count callbacks are `OnRefereeCountTick(slot,seconds)` and `OnRefereeCountCleared(slot)`. First snapshot replay is suppressed; types Slip, SlipEStop, Knockdown, and DoubleKnockdown additionally require a nonzero count mask. `ResetClientRefereeReplay` begins at line 34896. Server fields `refereeLastCall`, `refereeLastCallFaller`, and `refereeLastCallPoints` are not client mirrors and should not be treated as live received labels.
- `FightCoordinator.txt:37712` implements `RaiseRefereeCall` and byte wrap with zero skipped. `FightCoordinator.txt:40096` implements `ResolveCountExpiry`; the single and double paths raise Knockout and DoubleKnockout. The checked-in `ocean/rek/evidence/G1_FIGHT_SEMANTICS_CURRENT_BUILD.md:175` documents the recovered call/count/score/reset flow, including continued G1 rounds after countout.
- `ocean/rek/evidence/windows/RekUiBridgeAgent/Plugin.G1PolicyStream.cs:355` already obtains the coordinator/round for the policy snapshot. Its round projection begins at 404, while line 451 exposes robot `IsFalling`/`IsFallen`. A future version should add a separately named received-referee snapshot with client count fields plus a receipt-time cache of the full call payload, bound to coordinator/fight/round reset identity. Reading the current visual robot flags cannot recover this state. Hooking the full existing received snapshot avoids losing call payload when only `clientRefCallSeq` remains. Any collector change needs its own tests and runtime evaluation.

The inspected build is bound to GameAssembly SHA-256 `6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412` and global metadata SHA-256 `e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd`.

## Reproduction

Run `node native_referee_data.cjs NATIVE_DIRECTORY RECEIPT_AUDIT NEW_PRIVATE_OUTPUT`. It imports the colocated `native_hit_receipt_data.cjs` read-only helpers. Final private output is `/home/spark-advantage/rek-training/native-referee-20260919-r2`, with directory mode 0700 and files mode 0600. Only the aggregate report/schema are checked in. Report SHA-256: `826f339e9494f32b1bf6b5f3dea41ca8903936966f75f42122c00893ac4287d8`.

Ten referee tests and nine hit-receipt tests pass on Windows Node 25.2.1 and Spark Node 18.19.1. They cover repeated packets, resets, wraps, censoring, countout/recovery distinction, ambiguous score matches, double countouts, exact wire hashes/bytes, and unchanged source binding. No model was fitted with these new labels.
