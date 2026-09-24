# Post-stop terminal readback, staged driver variant

This driver-only patch targets one measured match-end serialization failure. It does not rebuild the bridge, change the policy, infer a terminal result from a clock or score, or rewrite historical summaries. Native validation of the new recovery path remains pending at this publication cutoff.

## Observed failure

In `persistent-runtime-any-ai-20260924-r1/live/any-s1803`, the last policy source showed native round3 active at0.0741855s and19:9. The next relay diagnostic was `policy_state_unavailable:policy_opponent_identity_changed`. The coordinator had advanced Bot1 to Bot2 at a full-match win. The existing serializer checks the pinned opponent before exposing the final round. Its generic stream-end event retained only `InvalidDataException`. The process remained alive.

An independent native recorder captured and applied the actual terminal packet: round3 inactive0s,19:9, WonByPoints/winner0; match2:1, WonByRounds/winner0, next AI level1. Native file SHA256 `b35d7b6eec6efd556d92420241fd28ac1fa0b5094ecacd61a8306e7c677a83b7`; terminal packet SHA256 `845833b84760d71a06a6a17ec964283af95ee3f73273177b770c77c08653d0c8`. The original attempt remains incomplete in its original ledger. Raw packet bytes, game binaries and recovered game bodies are excluded here.

## Bounded behavior

After acknowledged `StopG1PolicyStream`, only this exact diagnostic paired with the same stream-end round identity triggers one passive `get_state` and one passive `get_policy_state`, before releasing the lease. The action callbacks stay stopped. No action, stream restart, retry loop, menu command or process operation is introduced.

Recovery requires all of the following:

- Same native round identity hash, local/opponent slot, fight epoch and round number; isolated known-private-AI/no-human proof.
- Actual inactive120s/nonredo round at0s with a score-consistent WonByPoints win and matching WonByRounds match result.
- Exactly the next known bot identity, with the original policy opponent retained separately.
- Source readback within500ms and an available, hook-verified, fresh received-referee receipt bound by the bridge to the native round/fight lifecycle.
- A SHA-verified33-byte packet whose phase, round, time, scores, winner, match wins and next bot level agree with the native readback fields.

Any missing, stale, changed or ambiguous evidence leaves the attempt incomplete. Successful proof adds `terminal_recovery` and its directly observed final round, while preserving the original stop reason. Existing return/reward/export code is untouched. Policy callbacks, masks, recurrent state and normal exit behavior are unchanged.

## CPU reproduction

Run from this directory:

```text
node verify.cjs
```

The verifier checks the committed handoff driver, reconstructs the published redo driver, applies this patch in a fresh retained temporary directory, checks exact runtime hashes and runs eight self-contained synthetic CPU tests. Node.js and Git are required. It does not read private fixtures or start a bridge, controller, game or GPU workload.

Synthetic tests cover accepted terminal fields and rejections for unrelated diagnostics, wrong epoch/round/hash/slot, active state, human scope, wrong bot progression, stale/missing referee, wrong packet scores and changed match result. The private test suite additionally uses recorded packet/postfix facts, but its expected post-stop policy envelope remains explicitly synthetic because that request was not issued in the failed historical attempt.

Baseline driver SHA256: `f1f25159c5155f15ce0a6035236c684fcc4f62f8f9843c54884bebdc59a33dab`. Patched driver SHA256: `9c55723297e8abeb10ea2a98ab9cd19978e2b97285b78b8f750b04214d73ca83`.

## Native validation status

A separate passive probe at13:45:12Z on the same client observed inactive Idle/Bot4 and successfully closed its relay. Its policy-state request was rejected because local G1 bone bindings were unavailable. This does not validate the proposed immediate FightOver readback, and no recovered-success claim is made.

The same-client test started at13:47:13.521Z with plan SHA256 `4aa14abf29a28822fa6f738ebdee11adb6f29e6aba405eb2f316d517fae497da`, bounded to at most four regular full rounds, two attempts per label and zero automatic relaunches. It retained native redo opt-in, opponent-stratified runtime-only mode and frozen checkpoint `f147bdc358261e272c58e70167cd8b0b9e4953806c417d8891d94a3ad083ac84`.

It stopped after one completed Bot4 round and one incomplete round:

| Attempt | Initial native state | Final observed state | Local applied ACKs |
|---|---|---|---:|
| terminal-s1801 | Round1,120s/nonredo,119.849754s,0:0 | Inactive0s,5:8,WonByPoints | 5962 |
| terminal-s1802 | Round2,120s/nonredo,119.783195s,0:0 | Active22.07044s,18:18,InProgress | 4861 |

The first round used the normal terminal path, with no recovery call. Its endpoints closed cleanly at13:49:28.974Z, handoff proof followed at28.984Z, and the next driver started13:49:29.010Z. The first MP4 validated at13:49:32.558Z. Round2 failed with `source_stream_missing`; stop/release acknowledgements were unavailable after client loss. The controller logged `game_died_during_attempt` at13:51:31.929Z and `relaunch budget exhausted` at13:51:36.957Z. No relaunch occurred. The incomplete18:18 state is not a tie result.

`EXECUTION.json` contains compact owned-log evidence and hashes. Neither a terminal-recovery event nor controlled30s redo occurred. The patched readback therefore remains CPU-tested but unvalidated in native execution. This was a runtime correction test, not evidence of improved policy performance or a crash fix. Local applied ACKs do not independently prove server attack execution.
