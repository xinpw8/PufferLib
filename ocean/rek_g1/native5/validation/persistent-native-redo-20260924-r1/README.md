# Native redo and known-private-AI runtime continuation

These patches extend the committed persistent-session handoff sources. They retain the same client, policy checkpoint, observation/action path, source freshness, watchdog, no-human proof, transport-close proof, and media validation. No game binary or recovered game body is included.

## Behavior

1. Driver config `play_native_redo: true` permits the native 30 s redo contract. It still requires an active untouched 0:0 round, measured clock/identity, warmup within 2.5 s, and controlled startup within 3 s. The default remains exactly 120 s/nonredo. Warmup latches the native round kind; it cannot change at stream start.
2. A completed 30 s redo is an auxiliary result. It never fills a planned 120 s label or enters that win rate. Its clean handoff continues with the same pending label, seed and checkpoint. Every attempt consumes the existing finite retry budget. The finite campaign ends after its last planned regular label; it does not drain an additional final tiebreaker.
3. The controller-only any-AI diagnostic requires all of `plan.runtime_only: true`, `plan.runtime_any_ai: true` and `AB_RUNTIME_ANY_AI=1`. Runtime plans remain bounded to 1..4 regular rounds. Default Bot1 policy cohorts retain their old verdicts and criterion.
4. The any-AI watcher reuses the unchanged driver's `privateArena(rawState)` proof. Its separate `canReadyPrivateAiSession(rawState)` proof is permitted only for inactive Idle before fighter bindings exist. Both preserve the native isolated private/no-human checks. No added command or input is emitted by the watcher.
5. `same_bot` still means Bot1. `eligible_opponent`, measured difficulty/bot identity, and `runtime_any_ai` are distinct fields. Final runtime metrics separate each opponent and separate regular 120 s from auxiliary 30 s results. The existing Bot1-only G1/T800 recovery is unchanged.

Driver, controller and recorder endpoints retain their existing lifetime contracts. The passive launch watcher resolves after its owned relay's `exit` event, as before this patch. The fast policy handoff separately requires the existing actual endpoint `close` receipts. The publication test title states this distinction explicitly; it has no runtime effect.

## Reconstruct and test

From this directory run:

```text
node verify.cjs
```

This requires Node.js and Git. The verifier checks the four committed source hashes in the sibling `persistent-private-session-20260924-r1/handoff` directory, copies them to a new OS temporary directory, applies the three patches, checks exact reconstructed runtime hashes, and runs all 22 CPU tests. It prints the retained scratch path. It does not launch a policy worker, bridge, game, controller, or GPU workload. The repository and private/live runtime files remain untouched.

Patch order: `redo-controller.patch` and `redo-driver.patch` apply to the committed handoff controller/driver. `runtime-any-ai.patch` applies only to the reconstructed redo controller. Recorder and handoff helper stay byte-identical. Test source is the staged test source; only the published any-AI test title changes “closes owned relay” to “observes owned relay exit”.

## Runtime source pins

| Source | Baseline SHA256 | Result SHA256 |
|---|---|---|
| Redo driver | `f6d31931186da3dce3621b4a952b7e853055446c8ab5aa934060d3263cfb6136` | `f1f25159c5155f15ce0a6035236c684fcc4f62f8f9843c54884bebdc59a33dab` |
| Redo controller | `fe839414eca8d366297e877988b569e714ac30b6d0cd1176063ca683f3da4b8e` | `5c0af77071fe561e9da70fc41c130adcebab3dc157927673bafbd7b29fbd22b7` |
| Any-AI controller | `5c0af77071fe561e9da70fc41c130adcebab3dc157927673bafbd7b29fbd22b7` | `77326a4d18296af696715c6c29e2335afe212545ef03c30be5ca58750361d926` |

Unchanged recorder: `9a29c88f30886654f5afe57e573245e2005ef790909ae656ec67114c19902e07`. Unchanged handoff helper: `482096b5b0a4e7473ee2f2ec3b9e96036bb4159954d560d1e23a67292d5daa72`. Frozen checkpoint: `f147bdc358261e272c58e70167cd8b0b9e4953806c417d8891d94a3ad083ac84`.

## Native observations, interim cutoff

The EventPipe diagnostic stage `/home/spark-advantage/rek-training/persistent-baseline-eventpipe-20260924-r1/live` started `2026-09-24T13:09:50.941Z`, plan SHA256 `a51834d43b18b487d253c61dbd080ac9411a4099202d429488bf0810948ef052`. Its `trace-s1801` round was a valid native 120 s/nonredo Bot2 round: initial 119.64988 s and 0:0; final 11:21 at 0 s. It recorded 5,888 predictions, 5,887 locally applied ACKs, and one rejection. Source warmup used four samples over 325 ms. Control began at 13:10:07.056Z and stop was acknowledged at 13:12:06.719Z, approximately 49.2 predictions per wall-clock second. These ACKs do not independently prove server attack execution. Startup was already at Login; no Skip command occurred.

The old Bot1-only controller excluded that result and then ignored the private Bot2 boundary. Original ledger and evidence were preserved. Its watcher was closed without restarting the client. The any-AI stage `/home/spark-advantage/rek-training/persistent-runtime-any-ai-20260924-r1/live` started `2026-09-24T13:18:28.984Z`, plan SHA256 `0281a096e0d84212d13eda02987929f9dc97e219b064348e1a790badecee5e9c`, on the same client.

At the 13:19:51Z transcript cutoff, the new controller accepted native Idle and later BetweenRounds. Attempt one failed the existing ready timeout with zero policy actions: native Bot3 spawned a G1/T800 mixed pair, so active-gameplay proof remained false despite the visual pair becoming available. Attempt two ended the existing active-round timeout. No crash or controller input fault is established by those timeouts. The general known-bot watcher does not broaden the exact G1 execution requirement or the Bot1-only mixed-pair recovery.

The reconstruction command actually reproduced all three pinned runtime hashes and passed all 22 CPU tests. Native opted-in 30 s control remains unobserved at this cutoff. These are runtime diagnostics with multiple instrumentation changes, not a policy comparison, a consistency claim, or a demonstrated crash fix.
