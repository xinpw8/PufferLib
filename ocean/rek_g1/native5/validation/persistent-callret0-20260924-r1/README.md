# CALLRET=0: closed runtime test

`BOX64_DYNAREC_CALLRET=0` did not prevent a crash. The verified initial client exited 5 at 2026-09-24 12:56:43 UTC after attempting execution at a null address. The origin of that control transfer remains unknown. This differs from the prior bad-pointer read faults and does not identify a faulty component.

The four-label runtime-only plan was stopped after the failed hypothesis and completion of the already-owned round. Two full rounds completed: one 11:11 tie and one 12:10 win, aggregate 23:21. This is not a completed four-label test, policy acceptance result or evidence of improved fighting.

## Preserved attempts

| Attempt | Outcome | Evidence |
| --- | --- | --- |
| callret-s1801 | Complete, tie 11:11 | Native 120 s round 1; zero-score start, terminal native fields verified |
| callret-s1802 | Excluded, zero predictions | Native round 2 was a 30 s redo; full-round startup contract rejected it |
| callret-s1802-retry2 | Crash, incomplete 5:3 | Native round 3, 85.6994 s remaining; 317 predictions/applied ACKs |
| callret-s1802-retry3 | Excluded, zero predictions | Unsupported G1/T800 pairing after automatic crash recovery |
| callret-s1802-retry4 | Complete, win 12:10 | Recovered from child-owned terminal/media receipts after outer-controller retirement |

One actual crash caused one automatic relaunch, r100. The two completed native terminal rounds exactly match their driver summaries. Both began at 0:0, with 118.46533 and 118.46494 s remaining respectively. Both completed videos validated below 20,000,000 bytes. Local applied ACKs do not prove server execution or individual attack scoring; no full native score-packet audit is claimed.

## Timing and runtime identity

The first process was host PID 3110517, namespace PID 56250, start ticks 94209601. Saved process receipts verify CALLRET=0, STRONGMEM=2, WEAKBARRIER=0 and the existing fault-log flags on the exact isolated REK target. The r100 launch-script copy has the same CALLRET=0 launcher hash; its replacement-process environment was not independently sampled by these receipts.

The first completed round had 1,071 native source states at 9.026324 Hz, p95 interval 128.656 ms and Unity-time/wall-time ratio 0.999015. The crash partial had 318 source states at 9.354589 Hz, p95 123.9687 ms. The previous instrumented baseline's first two completed rounds measured 24.357 and 24.320 Hz. This substantial cadence difference confounds policy comparisons. These observations do not identify the cause of the slowdown.

The original fault log is private; `receipts/FAULT-FINDINGS.md` records its SHA256 and selected interpretation without raw registers or stack dumps. The earlier host-resource schema-v1 process CPU/RSS parser was faulty for multiword names; those fields are not used here. No resource-saturation or crash-mechanism conclusion is inferred from a single snapshot.

## Stop and recovered final round

Only controller PID 3110969/start ticks 94233108 was verified paused at 12:59:35.248 UTC. That receipt records `already_stopped=true`; the initial SIGSTOP occurred during an earlier invocation whose immediate state assertion failed before saving a receipt. Its exact signal timestamp is not established. The owned wrapper continued. At 13:01:01.937 the controller was retired with SIGKILL after the wrapper was a zombie with kernel exit 0, the driver and all three endpoints had closed cleanly, and MP4 validation had completed. The game was not signalled.

The original ledger still contains only the first completed round. It was not rewritten, the original four-label plan was not shortened, and the stale lock was preserved. `retired-controller.json` binds the final 12:10 summary SHA `63e7cbd13b0472d27a9a91910d518fed8dc678936493194995118725427f202f` and validated video SHA `eb8b5a2e01403c0da21f27136b137c4a9f6035d0e3cfc051370cd9a3bab44895`. Parent-owned wrapper stdout may lack an unread tail because the parent was paused; child-owned terminal/media receipts and kernel exit establish closure. `receipts/runtime-result.json` preserves this distinction and the source hashes.

## Source and checks

The exact launcher and complete one-line diff are under `source/`. Its parent is the [persistent-client launcher](../persistent-private-session-20260924-r1/launcher/relaunch.sh), SHA256 `41d685616f90c1f78e70a744ee7e7059537ce33d2085cc4da87b96f8163483d7`. The variant SHA256 is `452876dff3e6f29284bce16c0ea7c53eedd558dfb743b6fda8c47e7623bfc400`. Removing precisely the added CALLRET assignment reconstructs the parent byte-for-byte. All other launch settings, guards, bridge and game pins remain unchanged.

The policy stayed frozen f147 with the balance8 schema, all-ones feature mask and all 17 attacks. The [endpoint-close handoff source](../persistent-private-session-20260924-r1/README.md) was reused. The staging receipt remains verbatim preparation history, superseded here for execution status.

Focused CPU checks:

```sh
node --test launcher.test.cjs
bash -n source/relaunch.sh
```

The launcher is preserved reproduction source, not an instruction to restart the current game. This publication changed no production source, original ledger, plan, runtime, policy or active game. No raw fault log, game binary, full observation stream or proprietary source is published. No archive or GPU work was performed for this publication.
