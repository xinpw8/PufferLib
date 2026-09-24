# Persistent private client and native startup menus

One planned maintenance restart installed the reviewed startup bridge on 2026-09-24 at 12:08:52–12:08:55 UTC. The subsequent client stayed alive after completed controlled rounds instead of being recycled. Native Skip was accepted 13 ms after lease acquisition; the existing login callback observed Home 181 ms after Skip.

This fixes the deliberate per-round teardown and adds guarded native startup navigation. The original crash cause remains unknown. The diagnostic client later crashed during its third controlled attempt; this is not a crash-fix claim.

## Deployed changes

- `SkipIntro` requires the current pipe connection and exclusive lease, exact Spark Wine/X98 isolation, no active control or connected network session, active native Intro state, and a visible/enabled Skip button. It invokes the native Skip callback. Authentication is unchanged.
- `ExitUnsupportedPrivateAiPairing` requires the same Spark isolation and lease, idle controls, proven private unranked Bot1 with no human opponent, exact local G1/opponent T800 semantics and bone signatures, matching slots and client-visual fighters. It opens the native menu and requests Home. A Forfeit confirmation is invoked only for a revalidated scope with native pending Home, open Forfeit pane and visible/enabled confirmation button. This path subsequently completed live in `handoff-s1801-retry2`: native exit accepted at 12:26:13.921, native confirmation accepted at 12:26:14.062, Home observed at 12:26:14.398, then ordinary FreePlay/EnterSolo and a new private Bot1 stream. No process restart was requested. The separately hashed source-prefix receipt is `receipts/unsupported-home-recovery.json`; this is startup recovery evidence, not a scored result.
- The controller removes completed-round and unsupported-pair process recycling. Native round transitions and scoped menu recovery preserve the client. Automatic relaunch remains limited to an already absent game; its existing guard refuses prefix cleanup with a live game.
- The launcher adds `BOX64_SHOWSEGV=1` and `BOX64_SHOWBT=1`, retaining S2W0 and all other runtime settings. These flags request fault evidence and can affect timing. They do not establish a root cause or repaired crash behavior.

Both new commands are unavailable on the Windows execution surface. No general public-session forfeit, synthetic keyboard/mouse/gamepad input, policy/action-gate change, reward change or model update was added. The 500 ms action-age and 750 ms watchdog settings remain unchanged.

## First smoke result and measured handoff

The three-label runtime smoke closed at 12:22:16.590 UTC. These controller/summary outcomes are not an 18/20 evaluation or native score-packet audit:

| Attempt | Completed | Score | Observation |
| --- | --- | ---: | --- |
| persistent-s1701 | Yes | 2:14 | Original client preserved |
| persistent-s1702 | Yes | 4:13 | Same original client preserved |
| persistent-s1703 | No | 0:7 | Crash with 78.069824 s remaining |
| persistent-s1703-retry2 | Yes | 7:5 | One genuine crash recovery/relaunch |

Completed totals are 1W2L, 13:32. The policy remained checkpoint `f147bdc358261e272c58e70167cd8b0b9e4953806c417d8891d94a3ad083ac84`, with unchanged full observations/mask/attack set. Host PID 3089342 and namespace PID 54459 shared start ticks 93969025 across the first two completed controlled rounds. The 12:19:07 receipt observed the same process at approximately 609 s age; that receipt predates the subsequent crash. First-two source cadences were 24.357 and 24.320 Hz. None of these observations proves future stability or improved fighting.

The recorder's media cleanup delayed control handoff, so following native rounds were missed and the next controlled attempts began after normal menu recovery. The no-recycle change alone did not establish contiguous control of consecutive native rounds. The separately reviewed handoff source under `handoff/` subsequently completed two native round-1-to-round-2 pairs. Its four-label runtime smoke closed at 12:36:31.967 UTC.

`receipts/deployment.json` records the selected startup commands, exact summary hashes, closed smoke, maintenance timestamps and failure. Process receipts contain selected metadata only. The prepared-only README/build/staging receipts remain verbatim as historical preparation records and are superseded by the separate deployment receipt for execution status. No raw stack, register, game observation or proprietary recovered source is included.

## Endpoint-close handoff variant

The new handoff separates finished policy control from unfinished media encoding. It requires actual clean closes of relay, worker, encoder and driver, no sent termination signals, acknowledged stream stop/lease release, and exact checkpoint/config/driver/trial/summary bindings. A valid receipt permits one successor attempt while prior media finishes. A fresh worker and encoder reset recurrent/projection state; normal private-scope, zero-score and actual fair-start checks remain unchanged. The native completion predicate is byte-identical to the first persistent controller.

The ledger marks media pending, then updates the same record on wrapper completion. Late media failure remains a campaign failure at the next boundary without stopping an already active successor. Without a valid receipt, the controller uses the ordinary watcher; its initial poll is now immediate. Consecutive native round numbers are reported explicitly instead of assumed. The ordinary third-nonwin policy criterion remains unchanged; a separate explicit runtime-only smoke plan is limited to one through four rounds.

Twelve focused handoff tests pass, including an actual CPU child-close test, missing/wrong receipt, wrong identities, sent signals, late media success/failure, unchanged verdict and fresh-driver path. Tests do not establish real consecutive-round capture. The source and three patches are against the exact first persistent controller, startup driver and recorder pinned in `SOURCE-MANIFEST.json`.

### Closed four-label handoff smoke

`receipts/handoff-final.json` records all six attempts, the four completed outcomes, exact source hashes and final campaign/ledger hashes. Every completed attempt began with an actual 0:0 encoder input; its final native relay round exactly matched the terminal summary, inactive at 0 s, duration 120 s and no redo. The last encoder input remained preterminal, so this does not claim the terminal state was encoded. `receipts/handoff-execution.json` is retained as the earlier prefix-bounded receipt through 12:31:39.447 UTC, superseded by the closed receipt for final status.

| Completed attempt | Runtime / native round | First controlled time remaining (s) | Final score | Validated video bytes |
| --- | --- | ---: | ---: | ---: |
| handoff-s1801-retry2 | r98 / 1 | 119.699936 | 22:9 | 10,922,680 |
| handoff-s1802 | r98 / 2 | 119.799930 | 10:16 | 10,881,703 |
| handoff-s1803-retry2 | r99 / 1 | 118.683050 | 20:19 | 10,896,037 |
| handoff-s1804 | r99 / 2 | 119.599920 | 7:9 | 10,893,904 |

- `handoff-s1801-retry2` completed native round 1 at 22:9. Its policy stream stopped and lease released before encoder, worker and relay each closed with exit 0 and no sent signals. The driver-close handoff receipt arrived at 12:28:25.659 UTC; the successor launched at 12:28:25.678.
- `handoff-s1802` started native round 2 at 0:0 with 119.79993 s remaining at source UTC 12:28:34.7899444. Prior media finished at 12:28:29.205, with wrapper exit 0 and validated MP4. Round 2 completed at 10:16, and its media also finalized successfully. This is observed consecutive-round control.
- `handoff-s1803` started native round 3 at 0:0 with 119.71661 s remaining, then the client crashed. Its last measured state was 0:3 with 91.3091 s remaining. The controller recorded game death at 12:31:30.242 and one unplanned r99 relaunch. There is no completed third-round or repaired-crash claim.
- After crash recovery, `handoff-s1803-retry2` completed native round 1 at 20:19. Its clean driver-close receipt at 12:34:18.723 enabled the next handoff. `handoff-s1804` then completed native round 2 at 7:9. All four completed wrappers exited 0, all media finalized and all videos were below 20,000,000 bytes.

Final totals are 2W2L, 59:53, with two successful native 1-to-2 handoffs on two client processes, separated by one genuine crash and relaunch. The other incomplete attempt was an unsupported G1/T800 pairing with zero policy predictions. Saved launch stats identify namespace PIDs 54921 and 55454 with start ticks 94032484 and 94105181; the final proof also recorded host PID 3102832/start ticks 94105181. Continuity is supported by the campaign event chain and source clocks. No immutable contemporaneous JSON process sample exists for both r98 rounds, and none is invented here.

These are runtime findings with verified native terminal fields, not a strict native score-packet audit or policy acceptance result. The policy, schema, all-ones feature mask and full attack set remained unchanged. No crash cause or fighting improvement is established. `handoff/prepare_handoff_smoke.cjs` is the exact four-label runtime-only preparer; it refuses an existing output stage and verifies the frozen source/checkpoint/bridge pins before producing configs. It was copied for reproduction and was not rerun during publication.

## Source and exact baselines

`startup/` contains the original-authored new bridge helpers, current driver, pure guard tests and the small protocol/relay sources needed by those tests. `startup/integration.patch` patches the original bridge dispatcher, protocol and relay. Copy the two new bridge `.cs` files into the same project when reconstructing a private build. `startup/driver.patch` is against the original driver. `controller/` contains the deployed persistent controller, smoke preparer and controller patch/tests. `launcher/` contains the reviewed launcher, exact patch and historical one-time maintenance installer.

| Component | Baseline SHA256 | Deployed variant SHA256 |
| --- | --- | --- |
| Bridge DLL | `11fcfa2bea541f2829553c3833c301717c0182a59eebb0020f5f7b8cb066ca3d` | `5a2edac6c586f1ea401d92e0086ebfc468dbc591e2bb11115aef056682280e7a` |
| Driver | `b57654d54a76c80b38c9e44f59f53efb2502959a70dc568835ec5a8f6d238214` | `1fff2cb9f3ebc43fc3e9b27b1b390e17e1e8805f157c0cf5ab0be09cec5a2489` |
| Controller | `160299c466f6aeb8b0f9cebfd63132316f98625fe89af5f851a505472e44b151` | `bc164ed70fb323a781f0c24a0f9867b4c85e805d2dbd392c4b440463bb83e190` |
| Launcher | `ca556b4688ee2d14ea2e0e3baf0d0c23f20d3f79a1375943ff145c572f96793b` | `41d685616f90c1f78e70a744ee7e7059537ce33d2085cc4da87b96f8163483d7` |

The new relay executable SHA256 is `b7059eaf024c62465e752edd9b72a3026e6c67733052b8da181bd3f73bdaaa1f`. Binaries remain private. Exact source hashes and copy provenance are in `SOURCE-MANIFEST.json`.

Bridge baseline: `C:/rekagent/work/policy-timing-500-750-20260924-r1/variant/windows`, corresponding to the [timing variant](../policy-timing-500-750-20260924-r1/README.md). Driver/controller baselines: `C:/rekagent/work/persistent-private-session-20260924-r1/original_driver.cjs` and `original_campaign.cjs`. Launcher baseline: sibling `relaunch.baseline.sh`. The bridge build needs its existing BepInEx and private REK interop references; these are not redistributed. Production source in this repository remains untouched.

## Checks and safe reproduction

Executed against this publication: 12 startup/controller Node tests, 12 handoff tests, 30 C# guard checks, both launcher shell syntax checks, the handoff preparer's JavaScript syntax check, and all seven exact-baseline patch checks passed. Existing private bridge/relay builds had zero errors; their build receipt is preserved. These CPU checks do not substitute for the native UI observations above.

```sh
node --test startup/startup.test.cjs controller/persistent_campaign.test.cjs
node --test handoff/handoff.test.cjs
node --check handoff/prepare_handoff_smoke.cjs
dotnet run --project startup/tests/StartupMenuTests.csproj -c Release
bash -n launcher/relaunch.sh
bash -n launcher/install_once.sh
```

Use `git apply --check` for each patch in a disposable copy of its exact baseline. The dispatcher patch expects the `variant/windows/` layout, while the driver/controller/launcher patches expect their respective file at the copied stage root. No patch was applied to production source by this publication task.

The maintenance installer records the already executed one-time replacement and backup. It is not a recurring recovery command and must not be rerun against the running client. The launcher and smoke preparer are preserved reproduction source, not instructions to restart an active session. This publication performed no game interaction, GPU work or process control.
