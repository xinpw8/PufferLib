# Private-AI policy timing experiment

Installed on the dedicated Spark client at **2026-09-24 09:51:34 UTC**. Root reported the prospective cohort launched at 09:52 UTC. **Live completion is not yet verified by this package.** CPU tests and installation do not prove game success or improved fighting.

## Change and containment

Action freshness increases from 250 to 500 ms. The established-action watchdog is separated into a 750 ms constant. Startup grace remains 1 s; the unchanged host still discards predictions taking 200 ms or longer. Hello capabilities advertise both budgets. All lease, identity, private-arena, account, opponent and isolated-control checks remain unchanged.

`source/` contains the two variant production files. `variant.patch` shows their changes and the four updated expectations in the existing relay tests. Original production files remain untouched in this repository. This DLL is operationally restricted to the dedicated Spark experiment; its constants apply to every G1 policy stream if the DLL is used elsewhere.

Installed DLL SHA-256: `11fcfa2bea541f2829553c3833c301717c0182a59eebb0020f5f7b8cb066ca3d`. Original `ea8511a87b9e456547f13ffdb2f9af5e941b8b04f97bf40419d38e1c9b31ff3e` was backed up outside the plugin directory. Hashes of all 590 dedicated runtime files were compared: only this DLL changed; 589 others stayed identical. See `receipts/installation-receipt.json`. No DLL, proprietary reference assembly, credentials, raw game log or dump is included here.

## Evidence and tests

The preceding BC attempt averaged 6.889 source Hz, declining from 10.70 to 4.87 Hz across successive wall-time windows. Its final action was processed one Unity frame after the source, at 262.3265 ms age; worker inference took 2.428 ms. The original watchdog then stopped 278.5292 ms after the preceding accepted action. Pipe-arrival timing was not recorded, so the transport/queue split is unknown.

The variant built with zero warnings/errors. **13,718 checks passed**: focused timing 40; relay 6,447; isolation/authenticated continuation 5,930; referee 860; protocol 441, plus a test-only local pipe roundtrip. Cohort preparation adds two passing Node tests. Focused tests cover exact 500/501 ms and 750/751 ms boundaries, unchanged startup grace, rejected identity/clock cases, a CPU model proving stale rejections do not refresh the accepted clock, and subsequent fresh-source eligibility. Guard-source hash equality is recorded; these tests do not exercise Unity lease transitions.

Run the focused CPU tests with `dotnet run --project TimingTests/TimingTests.csproj -c Release` from this directory. Run preparation tests with `node --test cohort/prepare_live.test.cjs`. `cohort/prepare_live.cjs` is a preserved Spark-only preparation script, not a command to run against an existing cohort. `receipts/verification-receipt.json` is the historical pre-install build receipt; the installation receipt records the later approved deployment.

## Prospective evaluation and limits

The frozen checkpoint remains `5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4`; seeds are 1201–1220. Driver, controller, worker, encoder, recorder, masks and S2W0 settings remain unchanged. Only the bridge timing/advertisement and exact plugin pins change. Reviewed preparation code and compact metadata are under `cohort/`; live outcomes must be recorded separately.

750 ms is checked on Unity Update, not by an independent real-time timer. It cannot guarantee wall-clock neutralization during a stalled frame. The change permits longer stale-action age and held-input duration; it does not restore prior 25/48 Hz cadence, explain the slowdown or repair CoreCLR crashes. Stale actions remain unapplied and do not refresh the watchdog. The host can accept a subsequent fresh source without replaying the rejected action. Recurrent history remains continuous; unapplied rows receive zero actor weight. Earlier runtime-aborted partials remain incomplete, including their actual score snapshots.
