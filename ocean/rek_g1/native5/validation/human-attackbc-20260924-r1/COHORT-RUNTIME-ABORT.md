# Human-attack BC: live cohort stopped for runtime failures

The cohort was explicitly stopped after three incomplete watchdog failures. **Zero rounds completed. No win/loss result or completed-round point total is available.** These attempts must not be reported as three losses or as evidence of improvement.

Source stage: `/home/spark-advantage/rek-training/human-attackbc-live-20260924-r1`.
Checkpoint SHA256: `5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4`.
All three attempts used policy seed 1101 and that checkpoint. Planned seeds 1102 through 1120 were not reached.

## Attempt accounting

Every ledger entry has `same_bot` and `fair_start` true, but `complete` and `terminal` false. Each last observed round remains active with result `InProgress`; the scores below are partial learner:Bot1 points, not outcomes.

| Attempt | Attempt end UTC, 2026-09-24 | Partial points | Time remaining, s | Predictions | Local applied ACKs |
| --- | --- | --- | --- | --- | --- |
| humanbc-s1101 | 09:39:19.248 | 5:19 | 4.4454923 | 787 | 786 |
| humanbc-s1101-retry2 | 09:42:24.242 | 5:3 | 82.80826 | 200 | 199 |
| humanbc-s1101-retry3 | 09:44:03.053 | 0:0 | 113.20902 | 36 | 35 |

All three stop reasons are `stream_end:policy_action_watchdog_expired`, with one rejected action apiece. The operator-stop receipt is timestamped 09:44:24.953 UTC: the already-started third attempt was allowed to close before the controller was stopped. This is an interrupted campaign, not a normal criterion-completion event. The original 18-of-20 criterion was not evaluated by completed results.

The preserved `root-campaign/campaign.lock` contains PID 2886688. A read-only `/proc/2886688` check found no process before archival. The stale lock was retained; no lock cleanup or runtime change was performed for this report.

## Measured timing failure

The final rejected ACK in each attempt is `stale_observation`. Its source and ACK are on consecutive Unity frames:

| Attempt | Source / ACK frame | Source age at ACK, ms | Final worker latency, ms |
| --- | --- | --- | --- |
| humanbc-s1101 | 3466 / 3467 | 262.3265 | 2.428346 |
| humanbc-s1101-retry2 | 9192 / 9193 | 252.3643 | 1.175821 |
| humanbc-s1101-retry3 | 12218 / 12219 | 258.4892 | 1.465487 |

The first attempt has 788 controlled-stream source observations, 6.888698 Hz measured from QPC, a median interval of 157.1666 ms and a p95 interval of 217.1594 ms. Its worker latency p95 was 3.030562 ms, maximum 3.705385 ms. These measurements support a frame/dispatch timing incompatibility with the 250 ms freshness contract; a long model inference is not observed at the rejected actions. They do not identify the underlying cause of slow Unity frames or establish GPU-scheduling causation. No timeout, freshness, watchdog, fairness, or policy rule was changed in this cohort.

Timing sources are each attempt's `trial/relay.stdout.jsonl`, `trial/worker.stdout.jsonl`, and `trial/summary.json`. QPC differences measure source age; worker latency is the worker's recorded inference/request latency.

## Facing-sign audit, separate baseline cohort

A read-only audit of the earlier closed `scorecredit5s-live-20260924-r2` cohort found no evidence of a reversed sign mapping. Category 6 maps to Q/positive yaw; 7 maps to E/negative yaw in `G1PolicyStreamContract.cs:126` and `fast_runtime.cu:259,314`. The live encoder's signed bearing is opponent direction minus actor heading (`live_transfer/encode_live.cpp:219`). All 1,022 ready encoder outputs in `credit5s-s1003-retry2` matched independently derived bearing/pi at exact source QPC, maximum difference `9.99e-16`, with no sign disagreements.

For sustained pure-turn requests lasting at least 0.4 s, balanced poses and more than 3 s since an own attack request, six category-7 snippets (34 observations, 3.253 s) all had negative measured heading change. No category-6 snippet met those criteria. A shorter 0.2467 s category-6 snippet had positive heading change but was close-contact confounded. Contact, opponent motion, and unavailable server attack state limit causal interpretation. This is evidence about the earlier checkpoint's mapping and recorded geometry, not a result for the human-attack BC candidate or a guarantee of correct turn selection.

## Provenance and preservation

Authoritative files relative to the source stage:

- `root-campaign/ledger.json`: SHA256 `53a9360519f6bbe17d2399867ae48961e410c1c699f84ee947b513515e5e0c71`.
- `root-campaign/operator-stop.json`: SHA256 `d0dda4db674adb01eeef96c8546a7f76036128bd5643e2db27bf716476da704f`.
- `root-campaign/campaign.log`: SHA256 `f971712a87948e9995deedf82877596e26143e38f005de913f9d41c389644560`.
- `root-campaign/campaign.lock`: SHA256 `8576240bdd912f9c8b6f255285435d5206254bcc8caf9f5ea2f635127f6ccce0`.

The entire interrupted source stage was archived at 2026-09-24 09:49:01.359 UTC to:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\human-attackbc-live-20260924-r1-interrupted-20260924T094424Z\evidence.tar.gz`

- Size: 23,450,997 bytes; 150 archive entries.
- SHA256: `75503f3019c75b7e1f7beec4eb6f23bd2b95902185b5708136e3b92d1f658d4d`.
- Transfer exit code 0; NAS readback hash and archive listing verified.
- All four source hashes above were unchanged before and after transfer; PID 2886688 was absent at both checks.
- Receipt: `receipt.json` alongside the archive; local copy at `C:\rekagent\work\human-attackbc-publication-20260924-r1\closed-archive-receipt.json`.

Source files and the stale lock were retained. This archive covers the cohort stage; recorder files elsewhere on Spark are outside its scope. Existing training receipts and publication files remain unchanged; this live cohort is separate from the training archive in `RESULTS.md`.
