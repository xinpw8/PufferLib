# Score-credit 5 s: closed live cohort

Closed at 2026-09-24 09:21:20.233 UTC with **1 win, 3 losses and 30:45 points**. The original 18-wins-in-20 criterion failed at the third completed nonwin. There were 17 attempts: four counted completions and 13 excluded incomplete attempts. Seeds 1005 through 1020 were not reached. No consistent-beating or improvement claim is supported.

Source stage: `/home/spark-advantage/rek-training/scorecredit5s-live-20260924-r2`.
Frozen checkpoint SHA256: `0daf90a96442d541d38d9dd4f8fe765917c3c766ca80019855cfd7d358d9fe12`.
Planned policy seeds: 1001 through 1020, with the same checkpoint throughout.

## Counted rounds

Scores are learner:Bot1. All four ledger entries have `complete`, `terminal`, `same_bot`, and `fair_start` true, initial 0:0 scores, a 120 s non-redo round, and received terminal `WonByPoints` results at time zero. The existing G1-vs-G1 private-AI and fair-start requirements were retained.

| Seed | Counted attempt | Result | Points | Local applied ACKs | Attempt end UTC |
| --- | --- | --- | --- | --- | --- |
| 1001 | credit5s-s1001-retry3 | Win | 15:7 | 1701 | 08:49:50.657 |
| 1002 | credit5s-s1002-retry8 | Loss | 0:12 | 953 | 09:06:54.632 |
| 1003 | credit5s-s1003-retry2 | Loss | 2:11 | 1021 | 09:12:34.295 |
| 1004 | credit5s-s1004-retry4 | Loss | 13:15 | 2240 | 09:21:15.287 |

Local applied ACKs count accepted local commands; they do not establish server-executed attacks or scored-hit attribution. The ledger's `clean_hits` arrays are the point totals reported here, not a count of distinct contacts.

## Incomplete attempts

| Ledger stop reason | Count | Attempt suffixes, all prefixed credit5s- |
| --- | --- | --- |
| unsupported_pairing:local_g1_opponent_t800 | 7 | s1001; s1002, s1002-retry2, s1002-retry3, s1002-retry4; s1004, s1004-retry2 |
| stream_end:policy_action_watchdog_expired | 4 | s1001-retry2; s1002-retry5; s1003; s1004-retry3 |
| source_stream_missing | 2 | s1002-retry6; s1002-retry7 |

The two missing-source attempts were separately diagnosed as CoreCLR client exits with exit code 5. Their last observed scores were 1:7 with 58.28685 s remaining and 6:11 with 58.986782 s remaining. The ledger proves loss of the source stream and absence of terminal evidence; it does not identify the underlying crash cause. These partial scores are excluded from both W/L and the 30:45 total.

Retries retained the planned policy seed and checkpoint, but could encounter a different server round or client lifecycle. Excluding interrupted rounds and conditioning on eventual completion introduces selection bias, especially because both crash-interrupted attempts were behind. The four completed results are not an unbiased estimate of deployment win rate. Incomplete attempts are not complete training episodes.

No live range gate, forced attack, or attack cooldown was introduced. Existing estimated busy-duration masking is documented in [startup/README.md](startup/README.md).

## Provenance and archive

Authoritative files relative to the source stage:

- `root-campaign/ledger.json`: SHA256 `1069bcc2a335c9a77d07423b16fbbbdb98de5dff817a6df9cd28c0fc2386e433`.
- `root-campaign/campaign.log`: SHA256 `fa3a16a66b5fbbd1f2aae1d117558d8f3cf0b554f13a3169514a6f13bb888411`.

The campaign log records `criterion_failed` at 09:21:20.232 UTC and `campaign_end` at 09:21:20.233 UTC. This report supersedes the early live progress snapshot. The existing [ARCHIVE.md](ARCHIVE.md) receipt covers the separate training stage and remains unchanged.

The entire closed source stage was archived at 2026-09-24 09:32:41.412 UTC to:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\scorecredit5s-live-20260924-r2-closed-20260924T092120Z\evidence.tar.gz`

- Size: 120,398,104 bytes; 533 archive entries.
- SHA256: `e2cf623cfab8a5ad743359943402e5c2c445f9e8a757ec5892a28c96e1d1605e`.
- Transfer exit code 0; NAS readback hash and archive listing verified.
- The ledger and campaign-log hashes above were verified before and after transfer.
- Receipt: `receipt.json` alongside the archive; local copy at `C:\rekagent\work\scorecredit5s-publication-20260924-r2\closed-archive-receipt.json`.

Source files were retained. Raw evidence remains outside this publication directory. This archive covers the closed cohort stage; recorder files elsewhere on Spark are outside its scope.
