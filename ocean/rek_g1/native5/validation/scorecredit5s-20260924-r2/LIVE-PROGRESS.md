# Closed live evaluation

Closed: 2026-09-24 09:21:20 UTC. Final result: **1 win, 3 losses; 30:45 points**, with 13 incomplete attempts excluded. This supersedes the early 1W0L progress snapshot.

Stage: `/home/spark-advantage/rek-training/scorecredit5s-live-20260924-r2`.
Checkpoint: `0daf90a96442d541d38d9dd4f8fe765917c3c766ca80019855cfd7d358d9fe12`.
Fixed policy seeds: 1001 through 1020. The target of 18 wins in 20 completed rounds failed at the third completed nonwin. The controller stopped after seed 1004; seeds 1005 through 1020 were not reached. Incomplete attempts do not count as wins, losses, or complete training episodes.

| Seed | Counted attempt | Result | Points |
| --- | --- | --- | --- |
| 1001 | credit5s-s1001-retry3 | Win | 15:7 |
| 1002 | credit5s-s1002-retry8 | Loss | 0:12 |
| 1003 | credit5s-s1003-retry2 | Loss | 2:11 |
| 1004 | credit5s-s1004-retry4 | Loss | 13:15 |

Excluded attempts: seven unsupported G1-vs-T800 pairings, four startup watchdog failures, and two missing-source-stream failures associated with client crashes. Both crash-interrupted attempts were behind when observations stopped. Retrying incomplete rounds introduces completion-selection bias. These results do not establish improvement or consistent Bot1 wins. See [COHORT-FINAL.md](COHORT-FINAL.md) for the full accounting, provenance, and archive receipt.

No live range gate, forced attack, or attack cooldown was introduced. Existing estimated busy-duration masking remains and is documented in `startup/README.md`.
