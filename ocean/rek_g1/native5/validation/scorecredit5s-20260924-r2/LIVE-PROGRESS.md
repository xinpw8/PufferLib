# Prospective live evaluation

Cutoff: 2026-09-24 08:52:14 UTC. Cohort is still running.

Stage: `/home/spark-advantage/rek-training/scorecredit5s-live-20260924-r2`.
Checkpoint: `0daf90a96442d541d38d9dd4f8fe765917c3c766ca80019855cfd7d358d9fe12`.
Fixed policy seeds: 1001 through 1020. Target: 18 wins in 20 completed rounds, stop after the third completed nonwin. Incomplete attempts do not count as wins, losses, or training episodes.

| Seed | Counted attempt | Result | Points |
| --- | --- | --- | --- |
| 1001 | s1001-retry3 | Win | 15:7 |

At cutoff there were three excluded attempts: two server-spawned G1-vs-T800 pairings unsupported by the current encoder, and one freshness/watchdog failure. G1-vs-G1 private AI scope and received terminal score evidence are required. Seed 1002 is underway. A single completed win does not establish improvement or consistent Bot1 wins.

No live range gate, forced attack, or attack cooldown was introduced. Existing estimated busy-duration masking remains and is documented in `startup/README.md`.
