# Live progress: two completed rounds

Cutoff: **2026-09-24 10:02:27 UTC**. The cohort remains active. At this cutoff: **1 win, 1 loss, 22:23 points**, plus seven incomplete attempts excluded. This supersedes the README's pre-completion status; two rounds do not establish improvement or consistent Bot1 wins.

Stage: `/home/spark-advantage/rek-training/humanbc-timing500-live-20260924-r1`.
Checkpoint: `5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4`.
Bridge DLL: `11fcfa2bea541f2829553c3833c301717c0182a59eebb0020f5f7b8cb066ca3d`.

| Counted attempt | Outcome | Total points | Ordinary +1/+2 points | +5 awards, learner:Bot1 |
| --- | --- | --- | --- | --- |
| timing500-s1201-retry8 | Win | 16:10 | 6:10 | 2:0 |
| timing500-s1202 | Loss | 6:13 | 6:8 | 0:1 |

Both are fair-start, 120 s non-redo G1-vs-G1 Bot1 rounds with received terminal `WonByPoints` evidence. Independent native score packets reconcile exactly to both totals, and both native referee validations pass. The three +5 awards have uncensored received `Knockout` calls; these calls do not mean the rounds ended by knockout. Both learner ordinary totals consist of six +1 awards, with no learner +2 award.

The seven excluded attempts precede seed 1201's completion: one private-practice entry timeout, five reservation-already-in-progress failures, and one unsupported G1-vs-T800 pairing. No watchdog stopped either counted round. Each has one non-applied terminal action ACK, not a watchdog failure. Seeds 1203 onward are outside this cutoff report.

Measured controlled-stream cadence was **23.76 Hz** and **24.95 Hz**, respectively; p95 source intervals were 57.13 and 54.67 ms. Local applied ACK counts were 2820 and 2963. Locally accepted attack requests numbered 63 and 61; category 17 was unused in both. These are requests, not confirmed attack executions or successful contacts.

Root-facing error was within pi/4 for 52.77% and 46.71% of observed controlled wall time. Median root separations across source observations were 0.8251 and 0.7357 Unity units, without a metre calibration claim. At request-observation time, 22/63 and 23/61 attacks had abs(bearing) greater than pi/2. Geometry describes observed states; it does not prove that a request caused a later score or missed.

The faster observed cadence and two completions do not isolate a timing-change effect from client lifecycle or workload changes. Read-only resource snapshots began around 10:00 UTC; they do not establish causation. The earlier runtime-aborted cohort stays excluded.

Private derived report: `C:\rekagent\work\timing500-closed-two-20260924-r1\REPORT.md` and `closed-round-geometry.json` (SHA256 `75e76a3782bcebfbe3ac9829a8128dadda0274037c17e73c93713a44eca6188e`). Full per-request geometry, native point ledger, source paths/hashes and category counts remain outside Git. Raw sources were retained.

The cutoff ledger SHA256 was `e2fea85b374b867f96f395078ba8eb2e1e417d7654e63e5500565c5e94edc652`. The active ledger may subsequently change. See [ATTACK-READINESS.md](ATTACK-READINESS.md) for the separate closed-cohort legality audit.
