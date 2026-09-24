# Actual REK live-validation results, 2026-09-24

One completed private G1 round against authentic Bot1 ended **12:19, loss**. The second round was interrupted by a runtime crash at **1:7**, with **69.30366 s remaining**. No complete match result was observed. This block does not demonstrate a live improvement or justify promotion.

## Scope and policy

- Stage: `/home/spark-advantage/rek-training/native-hparam-live-retry-20260924-r1`.
- Candidate SHA-256: `fa6f760a5823d904b9f636df6421426e21344c03767ffa8ed0fe776c400b75ad`.
- Observation schema: `rek.native5.scaled_polar_xy.v1`, all-ones feature mask.
- Local slot 0; authentic private G1, exact sparring Bot1, no human opponent.
- Four rounds planned, two attempted, one completed. One continuous game process and native between-round handoff. No runtime restart after the crash.

## Confirmed live results

| Trial | Controlled start UTC | Result | Applied actions | Armed attack requests |
|---|---|---|---:|---:|
| hparam-s1901 | 17:08:41.440 | Round 1 loss, 12:19, timer expired | 2,991 | 59 |
| hparam-s1902 | 17:10:50.118 | Round 2 incomplete, 1:7, 69.30366 s left | 1,274 | 24 |

Round 1 had 2,992 predictions and one terminal-race rejection, `policy_stream_not_owned`. Its final source receipt was 17:10:40.0268525 UTC. Stop and lease release succeeded, endpoints closed cleanly, and the controller preserved the game. Round 2 had 1,274 predictions and zero rejections; its last source receipt was 17:11:40.3668776 UTC. Its final summary reason is `source_stream_missing`.

Completed-round aggregate: **0 wins, 1 loss; 12 own points and 19 conceded; own points/round 12; margin/round -7**. The partial second round is excluded. The last native fight receipt reports rounds won `[0,1]`, result `InProgress`, winner `-1`. **Completed matches: 0; match win rate: unknown/null.**

## Point and action evidence

Adjacent-state point increments, in observed order per side:

- Round 1 own: `1,1,5,1,1,1,1,1`; opponent: `2,5,1,1,2,2,2,2,1,1`.
- Round 2 own: `1`; opponent: `1,2,1,1,1,1`.

Round 1 therefore contains one observed +5 increment for each side. This is not, by itself, a physical-knockdown count. Both round summaries retain falls `[0,0]`. The final round-1 referee receipt contains a latched `Knockout` call, faller 1, points 5. Its schema explicitly states that a Knockout call does not imply a terminal round KO, and provides no attacker/action causality. The completed round explicitly reports `WonByPoints`, knockout false.

Action 16 was selected **0 times in round 1 and 1 time in round 2** across all worker decisions. Its round-2 request was `accepted_locally_and_armed` at 17:11:31.1318655 UTC. No move name is inferred without confirming the exact active bridge map. Armed requests establish local command acceptance, not scoring hits. Full 33-action histograms are in the companion JSON.

## Runtime failure

Runtime PID 3162378, start-time ticks 95712930. Existing launcher receipts record **2026-09-24T17:11:44Z, exit code 5**.

Filtered native stderr evidence:

- Line 24219: `SIGSEGV`, access address `0x5d415c415d5b66`, x64 PC `0x6fff9639de27`.
- Subsequent Wine record: `EXCEPTION_ACCESS_VIOLATION`, code `c0000005`.
- CoreCLR 6.0.722.32202 / .NET 6.0.7 reports process termination from an internal .NET runtime error at IP `00006FFFF51B1FDD`, exception `c0000005`, followed by fail-fast/unhandled-exception records.

Runtime evidence directory: `/home/spark-advantage/codexrook-runtime/live-transfer-20260915/live-attack-gate-20260921-native-hparam-baseline-20260924-r1`. Exact exit time comes from `game.exited.utc.txt`; the SIGSEGV line itself has no UTC timestamp. This fault address differs from the previously reported old-Box64 `0x80` signature. A shared underlying cause and policy causality remain unproven.

## Existing video evidence

Both videos passed the existing full decode check, exit code 0, and are below 20,000,000 bytes. No new capture was performed for this report.

| Trial | Coverage | Bytes | SHA-256 |
|---|---|---:|---|
| hparam-s1901 | Completed round | 10,882,438 | `7c7c65dbe7f426702e1011135db1941f9d1e634be6525279490bc3d86e4c13de` |
| hparam-s1902 | Partial round before crash | 4,792,699 | `d94f61b85e0f98b4b7da72565191ec85a582d5888cfab63b8f6ac218141fc7fd` |

Stage-relative paths: `hparam-s1901/media/authentic-rek-policy-fight.mp4` and `hparam-s1902/media/authentic-rek-policy-fight.mp4`.

## Provenance and limits

Results were reconstructed read-only from native relay state/action receipts, worker decisions, trial summaries, campaign log, runtime stderr/exit receipts, and existing capture manifests. Source relay SHA-256 values and structured metrics are in `LIVE_RESULTS.json`. No credentials, full Unity logs, or account identifiers are included.

Only one completed round was obtained, with no live baseline block. Simulator improvements cannot establish a live improvement. The runtime crash prevented the planned four-round evaluation and any completed-match measurement.
