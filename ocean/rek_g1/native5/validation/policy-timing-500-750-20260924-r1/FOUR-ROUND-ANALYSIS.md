# Four-round cutoff, not a final cohort result

Cutoff: **2026-09-24 10:08:05.755 UTC**, the close of seed 1204. The first four counted rounds are **3 wins, 1 loss, 37:30 points**, with seven earlier incomplete attempts excluded. The cohort remains active; seed 1205 and later outcomes are outside this analysis. The earlier [two-round snapshot](LIVE-PROGRESS.md) is preserved. Four rounds do not establish consistent Bot1 wins or an isolated benefit from the timing change.

Checkpoint remains `5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4`; bridge DLL remains `11fcfa2bea541f2829553c3833c301717c0182a59eebb0020f5f7b8cb066ca3d`. Source stage: `/home/spark-advantage/rek-training/humanbc-timing500-live-20260924-r1`.

## Native point accounting

Scores and award counts are learner:Bot1. Each counted round retains fair-start, same-Bot1, 120 s non-redo and received terminal `WonByPoints` evidence. Native referee audits pass and native score packets reconcile exactly for all four.

| Completed attempt | Result | Total points | Ordinary +1/+2 points | +5 award counts |
| --- | --- | --- | --- | --- |
| timing500-s1201-retry8 | Win | 16:10 | 6:10 | 2:0 |
| timing500-s1202 | Loss | 6:13 | 6:8 | 0:1 |
| timing500-s1203 | Win | 4:3 | 4:3 | 0:0 |
| timing500-s1204 | Win | 11:4 | 6:4 | 1:0 |
| Four-round total | 3W1L | 37:30 | 22:25 | 3:1 |

Seed 1203 won on ordinary points alone. Seed 1204's learner ordinary points comprise four +1 awards and one +2 award; its +5 award accompanies an uncensored received `Knockout` call for opponent slot 1 at observed elapsed 113.464467 s. That call is not a terminal knockout result. The +2 award is not causally assigned to a requested move.

## Newly analyzed rounds

| Metric | s1203 | s1204 |
| --- | ---: | ---: |
| Locally accepted attack requests | 80 | 77 |
| Category 17 requests | 0 | 0 |
| Requests with abs(bearing)>pi/2 | 32/80 | 3/77 |
| Requests at root distance >1 Unity unit | 18/80 | 12/77 |
| Facing within pi/4, controlled QPC-time fraction | 47.67% | 61.13% |
| Median root distance across source observations, Unity units | 0.7514 | 0.7519 |
| Held translation, worker decision rows | 472/3064 (15.40%) | 566/3100 (18.26%) |
| Held translation, controlled QPC-time fraction | 16.59% | 18.49% |
| Attack-legal worker rows | 457/3064 (14.92%) | 125/3100 (4.03%) |
| Controlled source cadence, Hz | 25.8039 | 26.0949 |
| Source interval p95 / maximum, ms | 54.0808 / 129.0152 | 53.3569 / 151.3786 |
| Local applied ACKs | 3063 | 3099 |

Category 21 accounts for 60 of seed 1203's 80 attack requests. Seed 1204 has 23 category-21 and 30 category-23 requests. Category 17 was unused across all four rounds. The full category counts and request geometry are in the private report.

Held translation means retained ASDW, including combined translation/yaw categories; Q/E alone are not translation. Readiness counts use the actual saved worker masks. The remaining busy/transport restrictions are retained, not removed for this analysis. Request geometry is measured at the exact policy observation; an ACK is local acceptance, not proof of server execution, a hit, or a miss. Contact attribution and physical metre calibration remain unavailable. Time-weighted geometry holds each observation's classification until the next source QPC timestamp.

## Preserved evidence

New private report and derived data: `C:\rekagent\work\timing500-four-round-cutoff-20260924-r1\REPORT.md` and `closed-round-geometry.json`, SHA256 `0bb63ae112c1716d97056274b981084408fe0ff0d725688cbe0c6a29e05e5d9f`. The derived JSON covers only newly analyzed seeds 1203 and 1204, with native/relay/worker/encoder hashes, full point ledgers, category counts and request geometry. Its Spark copy is `/home/spark-advantage/rek-training/timing500-four-round-cutoff-analysis-20260924-r1/closed-round-geometry.json`.

Selected summary hashes: seed 1203 `91c4ebb7ea2727efd78712ba9856e59a06d1219784220a16fd7047c3209b55f8`; seed 1204 `4696d1d56b170afe762c80ca3ee417eae468a6df92631b607fb70529c9ced727`. The prior two-round derived JSON remains unchanged at SHA256 `75e76a3782bcebfbe3ac9829a8128dadda0274037c17e73c93713a44eca6188e`.

No active trial contents, game connection, GPU job, controller, configuration or policy was used or changed. Raw sources and earlier reports were retained.
