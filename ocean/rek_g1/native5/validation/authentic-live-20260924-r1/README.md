# Authentic live evaluation, 2026-09-24

Offline analysis of completed cp056818f3 rounds `authentic-s801-retry9` and `authentic-s802-retry2`, private Bot 1, own slot 0. No game connection, GPU work, Windows launch, or runtime change.

| Recorded measurement | s801-retry9 | s802-retry2 |
| --- | ---: | ---: |
| Final own:Bot1 points | 3:12 | 13:14 |
| Own +1 / +2 / +5 counter changes | 3 / 0 / 0 | 3 / 0 / 2 |
| Bot1 +1 / +2 / +5 counter changes | 8 / 2 / 0 | 3 / 3 / 1 |
| Points excluding +5 changes | 3:12 | 3:9 |
| Own / opponent received count-flag exposure, s | 0 / 0 | 3.007 / 8.272 |
| Attack proposals / local acceptance / dispatch | 56 / 56 / 56 | 67 / 67 / 67 |
| Own absolute bearing at dispatch, median | 113.18 degrees | 118.91 degrees |
| Dispatches above 90-degree bearing | 29/56 | 42/67 |
| Median XZ root separation, Unity units | 0.679 | 0.803 |
| Source observations/s | 25.866 | 24.536 |

All **6,047** available referee payloads passed body-hash, field, and freshness checks. In the second round, received `DoubleKnockout` and opponent-faller `Knockout` calls corroborate the three +5 changes. These calls resolve count episodes; final result remains `WonByPoints`, with `knockout=false`. One earlier call-sequence gap is explicitly censored. Visual falling/fallen flags remain false throughout both rounds.

The requested straight-left kick is **Left Front Kick, action 17/native move 7: zero proposals, acceptances, or dispatches in either round**. Left Side Kick is action 16/native move 6: zero/two. Left Jab is a separate action 21/native move 1: three/eight. Most dispatches were Right Hook (21/23) and Six Punch (18/21). Full mix and mapping sources are in `analysis-details.md`.

Facing was checked against every recorded encoder-ready observation: maximum difference from feature `[87] * 180` was `1.14e-13` degrees. Existing native getter/G1 asset evidence supports projected root-local +X. The current live instance's forward offset was not directly sampled.

Limits: +1/+2 changes are sampled cumulative point-counter changes, not raw strike-event counts. Root gap is not collider clearance or proven reach. Pose at dispatch is 3.5 ms old at the median. Server acceptance, actual attack starts, playback, and causal hit/miss outcomes remain unknown. No forced action rule or reward conclusion follows from this report.

## Evidence

`analysis-details.md` contains methods, source paths, timestamps, complete attack mix, and interpretation limits. Per-round JSON/JSONL contains derived measurements only. `receipts.json` hashes eight source logs and thirteen original analysis artifacts; its original `README.md` entry corresponds byte-for-byte to `analysis-details.md` here. Scripts are offline only and refuse existing output files/directories. `native-windows-fallback.md` records the separate read-only fallback investigation and unproven Steam relaunch containment.

Raw private logs for all 13 attempts remain on NAS, outside Git:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\authentic-ppo-live-closed-20260924T0703Z\evidence.tar.gz`

Independently verified: 134,524,269 bytes; SHA256 `0008d1980ff5603086c2b3595674bead845ce75205823d0b2e7f8f1a003816fd`.
