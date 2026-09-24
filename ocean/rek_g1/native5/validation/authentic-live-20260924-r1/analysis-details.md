# Completed authentic policy rounds: recorded-log analysis

2026-09-24. CPU-only offline analysis. No game or relay connection, GPU work, Windows launch, input, or runtime configuration change.

Analyzed checkpoint `056818f3947c3e7efb1a8050521cfed72f6147c2444b21b02c7aa7ef0166b5d7`, local fighter slot 0, private Bot 1, two completed 120-second rounds. Inputs are under `/home/spark-advantage/rek-training/authentic-ppo-live-20260924-r1/{authentic-s801-retry9,authentic-s802-retry2}/trial`.

## Main measurements

| Measurement | s801-retry9 | s802-retry2 |
| --- | ---: | ---: |
| Final points, own:Bot1 | 3:12 | 13:14 |
| Own observed +1 / +2 / +5 counter changes | 3 / 0 / 0 | 3 / 0 / 2 |
| Bot1 observed +1 / +2 / +5 counter changes | 8 / 2 / 0 | 3 / 3 / 1 |
| Remaining point increments after excluding +5 | 3:12 | 3:9 |
| Own / opponent received count-flag exposure, s | 0 / 0 | 3.007 / 8.272 |
| Model attack proposals / local accepted / dispatch returns | 56 / 56 / 56 | 67 / 67 / 67 |
| Own absolute bearing at dispatch, median | 113.18 degrees | 118.91 degrees |
| Dispatches with own absolute bearing above 90 degrees | 29/56 | 42/67 |
| Dispatches with own absolute bearing above 45 degrees | 39/56 | 52/67 |
| XZ root separation at dispatch, median | 0.679 | 0.803 |
| Dispatches at root separation above 1.0 | 7/56 | 19/67 |
| Source observations/s | 25.866 | 24.536 |
| Applied local actions/s | 25.850 | 24.520 |
| Maximum source interval, s | 0.1018 | 0.1360 |

Separation uses the existing 1 Unity unit = 1 candidate metre convention. It is not a fresh physical scale calibration, collider clearance, strike reach, or contact test. Geometry is the nearest preceding recorded pose at dispatch receipt: median age 3.49/3.50 ms, maximum 6.16/23.20 ms. This is measured dispatch context, not an authoritative server attack-start pose.

The policy requests many attacks with substantial measured bearing offset. The close second score is substantially affected by referee awards. These are observed mechanisms worth testing; these two rounds do not establish why the policy chooses those actions, causal contact outcomes, a reward-design defect, or a corrective control rule.

## Referee and scoring evidence

All 6,047 source observations contain available referee data. The analysis validates the 33-byte body SHA256, decoded count and call fields, hook/provenance labels, and QPC freshness. There are 1,199 and 1,198 distinct recorded referee receipts. This is validation of the bridge-recorded received body, without an independent native recorder capture for these two rounds.

Calls are deduplicated by round/lifecycle/call-observation identity, not counted once per latched sample. In s801-retry9 there are no count flags or referee calls. In s802-retry2:

- Opponent count begins at receipt elapsed 50.147 s with `Slip(faller=1)`.
- Both counts are observed at 52.432 s with `DoubleKnockdown`; the call sequence jumps 1 to 3, so one intermediate call is unobserved and this call's history is censored.
- A new uncensored `DoubleKnockout(points=5,faller=-1)` at 55.439 s clears both count bits and carries counters 7:9. Both preceding +5 counter changes have a unique nearby matching call and exact new counters.
- Opponent count begins again at 114.892 s with `Slip(faller=1)`.
- A new uncensored `Knockout(points=5,faller=1)` at 117.882 s clears that count and carries counters 13:14. The own +5 counter change has a unique nearby matching call and exact new counter.

Three observed fighter count episodes resolve in two explicit countout calls: own 3.007 s; opponent 5.291 s and 2.989 s. Final round result remains `WonByPoints`, `knockout=false`. A referee `Knockout` call is not synonymous with terminal-round knockout. Visual `falling` and `fallen` flags remain false for both fighters in every sample, so those visual flags do not substitute for received referee count state.

Point +1/+2 entries are changes between observed cumulative `round.clean_hits` counters, which hold awarded points. They are not raw award packets or confirmed counts of distinct strikes; multiple awards within one observation interval cannot be excluded. Five-point associations are corroborated by explicit received calls, counter agreement, and a 0.25-second receipt window. No causal hit or executed action is inferred. Count exposure is left-held between source observations; precise server onset/deadline times are unavailable.

## Attack mix and the requested straight-left kick

Counts below are identical across model proposals, locally accepted requests, and matching dispatch returns.

| Native registry move | Policy action | s801-retry9 | s802-retry2 |
| --- | ---: | ---: | ---: |
| Left Front Kick, native 7, requested straight-left kick | 17 | 0 | 0 |
| Left Side Kick, native 6 | 16 | 0 | 2 |
| Left Jab, native 1 | 21 | 3 | 8 |
| Right Hook, native 3 | 23 | 21 | 23 |
| Left Jab Right Uppercut, native 5 | 25 | 4 | 7 |
| Six Punch, native 10 | 26 | 18 | 21 |
| Run And Punch, native 11 | 27 | 1 | 0 |
| Left Hook Right Jab, native 14 | 30 | 9 | 6 |

Every other discrete attack category has zero proposals. The left-front kick's absence is already present at model proposal output, so it was not lost between a proposal and bridge dispatch in these rounds. The left jab is a separate registry move and must not be labeled as the requested kick.

Mapping source: `C:\Users\Daniel\codex-rek-puffysics-training-profile\ocean\rek\evidence\windows\RekUiBridgeAgent\G1PolicyStreamContract.cs:78`. Registry names: `C:\Users\Daniel\codex-rek-puffysics-training-profile\ocean\rek_g1\native_motion_routes.c:49` (left side), `:56` (left front), `:82` (left jab), `:93`, `:103`, `:109`, `:116`, `:137`.

Actual server starts remain unknown: all 6,047 samples lack local runner move identity, motion name, and native action-busy status. Local acceptance and `SendMoveEvent` return do not prove server acceptance, playback, hit, or miss. In s802-retry2, one dispatch occurred while the own received count flag was active and six while the opponent count flag was active.

## Facing convention cross-check

For normalized Unity XYZW quaternion `(x,y,z,w)`, use projected root-local +X:

`forwardXZ = (1 - 2*(y*y+z*z), 2*(x*z-w*y))`

`bearing = wrap(atan2(opponentZ-ownZ, opponentX-ownX) - atan2(forwardZ,forwardX))`

This equals `encode_live.cpp:57-59,104,219`: Unity XYZ to common XZY and XYZW to WXYZ `(-w,x,z,y)`. Empirically compared all 3,102 and 2,941 recorded ready observations against `worker_request.observation[87] * 180` and `[86]`: maximum angular disagreement `1.14e-13` degrees, maximum separation disagreement `8.89e-16`.

Existing source/asset proof: `C:\Users\Daniel\codex-rek-puffysics-training-profile\ocean\rek_g1\native5\validation\observable-balance-heading-20260921.md:14`. Recovered `Robot.Forward` negates `RootTransform.right` then applies serialized G1 `forwardYawOffset=-180`, yielding horizontal root +X. That proof matches the pinned assembly/assets and rejects an arbitrary 90/180-degree correction. It does not directly read the current live instance's offset or equate rendered root heading with server/controller heading during motion.

## Artifacts and reproducibility

Remote analysis directory: `/home/spark-advantage/rek-training/authentic-ppo-live-20260924-r1/analysis-s801r9-s802r2-20260924-r1`.

Local copy: `C:\rekagent\work\authentic-live-failure-analysis-20260924-r1`.

- `analyze_completed.py`: score changes, independently checked referee payloads and deduplicated calls/count episodes, dispatch geometry and mix. Fails if its output directory already exists.
- `verify_geometry.py`: numerical comparison against recorded encoder input features and worker proposals; fails on existing result files.
- Per-round summary JSON, geometry-verification JSON, dispatch JSONL, and score-delta JSONL contain derived fields only. Input file hashes are included.
- `native-windows-fallback.md`: separate read-only fallback investigation with source references and containment limits.

Referee decoding and deduplication semantics were checked against `/home/spark-advantage/rek-training/referee-bridge-spark-20260924-r1/validate_live_referee.cjs` and `native_referee_data.cjs`. No independent native-capture correlation is claimed.
