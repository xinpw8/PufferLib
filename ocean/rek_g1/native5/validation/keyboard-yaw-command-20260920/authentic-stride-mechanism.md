# Reset-trained stride1 versus stride5: authentic command behavior

Stride5 produced substantially longer held turns and larger outgoing yaw
commands in the three completed development rounds. The fighting results did
not improve in this sample. This establishes the intended command change,
without establishing a benefit to contact accuracy or winning.

The comparison uses reset-trained stride1 r38/r40/r42, checkpoint
`85808a6731f4f40f5756800a9edf93faa5c312aa7d5f7468cafdd4ae3a5c381f`,
and reset-trained stride5 r51/r52/r53, checkpoint
`dafda776f7898bd26f5b99f313168656a61436ada999b700e0799f75f5d01bf8`.
No legacy velocity-slew checkpoint is included. Both use keyboard-reset training;
stride5 also applies the corresponding five-ready-observation live decision
cadence. Thus this is a comparison of the trained policies with their intended
live cadence, not a factorial separation of training and deployment cadence.

Attempt r50 failed before gameplay with zero policy sources, predictions and
actions. It remains a recorded infrastructure failure and contributes no command
samples. Every selected client was closed before measurement began.

## Held turns and outgoing requests

| Round | Stride | Retained sign-run median, ms | Retained runs >=500 ms | Nonzero outgoing yaw median | Nonzero yaw p95 | Commands with absolute yaw >=0.9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r38 | 1 | 34.542 | 0 | 0.068653 | 0.217053 | 0 |
| r40 | 1 | 33.705 | 2 | 0.068736 | 0.269103 | 37 |
| r42 | 1 | 34.575 | 1 | 0.069728 | 0.263138 | 9 |
| r51 | 5 | 194.681 | 43 | 0.286118 | 1.000000 | 783 |
| r52 | 5 | 192.770 | 38 | 0.276341 | 1.000000 | 610 |
| r53 | 5 | 129.667 | 31 | 0.237604 | 0.999531 | 303 |

| Pooled measurement | Reset stride1 | Reset stride5 |
| --- | ---: | ---: |
| Ready source rows | 17,473 | 17,566 |
| Retained nonzero sign runs | 5,346 | 1,003 |
| Retained sign-run median / p95, ms | 34.317 / 118.023 | 191.526 / 706.663 |
| Retained runs below 100 ms | 4,916 | 123 |
| Retained runs >=500 ms | 3 | 112 |
| Outgoing movement commands | 19,565 | 20,152 |
| Commands with nonzero yaw | 12,935 | 14,046 |
| Absolute nonzero outgoing yaw median / p95 | 0.069034 / 0.245635 | 0.271373 / 1.000000 |
| Commands with absolute yaw >=0.9 | 46 | 1,696 |
| Native completed sign-run median / p95, ms | 34.141 / 112.406 | 136.247 / 596.385 |
| Native completed sign runs >=500 ms | 3 | 94 |
| Outgoing commands containing translation | 2,349 | 6,101 |
| Observed source time / projected-busy time, s | 360.957 / 301.145 | 358.732 / 194.625 |

The retained median increased about 5.58 times and outgoing nonzero-yaw median
about 3.93 times. Every stride5 round showed the longer-turn tail, including both
losses. Retained desired input and outgoing native input remain distinct: busy
suppression, request timing and zero commands can split a native sign run while
desired yaw remains held. That explains why their run counts need not match;
neither is a direct measurement of physical rotation.

## Existing attack context

All six finalized contact summaries report zero category17 requests, so their
`remaining_attacks` group includes every attack request.

| Round | Stride | Attack requests | Median gap, captured Unity units | Median absolute rendered-root bearing, degrees |
| --- | ---: | ---: | ---: | ---: |
| r38 | 1 | 90 | 0.839140 | 46.5961 |
| r40 | 1 | 66 | 0.721273 | 63.2000 |
| r42 | 1 | 78 | 0.797086 | 49.5253 |
| r51 | 5 | 43 | 0.740413 | 42.7596 |
| r52 | 5 | 58 | 0.673308 | 78.0671 |
| r53 | 5 | 48 | 0.781305 | 35.8973 |

There is no consistent reduction in the rendered-bearing proxy: stride5 r52
has the largest median of the six rounds. Stride5 requested 149 attacks versus
234 under stride1, while translation requests increased and projected-busy time
decreased. Positions, attack mix and trajectories also differ. The observed
change therefore includes more than yaw amplitude.

The [outcome report](../consistent-fighter-20260919/keyboard-yaw-cadence-development-r50-r53.md)
records stride5 1W/2L with 30:44 points. Its own points comprise 5 from
non-five-point awards and 25 from five-point awards. The earlier stride1 development
cohort was 2W/1L with 46:36 points, comprising 16 from non-five-point awards and 30
points from five-point awards. Longer commands alone did not produce better
observed results. Three rounds per arm do not establish a stable treatment effect.

## Methods, limits and evidence

The adaptation preserves the existing
[yaw measurement](../consistent-fighter-20260919/yaw-cadence-evidence.cjs)
formulas. Retained yaw uses desired held categories, preserving category0 holds.
Runs end at changed sign; gaps above 250 ms, nonincreasing timestamps and the
final right-censored run are excluded. The first observed sign starts a run and
can left-censor an already active turn. Quantiles use sorted index
`floor((N-1)*p)`. Arm samples pool within-round runs without joining across
round boundaries. Ready snapshots can miss intermediate desired-input changes.

Native prefixes are matched to the strictly preceding ready source within 50 ms.
Six prefixes per arm were unjoined and remain only in overall native statistics.
Busy is saved `dispatched_request_v4_duration`, not authoritative playback or
the bridge's local busy flag. Request invocation and projected bodies establish
neither server execution nor physical turning/contact. Attack context is copied
unchanged from validated summaries and retains their linearly interpolated
quantiles. Rendered pelvis/root +X bearing is not independently verified
controller heading, and captured Unity distance units lack physical calibration.
No score is assigned causally to a particular preceding attack.

Private directory:
`C:\rekagent\work\consistent-fighter-20260919-r1\yaw-command-stride-mechanism-r1`.
JSON outputs retain every source path, byte count and SHA256 value, including
result/ownership records, encoder streams, captures and existing contact summaries.
No live stream or human recording was read.

- `measurement-r1.json`: `6ebe345ef4746048a9a128773de7f8c5b0fabf3c075853dbefaa5d407136ca65`.
- `measure-closed-reset-stride.cjs`: `c64380dafb1a849f2a851e3f9a8a24287195910616e7a1bb264502b694d9e7fa`.
- `attack-context-r1.json`: `b20718b93f09bbe044de7cb14b17e2c1e7317aa765c2d22b6b7ba6a5910c5a56`.

The three new native-capture hashes are:

| Round / PID | SHA256 |
| --- | --- |
| r51 / 83448 | `bbb78699b0de5f163ee10c2fce4c1873adabc5848234d7f45b3441675174b49b` |
| r52 / 314380 | `ef64cacf27745bbe91bfeda9458e603308ceff7f9ba2b5713888b84aa5a1a92a` |
| r53 / 359124 | `fd97c5901e77555f7226a253bb0d38488358630731bd667526b5754edf6e9c32` |
