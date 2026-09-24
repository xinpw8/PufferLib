# First three counted live results

At the closed-round cutoff after `balance8-s903-retry3` on 2026-09-24, the candidate had **1 win, 2 losses, 27:39 points** against private Bot1. The campaign continued beyond this report's cutoff. Three rounds do not establish efficacy.

| Closed attempt | Seed | Result | Starting seconds | Worker inputs | Locally applied | Armed attacks |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| balance8-s901-retry4 | 901 | loss 5:13 | 119.88331 | 3,128 | 3,127 | 85 |
| balance8-s902-retry3 | 902 | loss 9:23 | 119.76656 | 2,921 | 2,920 | 65 |
| balance8-s903-retry3 | 903 | win 13:3 | 119.79991 | 3,080 | 3,079 | 37 |

All three were completed 120-second, non-redo rounds starting at 0:0, with terminal `WonByPoints` results. Frozen checkpoint: `9a875c347b512bb46dde25481be75d276ea1b4c0799886b30aaf03be6ae1c5ce`; schema: `rek.native5.scaled_polar_xy.balance8_v1`; all-ones feature-mask SHA256: `59158bfdf9ddb9a38686f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b`.

## Actual policy inputs and actions

All **9,129 worker inputs** matched their encoder outputs across all 223 FP32 cells and every action-mask entry. Each encoder manifest, worker ready message, and action carried the expected schema/checkpoint/mask identity. Worker seeds were checked against 901, 902, and 903. Each round had one rejected final request (`policy_stream_not_owned`), for 9,126 locally applied actions in total.

Each cell below is the observed FP32 minimum to maximum, followed by the nonzero count. Denominators are the worker-input counts above.

| Input | s901: range; nonzero | s902: range; nonzero | s903: range; nonzero |
| --- | --- | --- | --- |
| 9: actor vertical-root rate | -5.573691 to 9.451688; 3,128 | -3.436294 to 13.134561; 2,921 | -1.043296 to 2.351991; 3,080 |
| 95: opponent vertical-root rate | -1.663713 to 1.722613; 3,128 | -4.483126 to 8.452108; 2,921 | -2.935716 to 14.775940; 3,080 |
| 72: actor root tilt / pi | 0.001584 to 0.704354; 3,128 | 0 to 0.753765; 2,920 | 0.000651 to 0.111545; 3,080 |
| 158: opponent root tilt / pi | 0.000332 to 0.279761; 3,128 | 0 to 0.586904; 2,920 | 0.000160 to 0.537574; 3,080 |
| 202: fresh referee available | 1 to 1; 3,128 | 1 to 1; 2,921 | 1 to 1; 3,080 |
| 203: observation history available | 1 to 1; 3,128 | 1 to 1; 2,921 | 1 to 1; 3,080 |
| 204: actor count active | 0 to 1; 83 | 0 to 1; 150 | 0 to 0; 0 |
| 205: opponent count active | 0 to 0; 0 | 0 to 1; 78 | 0 to 1; 77 |

These are measured input values. LeftFrontKick category 17 was sampled, locally applied, and locally armed **zero times in all three rounds**. It was legal in 134, 194, and 84 input masks, respectively, totaling 412. All 17 attack categories remained enabled, with no range gate, cooldown masking, or forced attacks. There were 187 locally armed attack requests; local acknowledgment does not prove server execution.

## Received points and referee calls

The existing native score/referee analyzer was reused with only selected trial and capture paths changed. It validated all **9,135 bridge observations** (3,130 / 2,923 / 3,082) against the respective completed native captures, including exact received packet and clock bindings. Every native score receipt had a unique matching bridge counter increment, and received totals matched the final scores.

| Round | Actor received awards | Opponent received awards | Awards below five points | Five-point receipts, actor:opponent |
| --- | --- | --- | --- | --- |
| s901 | 5 x 1 | 6 x 1 + 1 x 2 + 1 x 5 | 5:8 | 0:1 |
| s902 | 4 x 1 + 1 x 5 | 5 x 1 + 4 x 2 + 2 x 5 | 4:13 | 1:2 |
| s903 | 8 x 1 + 1 x 5 | 1 x 1 + 1 x 2 | 8:3 | 1:0 |
| Total | 27 points | 39 points | 17:24 | 2:3 |

Explicit uncensored received calls, with elapsed round seconds and native actor slot 0 / opponent slot 1:

- s901: actor `Knockdown` at 67.072758, then actor `Knockout` at 70.073532. Its separate opponent five-point score increment appeared at 69.973476.
- s902: opponent `Slip` at 56.220980 and `Knockout` at 59.221348; actor `Slip` at 98.394577 and `Knockout` at 101.412160; actor `Slip` at 110.331821 and `Knockout` at 113.333929.
- s903: opponent `Slip` at 112.879040 and `Knockout` at 115.880212.

Each listed Knockout call reported five points. No attack-causes-fall attribution is made. Referee receipt time and score receipt time do not supply a shared causal event ID. Mid-round Knockout calls did not make these terminal match-KO results.

## Excluded redo

Earlier attempt `balance8-s901-retry3` completed a **30-second redo**, native round number 2, losing **0:3** with 711 predictions and 710 locally applied actions. It is recorded as observed but excluded from the prespecified 120-second non-redo criterion. Zero-action infrastructure attempts are also uncounted.

## Compact evidence

- s901: [receipt](live-eval-s901-r4/live-eval-s901-retry4.json), [input/identity inspector](live-eval-s901-r4/inspect_first_round.cjs), [reused point analyzer](live-eval-s901-r4/analyze_points.cjs).
- s902: [receipt](live-eval-s902-r3/live-eval-s902-retry3.json), [input/identity inspector](live-eval-s902-r3/inspect_closed_round.cjs), [reused point analyzer](live-eval-s902-r3/analyze_points.cjs).
- s903: [receipt](live-eval-s903-r3/live-eval-s903-retry3.json), [input/identity inspector](live-eval-s903-r3/inspect_closed_round.cjs), [reused point analyzer](live-eval-s903-r3/analyze_points.cjs).

Receipts include full-precision ranges, identities, point/call records, hashes, and exact paths for closed source files. Inspectors assert encoder-to-worker equality, legal sampled actions, action acknowledgment joins, recorded seed/schema/checkpoint/mask, closed source stability, received referee validation, and point totals. The repeated excluded-redo field in each receipt refers to the same s901 attempt and is counted only once here.

This follow-up used no GPU, bridge connection, game input, or runtime changes. All 63 files in the frozen source manifest were preserved. No later round is included.
