# Diverse frozen policy evaluation

Selected checkpoint: mixed-r1, SHA256 `0d612521839298ffbe5783ed4fa449286a940b62a3a6768da7cc5ab248eb843b`. The subsequent long-r1 checkpoint is rejected by the measured comparisons below. Selection does not imply universal wins or authentic REK parity.

The evaluator measures raw combat performance separately from shaped training return. It links the exact native CUDA runtime objects used by the training build. Both fighter sides are evaluated, with terminal-triggered recurrent resets and explicit BF16 sampled policy inference.

Source tools are `../../diverse_policy_eval.cu`, `../../build_diverse_policy_eval.sh`, `../../run_diverse_policy_eval.sh`, `../../run_diverse_policy_suite.sh`, and `../../summarize_diverse_policy.cjs`.

## Predeclared suite

- Opponents: neutral, original scripted, retreat, strafe, and a hash-verified older learned checkpoint.
- Twenty-second cases: fixed starts and held-out seeded starts, 128 arenas and four rounds per fighter side, totaling 1,024 matches per condition.
- Three-hundred-second cases: fixed neutral and scripted opponents, 32 arenas and one round per fighter side, totaling 64 matches per condition.
- Held-out seed: 10001. Random reset root gap: 0.55–2.5 m. Heading spread: pi radians. These fixtures differ from seed73 training.
- Training potential shaping is explicitly disabled in every evaluation. Scores remain the raw integer contact scores.

The learned-opponent case uses two independent native policy instances with separate recurrent state and random streams. No policy rows are replaced by scripted actions. All state evolution, inference, score accumulation, and behavioral measurements execute on the GPU.

## Measurements

Each match records scores, result, duration, first-hit latency or null if no hit occurred, zero-hit status, root path, root gap, facing fraction, move-locked fraction, and attack commands. Each side's point deltas are accumulated on the GPU and must equal the terminal score exactly. Terminal accounting, missed rounds, invalid values, and metric inconsistencies fail evaluation.

First-hit latency is the first positive point delta. Root path begins at the first post-reset state, excluding cross-round teleport distance and omitting that round's first20ms displacement. Facing means logical-heading error below0.16rad. Gap below1.05m is a geometric descriptor, not certified attack reach. These compact state transitions are not authentic REK parity measurements.

## Initial close-range curriculum result

The first fresh close-neutral 33M checkpoint, SHA256 `93f767f6fcc72738cb5e310ef351f8d3f3635a2e8d70a6129d88857b408fa48a`, was screened before a complete suite. Each initial condition used 64 arenas and two rounds per side: 256 matches, seed10001,20seconds, shaping0.

| Condition | Wins / losses / draws | Raw points, policy / opponent | Zero-hit matches |
| --- | ---: | ---: | ---: |
| Fixed neutral | 0 / 0 / 256 | 0 / 0 | 256 |
| Fixed scripted | 9 / 199 / 48 | 75 / 287 | 195 |
| Held-out neutral | 22 / 0 / 234 | 319 / 0 | 234 |

This checkpoint is not accepted as a fix. It remains move-locked approximately97–98% of ticks and travels less than0.03m on average in these tests. The evidence is under `initial-close-stage/`. No readiness conclusion is based on its training return.

## Pursuit curriculum screening

The next checkpoint, SHA256 `49216e495051709e1d148f9e4ebc3a01b95ffadd87a0c85ef27821f6d2623e59`, completed its134,217,728-transition stage. The same three256-match screening conditions produced:

| Condition | Wins / losses / draws | Raw points, policy / opponent | Zero-hit matches |
| --- | ---: | ---: | ---: |
| Fixed neutral | 248 / 0 / 8 | 8,399 / 0 | 8 |
| Fixed scripted | 231 / 16 / 9 | 5,037 / 2,485 | 6 |
| Held-out neutral | 103 / 0 / 153 | 2,084 / 0 | 153 |

Fixed neutral first-hit latency averaged approximately4.35seconds among successful matches; mean root path increased to approximately1.78m. Held-out neutral paths averaged approximately9.89m, with only7.0% of ticks facing within0.16rad. This checkpoint substantially improves approach from the aligned fixed start, but does not satisfy reliable stationary-target performance across held-out headings. A full suite was not run on this screened-out candidate. Raw results are under `pursuit-stage/`.

## Orientation curriculum screening

The subsequent checkpoint, SHA256 `65d1d31e8d99df825a0db5e34b370a697b4b26476f9a73b8db63f7fed87080a8`, completed its268,435,456-transition stage. The same256-match screens produced:

| Condition | Wins / losses / draws | Raw points, policy / opponent | Zero-hit matches |
| --- | ---: | ---: | ---: |
| Fixed neutral | 256 / 0 / 0 | 16,362 / 0 | 0 |
| Fixed scripted | 240 / 5 / 11 | 9,259 / 3,627 | 1 |
| Held-out neutral | 256 / 0 / 0 | 14,721 / 0 | 0 |

Mean first-hit latency against neutral was approximately2.787s fixed and2.818s held-out. This screen shows successful approach and scoring across the tested headings. It does not certify arbitrary opponents or100%generalization. Strict logical-facing fractions remained approximately9% fixed and2% held-out; successful scoring is not evidence that the robot always faces the opponent. Logs are under `orientation-stage/`.

## Mixed-opponent full suite

The mixed-stage checkpoint SHA256 is `0d612521839298ffbe5783ed4fa449286a940b62a3a6768da7cc5ab248eb843b`. The complete predeclared suite finished with zero failure bits in all 10,368 matches. `mixed-stage/summary.json` contains the complete side-balanced raw-score and behavioral results.

| Opponent | Fixed 20 s W/L/D | Held-out 20 s W/L/D |
| --- | ---: | ---: |
| Neutral | 1,024 / 0 / 0 | 1,024 / 0 / 0 |
| Original scripted | 1,022 / 0 / 2 | 1,011 / 3 / 10 |
| Retreat | 1,021 / 0 / 3 | 1,017 / 0 / 7 |
| Strafe | 1,024 / 0 / 0 | 1,024 / 0 / 0 |
| Older V3 checkpoint | 1,023 / 1 / 0 | 1,020 / 4 / 0 |

The smaller 300 s fixed-start checks produced 64/0/0 against neutral and 61/3/0 against the original scripted opponent. Zero-hit matches against neutral were zero in all tested durations and reset distributions. All rewards used for evaluation were unshaped raw contact scores. These results are benchmark-specific; they do not establish universal 100% wins, human superiority, or authentic REK parity.

## Rejected longer-round fine-tune

The subsequent 134,217,728-transition fine-tune, SHA256 `b20831647814b0336e582ae3c513543d3b65d8064359e635e6b38b92001104e4`, regressed across the same suite. Its full evidence remains under `long-stage/summary.json`.

| Condition | Selected mixed-r1 W/L/D | Rejected long-r1 W/L/D |
| --- | ---: | ---: |
| Fixed neutral 20 s | 1,024 / 0 / 0 | 949 / 0 / 75 |
| Held-out neutral 20 s | 1,024 / 0 / 0 | 968 / 0 / 56 |
| Fixed scripted 20 s | 1,022 / 0 / 2 | 649 / 251 / 124 |
| Held-out scripted 20 s | 1,011 / 3 / 10 | 696 / 265 / 63 |
| Fixed scripted 300 s | 61 / 3 / 0 | 8 / 54 / 2 |

The head-to-head tests in `long-vs-mixed/` independently favored mixed-r1. From long-r1's perspective: fixed 20 s was 266/734/24, held-out 20 s was 291/723/10, and fixed 300 s was 20/44/0. All were sampled BF16, both fighter sides, seed 10001, zero shaping and zero failure bits. The longer-round training stage changed several training conditions together; this evaluation establishes regression without attributing it to one cause.

`selection.json` records the selected/rejected hashes and their raw conditions. `validate_evidence.cjs` checks the packaged summaries and produces `artifact-hashes.txt`. The marker-order regression test separately verifies that completed-training monitoring waits for a delayed warm-start verification file instead of incorrectly aborting; it invokes no GPU.

This directory includes derived measurement logs, commands, hashes, and process status only. Checkpoints, proprietary assets, and detailed private match records remain on Spark. No production interaction or deployment is performed by these tools.
