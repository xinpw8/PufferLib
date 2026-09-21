# Keyboard-yaw stride-5 development, r50–r53

The frozen stride-5 candidate completed three authentic private Sparring Bot 1 rounds: **1 win, 2 losses, 0 draws; received points 30:44**. All three passed the unchanged strict checks. The preceding r50 entry failure is preserved separately, with no assigned win, loss, or draw. This is development evidence, with no acceptance or improvement claim.

Checkpoint SHA256: `dafda776f7898bd26f5b99f313168656a61436ada999b700e0799f75f5d01bf8`. It was freshly trained from the same parent and native training budget as the prior keyboard-reset checkpoint, with action stride 5. Live configuration SHA256 was `914d9d539fea0ac849e89241ee61e971369ec292cc70702f0c7b1e02cac74638`. The encoder used explicit `--action-stride 5`; the worker, sampled seed 73, v1 observations, 223 unmasked features, BF16/native CUDA, and private-AI route were retained. The candidate and configuration remained frozen through all four attempts. Stride governs action eligibility, while observation/prediction traffic continues at the existing base cadence.

## Preserved entry failure

r50 prepared native inference but never began policy streaming. The driver exited 2 after `private-practice entry timeout`, with zero sources, predictions, applied actions, or initial/final round evidence. Its owned PID 198016 was closed before analysis. `EnterSolo` returned the local reason `private_practice_reservation_requested` at 23:43:41.865 UTC; timeout was recorded at 23:44:26.685 UTC. That local acknowledgement does not prove a server reservation. Passive triage found no fight coordinator; the actual cause of entry failure is unknown.

The original records and a separate failure summary were preserved. Full-round validators were not run on this non-round. The declared collection was extended through r53 to obtain three completed rounds without changing the candidate or discarding r50.

## Received outcomes and requested actions

| Round | Status | Own:opponent points | Own non-five + five-point total | Opponent non-five + five-point total | Attack requests | Right-hook category 23 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| r50 | Entry failure | Unavailable | Unavailable | Unavailable | 0 | 0 |
| r51 | Loss | 11:19 | 1 + 10 | 9 + 10 | 43 | 26 |
| r52 | Loss | 2:12 | 2 + 0 | 7 + 5 | 58 | 26 |
| r53 | Win | 17:13 | 2 + 15 | 8 + 5 | 48 | 11 |
| Completed total | 1W, 2L, 0D | 30:44 | 5 + 25 | 24 + 20 | 149 | 63 |

All 149 attack requests were locally applied, returned true from `ExecuteMove`, had a native dispatch return, and had one matching outbound move-request projection. None requested LEFT_FRONT category 17. These observations do not establish server acceptance, playback, contact, hit, miss, or trip. Received five-point awards are not assigned to preceding attacks.

The [prior stride-1 keyboard-reset development rounds](yaw-command-development-r37-r42.md) were 2W, 1L with 46:36 points; its separate [frozen evaluation](keyboard-yaw-frozen-evaluation.md) failed the predeclared criterion at 4W, 3L. These small chronological cohorts are kept separate. The new stride-5 result does not establish improved fighting or isolate a causal effect of stride from the separately trained actor.

## Pre-request geometry and coverage

Geometry was available for every attack. Gaps retain captured Unity numeric units because physical metre calibration is unverified. Absolute bearing projects rendered root-local +X into Unity XZ; it is not authoritative controller heading or contact-time alignment. Medians use linear interpolation at `(n−1)q`.

| Round | Median gap, captured units | Median absolute rendered bearing | Predictions / locally applied | In-flight sources skipped | First applied after initial observation, s | Maximum applied-action gap, s | Maximum referee receipt age, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r51 | 0.740413 | 42.7596° | 5880 / 5879 | 27 | 0.200284 | 0.058470 | 0.126216 |
| r52 | 0.673308 | 78.0671° | 5838 / 5837 | 35 | 0.199865 | 0.065393 | 0.122037 |
| r53 | 0.781305 | 35.8973° | 5848 / 5847 | 40 | 0.207836 | 0.070539 | 0.137541 |

Each completed round had one terminal-race rejection; all attack requests were locally acknowledged. Across them, 17,671/17,671 source referee payloads were available and verified. r53 retains one explicitly censored call-sequence gap, so complete call history is not claimed. The largest last-applied-to-terminal gap was 0.001788 s.

The existing referee and contact tools exited 0 after each exact owned client closed. Owned-PID capture provenance, frame/QPC/root-position binding, complete native capture, terminal consistency, full-policy coverage, and received point-counter reconciliation from zero all passed. Existing limits remained 1 s for coverage and 0.5 s for referee receipt age. No gate, runtime, reward, or schema was changed during validation; analysis used no GPU or game connection.

## Reproduction and provenance

Private root: `C:\rekagent\work\consistent-fighter-20260919-r1`. Original attempts are `live-round_outcome_v1-r50` through `r53`. Each completed round contains `referee-validation` and `contact-analysis`. The originally named `keyboard-yaw-cadence-development-r50-r52-r1` directory was retained when the collection extended: it contains the r50 failure summary, r51–r53 derived summaries and stdout/stderr, and `validate-completed-round.ps1 -Round NN`. That wrapper reuses the unchanged validators and private `left-front-development-r30-r32-r1\summarize-round.cjs`, refusing existing output.

Exact completed-round captures are under `C:\rekagent\evidence\runtime\rek-private-ai-protocol-v7`:

| Round | Capture filename | SHA256 |
| --- | --- | --- |
| r51 | `rek-private-ai-root-motion-20260920T234624.3608393Z-pid83448-32b185e12e594c98a7f12ccf499ff89b.jsonl` | `bbb78699b0de5f163ee10c2fce4c1873adabc5848234d7f45b3441675174b49b` |
| r52 | `rek-private-ai-root-motion-20260920T234944.8438214Z-pid314380-b22c5f84df744fc4b5e77eb206d570d5.jsonl` | `ef64cacf27745bbe91bfeda9458e603308ceff7f9ba2b5713888b84aa5a1a92a` |
| r53 | `rek-private-ai-root-motion-20260920T235313.4097721Z-pid359124-63f8508a5c1a4c009b497b28e1693a1d.jsonl` | `fd97c5901e77555f7226a253bb0d38488358630731bd667526b5754edf6e9c32` |

| Artifact | SHA256 |
| --- | --- |
| Existing referee validator | `22bbb6652f8cf333c1c832f8719f9e37da20b66bfecd85185e228d2183c9fe89` |
| Existing contact analyzer | `f0bfc8117d62442c7b3f9dfcf50cb2084c324561826c9c5bb7b301c54b9310c4` |
| Unchanged private summary helper | `7a2cd333999dc4114ac600d39c660b6820654f3801db4a0c29b717de1da62706` |
| r50 result | `d28da1ec16f74ffa8dc81d9f0593ec270a333337c3f06882600a9e647d19e688` |
| r50 failure summary | `c8aedd68d09b122b6cad8f3481218c56d8e4f918445bc1f3c34a788e836c9244` |
| r51 derived summary | `9bad6ffd3f5780cc6b9ccaabc7ff5e4acc7f0e9971580f487f5c5b1f1b9eb31f` |
| r52 derived summary | `65edb6391a21fef92a7fdd0b8f0783e53493c159fbc9f07dd08c1f4e7ee2312b` |
| r53 derived summary | `0de1c5e3f9b0922ea46137d0dda1b79e4cafc7fcc1aaf8ce12d1b89be3ece223` |

Raw state records and account information are excluded from this report.

The four attempts, completed native captures, training results and stride
measurements were copied to the existing private evidence server under
`2026-09-19/consistent-fighter-r1/windows-keyboard-cadence-development-r1`.
All 221 files (1,211,167,009 bytes) matched source-before, destination and
source-after SHA256 checks. The archive manifest SHA256 is
`7551acb011c7fa102fc734c52f397970dd34db858508e641b8ccac4252e2eb55`.
Original files were preserved.
