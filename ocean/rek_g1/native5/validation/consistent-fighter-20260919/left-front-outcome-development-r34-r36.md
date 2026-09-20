# Left-front outcome-PPO development rounds r34–r36

The updated frozen checkpoint completed three authentic private Sparring Bot 1 rounds: **1 win, 2 losses, 0 draws; received points 40:47**. It requested LEFT_FRONT category17 twenty times among 243 attack requests (8.230%). All twenty were acknowledged locally and observed in native sends. The [parent's prior development cohort](left-front-development-r30-r33.md) was also 1W–2L, with 41:34 points. These small chronological cohorts do not establish improvement, consistent winning, or kick-specific efficacy.

Updated checkpoint SHA256: `add8d59367caf2487a550308dcf535409da9dbdbb4fe52e57cd6b4aba2ce168f`. Its authentic outcome-PPO training followed the parent exploration checkpoint's r31–r33 collection. The updated checkpoint stayed frozen across r34–r36; these rounds used unchanged v1 observations, 223 unmasked features, BF16/native CUDA, sampled seed73, private-AI route, and 20 ms target cadence. No command-ramp or other concurrent engineering change was applied to these rounds.

## Outcomes, awards, and requests

| Round | Result | Own:opponent points | Own non-five + five-point total | Opponent non-five + five-point total | Left-front / all attacks | Right-hook23 requests |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| r34 | Loss | 3:15 | 3 + 0 | 10 + 5 | 6 / 70 | 54 |
| r35 | Loss | 7:10 | 7 + 0 | 10 + 0 | 6 / 96 | 82 |
| r36 | Win | 30:22 | 10 + 20 | 12 + 10 | 8 / 77 | 50 |
| Total | 1W–2L–0D | 40:47 | 20 + 20 | 32 + 15 | 20 / 243 | 186 |

All three completed by points, not terminal knockout. r34's received referee records show one local countout. r35 has no observed referee calls/count episodes. r36 contains two opponent `Knockout` calls and two `DoubleKnockout` calls, corresponding to four opponent and two local count-mask resolutions. Its two call-sequence gaps remain explicitly censored; this is not a complete call-history claim. The received score packets independently reconcile fully to the final totals, including r36's four own and two opponent five-point awards. No fall or award is assigned to a preceding attack.

All 243 attack requests, including the 20 left-front and 223 remaining attacks, were locally applied with `ExecuteMove` returning true. Each has a native dispatch return and one matching outbound move-request projection. Those facts do not establish server acceptance, playback, executed contact, hit, miss, or trip. The observed 20/243 request fraction is distinct from the earlier fixed-history probability-mass calibration.

## Pre-request geometry

Ground-plane gaps below retain captured Unity numeric units; physical metre calibration is unverified. Absolute bearing uses rendered root-local +X projected into Unity XZ, not authoritative controller heading or contact-time alignment. All values were available; medians use linear interpolation at `(n−1)q`.

| Round | Left-front median gap | Remaining attacks median gap | Left-front median absolute bearing | Remaining attacks median absolute bearing |
| --- | ---: | ---: | ---: | ---: |
| r34 | 0.884266 | 0.708125 | 52.2718° | 100.0419° |
| r35 | 0.727655 | 0.650414 | 83.9382° | 78.0677° |
| r36 | 0.626396 | 0.756589 | 15.5114° | 22.4631° |

These are request-context measurements. They do not explain the score causally, establish kick stability or reliability, or turn absent hit-effects packets into misses.

## Existing strict validation and coverage

The unchanged referee and contact tools both exited 0 for every round, after the exact owned process closed. All three passed existing full-policy-round coverage, exact owned-PID capture provenance plus frame/QPC/root-position binding, complete capture footers with no errors, consistent terminal state, and received point-counter reconciliation from zero.

| Round | Predictions / locally applied | Sources skipped while action in flight | First applied after initial observation, s | Maximum applied-action gap, s | Referee sources verified | Maximum receipt age, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r34 | 5152 / 5151 | 474 | 0.201886 | 0.135170 | 5627 | 0.150686 |
| r35 | 5650 / 5649 | 47 | 0.212983 | 0.160851 | 5698 | 0.194707 |
| r36 | 5809 / 5808 | 52 | 0.186463 | 0.054539 | 5862 | 0.130677 |

Each rejected request was a terminal race; every attack request was locally acknowledged. The existing coverage limit is 1 s, and the existing referee receipt-age limit is 0.5 s. There were no unavailable referee payloads. r34's unusually high 474 skipped sources are retained as observed, despite passing the unchanged coverage criterion; no new threshold or causal explanation was introduced. All analysis was CPU-only, with no game connection, GPU work, schema change, or runtime edit.

## Reproducibility

Private root: `C:\rekagent\work\consistent-fighter-20260919-r1`. Each `live-round_outcome_v1-rNN` contains the unchanged trial evidence, ownership/result records, and new `contact-analysis` / `referee-validation` outputs. Detailed derived summaries and stdout/stderr are in `left-front-outcome-development-r34-r36-r1\r34`, `r35`, and `r36`. The existing private `left-front-development-r30-r32-r1\summarize-round.cjs` was reused without modification. Commands and interpretation limits are the same as the [prior cohort](left-front-development-r30-r33.md#existing-strict-checks-and-provenance).

Exact captures are under `C:\rekagent\evidence\runtime\rek-private-ai-protocol-v7`:

| Round | Capture filename | SHA256 |
| --- | --- | --- |
| r34 | `rek-private-ai-root-motion-20260920T223044.6493464Z-pid361920-f8221a1fa3574d158e3f610917039f43.jsonl` | `2982d13867c30cd3600f9bec86e128c2d74905d65ef493a9f689da061f68f973` |
| r35 | `rek-private-ai-root-motion-20260920T223400.6440635Z-pid257004-c04fe80450134d9da5820d1a79d3f1d1.jsonl` | `496a46e418412ccd5de2952297212eca6b444d82809286ae1cd9bd35bb63d86a` |
| r36 | `rek-private-ai-root-motion-20260920T223708.1718190Z-pid154276-1d70e09c4a2b465797c03a3d0e50b025.jsonl` | `410cc87c7f509ce2c2c76282ebb8ebda1b637677714293af2bc04825da20dc3c` |

| Artifact | SHA256 |
| --- | --- |
| Existing referee validator | `22bbb6652f8cf333c1c832f8719f9e37da20b66bfecd85185e228d2183c9fe89` |
| Existing contact analyzer | `f0bfc8117d62442c7b3f9dfcf50cb2084c324561826c9c5bb7b301c54b9310c4` |
| Unchanged private summary helper | `7a2cd333999dc4114ac600d39c660b6820654f3801db4a0c29b717de1da62706` |
| r34 derived summary | `4d1c244d8674ea677b19529d0bcdf9bbcd4cb1d6e3c4baacb8608e2d1dce7384` |
| r35 derived summary | `75a7ccaaf6deefa00923f413d63db0d53448c919a793db05df1cc7a246e8a664` |
| r36 derived summary | `5516cbfefab77156429b76d65e3790b22e12a2de4acfafb0058d73cca9667bc4` |

Raw state records and account information are excluded from this public report. This cohort is development evidence, with no frozen acceptance or promotion claim.

## Archive

The coordinator verified the closed trial trees and exact captures on the physical server at `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\windows-development-r34-r36`: 127 files, 1,172,960,298 bytes. `archive-manifest.json` SHA256: `5404698c6c1d0380b326a6f0f3522a970d077f4bbe65e26b46429897b04d588f`. Existing source records were preserved.
