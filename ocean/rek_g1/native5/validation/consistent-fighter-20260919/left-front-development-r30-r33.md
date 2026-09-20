# Left-front exploration development rounds r30–r33

The frozen exploration checkpoint completed three authentic private Sparring Bot 1 rounds: **1 win, 2 losses, 0 draws; received points 41:34**. It requested LEFT_FRONT category17 twenty times among 259 attack requests (7.722%). All twenty were acknowledged locally and observed in native sends. These are development results, not evidence that the exploration intervention improved fighting or that particular kicks scored or caused falls.

Checkpoint SHA256: `6e4100648c17c06142d8fe4f997b6e1e7fd958dd2f0d3a90a7de367f370ab245`. The [offline intervention](../left-front-exploration-20260920/README.md) changed only decoder row17. All completed rounds used that same checkpoint, v1 observations, 223 unmasked features, BF16/native CUDA, sampled seed73, the existing private-AI route, and 20 ms target cadence. No policy update occurred between these rounds.

## Attempts and received outcomes

| Attempt | Status / outcome | Own:opponent points | Own non-five + five-point total | Opponent non-five + five-point total | Left-front / all attack requests |
| --- | --- | ---: | ---: | ---: | ---: |
| r30 | Startup failure before policy streaming; preserved | Not observed | Not observed | Not observed | No gameplay requests |
| r31 | Strict-complete loss | 11:14 | 11 + 0 | 9 + 5 | 9 / 85 |
| r32 | Strict-complete loss | 5:10 | 0 + 5 | 10 + 0 | 5 / 82 |
| r33 | Strict-complete win | 25:10 | 15 + 10 | 10 + 0 | 6 / 92 |
| Completed total | 1W–2L–0D | 41:34 | 26 + 15 | 29 + 5 | 20 / 259 |

r30 failed authenticated continuation at `2026-09-20T22:10:18.2000162Z`, before policy streaming, gameplay, or `ConfirmLoggedIn` began. The driver process had run for prewarm and inference preparation. Its failure record remains preserved. The coordinator subsequently closed that owned process at 22:13:31Z and extended the declared collection to r33 to obtain three completed rounds. r30 is retained as a failed startup attempt, not silently replaced and not assigned a fighting loss or a zero score.

All three completed rounds ended by points. The received referee recorded one local countout in r31 (2.995432 s), one opponent countout in r32 (3.015693 s), and two opponent countouts in r33 (3.008455 and 3.001804 s). Each count episode was uncensored, resolved by an explicit `Knockout` referee call, and accompanied the reported five-point award totals. A referee `Knockout` call here does not mean the round ended by knockout. These records do not identify an attack that caused the fall.

## What the requests establish

For every one of the 259 attack requests, including all 20 LEFT_FRONT requests:

- The bridge reported a locally applied action and `ExecuteMove` returned true.
- A native dispatch return was observed, with one matching outbound move-request projection.
- Server acceptance, server playback/executed move, and individual contact outcome remain unknown.

RIGHT_HOOK category23 remained the most requested attack: 67, 51, and 67 requests, respectively (185 total). There were 239 non-left-front attack requests. Driver prediction/applied counts were 5625/5624, 4815/4815, and 5698/5697; the two rejected requests occurred at terminal races. All attack requests were locally acknowledged.

The observed 20/259 request fraction is not the offline 9.9943% attack-probability-mass calibration statistic. The latter was computed on fixed older histories; these fights supplied new histories, masks, and sampled draws.

## Pre-request geometry

These are median ground-plane root separations in captured Unity numeric units and absolute rendered-pelvis bearing angles. Physical metre calibration is unverified. Bearing is projected root-local +X in Unity XZ; it is not a measurement of the low-level controller's desired heading or contact-time alignment. Quantiles use linear interpolation at `(n−1)q`. All request geometry values were available.

| Round | Left-front requests | Left-front median gap | Other attacks median gap | Left-front median absolute bearing | Other attacks median absolute bearing |
| --- | ---: | ---: | ---: | ---: | ---: |
| r31 | 9 | 0.767908 | 0.723518 | 34.0025° | 24.8537° |
| r32 | 5 | 0.699184 | 0.607034 | 137.1820° | 111.4871° |
| r33 | 6 | 0.767434 | 0.873666 | 20.2627° | 42.9209° |

r32's large rendered-pelvis bearing occurs across both kick and other-attack requests. Geometry provides context for subsequent analysis; it does not label successful execution, misses, reliable trips, or the cause of the round result. Missing hit-effects packets are never converted into misses. No score or referee award is assigned causally to an attack request.

## Existing strict checks and provenance

Analysis was CPU-only and began only after each owned process closed. The unchanged `validate_live_referee.cjs` and `analyze_live_contacts.cjs` both exited 0 for r31, r32, and r33. Each round has:

- Exact native capture PID equality with its ownership record, plus the existing concurrent frame/QPC/root-position binding.
- A complete capture footer, no capture errors, consistent terminal evidence, and all received point counters reconciled from zero to the terminal score.
- Full existing policy-control coverage, starting within one second; maximum applied-action gaps were 0.061353, 0.103448, and 0.061466 s.
- Every referee-bearing policy source validated against exact 33-byte received packets: 5673, 4839, and 5732 sources, with no unavailable payloads. Maximum receipt ages were 0.212020, 0.245841, and 0.130059 s, within the existing 0.5 s budget.

No validator, runtime, observation schema, or evaluation gate was changed. These are chronological development rounds available for later authentic outcome training. No held-out acceptance or improvement claim is made.

Private root: `C:\rekagent\work\consistent-fighter-20260919-r1`. Each `live-round_outcome_v1-rNN` contains `result.json`, `ownership.json`, `trial`, `contact-analysis`, and `referee-validation` as applicable. Detailed derived counts/quantiles and tool stdout/stderr are in `left-front-development-r30-r32-r1\r31`, `r32`, and `r33`; that directory retained its preparation name when the collection extended to r33.

Exact native captures are under `C:\rekagent\evidence\runtime\rek-private-ai-protocol-v7`:

| Round | Capture filename | SHA256 |
| --- | --- | --- |
| r31 | `rek-private-ai-root-motion-20260920T221428.9736830Z-pid270592-818a6a95ed7a4c95b6b715350aaed58b.jsonl` | `ef84dcc3b557ca798ad5c298d48f8aeb8a37c304fe3f5e7baa6b5d010ab66763` |
| r32 | `rek-private-ai-root-motion-20260920T221751.2087389Z-pid295856-062c5128e3a048c3929826c3a5282239.jsonl` | `c8634e689510f59e4e0ff287c342f0a6fada5e560d9eb8b00fcae5129525daa5` |
| r33 | `rek-private-ai-root-motion-20260920T222510.4704721Z-pid340616-96110d26a35c499eaeb1f19947add18c.jsonl` | `15847a091d0a091f647a21221425e8ecaf5a0e41dc4716a512b976067c06d6d5` |

| Reproducibility artifact | SHA256 |
| --- | --- |
| Existing referee validator | `22bbb6652f8cf333c1c832f8719f9e37da20b66bfecd85185e228d2183c9fe89` |
| Existing contact analyzer | `f0bfc8117d62442c7b3f9dfcf50cb2084c324561826c9c5bb7b301c54b9310c4` |
| Private `summarize-round.cjs` | `7a2cd333999dc4114ac600d39c660b6820654f3801db4a0c29b717de1da62706` |
| Preserved r30 `result.json` | `dbdec5173ecdf3a174ca1150dc987244cea0009df0719dc9b45d5eb8677a08b1` |
| r31 derived summary | `d0b0e61ed67573c02baf4e7482707ef8f26688546b3561486a8915ecb6dfba0a` |
| r32 derived summary | `75ccca0bc8ade212d8cafdc85ecd735eb0e29b5587d54b0fec2d8bf813f585d5` |
| r33 derived summary | `bb80f0e20f7edc2a9f87e3eaecd74d536d27f3999ccf48e9143f18539ee0eba7` |

Reproduce each closed round using its exact capture and fresh output directories:

```text
node validate_live_referee.cjs TRIAL_DIRECTORY EXACT_NATIVE_CAPTURE NEW_REFEREE_OUTPUT
node --max-old-space-size=4096 analyze_live_contacts.cjs TRIAL_DIRECTORY EXACT_NATIVE_CAPTURE NEW_CONTACT_OUTPUT
node summarize-round.cjs ROUND_DIRECTORY NEW_CONTACT_OUTPUT NEW_REFEREE_OUTPUT
```

The summaries retain exact input hashes, source counts, receipt ages, request geometry, acknowledgement/send distinctions, and unknown causal fields. Raw captures and account information are not included in this public report.

The coordinator archived the closed r30–r33 trial trees and exact native captures to fresh physical-server directory `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\windows-development-r30-r33`: 155 files, 1,153,334,367 bytes. All source-before/after and NAS readback hashes matched; sources were preserved. `archive-manifest.json` SHA256: `bb4966451baa50d29433bc5ea5984e6c53c919072cb03b74f67a6fd5b8a0080e`.
