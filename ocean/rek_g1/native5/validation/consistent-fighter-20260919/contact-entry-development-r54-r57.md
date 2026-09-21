# Persistent contact-entry development, r54–r57

The frozen geom-pair-trained candidate completed three authentic private Sparring Bot 1 rounds: **0 wins, 3 losses, 0 draws; received points 24:38**. All three passed the unchanged strict checks. r54 failed private entry before policy streaming and is preserved separately. This development cohort provides no evidence of improved fighting or acceptance.

Checkpoint SHA256: `07fdce5f1d14833103189ad424ff2c6f25d37166e259a9e3ba9230680cb61766`. Configuration SHA256: `f7d8f3c226467bc2f95182e0979a247d71e491cf3f5f856447e0f3052e6eaef8`. Both remained frozen through all four attempts. The [native training treatment](../contact-entry-20260920/README.md) changed compact contact-entry history to persistent geom pairs, retaining keyboard-reset yaw, stride 1 and the previous velocity proxy. Live encoder, worker, sampled seed 73, BF16/native CUDA, unmasked v1 observations and private-AI route were unchanged. No compact scoring flag was applied to the authentic client.

## Outcomes and requested actions

| Attempt | Status | Own:opponent points | Own non-five + five-point total | Opponent non-five + five-point total | Attack requests | LEFT_FRONT category 17 | Right-hook category 23 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r54 | Entry failure | Unavailable | Unavailable | Unavailable | 0 | 0 | 0 |
| r55 | Loss | 8:12 | 3 + 5 | 12 + 0 | 87 | 0 | 55 |
| r56 | Loss | 6:10 | 1 + 5 | 10 + 0 | 86 | 0 | 24 |
| r57 | Loss | 10:16 | 5 + 5 | 11 + 5 | 81 | 0 | 27 |
| Completed total | 0W, 3L, 0D | 24:38 | 9 + 15 | 33 + 5 | 254 | 0 | 106 |

All 254 attack requests were locally applied, returned true from `ExecuteMove`, had a native dispatch return and one matching outbound request projection. This establishes request evidence, not server acceptance, playback, contact, miss or trip. No received point award is assigned to a preceding attack. Five-point awards retain unresolved causes.

The prior keyboard-reset stride 1 [development](yaw-command-development-r37-r42.md) and [frozen evaluation](keyboard-yaw-frozen-evaluation.md) remain separate chronological cohorts. A source-backed contact-history correction did not produce a successful fighter here. These three rounds do not isolate its causal effect from the separately trained policy or establish physical contact parity.

## Coverage and pre-request geometry

| Round | Predictions / applied | In-flight sources skipped | First applied after first observation, s | Maximum action gap, s | Maximum referee receipt age, s | Median gap, captured units | Median absolute rendered bearing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r55 | 5720 / 5719 | 45 | 0.186080 | 0.060814 | 0.162941 | 0.619412 | 120.4492° |
| r56 | 5813 / 5812 | 20 | 0.188754 | 0.060899 | 0.129961 | 0.753502 | 97.4069° |
| r57 | 5776 / 5775 | 35 | 0.184187 | 0.059603 | 0.132372 | 0.743905 | 101.0366° |

All 17,412 source referee payloads were available and verified; no cached call-history censoring was recorded. Each round had one terminal-race rejection. The maximum last-applied-to-terminal gap was 0.001294 s. Exact owned/native PID, complete capture, terminal consistency, full-policy coverage and received point-counter reconciliation from zero all passed. Existing limits remained 1 s for coverage and 0.5 s for referee receipt age.

Geometry was available for every attack. Distances retain captured Unity numeric units; physical metre calibration is unverified. Bearing uses rendered pelvis/root-local +X projected into Unity XZ, not authoritative controller heading or contact-time alignment. Medians use linear interpolation at `(n−1)q`.

## Preserved r54 entry failure

r54 exited the driver with status 2 after `private-practice entry timeout`, with zero sources, predictions or applied actions, and no initial round, final round or opponent evidence. Local reservation request acknowledgement occurred at 2026-09-21 00:13:06.960 UTC; timeout at 00:13:51.915 UTC. This acknowledgement does not establish a server reservation. The underlying reservation/entry cause is unknown.

Owned PID 257832 was closed; result timestamp is 00:13:52.9355071 UTC. No win, loss or draw was assigned, and full-round validators were not run on this non-round. Its original result, ownership, request logs and closed-client `Player.log`/`BepInEx.log` copies are preserved and hashed in the private failure summary. Raw client-log contents were not read or published by this analysis. Completed r55–r57 have exact native captures but no `closed-client-logs` directories; current logs were not substituted for them.

## Reproduction and provenance

Private root: `C:\rekagent\work\consistent-fighter-20260919-r1`. Original attempts are `live-round_outcome_v1-r54` through `r57`. Completed trial directories contain `referee-validation` and `contact-analysis`. Derived stage `contact-entry-development-r54-r57-r1` contains the failure summary, per-round summaries, stdout/stderr and aggregate `cohort-summary.json`.

The stage's `validate-completed-round.ps1 -Round NN` is the prior wrapper with only round range, output path and declared checkpoint changed. It requires exact owned-process closure and selects the single matching owned-PID capture. It runs the unchanged `validate_live_referee.cjs`, `analyze_live_contacts.cjs` and private `left-front-development-r30-r32-r1\summarize-round.cjs`, refusing existing outputs. Validation changed no thresholds, runtime, reward or schema, and used no client connection or GPU execution.

Exact captures under `C:\rekagent\evidence\runtime\rek-private-ai-protocol-v7`:

| Round | Capture filename | SHA256 |
| --- | --- | --- |
| r55 | `rek-private-ai-root-motion-20260921T001546.9242020Z-pid92856-02f8cd8d93544c539a468983aff3d8b4.jsonl` | `cd5f8c636f310e4bf8ab81e4d49e865fd4d44393e52108168babc2927ccb80cf` |
| r56 | `rek-private-ai-root-motion-20260921T001919.1784054Z-pid153612-f310596c2c104d5db6c01b4159a3f9b7.jsonl` | `f90546672eb878ab174934266ac6db52b10308d2f383956b153f77de79272ca9` |
| r57 | `rek-private-ai-root-motion-20260921T002230.7684141Z-pid250552-939b596087ec483c8c5c4fc5b2c3dae5.jsonl` | `063830db93c38f24706c564a486c7e8cf831f5bf4c77e106e42829d02d821b9d` |

| Artifact | SHA256 |
| --- | --- |
| Existing referee validator | `22bbb6652f8cf333c1c832f8719f9e37da20b66bfecd85185e228d2183c9fe89` |
| Existing contact analyzer | `f0bfc8117d62442c7b3f9dfcf50cb2084c324561826c9c5bb7b301c54b9310c4` |
| Unchanged private summary helper | `7a2cd333999dc4114ac600d39c660b6820654f3801db4a0c29b717de1da62706` |
| r54 result | `fd4e6b2efda0cd13b9ed413844dc6a3f5c485252ef4a7b87c7592c31b8593e30` |
| r54 failure summary | `00662bb287c2005be57b5526448004cb51bae086ac5b89378a6bff4428d5c973` |
| r55 derived summary | `96d58d474da89ede0259aa41cececfd4008c351fc37adb256cf52018d0975a0a` |
| r56 derived summary | `d6d9d2df46d6b49faab18f45d35dda4f977b3e86382140fed305fe724fc597ff` |
| r57 derived summary | `bd3c4fdbb8c04151863b4cf98b5f3e4e44f8bac0e1766a0a0d573d21e5167070` |

Raw state records and account information are excluded from this report. Original captures were not edited.

The completed training, four trial attempts, exact native captures, derived
round results and private velocity-source investigation are preserved on the
existing evidence server under
`2026-09-19/consistent-fighter-r1/windows-contact-entry-development-r1`.
All 280 copied files (1,229,111,260 bytes) passed source-before, destination and
source-after SHA256 checks. Manifest SHA256:
`71312ed6891b917db5412087e43b8fbf639b520513d0c409e9b5aaed9aff239a`.
The subsequently completed 3,204-byte cohort summary is preserved separately
under `windows-contact-entry-cohort-summary-r1`, with matching source/readback
SHA256 `42cd84c7aa7578a5da2207214d52ba3e476c643ed4b8c7fa4910d221477651ab`.
