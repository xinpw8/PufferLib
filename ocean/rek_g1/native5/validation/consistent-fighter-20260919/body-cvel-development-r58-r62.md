# Body-cvel development, r58-r62

The frozen body-cvel-trained candidate completed three authentic private Sparring Bot 1 rounds: **0 wins, 3 losses, 0 draws; received points 35:54**. Non-five-point awards were 15:24 and five-point awards were 20:30. All three completed rounds passed the unchanged strict checks. Two intervening sessions spawned G1 versus T800; the exact-G1 gate correctly withheld policy input. They are preserved without a scoring assessment or W/L/D.

This is a small chronological development cohort, not acceptance or evidence of improved fighting. No completed loss was omitted. The [native training treatment](../body-cvel-20260920/README.md) changed the compact contact-speed producer to each entered geometry pair's body-relative cvel approximation. It retained geometry-pair entry, keyboard-reset yaw, stride 1, reward and training settings. It still lacks controller/contact-response/balance dynamics.

Checkpoint SHA256: `08f717ff7945ea0a75779ca5e48ed99bbbf244956c5753e1ffa3136f04435249`. Configuration SHA256: `3fa3ce18f02f2d24ef587dae62f01b4553235047ed9d4a6bf161e689de34fda3`. The live worker, encoder, sampled seed 73, BF16 native inference, unmasked 223-column v1 observations and bridge were unchanged. No compact velocity-mode flag was applied to the authentic client.

## Outcomes and requests

| Attempt | Status | Own:opponent points | Own non-five + five total | Opponent non-five + five total | Attack requests | LEFT_FRONT category 17 | Right hook category 23 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r58 | Loss | 5:12 | 5 + 0 | 12 + 0 | 79 | 0 | 14 |
| r59 | Loss | 16:22 | 6 + 10 | 7 + 15 | 69 | 0 | 22 |
| r60 | Mixed-model session; zero policy | Not assessed | Not assessed | Not assessed | 0 | 0 | 0 |
| r61 | Mixed-model session; zero policy | Not assessed | Not assessed | Not assessed | 0 | 0 | 0 |
| r62 | Loss | 14:20 | 4 + 10 | 5 + 15 | 71 | 0 | 9 |
| Completed total | 0W, 3L, 0D | 35:54 | 15 + 20 | 24 + 30 | 219 | 0 | 45 |

Final terminal packets are used. In r59 the last active snapshot was 16:17 before a late five-point award; the final result is 16:22. In r62 the last active snapshot was 14:18; the final result is 14:20.

All 219 attack requests were locally applied, returned true from `ExecuteMove`, had a native dispatch return and exactly one matching outbound request projection. These observations do not establish server acceptance, playback, contact, miss or trip. No score award is attributed to a request; five-point award causes are not inferred.

Full attack-request histogram, using the pinned category-to-native-move table:

| Policy category / native move | r58 | r59 | r62 | Total |
| --- | ---: | ---: | ---: | ---: |
| 16 / 6 | 7 | 8 | 6 | 21 |
| 17 / 7 | 0 | 0 | 0 | 0 |
| 18 / 8 | 2 | 1 | 1 | 4 |
| 19 / 9 | 0 | 0 | 0 | 0 |
| 20 / 0 | 0 | 3 | 3 | 6 |
| 21 / 1 | 6 | 12 | 6 | 24 |
| 22 / 2 | 1 | 0 | 0 | 1 |
| 23 / 3 | 14 | 22 | 9 | 45 |
| 24 / 4 | 3 | 3 | 0 | 6 |
| 25 / 5 | 16 | 3 | 17 | 36 |
| 26 / 10 | 3 | 7 | 10 | 20 |
| 27 / 11 | 0 | 0 | 0 | 0 |
| 28 / 12 | 1 | 0 | 0 | 1 |
| 29 / 13 | 0 | 1 | 0 | 1 |
| 30 / 14 | 25 | 9 | 18 | 52 |
| 31 / 15 | 0 | 0 | 1 | 1 |
| 32 / 16 | 1 | 0 | 0 | 1 |

Category 17 maps to native move 7, LEFT_FRONT; category 23 maps to native move 3, right hook. Zero left-front selections do not establish that the move is ineffective.

## Coverage and request context

| Round | Predictions / applied | Skipped in-flight sources | First applied after observation, s | Maximum action gap, s | Maximum referee receipt age, s | Median gap, captured units | Median absolute rendered bearing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r58 | 5627 / 5627 | 36 | 0.190565 | 0.057242 | 0.124798 | 0.704687 | 101.0356 degrees |
| r59 | 5829 / 5828 | 35 | 0.194977 | 0.059673 | 0.251043 | 0.729987 | 42.5878 degrees |
| r62 | 5792 / 5791 | 26 | 0.185061 | 0.060686 | 0.134665 | 0.820906 | 32.8252 degrees |

All 17,349 source referee payloads were available and verified. r59 reported one cache-censored call; r58 and r62 reported none. This bounded-history censoring remains recorded and did not fail the existing verification contract. There were two terminal-race rejected actions, one each in r59 and r62. Maximum last-applied-to-terminal gap was 0.001797 s. Owned/native PID, complete capture, terminal consistency, full policy coverage and received point-counter reconciliation from zero passed. Existing coverage and receipt-age limits remain 1 s and 0.5 s.

Pre-request geometry was available for every attack. Distances retain captured Unity numeric units; physical metre calibration is unverified. Bearing uses rendered pelvis/root-local +X projected into Unity XZ, not authoritative controller heading or contact-time alignment. Quantiles use linear interpolation at `(n-1)q`. The private summaries retain full distributions separately for LEFT_FRONT, right hook and remaining requests.

## Preserved mixed-model sessions

r60 and r61 both reached `solo_route_proven` and `RoundActive`. Their local slot 0 had the exact G1 30-bone signature; opponent slot 1 had the exact T800 26-bone signature and semantic ID `t800`. The bridge reported `mixed_supported_runtime_models_rejected` and `exact_g1_pairing_not_proven`. They were not no-spawn sessions. The driver eventually reported `private AI ready bootstrap timeout; policy input withheld`, with zero sources, predictions and applied policy actions.

The first mixed-pair observations were 2026-09-21 01:09:51.808166 UTC and 01:12:24.0297498 UTC. Owned PIDs 376448 and 248788 were closed at 01:10:19.5691113 and 01:12:51.6291540 UTC. The successful r59 instead observed exact G1-versus-G1 pairing at 01:06:17.1800259 UTC.

Existing bridge commands expose no opponent-model selector. `EnterSolo` calls the native solo entry; `ReadyPrivateAiSession` emits the native ready request. The pinned `FightCoordinator.PickAiRobotId` dump contains `aiRobotPool`, `aiMirrorChance` and `Random.get_value`. Exact remote server configuration and RNG state remain unknown. No model field was invented, gate relaxed or timeout changed. An unchanged fresh r62 completed the third eligible development round.

Both failed sessions' original result, ownership, request records and closed `Player.log`/`BepInEx.log` copies are retained and hashed. Raw client-log contents were not read or published. The completed rounds have exact native captures but no closed-client-log directories; current logs were not substituted.

## Reproduction and evidence

Private root: `C:\rekagent\work\consistent-fighter-20260919-r1`. Original attempts are `live-round_outcome_v1-r58` through `r62`. Completed trial directories contain `referee-validation` and `contact-analysis`. The consolidated `body-cvel-development-r58-r62-r1` contains per-round stdout/stderr, derived summaries, both failure summaries, a sanitized startup diagnosis and `cohort-summary.json`. Earlier temporary derived stages remain unchanged.

The existing-style `validate-completed-round.ps1 -Round NN` was run only for closed r58, r59 and r62. Only its allowed range, output stage and declared checkpoint differ from the previous cohort wrapper. It selects the exact owned-PID capture and invokes unchanged `validate_live_referee.cjs`, `analyze_live_contacts.cjs` and the private `left-front-development-r30-r32-r1/summarize-round.cjs`. No game connection or GPU execution was used.

| Artifact | SHA256 |
| --- | --- |
| Referee validator | `22bbb6652f8cf333c1c832f8719f9e37da20b66bfecd85185e228d2183c9fe89` |
| Contact analyzer | `f0bfc8117d62442c7b3f9dfcf50cb2084c324561826c9c5bb7b301c54b9310c4` |
| Private summary helper | `7a2cd333999dc4114ac600d39c660b6820654f3801db4a0c29b717de1da62706` |
| r58 native capture | `53ce44b8cbf4751d5ef516d60f4a72e6a1fc726a68d7f3c704832dfb54328b14` |
| r59 native capture | `6dbfa938b7a10ff1f3cb37ab95ac3d2bd35b5f20ee7cadd0c49c3c3710180d97` |
| r62 native capture | `47192041ede2fe0ac94ce711aff2462ba2c737a096ab39434b3e8a27af72149c` |
| r58 derived summary | `1107c5899e4bc720b89ec944116dfbcdbf7e28b690b09d40a6968908fa55fbc2` |
| r59 derived summary | `2ec47d59cd049c7ec1e664609124e0f6f9db124a6744221f9e2232b843f7c655` |
| r62 derived summary | `bebe1ff8f9fbddaa2b16e6fcfe5c9a7be4e8ce174888dc583474efeba4c35ccb` |
| r60 mixed-model summary | `15cb4b7ddb122d27cc31b052c52e5bcd618bd5ea2d379decd7ab658060da8869` |
| r61 mixed-model summary | `32ce0a89c0b1518a5f0857b8f84107254022888f6f81b5c92bc42d491673e9f0` |
| Consolidated cohort summary | `f676f9738be3e33fec2b561620ee1f2e1fa79d075736d09a3a31ead6f6aad61e` |

Native capture filenames and all original result/ownership hashes are retained in the consolidated summary. No proprietary capture contents, account secrets or model weights are published.

## Verified NAS archive

All five original attempts, the three exact completed-round native captures,
the consolidated and earlier derived stages, frozen configuration, unchanged
launcher helper and archive script are preserved at
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\windows-body-cvel-development-r1`.
All 271 payload/script files, totaling 1,201,377,966 bytes, passed source-before,
NAS readback and source-after SHA256 comparison. Source sizes and modification
times remained unchanged. Original ownership, launch and native/referee bindings
were verified. The fresh destination preserved all existing archives.

Manifest SHA256: `eb4ccc3ad0ddba4e5f05c08d5b59f20fa1f35529064d4ef5a9519121c338cc69`.
`archive-transcript.txt` records the copy operation; the source private
`archive-body-cvel-development.stdout.txt` and `.stderr.txt` preserve its output.
