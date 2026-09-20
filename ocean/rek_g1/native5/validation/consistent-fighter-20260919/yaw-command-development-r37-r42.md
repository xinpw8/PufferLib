# Yaw-command development comparison, r37–r42

Three alternating authentic private Sparring Bot 1 rounds per frozen checkpoint produced **legacy: 0 wins, 3 losses, 26:55 points; keyboard-reset: 2 wins, 1 loss, 46:36 points**. Every round passed the existing strict validation after its owned client closed. This small development comparison supports further frozen evaluation, not acceptance, consistent-winning, or causal mechanism claims.

The legacy-trained checkpoint was `93180341b685c99549d9f55693a56b8cf42dfc45857790d77d43920d35219ba3` in r37/r39/r41. The keyboard-reset-trained checkpoint was `85808a6731f4f40f5756800a9edf93faa5c312aa7d5f7468cafdd4ae3a5c381f` in r38/r40/r42. Each stayed frozen throughout this cohort. Live bridge/encoder behavior remained unchanged: v1 observations, 223 unmasked features, BF16/native CUDA, sampled seed 73, private-AI route, and 20 ms target cadence. Native training throughput and simulator win rates are separate from these authentic results.

## Received outcomes and awards

| Round | Arm | Result | Own:opponent points | Own non-five + five-point total | Opponent non-five + five-point total |
| --- | --- | --- | ---: | ---: | ---: |
| r37 | Legacy | Loss | 7:22 | 2 + 5 | 12 + 10 |
| r38 | Keyboard-reset | Win | 17:13 | 7 + 10 | 8 + 5 |
| r39 | Legacy | Loss | 12:24 | 7 + 5 | 14 + 10 |
| r40 | Keyboard-reset | Win | 18:9 | 8 + 10 | 4 + 5 |
| r41 | Legacy | Loss | 7:9 | 7 + 0 | 4 + 5 |
| r42 | Keyboard-reset | Loss | 11:14 | 1 + 10 | 14 + 0 |
| Legacy total | | 0W, 3L, 0D | 26:55 | 16 + 10 | 30 + 25 |
| Keyboard-reset total | | 2W, 1L, 0D | 46:36 | 16 + 30 | 26 + 10 |

Both arms received 16 own non-five points. The reset arm's five-point differential was +20, versus the legacy arm's −15; non-five differentials were −10 and −14 respectively. This decomposition does not assign an award, countout, or fall to an attack. All terminal results were by points. In r40 the stream continued through the briefly active zero timer and retained the final opponent five-point award; the round was not truncated at timer zero. r42's two opponent countout resolutions yielded its two own five-point awards, despite the final loss.

## Requests and pre-request geometry

All 487 attack requests were locally applied, returned true from `ExecuteMove`, had a native dispatch return, and had one matching outbound move-request projection. LEFT_FRONT category 17 was requested zero times in both arms. These are dispatch observations; server acceptance, move playback, successful contact, misses, and trips remain unconfirmed.

| Round | Attack requests | Right-hook category 23 | Median gap, captured units | Median absolute rendered bearing |
| --- | ---: | ---: | ---: | ---: |
| r37 | 82 | 27 | 0.652589 | 83.5185° |
| r38 | 90 | 26 | 0.839140 | 46.5961° |
| r39 | 93 | 37 | 0.776984 | 37.8145° |
| r40 | 66 | 15 | 0.721273 | 63.2000° |
| r41 | 78 | 32 | 0.729683 | 52.2569° |
| r42 | 78 | 11 | 0.797086 | 49.5253° |

Legacy requested 253 attacks, including 96 right hooks; keyboard-reset requested 234, including 52 right hooks. Geometry was available for every request. Gaps retain captured Unity numeric units because physical metre calibration is unverified. Bearing projects rendered root-local +X into Unity XZ; it is not authoritative controller heading or contact-time alignment. Medians use linear interpolation at `(n−1)q`. These per-round values do not show a uniform reduction in attack-facing error and do not explain the outcome causally.

## Existing strict validation and coverage

The unchanged referee validator and contact analyzer exited 0 for all six rounds. Exact capture PID matched owned-process records; native frame/QPC/root-position binding passed. Complete capture footers, terminal evidence, full-policy coverage, and received score-counter reconciliation from zero all passed. All 35,227 source referee payloads were available and verified. r37 retains one explicitly censored call-sequence gap; no complete call-history claim is made.

| Round | Predictions / locally applied | In-flight sources skipped | First applied after initial observation, s | Maximum applied-action gap, s | Maximum referee receipt age, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| r37 | 5804 / 5803 | 50 | 0.178423 | 0.054419 | 0.124765 |
| r38 | 5818 / 5817 | 44 | 0.206036 | 0.069745 | 0.142590 |
| r39 | 5840 / 5839 | 38 | 0.199998 | 0.060740 | 0.125094 |
| r40 | 5919 / 5918 | 45 | 0.189631 | 0.057140 | 0.127292 |
| r41 | 5845 / 5844 | 35 | 0.198641 | 0.062600 | 0.124830 |
| r42 | 5736 / 5735 | 47 | 0.178579 | 0.056283 | 0.141273 |

Each round had one rejected terminal-race request; every attack request was locally acknowledged. The largest last-applied-to-terminal gap was 0.017493 s. Existing limits remain 1 s for coverage and 0.5 s for referee receipt age. No new gate, runtime edit, GPU work, or game connection was introduced by this analysis.

## Reproduction and provenance

Private root: `C:\rekagent\work\consistent-fighter-20260919-r1`. Each `live-round_outcome_v1-rNN` contains ownership/result evidence plus `referee-validation` and `contact-analysis`. CPU-only commands, stdout/stderr, and `derived-summary.json` are in `yaw-command-development-r37-r42-r1\rNN`. The wrapper `yaw-command-development-r37-r42-r1\validate-completed-round.ps1 -Round NN` uses the existing tools and unchanged `left-front-development-r30-r32-r1\summarize-round.cjs`; it refuses existing per-round output. The commands and interpretation limits are also documented in the [prior development report](left-front-development-r30-r33.md#existing-strict-checks-and-provenance).

Exact captures are under `C:\rekagent\evidence\runtime\rek-private-ai-protocol-v7`:

| Round | Capture filename | SHA256 |
| --- | --- | --- |
| r37 | `rek-private-ai-root-motion-20260920T225107.1466790Z-pid5892-d095f90fd93342ebb6c496372e1a8b7d.jsonl` | `2af6c358954307f8d64dde5d9d0b7cde87deaa68935deb86e84a79bd41b35d38` |
| r38 | `rek-private-ai-root-motion-20260920T225422.4011201Z-pid17272-ce78584384b44146ad6a5945909bcec3.jsonl` | `f37e3ef045a3ccb789b98529f34e2330ab3c8da15029b7bab3c6cea72cd5988b` |
| r39 | `rek-private-ai-root-motion-20260920T225725.8567115Z-pid256488-9a44a1f4c48a4bcb97499c7e230f701a.jsonl` | `3fcc5b96f44206a61657be04abfcb385d4f5964abafac664cc83eca994966445` |
| r40 | `rek-private-ai-root-motion-20260920T230028.5255073Z-pid353040-321a1117ac0746a0b522d8ec711835e4.jsonl` | `42ccb0506178f7d3e1b729185172b7f484dcc3100e85d02181341e24e4a9a857` |
| r41 | `rek-private-ai-root-motion-20260920T230347.5209694Z-pid373992-b5a40d18e8464f70a67994e99a3ace49.jsonl` | `2776234af9720342fe4dc2fe3296ed4e709552109ef3c659beb9699ba3c7dba6` |
| r42 | `rek-private-ai-root-motion-20260920T231057.6044106Z-pid392048-82683eb437984528854c7513d4d82079.jsonl` | `5c7d8b899bd9bed21f237dbaaf018c3e2daaee33eb4648935f2beb7cd2ab8934` |

| Artifact | SHA256 |
| --- | --- |
| Existing referee validator | `22bbb6652f8cf333c1c832f8719f9e37da20b66bfecd85185e228d2183c9fe89` |
| Existing contact analyzer | `f0bfc8117d62442c7b3f9dfcf50cb2084c324561826c9c5bb7b301c54b9310c4` |
| Unchanged private summary helper | `7a2cd333999dc4114ac600d39c660b6820654f3801db4a0c29b717de1da62706` |
| r37 derived summary | `4c35b596aa632d3083678e72e2eac191c7f20866431c77482569b0b09b94f7fb` |
| r38 derived summary | `d119e2b8172b9553bbd360cbe1d4b62df030582d39f84594560ac69badcdd920` |
| r39 derived summary | `1d1d8de0e41f5b95504d9665109c27ac0c1760f1210e4f0f782c85e4327ab9c6` |
| r40 derived summary | `86cc40a7b1eb1e873d01a43ed52fa57cdf743eb60dc773ae240ad37867bff99a` |
| r41 derived summary | `a8ec78c7733cf67b85a4a849da1c9a709c4bd296050f41dec3dc258dca99afb3` |
| r42 derived summary | `24a29796b3bdee04c4c2d11fb759010ab39f0e14e6233e5b022e0a8c95d4c6a6` |

Raw state records and account information are excluded. Any later frozen acceptance cohort must be reported separately from these six development rounds.

## Preserved file-server copy

The six completed trial trees and exact owned-PID captures were copied to the
existing evidence server under `pufferlib/rek-evidence/2026-09-19/consistent-fighter-r1/windows-development-r37-r42`:
253 files, 2,397,447,442 bytes. Source-before, destination and source-after
SHA256 checks matched for every file; no source was removed. Archive manifest
SHA256: `2fc6a48eb63cb7e31a836fcef14dd163f6389b39e2f5ddc782e9a263c31a6354`.
