# Why the straight left kick was not selected

The six recorded rounds show very little policy probability assigned to the straight left kick, despite its availability. They do not show that the kick is ineffective. The policy selected zero category-17 requests, so these fights cannot establish its execution, stability, trip rate or scoring value.

The native action registry maps policy category 17 to runtime move 7, `LEFT_FRONT`, and category 23 to move 3, `RIGHT_HOOK`. This comparison uses actual saved worker masks and native replay logits, not inferred intent from visual motion.

## Current actors, kept separate

Control is checkpoint `f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4`, using legacy v1 observations on r24/r26/r28. Treatment is `c59a9f9044c8975a7cd790c5d8b44b253e6394fc2de93f2727b7d99879c4981e`, using owned-yaw-v2 on r25/r27/r29. Both descend from the unmasked f8 authentic-PPO actor, with the treatment's behavior-preserving column-187 migration before the matched update. Neither is the earlier BC candidate.

The cohorts contain 17,226 and 16,800 saved decisions respectively. Each contains three terminal-race requests with actor weight zero. Every saved sampled action was reproduced by the exact native BF16 replay, including those terminal-race requests. No optimizer ran during this diagnosis.

| Arm / action | Legal rows | Selected requests | Mean probability when legal | Mean per-state probability conditional on a legal attack |
|---|---:|---:|---:|---:|
| Control / left front 17 | 426 | 0 | 0.0607805% | 0.154415% |
| Control / right hook 23 | 426 | 242 | 56.0175% | 62.6611% |
| Treatment / left front 17 | 640 | 0 | 0.0235775% | 0.0675475% |
| Treatment / right hook 23 | 640 | 205 | 30.9651% | 41.4181% |

Categories 17 and 23 were legal on exactly the same rows in each arm. Thus there is no left-kick-specific mask exclusion in these saved inputs. Attack opportunities were sparse: 426/17,226 control decisions, or 2.4730%, and 640/16,800 treatment decisions, or 3.8095%, permitted attacks. Control selected 319 attack requests in total; treatment selected 297. All selected right-hook requests in the table received a local applied acknowledgement. That acknowledgement is not proof of server execution or contact.

The sum of left-kick probabilities over the recorded legal states was only 0.258925 for control and 0.150896 for treatment, versus 238.634667 and 198.176845 for the right hook. These sums describe the fixed recorded histories; they are not predictions for counterfactual fights. Zero observed left-kick selections is consistent with such small exploration mass.

The last table column averages each state's attack-only softmax. A different statistic, summing action probability and dividing by summed total attack probability, gives left-kick shares of 0.0820596% control and 0.0512013% treatment; corresponding right-hook shares are 75.6291% and 67.2445%. The distinction matters because attack probability varies across states.

Probabilities are computed on CPU in float64 from the exact native BF16 replay's FP32 logit copies, normalized over the actual legal mask. Attack-conditional probabilities normalize only legal categories 16..32. The replay itself retains the unchanged native float32 sampler reduction and verifies each selected action. These are policy request probabilities, not predicted hit probabilities.

## Parent and human-data context

The earlier unmasked f8 parent, on r21/r22/r23, had the same pattern: both actions legal on 401/17,558 decisions, zero left-kick requests and 265 right-hook requests. Its mean legal left-kick probability was 0.0250329%, with summed mass 0.100382. This is separately identified parent evidence, not the current arms' probability measurement.

The human recording did contain the kick. Independently checking `C:\rekagent\work\imitation-20260919-r1\dataset-r1\manifest.json` found 16 category-17 labels in the training capture and two in the held-out capture, among 77 and 109 one-shot requests respectively. All 18 were retained; this does not establish that all succeeded. The [separate native BC experiment](../native-bc-20260919/README.md) reported zero held-out kick-category recall at every listed endpoint. Its corrected balanced 20-epoch run reached 83.33% training kick-category recall but zero held-out recall. That small experiment did not establish generalization and is not the lineage of the current f3/c59 actors. Human manifest SHA256: `eb69ee0bfc75b5a3a0c4fe00a57c94bcf295884ea18240b98839052ad55d12a8`.

The supported diagnosis is policy concentration and very little exploration of the kick. These data cannot determine why its learned preference is so low, nor whether a broader, correctly matched demonstration or exploration intervention would improve fighting. No kick-specific reward, forced action or policy change was introduced.

## Training-model limitations relevant to this preference

The bulk `semantic_cuda` training runtime does not integrate balance, trips or
knockdowns. `fast_runtime.cu` explicitly leaves those dynamics unmodeled and
suppresses logical planar integration during attacks. Therefore it cannot
represent a kick's advantage from preserving balance or tripping the opponent.
Authentic-trajectory updates can receive real five-point awards, but the
recorded policy cohorts above provide no left-front-kick trials from which to
learn that move's actual consequences.

The native schedule used here commits move 7 for 145 ticks, or 2.90 s at 50 Hz,
versus 45 ticks, or 0.90 s, for move 3. This difference could affect preference
in the incomplete training dynamics. Its causal contribution has not been
isolated; it is not evidence that the kick is worse in REK.

The separate [native physical candidate probe](left-front-kick-candidate.md)
does execute move 7 with articulated physics. It starts from the candidate's
own reset and does not establish authentic hit, stability or trip parity.
Neither that probe nor these selection statistics justify a kick-specific
reward bonus. The missing evidence is the move's actual outcome by starting
geometry and motion state, including both fighters' subsequent balance.

## Reproduction and bindings

Private local root: `C:\rekagent\work\consistent-fighter-20260919-r1\authentic-owned-yaw-cohort-r1`. Private Spark root: `/home/spark-advantage/rek-training/authentic-owned-yaw-cohort-20260920-r1`. Exports preserve gamma 0.9998844821426083 and lambda 0.9978673240629938 per 20 ms, actual QPC durations and one checkpoint per arm. The frozen exporter and helper match commit `688db00a`; no source or live report was modified during export.

| Artifact | SHA256 |
|---|---|
| `data-v1/authentic-trajectories.bin` | `06bf216a33e57c2bd22cddeb23a914f41e3f6bf31ebd77a19a4b60cd15cef96e` |
| `data-v2/authentic-trajectories-owned-yaw-v2.bin` | `e36b17c0edcd78d3474746db060991ed1eaea70997852b4ccb0c0e0b1495277b` |
| `replay-v1-r1/behavior-replay.bin` | `6f21784c9e48aa2787c61ce28416ebc1f94a16c7efe8ab90bf0501a6e5f7ae9d` |
| `replay-v2-r1/behavior-replay.bin` | `cd259abf3cc007452c5598487a7b208f236b34449f01248e100326145e4cfac6` |
| Native replay executable | `52bdec853309d02b68288aae33acdb2086a5e837eb8478852aeb035f6928e0c1` |
| Exact policy object also linked by actual `worker-build` | `4ada3de760b5a00f7bb3d6592cd2da4ca48a4a196e774d40e6d220a57d1f574c` |
| `control-action-probabilities.json` | `57af49574ccc0ef42bd00522a1fc6efe9b8d2f2eb516be2457eb87d25fb42353` |
| `treatment-action-probabilities.json` | `cdf14d7650636f7cfd8647d46ea83e2bebef0913a1a7e4e917ff14c0d0366587` |

Both native replays ran once, 03:38:47.542121006Z through 03:38:56.016418685Z on 2026-09-20, then released the GPU. Process times were 5.54 s control and 2.85 s treatment. The v1 replay matched all 17,226 actions, and the explicit REKBR002 replay matched all 16,800 v2 actions. Input dataset/checkpoint hashes were unchanged afterward. Each replay directory contains exact commands, stdout/stderr, timing and result hashes. The directories were mirrored locally under `spark-results`, with both replay binary hashes independently verified.

The adjacent `left_front_kick_probabilities.cjs` reproduces the CPU probability summaries without a game connection:

```text
node left_front_kick_probabilities.cjs LABEL DATA_BINARY MATCHING_REPLAY_BINARY
```

It checks replay/dataset SHA binding and action/mask identity. Omitting the replay produces mask/selection counts only and explicitly marks probabilities unavailable. No raw state, account information or proprietary weights are included in this report.

## Verified NAS archive

The finalized local roots `authentic-owned-yaw-cohort-r1`, `physical-schedule-probe-r1` and `physical-left-front-probe-r1`, including the probability source and JSON reports, were copied without modifying their source files to `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\kick-diagnosis-and-physical-probes-r1`.

The archive contains 69 manifest-listed copied files, including the archive script, totaling 67,735,244 bytes, plus the manifest and transcript. Every copied file passed source-before/after SHA256 checks and NAS readback verification; the complete source file sets were unchanged. The destination was required to be absent and no prior archive was overwritten. `archive-manifest.json` SHA256: `15c590863ccdac5533aa2f0c528ac8ae038ab5e8dbbf833a4551e484c506d900`.
