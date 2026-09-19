# External Windows balance validation

The existing diagnostic root predictor does not qualify as a balance-dynamics replacement. Its one-step improvement transfers to the new Windows recordings, but every frozen historical model is worse than pose persistence at 0.5 s and 1 s planar prediction. Explicit referee-count windows expose additional drift. No model was fitted, selected or tuned on these captures, and no fall rule, runtime or encoder was changed.

## Data and separation

`validate_balance_transfer.cjs` evaluates all six unchanged fixed-lambda ridge models from Spark's `balance-transition-20260919-r3`. Each model was fitted on five of the original six process-clock session groups. The new Windows captures `task512-r1` through `task512-r4` all belong to **one** external process group: PID 319504, estimated launch origin `2026-09-19T21:12:12.334Z`. None overlaps an original session. These four adjacent rounds are not four independent test sessions, and the six overlapping historical models are not independent replications.

There are 22,560 paired observations and 22,534 valid local-actor transitions. Valid transitions require positive receipt intervals from 0.005 to 0.06 s, active neighboring observations, and no interpolation across gaps. All four captures are retained. Initial remaining times are 119.79965, 118.98223, 105.473206 and 119.86664 s. In particular, r3 begins late; this analysis does not reclassify its policy coverage or impute unobserved motion.

Native recorder binding uses the expected PID, Arena scene, native round number/local slot, shared QPC frequency, and concurrent Unity frame/QPC/both fighter root positions. Matching sampled pose anchors number 1057, 1051, 867 and 1015 respectively. Every 33-byte referee packet is checked against its wire size, hash and decoded values. Complete capture endings and packet totals are required. The validator records source and model hashes and refuses inputs changed while being read.

## Frozen-model results

Planar RMSE uses uncalibrated Unity distance units. The range shows all six frozen models; neither endpoint is selected as a deployed model.

| Horizon | Windows | Pose persistence | Constant world velocity | Frozen ridge range |
| --- | ---: | ---: | ---: | ---: |
| One received step | 22534 | 0.012297 | 0.011546 | 0.009959 to 0.010308 |
| 0.10 s | 4250 | 0.048220 | 0.047503 | 0.041937 to 0.043638 |
| 0.25 s | 1775 | 0.088854 | 0.101616 | 0.082878 to 0.085932 |
| 0.50 s | 902 | 0.137753 | 0.196923 | 0.140077 to 0.145401 |
| 1.00 s | 455 | 0.208774 | 1.070787 | 0.226550 to 0.250669 |

At 1 s, ridge yaw RMSE is 55.538 to 58.051 degrees versus persistence at 21.967 degrees. Height improves from 0.042757 to 0.038149 to 0.039102 units, and tilt improves from 7.340 to 6.342 to 6.521 degrees. There are no nonfinite predictions. Mixed coordinate improvements do not establish physical fidelity.

The open-loop protocol is unchanged from the original diagnostic: initialize root/velocity once, replay recorded local command/request context, freeze initial pose/opponent context, and advance point receipt age without future point receipts. No future observed root, joint pose or opponent state is injected. Recorded controls are conditional context, not counterfactual actions from a newly simulated policy.

## Received referee counts

The four rounds contain four explicit uncensored countout episodes: one local episode in r2, one opponent episode in r1, two opponent episodes in r3, and none in r4. Count-mask episodes and explicit resolution calls define evaluation strata only. They are never predictor inputs or fitted labels. Zero `Round.Falls`, `IsFalling` and `IsFallen` values are not used as fall truth.

During the local count episode, 150 one-step transitions have nearly static observed roots. Persistence planar RMSE is 0.000074 units; ridge produces 0.001015 to 0.002144. Persistence tilt RMSE is 0.042 degrees; ridge produces 0.969 to 1.206 degrees. Across seven 0.5 s windows overlapping that count, persistence planar error is 0.001314, versus ridge at 0.075197 to 0.155535. The four 1 s local-count windows include the subsequent visible repositioning and have large errors for every baseline. These windows are correlated observations of one event, not evidence for a new action-to-fall law.

All 45,120 fighter observations are visual-only. Falling/fallen are always false, raw root velocity and floor-contact counts are always zero, and all 22,560 policy snapshots have zero round fall counters. Their disagreement with explicit received countouts prevents treating these visual fields as negative balance labels.

## Inputs that can and cannot be supplied now

| Recovered balance input or useful observation | Current evidence and integration boundary |
| --- | --- |
| Root orientation, tilt, height, joint/bone pose | Measured rendered poses and derived tilt are available. They support observation and supervised root targets. They do not supply contact forces or coupled dynamics. |
| Root rates | QPC finite differences are measurable rendered-motion rates. Raw native velocity fields remain zero. Neither establishes authoritative substep velocity. |
| `CanGetUp` | All eight initial actor snapshots now contain complete direct runtime authority, cross-checking `FightCoordinator.CanFighterGetUp` and `PolicyRunner.CanGetUp`, with value false. Continuous sample records omit this authority. It supports the observed initial runtime configuration; changes require refreshed authority. |
| Received count mask/seconds and referee calls | Validated wire observations support live diagnostic/state features, score attribution and event-supervision targets. They describe received referee state, not the physical causes of an unseen future fall. |
| Round clock | Measured remaining time is available. Receipt deltas must not be relabeled as physical integration steps. |
| `PelvisHeightRatio` | A native initial value exists, but this predictor has no continuously validated floor/standing-height calibration. Refuse a continuous physical ratio derived from arbitrary root height. |
| `FeetOffFloor`, `FootBodyContact`, `NonfootBodyContacts` | No valid continuous authoritative support-contact stream exists in these visual observations. Zero contact diagnostics do not certify no contacts. Refuse completion of these fields. |
| `Tracking`, `DetectorTickEnabled`, recovery and estop gates | Initial visual/runtime flags are incomplete evidence for the full continuous native early-gate contract. Refuse declaring that contract measured from these recordings. |
| `CompleteContactStream` | Hit-effect packets and score receipts are incomplete effects/events, with no verified active strike intent or full contact stream. Refuse certifying an empty stream from absent packets or renderer overlaps. |
| Physical reset completion | Count-clear and pose changes are observations. They are not the caller's confirmed two-fighter physical reset acknowledgement. |

The immediate defensible integration is the separately implemented received-referee observation path, with explicit freshness, round identity and availability. Preserve the existing private pose/referee streams as supervisory evidence. Do not insert this ridge predictor into `fast_runtime.cu` or manufacture support contacts from height/tilt. The recovered rules adapter must continue returning `MissingInput` until an actual dynamics/contact producer supplies its full measured or explicitly modeled contract. A new source of dynamics must then be tested against held-out trajectories and event sequences; rule-fixture agreement alone does not validate that producer.

## Artifacts and verification

Full per-model, per-capture, per-session, per-horizon and referee-stratum metrics: `C:/rekagent/work/balance-heldout-windows-20260919-r1/validation-r2/balance-transfer-validation.json`.

Report SHA-256: `196d69f3f1c3a8a6c8eccd93aabea8e636b99ebc4fd3c2e8cf149f5af75c78fa`.

The manifest and original copied model weights remain in the same private work directory. Raw recordings remain in their existing locations. Fifteen tests pass across the new validator and prior transition tests, including historical-session exclusion, grouping adjacent rounds, count-mask strata, malformed models, timing gaps, future-pose exclusion, and exact agreement with the established unstratified evaluator.

Reproduce from the repository root:

```powershell
node --test ocean/rek_g1/native5/validate_balance_transfer.test.cjs ocean/rek_g1/native5/balance_transition_data.test.cjs
node ocean/rek_g1/native5/validate_balance_transfer.cjs C:/rekagent/work/balance-heldout-windows-20260919-r1/manifest.json NEW_OUTPUT_DIRECTORY
```

Existing output directories are refused. No game process, bridge pipe, deployment, policy training or commit is performed by this validator.
