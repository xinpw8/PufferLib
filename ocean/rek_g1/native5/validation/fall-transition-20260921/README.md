# Received fall-onset forecast, 2026-09-21

The tested model is unsuitable for enabling falls in the compact simulator. On
unseen complete processes, its calibrated probabilities score worse than the
constant prevalence baseline by Brier loss. Adding request history worsens the
opponent forecast relative to observed geometry alone. Weights remain private,
with `runtime_enabled=false`; no simulator or policy was changed by this probe.

## Sources and labels

The complete cohort contains **56** Windows agent-versus-AI rounds, each from a
different process, under `C:/rekagent/work/consistent-fighter-20260919-r1`.
Existing native/relay referee-validation reports are checked, and both full
source-file hashes are independently recomputed before exporting observations.
The early inventory estimate of 53 rounds is superseded by this exact count.

The native recordings contain 116 count episodes. The forecast dataset retains
105 uncensored, explicit onsets: **88 Slip and 17 Knockdown**. Eleven additional
DoubleKnockdown count rises have `sequence_gap_censored` receipt history and are
excluded. No double-fall forecast class has been validated. A five-point award,
rendered height or tilt, and always-false visual fall flags never create labels.

All 105 retained count episodes clear with an explicit Knockout or
DoubleKnockout call. Observed receipt durations range from 2.921 to 5.107 s.
These are referee count resolutions. They do not establish a physical body
reset, its completion, or its server timestamp. There is no attacker/contact
label identifying whether a fall was self-caused or induced by a specific move.

## Experiment

There are 12,352 eligible windows, spaced at least 0.5 s apart. Targets indicate
a newly received count onset during the following 0.5 s for the local fighter
and opponent separately. Current and preceding 0.2 s observations supply 42
geometry/rate features, including both roots, relative heading/displacement,
ankles and wrists. Another 25 features describe held commands, prior commands,
request age, recent request frequency, and the 17 requested move IDs. Request
and held-command observations are not executed-move or controller-phase labels.
Distance values remain Unity numeric units; metric calibration is unverified.

Entire chronological process groups are split 60/20/20. No process or round
crosses splits. This is an agent cohort, so no human person-held-out claim is
possible.

| Split | Processes | Windows | Local positive | Opponent positive |
| --- | ---: | ---: | ---: | ---: |
| Training | 33 | 7,313 | 19 | 35 |
| Calibration | 11 | 2,412 | 9 | 14 |
| Final test | 12 | 2,627 | 11 | 15 |

The single specified experiment fits native C++ logistic models with fixed
600-step full-batch Adam, learning rate 0.03, L2 coefficient 0.001, train-only
standardization, natural prevalence, and no oversampling or class reweighting.
Only an intercept correction uses the separate calibration processes. No
architecture search or follow-up fit was performed after viewing the test.

| Target | Model | Test Brier, lower is better | Test average precision |
| --- | --- | ---: | ---: |
| Local | Calibrated prevalence | 0.004170 | 0.00419 |
| Local | Requests/commands | 0.004177 | 0.01138 |
| Local | Geometry/rates | 0.005168 | 0.26670 |
| Local | Geometry/rates + requests/commands | 0.005118 | 0.25285 |
| Opponent | Calibrated prevalence | 0.005677 | 0.00571 |
| Opponent | Requests/commands | 0.005671 | 0.01096 |
| Opponent | Geometry/rates | 0.007609 | 0.35623 |
| Opponent | Geometry/rates + requests/commands | 0.010425 | 0.32675 |

The combined opponent model predicts mean onset probability 1.697%, compared
with observed 0.571%; local values are 0.779% and 0.419%. Geometry ranks imminent
received falls, but the calibrated hazards do not transfer reliably. Action 9
has zero observed support. Six other move IDs have fewer than 20 eligible
training windows each, and all action effects remain observational. There are
only 26 independent positive onsets represented in the final test.

## Verification and artifacts

Five exporter tests cover explicit onset deduplication, rejecting pose/award
proxies, censoring uncertain rises and missing observations, and excluding
future requests from features. Native metric tests cover constant-probability
Brier/AP/AUC, tied-probability calibration, and perfect ranking. All pass.
An independent evaluator uses the saved native weights without refitting and
reproduces Brier, log loss, average precision and AUC within 4.33e-13. Its
calibration bins keep tied probabilities together.

Dataset SHA-256:
`895f66784b51f333111d4d3f0231017fc9caeb0f49b49c7328865cbe0393e55c`.
Private fitted-weights SHA-256:
`89394a816b2db22948a4ea3f8fc7d1cac97c6a5a374aca30706cfe63beadc921`.

Private dataset, provenance, weights, original and independently verified
metrics, and source snapshots are archived under
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\fall-transition-probe-r1`.
The local scratch work is `C:/rekagent/work/fall-transition-20260921-r1`.

The most useful missing features are authoritative executed clip identity and
phase, controller/support-contact state, and the opponent's executed command
history, all timestamped before native fall classification. Current requests
cannot establish those quantities, and the captured visual robots do not expose
them. Additional architectures on the same 105 receipt events would not recover
that missing causal information.
