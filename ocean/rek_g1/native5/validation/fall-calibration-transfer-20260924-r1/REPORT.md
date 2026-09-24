# Frozen received-onset calibration transfer result

The prespecified calibration criterion failed.
No simulator hazard or policy change was enabled. This experiment measures received referee onset forecasting, not physical or causal state transitions.

## Frozen protocol and execution

Protocol frozen: 2026-09-24T04:36:49.563Z. Native execution started: 2026-09-24T04:36:49.626Z; exit 0; elapsed 1.337 s.
Freeze receipt SHA256: `c9d29b68fb06fffc488c384ed2542bade9540138f88bb3fb89e197a20d2bb209`.
Data SHA256: `43990a25e9e6dfe48d9f0e33130951e1fe9277d182b3fec406523322d192b455`.
Results SHA256: `08f344a9a7404c7bea5f82e7638c5f151fef9c867576f45b4efe31433cbb9a81`.
Compiler executable SHA256: `2360901d864cf10bfd6296e261cb2c14053552a80377761ab07146ec9ec9a2c0`.
Native executable SHA256: `1eef553bd4d04ea86d0d8e8b6914b8c80b4ca2b2dcfa56ab86e9808f5007ccd5`.
Frozen base weights SHA256: `89394a816b2db22948a4ea3f8fc7d1cac97c6a5a374aca30706cfe63beadc921`.
One fit only: mean Bernoulli NLL + 0.001*a*a/2, p=sigmoid(a*z+b), a>=0; intercept unpenalized.

Calibration: original rounds37–46, 10 processes, 2,208 windows, 7 local/12 opponent positives. Round36 excluded for checkpoint overlap with original training. Original33-trained geometry weights and scalers stayed frozen.
Transfer test: all17 later Windows rounds86–89,110,112–123, 3,693 windows,22 local/17 opponent positives. All relay and native hashes matched prior validation receipts. Source outcomes/counts had been inventoried, but no model predictions on these17 rounds were inspected before the freeze.
Original held-out test is historical development evidence; it was not rescored or reused for this result. Exact checkpoints/processes are disjoint between training, selected calibration, and new test. Related physical policy ancestry is not claimed independent.

## Pooled transfer metrics

| Target | Arm | Brier | Log loss | AP | Mean probability | Test prevalence |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Local | raw_geometry | 0.007634 | 0.027132 | 0.209928 | 0.011786 | 0.005957 |
| Local | original_intercept | 0.007504 | 0.026755 | 0.209928 | 0.011415 | 0.005957 |
| Local | slope_intercept | 0.006101 | 0.022267 | 0.209928 | 0.009324 | 0.005957 |
| Local | calibration_prevalence | 0.005929 | 0.037434 | 0.005957 | 0.003170 | 0.005957 |
| Opponent | raw_geometry | 0.005298 | 0.017636 | 0.295857 | 0.007131 | 0.004603 |
| Opponent | original_intercept | 0.006120 | 0.019967 | 0.295857 | 0.009121 | 0.004603 |
| Opponent | slope_intercept | 0.005892 | 0.019159 | 0.295857 | 0.008761 | 0.004603 |
| Opponent | calibration_prevalence | 0.004583 | 0.029430 | 0.004603 | 0.005435 | 0.004603 |

## Calibration and paired uncertainty

| Target | Slope | Intercept | Iterations | Positive processes | Brier difference 95% CI | Log-loss difference 95% CI | Criterion |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Local | 0.730541 | -0.834355 | 7 | 12/17 | [-0.001506, 0.002190] | [-0.024329, -0.007272] | fail |
| Opponent | 0.974488 | 0.509638 | 8 | 12/17 | [-0.001213, 0.004682] | [-0.020375, 0.001832] | fail |

Differences are new calibrator minus calibration-prevalence baseline. Intervals are percentile95% from2,000 complete-process resamples with fixed xorshift32 seed73, paired across arms and targets. They condition on the fitted calibrators, omit sparse calibration-estimation uncertainty, and are not independent-lineage intervals. Pooled metrics are window-weighted, separately for each target. Full metric intervals and undefined-AP replicate counts are in results.json.

## Checkpoint strata

| Checkpoint | Rounds | Target | Windows / positives | New Brier | Baseline Brier | New log loss | Baseline log loss | AP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 390007e25657 | r86,r87,r88 | Local | 661/3 | 0.003316 | 0.004520 | 0.011340 | 0.029276 | 0.442857 |
| ec640746b48c | r89 | Local | 220/2 | 0.021857 | 0.009043 | 0.066002 | 0.055455 | 0.162338 |
| a11ace1655cf | r110 | Local | 207/1 | 0.008623 | 0.004810 | 0.033413 | 0.030957 | 0.250000 |
| d34f3fefc71b | r112 | Local | 214/3 | 0.008082 | 0.013940 | 0.030251 | 0.083793 | 0.791667 |
| a2481a82dc2c | r113 | Local | 217/3 | 0.015044 | 0.013747 | 0.055686 | 0.082679 | 0.273504 |
| 7c34eaa9f00c | r114,r115,r116,r117 | Local | 873/4 | 0.004249 | 0.004563 | 0.019624 | 0.029525 | 0.314892 |
| ef85a01b207d | r118,r119,r120,r121,r122,r123 | Local | 1301/6 | 0.003875 | 0.004593 | 0.013536 | 0.029697 | 0.368318 |
| 390007e25657 | r86,r87,r88 | Opponent | 661/3 | 0.008536 | 0.004519 | 0.029963 | 0.029093 | 0.183824 |
| ec640746b48c | r89 | Opponent | 220/0 | 0.000001 | 0.000030 | 0.000370 | 0.005450 | undefined |
| a11ace1655cf | r110 | Opponent | 207/1 | 0.004399 | 0.004808 | 0.012271 | 0.030616 | 0.500000 |
| d34f3fefc71b | r112 | Opponent | 214/0 | 0.001206 | 0.000030 | 0.005974 | 0.005450 | undefined |
| a2481a82dc2c | r113 | Opponent | 217/0 | 0.000695 | 0.000030 | 0.003116 | 0.005450 | undefined |
| 7c34eaa9f00c | r114,r115,r116,r117 | Opponent | 873/5 | 0.010619 | 0.005695 | 0.032397 | 0.035286 | 0.226651 |
| ef85a01b207d | r118,r119,r120,r121,r122,r123 | Opponent | 1301/8 | 0.004247 | 0.006112 | 0.013906 | 0.037483 | 0.536308 |

## Limits

No hyperparameter search, classifier refit, or post-test refit occurred. A positive calibration slope preserves ranking, so AP improvement is not expected. Zero-positive strata report undefined AP. Checkpoint results are descriptive and no subset was selected for promotion.
Received onset time is client packet receipt. Executed action/phase, opponent command, support/contact causality and physical reset labels remain unavailable. The simulator feature distribution and any closed-loop transition or shaping use were not validated. Favourable forecasting would still require separate compact-support and authentic-policy tests.

Source code, protocol, input manifest, compiler/executable hashes, frozen receipt, metrics, and execution receipt are private artifacts in this directory. No credentials or raw proprietary game captures are copied into this experiment package.
