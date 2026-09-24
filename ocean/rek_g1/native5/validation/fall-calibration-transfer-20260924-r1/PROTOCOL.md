# Frozen received-onset calibration transfer experiment

Date: 2026-09-24. This is offline native C++ statistics, not policy training,
physical fall modeling, or authorization to enable a simulator hazard.

## Fixed inputs and split

Use the two `state_only` models in the existing private
`fall-transition-20260921-r1/weights.private.json`, SHA256
`89394a816b2db22948a4ea3f8fc7d1cac97c6a5a374aca30706cfe63beadc921`.
Their 42-feature weights and train-only scalers were fitted on the original 33
training processes. Do not refit them. Ignore their saved calibration offset
when fitting the new calibrator. Retain that offset only as a diagnostic arm.

Calibration: original dataset split 1, rounds 37 through 46, inclusive.
Exclude round 36 because its checkpoint also appears in training rounds 34/35.
Expected calibration size: 2,208 windows, local/opponent positives 7/12.

Transfer test: all later Windows rounds 86,87,88,89,110,112,113,114,115,116,117,
118,119,120,121,122,123. These were absent from the original dataset. Expected
size: 3,693 windows, local/opponent positives 22/17, 17 distinct processes.
Keep whole processes and exact checkpoints together. Do not split the related
7c34eaa9 and ef85a01b checkpoints across calibration and test. Exact checkpoint
disjointness is checked; independent training ancestry is not claimed.

The old test influenced this follow-up hypothesis and is historical development
evidence only. No model predictions or metrics on the new transfer test have
been inspected before this protocol. Its labels/counts and policy provenance
were inventoried before protocol registration.

Use the unchanged exporter compact()/buildRows() feature and onset definitions:
0.2 s history, 0.5 s horizon/spacing, explicit uncensored received referee onsets,
and the existing missing-window/count-active exclusions. Hash-check each relay
and native recording against its prior referee validation before preparing data.
No +5 award, pose threshold, or visual flag is a fall label.

## One prespecified fit

For each target independently, let z be the frozen raw geometry-model logit.
Fit p = sigmoid(a*z+b), a >= 0, by minimizing mean Bernoulli negative log
likelihood plus 0.001*a*a/2. The intercept is unpenalized. This fixed weak ridge
shrinks an overconfident slope, prevents unconstrained separation, and preserves
risk ordering. No parameter/penalty/architecture search or post-test refit.

Use deterministic double-precision Newton optimization with Armijo line search,
at most 200 iterations, projected slope boundary, and a 1e-10 gradient stopping
threshold. Start a=1 and b=logit(calibration prevalence)-mean(z). Record achieved
parameters, objective, iterations and convergence. Failure to converge is a
failed experiment, not permission to tune after evaluating the test.

Comparison arms: frozen raw geometry model; original intercept-only geometry
model; new slope-plus-intercept calibrator; constant calibration prevalence.
The constant baseline uses 7/2208 or 12/2208, never test prevalence.

## Evaluation and uncertainty

Report Brier, Bernoulli log loss, tie-aware average precision, mean prediction,
and observed prevalence, pooled and separately for all seven checkpoint groups.
Average precision is undefined for a zero-positive stratum and is reported null.
Compute sigmoid and Bernoulli log loss using stable double-precision formulas.
Rank AP by logits, with exact tied-logit groups. For a positive slope AP should
equal raw-model AP, apart from floating-point ties; a zero slope is one tie.
Pooled metrics are window-weighted and remain separate for each target.

For pooled new-calibrator and prevalence-baseline metrics, use 2,000 bootstrap
replicates resampling 17 complete test processes with replacement. Use a fixed
xorshift32 seed 73 and identical sampled process sets across arms and targets.
Report percentile 95% intervals and paired Brier/log-loss differences. These are
process-cluster intervals, not independent-policy-lineage confidence intervals.
Intervals condition on the fitted calibrator and omit calibration-estimation
uncertainty. Exclude undefined AP replicates and report their finite count.
Percentiles use sorted values at floor((n-1)*q), q=0.025 and 0.975. Record the
number of test processes containing positives for each target. At a=0 require
nonnegative slope gradient and zero intercept gradient; otherwise check both
gradients. Check fitted mean calibration probability against its prevalence.

Prespecified favorable result: both targets have lower pooled Brier, both paired
95% Brier-difference intervals have upper endpoint below zero, and neither
target has worse pooled log loss than the calibration-prevalence baseline.
Otherwise report which conditions failed; do not change the criterion.

Even a favorable result establishes only transfer of received-onset prediction.
Compact feature support, causal action effects, physical state transitions,
potential-based shaping, and authentic policy strength are separate untested
claims. All runtime_enabled fields must remain false.

## Execution order

Create source files and this protocol; prepare data without scoring; independently
review source; compile and self-test; write a freeze receipt binding protocol,
source, input manifest, dataset, compiler identity and executable hashes. Only
then execute the single fit/test command. Preserve every result, including
failure. No overwriting output, GPU, Python, bridge, or game interaction.
