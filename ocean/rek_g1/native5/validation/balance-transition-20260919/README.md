# Balance transition evidence audit

Existing REK sequences support a better short-horizon root predictor, but this diagnostic model does not pass a one-second open-loop replacement test. This audit does not change simulation physics, tune reward hyperparameters, or train an RL policy.

## Data and split

The audit uses the ten completed A/B policy trials from `authentic-policy-ab-20260919-r1`, plus passive defender rounds r4, r5, r11 and r12 from `passive-defender-20260917-r1`. It contains 81,105 paired-fighter observations spanning 1,667.6 s and 81,029 valid local-actor transitions. Passive r12 begins with points `[0,2]`; its observed sequence is usable, but its round-start coverage is partial. Human replay captures were not added to this export.

Captures are grouped by the stable process-clock origin `observed UTC - Unity unscaled time`, with a 1 s grouping tolerance. This identifies six launch groups. The ten A/B trials occupy four groups; six trials share one group. Passive r4/r5 share a group, and passive r11/r12 share another. Direct game PID is absent from these streams, so launch grouping is clock evidence rather than PID verification. Every model fold holds out a whole group. Adjacent frames and rounds from that group never enter its training set.

Raw pose-derived transitions and diagnostic model weights remain in `/home/spark-advantage/rek-training/balance-transition-20260919-r3` on Spark, outside Git. The adjacent JSON report contains aggregate measurements, source hashes and split membership. Its exporter SHA-256 is checked against the exact local script.

## Predictive result

A fixed regularization coefficient of 0.01 is used for a 29-feature standardized linear ridge model. Inputs include measured root height/tilt, finite-difference velocities, relative opponent position, root-relative ankle/wrist pose, held command, recent request presence/age and observed point-counter receipt context. The five targets are the next observed planar, vertical, yaw and tilt rates. Scaling and fitting use only each fold's training sessions. There was no hyperparameter search.

Planar RMSE in Unity distance units, pooled across held-out observations or windows:

| Horizon | Root persistence | Constant velocity | Ridge | Groups where ridge beats persistence |
|---|---:|---:|---:|---:|
| One received step, approximately 0.02 s | 0.018622 | 0.017248 | 0.015341 | 6/6 |
| 0.10 s | 0.061167 | 0.107517 | 0.057435 | 5/6 |
| 0.25 s | 0.111481 | 0.238248 | 0.116054 | 3/6 |
| 0.50 s | 0.165548 | 0.303685 | 0.166381 | 4/6 |
| 1.00 s | 0.250036 | 0.362096 | 0.268525 | 1/6 |

The one-step planar improvement is 17.6% against persistence and 11.1% against constant velocity. At 1 s the model is 7.4% worse than persistence in planar position. Its 1 s yaw RMSE is 46.844 degrees, versus 20.744 degrees for persistence. Its 1 s height RMSE improves from 0.053894 to 0.048909 units and tilt RMSE improves from 8.493 to 8.113 degrees. These mixed results do not establish sufficient multi-step fidelity.

For open-loop evaluation, each window is initialized once from observed root state and velocity. Only the recorded command/request schedule is replayed. Root state is then predicted recursively, while relative pose and opponent context stay at the initial observation. Future observed poses, root states and point receipts are not injected. Replayed controls were chosen in the real recorded world, so this remains conditional prediction rather than a new-policy counterfactual. The persistence baseline is a precisely defined zero-root-update diagnostic; it is not a replay of the full native5 simulator.

## Observable gaps and balance evidence

All 162,210 fighter observations are visual-only. Every raw root velocity and floor-contact count is zero; falling/fallen flags remain false; last-hit, joint-position and runner move/name fields are unavailable. Those fields cannot supervise physical contact, velocity, execution or referee falls here. The exporter derives velocity from timestamped root poses. Quaternion-derived tilt agrees with the provided tilt within 0.0117 degrees across the dataset.

There are 33 sustained geometric episodes with pelvis tilt at least 60 degrees for at least 0.2 s: six local-actor episodes and 27 opponent episodes. These are geometric proxies, with no assignment of referee fall or knockout labels. Thirty episodes have 0.5 s precursor/control comparisons matched within capture and actor on initial tilt, height, opponent distance and local requested-action context. Median precursor tilt change is +41.26 degrees versus -5.87 degrees in controls; median height change is -0.3203 versus -0.0070 units. Matching distances range from 0.036 to 3.143 standardized units. Controls can repeat, matching is approximate, and episodes are not independent causal trials.

The ten policy captures provide 927 one-second windows after distinct observed client attack requests. Root displacement exceeds 0.01 units in 907 windows. Per-trial median planar displacement is 0.0668 to 0.1025 units, with signed forward and per-requested-move summaries in the report. However, 755 windows contain a newer request. Later controls, interactions, client interpolation and resets can contribute. Requested move identity is retained as observed context, and executed move identity stays null. No request is relabeled as a hit, miss, fall or verified playback.

`CleanHits` changes are awarded-point receipt context. They are not contact labels. Five-point increments are enumerated separately without assigning a cause. Existing event-rule recovery is not refitted by this audit.

## Reproduction and bounds

From `ocean/rek_g1/native5`, run `node --test balance_transition_data.test.cjs`. All eight tests pass under local Node v25.2.1 and Spark Node v18.19.1, covering process-group separation, coordinates, timing gaps, serialization of missing history, held-out prediction, absence of future-pose injection, sustained-tilt labeling and overlapping request windows.

On Spark, run `node balance_transition_data_20260919.cjs /home/spark-advantage/rek-training NEW_PRIVATE_OUTPUT_DIRECTORY`. Existing outputs are refused. Source hashes and unchanged-file checks bind the report to the recordings. Raw transition files and diagnostic weights are created with private permissions.

The useful supervision product is the continuous observed root transition dataset and its explicit uncertainty schema. Missing active playback identity, support contacts, impulse/force state and coupled opponent dynamics remain material limitations. The short-horizon gain warrants further sequence-model work with these session splits preserved. The one-second result does not justify loading this fitted diagnostic into the training simulator.
