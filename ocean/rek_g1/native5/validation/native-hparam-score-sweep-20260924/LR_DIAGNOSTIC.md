# Native Muon learning-rate diagnostic

Read-only analysis of the completed 8,388,608-transition screen. No training, evaluation, inference, or source changes were performed for this diagnostic. The evidence supports excessive updates at the largest learning rates. It does not establish which actions the degraded policies select.

Paths on Spark:

- `S=/home/spark-advantage/rek-training/native-hparam-score-sweep-20260924-r1`
- `P=/home/spark-advantage/rek-training/f7-action-id-fix-20260924-r1/build/trainer/src`
- `R=/home/spark-advantage/rek-training/f7-action-id-fix-20260924-r1/source/ocean/rek_g1/native5`

Evidence is in `S/screen-results.json`, each `S/screen/ARM/stdout.txt`, its original `logs/rek_native5/ARM-s73.ini`, `stderr.txt`, and final checkpoint. Every selected run loaded the same checkpoint SHA-256 `7561859790d0ed55a55471d96442fe954a593c52d8617b51c143687ba9d98f96`; the execution harness checked the exact step-zero readback. Each final native run exited 0 with finite weights and runtime failure bits 0.

## Matched screen comparison

Frozen evaluations below use policy side 0, 128 completed rounds, and the same two evaluation seeds. They measure simulator rounds, not authentic REK matches. The unchanged checkpoint scores 34.046875 points per round, margin 25.3671875, and 93.75% round wins.

| Arm | LR | Frozen own points | Round wins % | Peak logged KL | Maximum clipping fraction | Entropy minimum / final | Maximum logged value loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | .000055 | 33.5859375 | 89.84375 | .000 | .001 | 1.307 / 1.324 | .000 |
| arm-01 | .000055 | 31.640625 | 93.75 | .000 | .001 | 1.299 / 1.319 | .000 |
| arm-05 | .0003 | 31.796875 | 91.40625 | .002 | .013 | 1.277 / 1.313 | .000 |
| arm-09 | .001 | 31.296875 | 97.65625 | .004 | .034 | .938 / 1.095 | .000 |
| arm-13 | .003 | 23.1328125 | 74.21875 | .008 | .102 | 1.106 / 1.272 | .001 |
| arm-17 | .0075 | .1484375 | 1.5625 | .033 | .309 | .095 / 1.639 | .064 |
| arm-21 | .015 | .6640625 | 4.6875 | .103 | .448 | .036 / .767 | .198 |

The arm-01/05/09/13/17/21 sequence changes only LR: all use entropy .00017, horizon 256, policy clip .2, value coefficient .5, gamma .9997689776295918, lambda .9957391964326402, and seed 73. This comparison isolates LR within the screen. Control also changes horizon to 128, policy clip to .13, value coefficient to 1.02, gamma to .9998844821426083, and lambda to .9978673240629938. Control versus arm-01 cannot isolate clipping or value loss.

The .0075 and .015 runs already reach their maximum KL and clipping fraction at the first logged 131.1K transitions. Their entropy minima occur around 2.9M and 2.8M transitions. Arm-17 later recovers entropy to 1.639 while its frozen performance remains degraded, so persistent low entropy is not a sufficient explanation. Policy-loss ranges are [-.042, .024] for arm-17 and [-.024, .078] for arm-21, versus [-.001, .003] for control.

These loss/KL/entropy diagnostics come from 14 to 16 dashboard snapshots, rounded to three decimals. A displayed .000 does not mean exact zero. The native writer deliberately omits `loss/*` from INIs (`P/pufferl.cu:3554`); dashboard formatting is `%.3f` (`P/pufferl.cu:2408`). Each selected INI contains only one saved metric bin, ending around 6.2M to 6.7M transitions. It cannot reconstruct the full loss trajectory.

CPU-only comparison of final FP32 checkpoint arrays against the initial array also shows increasing displacement. Relative L2 changes in the action-decoder weights are 0.44%, 1.78%, 5.37%, 21.69%, 57.40%, and 128.66% for arm-01/05/09/13/17/21. Recurrent-weight changes are 0.60%, 3.11%, 10.27%, 32.52%, 84.92%, and 170.59%. These are parameter distances, not policy KL or action measurements.

## Mechanism supported by the implementation

Muon globally clips raw gradients, applies momentum/Nesterov, then normalizes each matrix by its own L2 norm before five Newton-Schulz iterations. It applies an aspect-ratio scale and finally multiplies the update by LR (`P/algo.cu:1051`, `:1066`, `:1144`, `:1181`, `:1205`, `:1209`). Small raw gradients and a .5 global gradient limit therefore do not impose an equivalently small parameter displacement. The early KL, clipping, transient entropy collapse, and growing weight displacement are consistent with excessive LR-scaled updates. They do not independently identify every cause of degradation.

Warm starts restore weights only, with fresh optimizer, RNG, RNN, and step count (`P/pufferl.cu:3254`). The momentum buffer is explicitly zeroed (`:2154`), and each invocation restarts cosine LR annealing (`:1654`). A pretrained actor is therefore exposed immediately to a fresh high-LR schedule. This reset is common to every arm and cannot by itself explain the LR ordering. No comparison restoring the original momentum state was performed.

The PPO policy gradient uses raw advantages (`P/algo.cu:1500`). Point differences are divided by 100; every accepted learner attack start additionally costs .15/100 = .0015 reward (`R/fast_runtime.cu:737`, `:743`). Attack acceptance is set at the settled attack-start branch (`:297`, `:300`). The value loss and entropy gradient share the same model update (`P/algo.cu:1511`, `:1595`). Changing value coefficient can change update direction even when Muon normalizes its magnitude.

At entropy approximately 1.35 nats, coefficient .00017 contributes about .0002295 to the per-sample entropy loss term; coefficient .002 contributes .0027. These are loss-term magnitudes, not equivalent awarded points or measured behavior. Entropy is evaluated every sampled step; the attempt penalty occurs only on accepted attack starts, and advantages include future rewards. Direct behavioral attribution needs action counts and advantage/gradient diagnostics.

Matched short-profile entropy comparisons offer no reason to increase entropy now: raising .00017 to .002 changes frozen own points from 31.640625 to 27.59375 at LR .000055, from 31.796875 to 14.0859375 at .0003, and from 31.296875 to 6.1484375 at .001. These are single-training-seed comparisons, not universal optima.

## Limits and next decision

No action histogram, accepted-attack counter, or per-action log-probability trace exists in the inspected screen/evaluation artifacts. Low score, fewer hits, and entropy do not establish a no-op policy. The displayed hit metric includes contact events from both fighters (`R/fast_runtime.cu:721`), so it cannot substitute for the learner's action distribution.

Retain the warm-start learning-rate scale while the full independent-seed confirmations finish. No further grid is justified solely by the screen. LRs below .000055 were not tested and remain hypotheses. If confirmations later justify a lower-LR study, isolate LR while holding the control's horizon, gamma/lambda, policy clip, value coefficient, entropy, reward, and initialization fixed; investigate clipping/value changes separately.

The confirmation snapshot observed during this analysis was incomplete. Against its separate 384-round unchanged baseline of 32.184895833333336 own points, 23.083333333333332 margin, and 91.92708333333334% wins, control seed 73 produced 33.3046875 / 23.838541666666668 / 94.01041666666666%, control seed 947 produced 33.239583333333336 / 23.580729166666668 / 91.92708333333334%, and arm-02 seed 73 produced 34.127604166666664 / 24.911458333333332 / 94.01041666666666%. These partial confirmations do not support the assumption that all shortlisted runs regress. Consult the completed `S/confirmation-results.json` for the final decision.

This analysis concerns optimization within the candidate simulator. It does not measure or resolve the gap between that simulator and live REK.
