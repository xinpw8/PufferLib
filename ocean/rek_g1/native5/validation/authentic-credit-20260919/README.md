# Authentic reward-credit and critic review

CPU review completed 2026-09-20 UTC. Four complete development rounds contain
22,585 decisions, of which 22,582 are actor-eligible. Three final ownership-race
requests have actor weight zero; their terminal transitions remain in returns.
These are four correlated episodes, not 22,585 independent evaluation samples.
No game interaction, GPU execution, optimization, or deployment was performed by
this review. The accompanying diagnostic reads private binary evidence and emits
aggregate numbers; no observations, credentials, weights, or proprietary assets
are included here.

## Findings

The r1 export used reference gamma `0.999` and lambda `0.995` per 20 ms. This was
a configuration mismatch. The corrected r2 uses the intended gamma
`0.9998844821426083` and lambda `0.9978673240629938`. Discounts are raised to
`actual_QPC_transition_seconds / 0.02`, then rounded to float32. Recomputing r1
with the intended constants produces the same diagnostic numbers as stored r2.

No further blocking defect was found in the reviewed reward chain, frozen
behavior-probability pairing, actual legality masks, recurrent boundaries,
current-weight prefix burn-in, or FP32 old-probability PPO loss adapter. Score
changes are observed rewards; local command acceptance does not establish a hit
or server-side execution. Missing source snapshots are not fabricated.

The initial BF16 training-batch versus sequential-inference discrepancy does
not itself invalidate PPO. The corrected diagnostic measured full legal
`KL(old || batch)` mean `4.51907388105e-8`, maximum `0.000156796977545`;
chosen-action ratios ranged from `0.98425141748` to `1.00618886839`, with median
and p95 equal to 1 and clipping fraction zero at the 20% clip setting. The
sequential Puffer path reproduced all teacher logits and values exactly.
Thus the previous `1e-4` maximum-ratio tolerance was an engineering threshold,
not a mathematical PPO requirement. These pre-update numbers do not establish
post-update behavior or game performance.

## Critic comparison

The following statistics use the 22,582 actor-eligible rows. MC means complete
discounted **shaped** round return. It is an observed sample return, not the true
conditional expected value. The frozen critic's mean is `7.91856505364` in both
exports.

| Quantity | r1 actual discounts | corrected r2 |
|---|---:|---:|
| MC mean | -0.10272005615 | 0.17281762408 |
| Frozen value versus MC MAE | 8.84088634468 | 8.62234882195 |
| Frozen value versus MC RMSE | 11.06896254567 | 10.79635136771 |
| Lambda-return mean | 6.09580589225 | 6.19306808703 |
| Lambda-return versus MC RMSE | 8.42259029775 | 8.00056236119 |

With corrected discounts, round-start frozen values are `14`, `13.9375`,
`13.8125`, `13.9375`. Corresponding MC returns are `0.50094212573`,
`-0.50127240984`, `0.50137266162`, `0.50141960399`, whereas lambda-returns are
`18.84313314621`, `17.49915873461`, `19.75576515461`, `19.62619806169`.
The critic remains substantially miscalibrated on these recorded rounds.

Corrected all-row GAE advantage mean is `-1.72551384915`, population standard
deviation `3.99731515362`, minimum `-31.46378565443`, maximum `13.42849287027`.
There are 6,617 positive advantages. All-row value-versus-GAE-target MSE is
`18.95592648096`, agreeing with the separate GPU report. Actor-only mean and
standard deviation are `-1.72549696661` and `3.99739296306`; 2,725 actor rows
(12.07%) have a different advantage sign from `MC - frozen_value`.

The intended lambda has a 9.368 s exponential decay time; gamma times lambda
has an 8.887 s decay time. Future critic estimates therefore materially affect
these targets. With lambda set to 1 in the CPU control, lambda-return versus MC
RMSE is approximately `1.98e-14` before float32/BF16 target casts. This motivates
a critic-independent control; it does not prove that such a control will win
more rounds.

## Complete-return control

For terminal potential zero and `Phi(s) = score_difference / (5 + abs(difference))`,
the discounted shaped return telescopes to:

`G_shaped(t) = discounted_terminal_outcome(t) - Phi(s_t)`.

The maximum observed float-rounding residual is `4.83e-6`. Using
`advantage = G_shaped` and value-loss coefficient zero removes the learned
critic while retaining a known action-independent state baseline. This is a
valid on-policy policy-gradient control. Lambda-1 GAE alone still produces
`G_shaped - frozen_value`, so it is not the zero-learned-baseline control.
The baseline invariance argument applies to the unclipped on-policy gradient;
it does not claim identical finite-update clipped PPO behavior.

Keep frozen actual behavior log probabilities, captured legal support,
per-round recurrence, and actor-ineligible final-row masking unchanged. Include
those final rewards in earlier returns. For scale comparison, corrected
actor-row `G_shaped` has mean `0.17281762408`, population standard deviation
`0.40177193541`, and range `[-0.95914219778, 1.02971319933]`.

## Reproduction and provenance

Run the standalone CPU diagnostic with Node:

```sh
node critic_reference.cjs DATASET.bin REPLAY.bin MANIFEST.json
node --test critic_reference.test.cjs
```

It reports stored discounts, recomputation with the intended discounts, and a
lambda-1 control. Recurrence uses float64 arithmetic on float32 rewards,
discounts, and frozen values, before training-target casts. Four unit tests
passed; the final diagnostic also ran successfully on the actual r2 files over
WSL SSH without copying or altering them.

Private Spark evidence root:
`/home/spark-advantage/rek-training/authentic-trajectory-20260919-r1/`.

| Relative file | SHA256 |
|---|---|
| `data/authentic-trajectories.bin` | `31e6f59c25fa834fa2c05cb12c60ca390186366b335abc5656e78ab4adf8a02e` |
| `data/behavior-replay-r1.bin` | `9b9c22f7e86ca98469ed1d16be7489d7761000eeb1c612ad92746b1f389f2133` |
| `data-r2/authentic-trajectories.bin` | `f2b67a599600bcca878085a942a6ae9b174e4d054aff0bd3cddf3d883c6d3b04` |
| `data-r2/behavior-replay-r2.bin` | `06c14f853cba325fd276fa9e97102f3917a623c6c9969722d43aca2c161cb171` |

Both replay headers identify frozen checkpoint
`61f97b0b0a4504c6bdd0ee16d369ad4c1915e3cdf73d6358bab01ce64c8fde3f`.
Exporter source `native5/authentic_trajectory_data.cjs` SHA256:
`ddeaa43e7296813458628125bd2dcc1b0fe0e720cb5c34d90f1edf648bcf80f8`.
CPU recurrence source `native5/authentic_gae.h` SHA256:
`b2b0c5311922c4b7972f4060297c63cecee960530e6309cbe27f26c6cd5a40e4`.

The separately executed GPU diagnostic was read from
`/home/spark-advantage/rek-training/authentic-ppo-20260919-r1/diagnostic-r2/parity.stdout.jsonl`,
SHA256 `bb5d939048a4fdfd0851520b22c7fa63dd7e2f162dc90d28ca837eca699def43`.
It explicitly reports zero optimizer updates. The reviewed pinned loss adapter
is `build-r2/source/puffer5_ppo_fp32.cuh` under that same PPO root, SHA256
`decb9d01af40979c966c6e7c302b05152417b579234a3e3ffcafc9f916848c2f`.
The adapter generator `native5/prepare_authentic_ppo_kernel.cjs` SHA256 is
`95e2a2c989d33384d56595405c6930ba612a8528558479e15cd67fba52910c3c`.
