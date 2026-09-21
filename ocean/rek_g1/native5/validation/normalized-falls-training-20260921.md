# Normalized points/falls integration and authentic evaluation

## Implemented reward contract

The opt-in `normalized_points_falls_v1` mode uses one fixed scale in the compact
and physical runtimes:

```text
reward = clamp((own_awarded_points - opponent_awarded_points
                - own_newly_confirmed_fall) / 100, -1, 1)
```

A one-point award is +0.01, a five-point award is +0.05, conceding five is -0.05,
and a newly confirmed own fall is -0.01. Simultaneous events add before scaling.
There is no repeated penalty while down, opponent-fall bonus, terminal bonus,
contact bonus, or additional spatial shaping in this mode. A reset does not
manufacture a fall. All saturation counts are reported explicitly.

This is fixed normalization, with no changing batch extrema or running reward
standard deviation. Relative weights still define the objective: the added fall
cost equals one point, while conceding a five-point countout costs five times
that amount. Multiplying every reward by a common positive factor preserves an
ideal expected-return ranking, but does not guarantee identical finite PPO
updates with a fixed critic loss, entropy coefficient, optimizer and clipping.
This PufferLib implementation already normalizes advantages by minibatch mean
and standard deviation in `src/pufferlib.cu`; the critic's targets and value loss
still depend on reward units. Reward normalization here is separate from that
advantage normalization.

The compact runtime still has no physical fall-event producer. It uses this
same point-reward scale and explicitly supplies no fall events. The physical
runtime consumes the recovered `BECAME_FALLEN` event, accumulated once across
the decision's substeps. Its reward snapshot and learner buffer agree. Neither
mode silently substitutes synthetic damage-threshold falls.

## Executed reward and integration tests

- CPU helper tests: 20,683,663 checks on each of WSL and Spark, with undefined
  behavior sanitization enabled.
- CUDA helper tests: 1,451,552 cases per replay, four graph replays, exact
  agreement with independent expected values.
- Compact runtime rollout: 70,400 checks, 58 nonzero point-award rows, 12 terminal
  steps and 12 automatic round resets. Scalar/warp contact-flow and training
  autoreset tests also passed.
- Physical action-only exposure: 20,000 fighter reward comparisons and 10,000
  learner-buffer comparisons. Both fighters became fallen, were counted out,
  and reset. Each fall was penalized once. Neutral control had no falls.

The physical exposure classified both falls as slips. It does not prove that a
specific strike induced a trip, or that this physical model has authentic REK
trajectory parity. Detailed evidence is in
[the physical exposure report](consistent-fighter-20260919/physical-fall-exposure.md)
and [the helper test report](normalized-reward-20260921.md).

## Compact training performance

Executed on `spark-4ae3`, NVIDIA GB10, native PufferLib C++/CUDA:

| Quantity | Measured value |
| --- | --- |
| Learner transitions | 33,554,432 |
| Environments / rollout horizon / updates | 512 / 512 / 128 |
| Full training-loop time | 35.7631254196 s |
| Complete-training SPS | 938,241.0404 |
| Process wall time including startup | 36.48 s |
| Startup-inclusive SPS | 919,803.5088 |
| Reward saturations / runtime failure bits | 0 / 0 |

This includes rollout, inference and PPO updates. It is not a physics-only or
rendered-client benchmark. Decision rate is 50 Hz, policy stride one, with eight
compact contact substeps. Physics, controller and PPO execution are native CUDA;
the Node summarizer runs only after training.

Configuration: 120 s rounds, minibatch 8192, learning rate 0.0001, entropy
coefficient 0.01, gamma 0.9998844821426083, lambda 0.9978673240629938,
environment seed 419 and base seed 73.
Contact mode is `body_cvel_v1` / `geom_pair_v1`, yaw is `keyboard_reset_v1`,
observations are `rendered_pose_v1`, and the opponent is `recovered_bot1_v1`.
Random reset separation is 0.55 to 2.5 m, with full heading spread. Additional
reward shaping is disabled. The 512-step horizon is 10.24 s; gamma has a 120 s
half-life, and the gamma-lambda trace has a 6.16 s half-life.

The changing policy completed 5,120 training rounds: 3,261 wins, 1,816 losses,
43 ties and aggregate points 338,670:173,886. These are training statistics,
not held-out skill or authentic-client wins.

Initial checkpoint SHA-256:
`f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4`.
The step-zero checkpoint matched it byte-for-byte.

Final checkpoint SHA-256:
`d4c9758b16b77ab404aedbcd076b6dfa7cd6b9ca15e5b2bf21c164caaca7c04d`.

## Authentic private-AI evaluation

The final compact checkpoint played one complete 120 s round on Windows host
`D21`, account `moogleod`, against private **Sparring Bot 1**, difficulty zero,
with G1 robots. Result: **loss, 11:18**. This does not establish improvement.

The process ran on the isolated `WinSta0\\RekPolicyEval` desktop. Native bridge
commands controlled only the owned REK PID 200788. No global keyboard or mouse
input was used. The owned process closed at 2026-09-21T08:11:10.8331356Z.

The authoritative received point packets reconcile all 11:18 points: six
ordinary points plus one five-point award for the policy; thirteen ordinary
points plus one five-point award for the opponent. The native referee records
one Slip then Knockout/countout for each fighter. All 4,614 policy observations
had matched native referee receipts, with maximum receipt age 0.1781389 s.
There were no censored call IDs in this trial. Five-point awards alone were not
used to infer fall causes.

The policy requested 65 attacks and requested the straight-left/front category
17 zero times. Median absolute pre-request opponent bearing was 65.97 degrees;
this describes request geometry, not server-confirmed executed attacks or
contact attribution. No reward, motion or live-play parity improvement is
claimed from this single result.

The existing trial harness directory is named `live-point_difference_v1-r63`.
That is a legacy harness bucket, not the training reward identity. The candidate
configuration, exact checkpoint hash and training log establish that the tested
candidate was trained with `normalized_points_falls_v1`.

## Physical fall-exposed PPO baseline

A separate native MuJoCo CUDA + SONIC PPO run completed 4,194,304 learner
transitions and 16 updates with 512 arenas and horizon 512. It trained through
actual physical fall transitions using the same normalized reward contract.
The runtime's cumulative counters recorded **208 learner falls and 1,222
opponent falls**, awarded points 6,715:4,375, and zero reward saturations.
These lifetime counters include all executed runtime transitions, including
warmup; they are not restricted to completed-round summaries.

One cohort of 512 complete 120 s rounds produced 374 wins, 105 losses and 33 ties,
with completed-round points 5,267:3,272 and zero runtime failure bits. These are
changing-policy training outcomes. The opponent is the physical runtime's
`CandidateApproachDummy`: approach, align and cycle attack categories 16 through
31, with no attack while falling. It is not authentic Sparring Bot 1 or the
compact recovered-Bot1 implementation.

Complete-training throughput was **6,632.2124 SPS** over 632.4140021801 s,
around 141 times slower than the compact run. Process wall time was 642.65 s,
or 6,526.5759 SPS including startup. No CPU physics or Python training was used. This
run proves learning rollouts encountered falls and their penalties; it does not
establish improved fall avoidance, better fighting, or acceptable performance
for the final high-throughput environment.

The initial checkpoint was the same compact checkpoint listed above. Its tensor
shape fits the physical actor, but the observation meanings differ. This is a
cross-model physical finetuning baseline, not a matched-observation deployment.
It has not been evaluated in authentic REK through the incompatible old encoder.

Physical final checkpoint SHA-256:
`a71f017f80d8e85a3dae520ba9c8e91410a6add059e5a29ddd9d83e6bcdd9f4a`.
All 17 saved checkpoints have distinct verified hashes. Private commands, source
bindings, exposure results and physical PPO artifacts are preserved in
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-fall-exposure-r1\physical-fall-exposure-20260921-r1-evidence.tar.gz`.
The archive is 75,761,181 bytes, SHA-256
`7afd5e3bf83505d3f9d97fb4969845f001443178bc5d2f371257b8749ede6603`.

## Remaining model and observation differences

A received-fall forecast was tested on 56 authentic rounds with process-held-out
splits. Its combined geometry/request probabilities had worse held-out Brier
loss than the constant-prevalence baseline. It was not enabled as a compact
fall generator. See [the forecast results](fall-transition-20260921/README.md).

The existing live encoder targets the compact observation contract, including
zeroed fall/support/referee fields and estimated request duration. The physical
runtime exports measured physical fields with different meanings. Matching the
223-column shape does not establish matching observations. A physical-model
checkpoint must not be deployed through that old encoder as a matched policy.
The standalone [observable-balance contract](../observable_balance.md) has
CPU/CUDA projection tests. It still requires adapter integration, fresh training
and authentic evaluation before acceptance. Neither running encoder was changed
by adding that standalone header.

## Artifact locations

Private Spark build, commands, tests and compact training:
`/home/spark-advantage/rek-training/normalized-falls-20260921-r1`.

Windows staging and derived authentic validation:
`C:/rekagent/work/normalized-falls-20260921-r1`.

Private authentic trial:
`C:/rekagent/work/consistent-fighter-20260919-r1/live-point_difference_v1-r63`.

Verified NAS archive root:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\normalized-falls-r1`.
New destinations were used with source-before/source-after/copy SHA-256 checks:

| Subdirectory | Files | Receipt SHA-256 |
| --- | --- | --- |
| `helper-tests` | 36 | `cf3c71069a7f72b66ae7eac76bcb6c4b993c7349ce63dcae4aa33f32dc987813` |
| `compact-training` | 387 | `17daef5f8710b898bd8070a50090768a8a8489cb82c35d9e6de0e0b3c116317d` |
| `authentic-r63` | 62 | `983d26ecf39fe466151bc0de504e06101c58ed7f5b15f5a00e7afecd5a95e2dd` |

Each directory contains `archive-receipt.json`. The compact archive includes
initial/final checkpoints, commands, compiled trainer, source snapshots, tests
and logs. The authentic archive includes the entire owned trial, analysis
inputs/outputs and exact native capture. Existing files were not overwritten.

Public source contains no REK binaries, controller weights, policy checkpoints,
credentials or raw captures.
