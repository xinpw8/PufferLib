# Native policy learning experiments

The target is a frozen learned policy winning every evaluated round against
the unchanged human-evaluation candidate dummy. This target has not yet been
met. It does not establish performance against authentic REK Bot 1, humans,
or untested starting states. See `TRAINING_PERFORMANCE.md` for the deployed
physics optimizations and earlier frozen comparisons.

## Optimizer calibration

A separate two-action reward-bandit fixture exercised the actual native
MinGRU, masked categorical sampler, PPO and Muon optimizer. The initial
policy selected the rewarded action on 50.073% of 8,192 frozen samples.
Each experiment then used 524,288 samples and 256 rollout horizons.

| Constant learning rate | Replay ratio | Final rewarded-action percentage |
| --- | ---: | ---: |
| 0.0003 | 1 | 59.058% |
| 0.003 | 1 | 98.889% |
| 0.015 | 1 | 100%, 8,192/8,192 |
| 0.003 | 4 | 100%, 8,192/8,192 |

These are controlled optimizer tests, not fighting results. The combat
experiment selects 0.003 and replay ratio 4; stability at 0.015 in combat
has not been measured. Native prioritized replay draws minibatches with
replacement, so ratio 4 does not mean exactly four visits to every sample.

## Matched credit windows

`run_gpu_credit_experiment.py` overlays either
`config/rek_g1_credit_short.ini` or `config/rek_g1_credit_long.ini` on the
existing configuration. It verifies the initial checkpoint hash and records
the resolved parameters. Both arms use 512 arenas, 3,276,800 learner steps,
3,200 optimizer minibatches, unchanged score-delta rewards and the same dummy.

| Setting | Short | Long |
| --- | ---: | ---: |
| Horizon | 64 ticks, 1.28 s | 256 ticks, 5.12 s |
| Gamma | 0.99 | 0.999 |
| GAE lambda | 0.95 | 0.995 |
| Direct GAE weight after 3 s, ignoring truncation | 0.000101 | 0.405775 |

Both explicitly reset recurrent state at each horizon. Frozen evaluation
must retain each policy's training horizon. The native GAE implementation
uses a final zero-advantage bootstrap slot; its boundary behavior predates
these experiments and is unchanged here.

The completed short run executed 3,276,800 steps in 571.691 s, or 5,731.76
learner steps/s. Its changing-policy training outcomes were 338 wins,
138 losses and 36 ties. These cannot substitute for frozen evaluation.
PPO occupied 1.563% and rollout 98.400% of the measured CUDA stream envelope.
Physics and controller execution remained on CUDA with no native CPU
environment workers. Host orchestration still uses CPU time.

## Policy input coordinates and units

The raw 223-float simulator observation contains world coordinates and
unscaled physical quantities. On 2,048 recorded replay observations, the
two round-time fields account for 94.505% of squared input magnitude and
the two tilt fields for 5.072%. Using the actual pristine checkpoint,
36.876% of first-layer update gates and 40.179% of highway gates have
sigmoid outputs outside [0.01, 0.99]. Dividing round times by 120 and tilt
by 180 reduces both fractions to zero in this trace. This is an offline
gate-sensitivity measurement, not a measured improvement in fighting.

Two explicit policy-only views are available:

- `polar_xy_v1`: replaces opponent world XY with horizontal range in metres
  and bearing relative to self-root heading in radians.
- `scaled_polar_xy_v1`: also divides bearing by pi, tilt by 180, and round
  duration/remaining time by 120. No clipping or adaptive statistics occur.

The raw simulator observation, opponent inputs, physical state, rewards,
legal masks and controller remain unchanged. `raw` remains the default.
Transformed checkpoints carry an encoder descriptor and fingerprint;
evaluation rejects an incompatible view even when the vector width matches.
Initialization from untrained raw weights requires the explicit
`raw-initial-weights` mode and a pinned source hash. It is a declared change
of input coordinates, not a compatible resume of a trained raw policy.

CPU geometry, inverse-coordinate, metadata and integration tests pass. Both
views also passed a CUDA capture fixture with changing observations, stable
buffers and invalid-input status checks. That fixture observed zero CPU/CUDA
output difference and unchanged raw state, opponent actions, rewards and
masks. Learned combat outcomes for these views remain to be measured.

## Frozen evaluation and scripted diagnostics

`evaluate_gpu_dummy.py --greedy` selects the highest-logit legal action.
Default evaluation retains stochastic categorical sampling. Native and
Python guards prohibit PPO updates from greedy rollouts. Native tests cover
changing masks, tie selection and immutable checkpoint hashes; the default
stochastic path matched the previous extension's 128 tested actions exactly.

`gpu_grouped_frozen_eval.py` partitions one physics batch among frozen
policies. Each retains separate state, sampling, encoder and horizon. All
policies select actions before the shared environment advances once. Its
synthetic CUDA regression matched all 512 actions against separate native
instances, including interior terminals and changing legal masks. Reported
wall time is shared batch time and must not be added across policy groups.

The separately labeled scripted diagnostic tested eight strategies with
16 completed rounds each. Facing without attacking won 15/16; facing and
front-kicking won 14/16; repeated right-side kicks lost 16/16. The face-only
policy's points came entirely from opponent falls. These are scripted
baselines with small, shared-start cohorts, not learned-policy results or
independent scenario trials.

## Evidence

Full commands, output, checkpoints and JSON results are external to Git:

- `C:/rekagent/evidence/rek-training-winrate-20260911-v1`
- `C:/rekagent/evidence/rek-training-opt-20260911/reward-bandit-v1`
- `C:/rekagent/evidence/rek-training-opt-20260911/reward-bandit-replay4-v1`
- `C:/rekagent/evidence/rek-training-opt-20260911/first-mingru-gates-cpu-v1`
- `C:/rekagent/evidence/rek-training-opt-20260911/policy-encoder-cuda-001`
- `C:/rekagent/evidence/rek-training-opt-20260911/scripted-baseline-summary.json`

The short training process produced its complete report and final weights.
Its outer transfer script subsequently failed after an in-flight script edit;
the separate `credit-short-collect-01` command recovered the unchanged output.
The shell failure remains in the command evidence and is not hidden.
