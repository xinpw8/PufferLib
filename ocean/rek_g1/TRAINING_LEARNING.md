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

The completed raw long-credit run executed the same 3,276,800 steps in
584.430 s, or 5,606.83 learner steps/s. Its changing-policy outcomes were
323 wins, 162 losses and 27 ties. Mean points increased from 11.68164 to
13.27930, but conceded points increased from 7.64258 to 10.19141. Facing
within 30 degrees increased from 26.784% to 35.069%. These observations do
not establish a better final policy. Its final checkpoint SHA-256 is
`81cb797090e329a02408c966cd30765ed16eceb9c2d97658c3394423210417b2`.
PPO occupied 2.134% and rollout 97.857% of its CUDA stream envelope.

Each run supplied only 128 simulated seconds per arena, while a completed
round averaged approximately 120.15 s. Most first-round completions appeared
at short epoch 94/100 or long epoch 24/25. Replay increases reuse, not the
number of fresh rounds experienced. Any longer-budget experiment should
retain its optimizer within the same process and evaluate frozen checkpoints
separately. Disabling recurrent horizon resets is not a compatible shortcut:
the current learner does not store replay-window initial recurrent states.

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
masks.

The completed `scaled_polar_xy_v1` long-credit run held the raw long arm's
initial weights, learner parameters, reward and environment fixed. It used
3,276,800 learner steps in 580.354 s, or 5,646.21 learner steps/s. Its online
round win percentage was 60.547%, with mean points 15.99219 against 13.02930.
Facing was 37.652% and attack-facing 50.523%. Own falls increased to 1.68359
per round. These changing-policy outcomes are separate from the frozen
screening below. The final checkpoint SHA-256 is
`0f53d7fc0018dee0756b90a0fc4f7f0577964cfd73aa0d5e3f2815c75f687154`.

## Optional training-only facing potential

`--facing-potential-scale` defaults to zero and allocates no shaper at zero.
A positive value applies `raw_reward + gamma*Phi(next) - Phi(current)` only
to the learner reward buffer, where `Phi=scale*cos(ego opponent bearing)`.
Raw game scores, opponent inputs, simulation state and evaluation rewards
remain unchanged. The scale is a training choice, not an extracted game rule.
Both current and next terminal potentials are zero, preserving the actual
delayed-reset transition boundary. The discount must match the native
learner's float32 gamma; combined reward clipping is prohibited.

In real arithmetic the discounted shaping sum telescopes to
`-Phi(initial) + gamma**T*Phi(final)`. This does not guarantee improvement
under finite-window, approximate PPO learning. CPU tests cover this identity,
reset boundaries, immutable raw buffers and independent checkpoint metadata.
An actual CUDA capture fixture passed 12 synthetic transitions with mixed
terminals and an explicit reset; CPU/CUDA reward error was zero. No learning
run has used this optional reward yet.

## Optional round-win training objective

`--reward-objective round-win` changes the learner's optimization target to
the completed round's win indicator. It requires native gamma exactly one
and zero reward clipping. The base reward is one only when the published
terminal outcome is a points or knockout win for the learner's side. Losses,
ties and redos return zero. Raw game scoring and evaluation are unchanged.

The training-only potential is
`Phi = margin_potential_scale*tanh((own_points-opponent_points)/margin_points)`;
the added reward is `Phi(next)-Phi(current)`. Both terminal potentials are
zero. Therefore a full round beginning at zero score returns its win
indicator in real arithmetic, independently of its length or winning margin.
Finite-window PPO, numerical arithmetic and bootstrapped values remain
approximate. This reward does not guarantee that PPO finds a winning policy.

`config/rek_g1_round_win.ini` uses constant learning rate 0.003, horizon 256,
lambda 0.995 and replay ratio four. Margin scale 0.5 and margin points 5 are
explicit training choices. Nonzero facing shaping cannot be combined with
this objective. Existing score-delta training remains the default. Checkpoint
sidecars record the reward objective independently of the input encoder.

CPU tests classified 384 previously recorded round outcomes. A CUDA fixture
covered 24 synthetic transitions and two captured graphs, then 18 transitions
through the production wrapper including an explicit reset. Reward error
against the CPU reference was zero. Game state, opponent actions, scores,
masks and metrics remained unchanged. This is a tensor/wrapper test; it does
not establish improved learning or exercise physical trajectories.

The first completed round-win run warm-started the scaled-polar checkpoint
`0f53d7fc0018dee0756b90a0fc4f7f0577964cfd73aa0d5e3f2815c75f687154`,
with a fresh optimizer retained for all 100 rollouts. It executed 13,107,200
additional learner steps in 2,320.452 s, or 5,648.56 steps/s. Physics and
controller execution remained CUDA-only; CPU environment workers, physics
steps and controller inferences were zero. Host orchestration consumed
approximately one CPU core. PPO occupied 2.160% of the measured CUDA stream
envelope; rollout occupied 97.825%.

Its changing-policy outcomes were 1,393 wins, 550 losses and 105 ties across
2,048 completed rounds, or 68.018% wins. Mean points were 14.78662 versus
10.63721; learner/opponent falls were 1.19141/1.80908 per round. Facing was
32.264%, attack-facing 46.919%, and mean root separation 1.10301 m. The
recorded 12,247 scored hits are arena totals, not learner-attributed hits.

At the completed 512-round boundaries, successive online win counts were
369, 353, 370 and 301. The last cohort was weaker. These observations do not
show consistent learning improvement and cannot replace frozen evaluation.
The final checkpoint SHA-256 is
`e71febdacba546dad91c5fe20cfa0f64a493d8fd3e4281a353fccda79757e316`.
Earlier verified checkpoints are retained for separately labeled selection.

The Python training process wrote the complete report and all checkpoints.
An edit to its running outer Bash launcher caused a trailing syntax error
after training. That shell failure is preserved in `round-win-long-run-01`.
The independent `round-win-completion-collect-01` command recovered the full
unchanged output under `round-win-long-r1-completion-recovery-v1/` in the
current evidence root. It is not recorded as a successful outer launch.

The completed greedy screening selected recorded checkpoints nearest 25%,
50% and 75% of this run, plus its final checkpoint. Each used 32 arenas,
6,400 control ticks, the actual 256-tick training horizon and unchanged
weights against the original candidate dummy.

| Additional trained steps | Wins / losses / ties | Own / opponent points |
| ---: | ---: | ---: |
| 3,145,728 | 25 / 7 / 0 | 20.59375 / 14.87500 |
| 6,291,456 | 23 / 6 / 3 | 14.96875 / 9.43750 |
| 9,437,184 | 27 / 5 / 0 | 17.62500 / 10.96875 |
| 13,107,200 | 26 / 5 / 1 | 14.37500 / 7.68750 |

None meets the 100% target. The best count in this selection is 27/32,
below the earlier scaled-polar screen's 29/32 and the raw initial screen's
31/32. These small, separately executed screenings do not isolate a causal
policy-strength difference. The selected round-win checkpoint is
`e8041a203d1588b2ddab10edf9fe3eb0742424ec07df13bf08bd31c73cf9756d`.
Selection is not held-out validation. Reports and immutable weight hashes
are under `round-win-screen-r1/evaluations/` in the current evidence root.

The declared next credit-assignment trial is
`config/rek_g1_round_win_lr0003_h1024.ini`: constant LR 0.0003, gamma 1,
lambda 0.999, horizon 1,024 ticks (20.48 s), replay ratio 4 and the existing
horizon memory reset. Its 7,340,032-step plan contains 14 complete rollouts
with 512 learner arenas and 7,168 optimizer minibatches of 4,096 samples.
The lower LR responds to the previous run's final approximate KL of 0.03278
and clip fraction 0.21147; the longer horizon extends temporal credit.
These are experimentally motivated settings, not demonstrated improvements.
Seven CPU runner tests passed, including exact rollout and minibatch counts.

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

The first complete grouped stochastic screening used 32 arenas per policy,
128 arenas total, seed 73, and 6,400 control ticks per arena. Every policy's
training horizon was retained; no updates occurred and every checkpoint hash
remained unchanged. Each group completed 32 rounds.

| Frozen policy | Wins / losses / ties | Win percentage | Own / opponent points |
| --- | ---: | ---: | ---: |
| Initial, raw | 22 / 6 / 4 | 68.750% | 13.06250 / 8.18750 |
| Short credit, raw | 23 / 7 / 2 | 71.875% | 12.21875 / 7.62500 |
| Long credit, raw | 21 / 11 / 0 | 65.625% | 13.21875 / 8.75000 |
| Long credit, scaled polar | 27 / 4 / 1 | 84.375% | 18.65625 / 11.46875 |

This small, shared-start screening favors the scaled-polar checkpoint. It
does not establish a robust multi-seed improvement or meet the 100% target.
Three of its four lost rounds included learner falls; the remaining loss
occurred without a learner fall. The tie had no points or contacts.
The evaluation is in `grouped-stochastic-r1/evaluations/` under the current
evidence root. Its instrumented evaluation speed is not training SPS.

The matching greedy screening completed 32 rounds per policy with unchanged
weights and the same seed, horizons and shared physical batch dimensions:

| Frozen policy | Wins / losses / ties | Win percentage | Own / opponent points |
| --- | ---: | ---: | ---: |
| Initial, raw | 31 / 1 / 0 | 96.875% | 11.40625 / 4.43750 |
| Short credit, raw | 23 / 9 / 0 | 71.875% | 11.90625 / 7.87500 |
| Long credit, raw | 23 / 7 / 2 | 71.875% | 9.90625 / 5.46875 |
| Long credit, scaled polar | 29 / 0 / 3 | 90.625% | 19.65625 / 10.40625 |

The scaled-polar policy's ties were 4-4, 13-13 and 15-15. Zero losses is not
100% wins. Its higher scoring does not beat the initial policy's 31/32 win
count. These small screenings justify neither superhuman performance nor
an established improvement across independent starting conditions. Full
reports are under `grouped-greedy-r1/evaluations/` in the evidence root.

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
