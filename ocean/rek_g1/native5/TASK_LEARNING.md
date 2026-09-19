# Task-specific credit assignment

The compact candidate runs at 50 control decisions per second. One normal round
lasts 120 s. The inspected motion catalog's longest move lasts 3.16 s. The
recovered no-recovery count lasts 3 s; the recovered KO count is separately 10 s
and double-knockdown count is 20 s. The compact runtime does not model those
knockdown dynamics. These time scales do not establish physics or live-game
parity.

`task_learning_profile.cjs` produces a reproducible, optional experimental
profile. Its reward discount half-life of 120 s and combined GAE trace half-life
of 6.16 s are declared hypotheses. They have not been demonstrated optimal.
For control interval `dt`, it derives

```
gamma = exp(log(0.5) * dt / reward_half_life_seconds)
lambda = exp(log(0.5) * dt / trace_half_life_seconds) / gamma
```

The combined `gamma*lambda`, rather than lambda alone, determines trace decay.
A 512-transition rollout covers 10.24 s and is the next power of two that covers
3.16 + 3 s. The profile keeps recurrent carry enabled and inherits the existing
entropy coefficient of 0.01. A cutoff still truncates gradient propagation even
when recurrent state is carried. A longer rollout or a different half-life is
an experimental choice, not a claim of improved fighting ability.

The prior 128-step profile covers 2.56 s, with only 2.54 s of internal transitions
in the unpatched learner. Gamma 0.999 has a reward half-life of approximately
13.86 s and retains approximately 0.00247 of a reward 120 s away. With lambda
0.995, the combined trace half-life is approximately 2.31 s.

## Verified learner boundary defect

The audited build is PufferLib commit
`773f923d80e73bdc255a2ba730c918b28e416aa1`, with the existing action-mask and
initial-checkpoint patches. The exact inspected sources on Spark are:

```
/home/spark-advantage/rek-training/contact-potential-20260918-r1/build-candidate/trainer/src/pufferl.cu
/home/spark-advantage/rek-training/contact-potential-20260918-r1/build-candidate/trainer/src/algo.cu
```

The original `algo.cu` SHA-256 is
`20c5f33b036ad43c66492ca6be82265675b8f620bfcb989eb433d2f04f27cd98`.
`pufferl.cu:818` stores the observation, preceding reward and terminal flag;
`:1296` performs H sample-and-step iterations. `algo.cu:1702` computes
advantages using reward[t+1] and done[t+1], but leaves advantage[H-1] zero.
The last executed action's reward is copied to the following batch's slot zero,
which the next advantage computation never consumes. This omits one actual
transition per rollout, 1/128 = 0.78125% for the baseline. The full PPO loss
still includes that slot for entropy and value clipping. It does not receive
its missing reward-driven advantage.

`probe_pufferlib5_temporal_credit.cjs` extracts and compiles the exact hashed
CUDA kernel. With H=8, a five-point terminal reward in observed slot seven
correctly credits action six. A five-point terminal reward from the final
executed action gives action seven an advantage of zero instead of five;
putting that reward in the next batch's slot zero does not recover it. Internal
termination correctly prevents rewards from the next episode leaking backward.

`pufferlib5_temporal_credit.patch` adds a separate boundary reward, done and
value tuple, then computes the final action's advantage. A value-only native
network forward uses copied recurrent state and consumes no action samples or
RNG. It adds no environment step. The opt-in macro is
`PUFFER_ENV_GPU_ROLLOUT_BOOTSTRAP`; apply this patch after the two existing
trainer patches. The implementation explicitly requires synchronous GPU mode,
one environment buffer and one learner policy. External frozen-opponent rows
managed by the REK runtime are compatible with that single learner policy.

The boundary value is an actor-weight snapshot from the end of the rollout.
The pinned learner recomputes internal values during each training minibatch;
the fixed boundary snapshot remains constant for that update. This should be
recorded when experimenting with replay ratios above one. The profile retains
the existing replay behavior.

The fixed-kernel regression checks all 40 advantages and returns against a
scalar reference, with both terminal and nonterminal batch endings. The actual
native network test checks that copied scratch state advances while actor
state, RNG, actions, observations, masks and parameters remain unchanged.
It also captures and replays the helper as a CUDA graph four times. Neither
test performs an optimizer update.

## Reward and boundaries

`task_learning_contract.h` keeps actual point differential as the base reward.
Its default terminal win bonus is zero. An explicitly configured W/L bonus is
an additional reward in game-point units, with the same value for every win
path. Actual KO points, if supplied by an authentic runtime, already occur in
the point delta and are never added a second time by this helper. Reward
scaling does not change scoreboard points. The terminal transition must be
reported exactly once.

The reward helper is a reference contract only in this change. The runtime
continues to emit raw point differential, and the profile does not export a
terminal-bonus environment setting. A nonzero bonus in the descriptive profile
requires separate runtime integration before an experiment can use it.

A round ending at its game-defined 120 s limit is a true episode termination:
zero next-state bootstrap, stop the GAE trace, and reset recurrent state before
the next initial observation. A rollout cutoff bootstraps and carries memory.
An external time-limit truncation needs the final observation's value and a
distinct episode-restart signal. The pinned API exposes only one terminal
flag, so separate truncations remain unsupported; do not label them terminal
or infer support from the reference contract test.

The runtime must emit the next initial observation and mask together with the
preceding terminal reward/done before selecting the first action of the new
episode. The parent integration supplies this separate autoreset fix. Standalone
evaluation can retain its final-state observation contract.

## Exploration interpretation

PPO entropy regularization is a loss term, not intrinsic environment reward.
The categorical policy has at most 33 legal actions, with maximum entropy
log(33). During a busy attack the mask permits four hold/yaw categories, with
maximum log(4); during reset wait, only hold is legal and entropy is zero.
Repeated held-action samples at 50 Hz do not constitute independent executed
strikes. Report normalized entropy conditioned on legal count, accepted attack
starts per second, busy fraction, and accepted executions per route when
assessing exploration. Increasing entropy alone does not demonstrate better
behavioral coverage.

The files here establish training accounting and testable timing hypotheses.
They do not qualify the compact physics, contact rules, or policy for authentic
REK transfer.
