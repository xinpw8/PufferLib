# Fixed older-policy mixture and weights-only warm start

With `env.opponent_checkpoint` supplied, `REK_FROZEN_OPPONENT_FRACTION` selects
which arenas use that checkpoint as fighter 1. The default is 1, preserving
the original all-frozen behavior. Fractions must be finite and in `[0,1]`.

Selection occurs once at startup using `env.seed`. A partial seeded shuffle
chooses `floor(arenas * fraction)` unique rows. A positive fraction selects at
least one frozen arena, even when the batch or fraction is small. At zero,
all opponents come from the runtime. At one, all are frozen. Thus a one-arena
batch with any positive fraction necessarily has one frozen opponent.

Fighter-0 learner overrides remain zero. Selected fighter-1 overrides are 1;
unselected fighter-1 overrides are 0, preserving the runtime's GPU opponent
selection/mix. The frozen policy still infers all arena rows, and unselected
outputs are ignored. There is no additional CPU work inside rollout steps.
The startup `native5_frozen_opponent_mix` JSON records requested fraction,
actual frozen/runtime counts, seed, and inference batch size.

Example additions to an existing native training command:

```sh
REK_FROZEN_OPPONENT_FRACTION=0.5 ./puffer-rek-native5 train --headless \
  --env.opponent_checkpoint=/private/older-policy.bin \
  --env.opponent_sha256=EXPECTED_SHA256 \
  --env.opponent_observation_encoding=scaled_polar_xy \
  --env.opponent_hidden_size=256 --env.opponent_num_layers=2 \
  --env.opponent_precision=0 --env.opponent_deterministic=0
```

These are extra arguments, not a complete runtime configuration. Precision 0
means BF16 inference. Checkpoints store FP32 coefficients. Existing native
masking and terminal recurrent-state resets are unchanged. This is a fixed
older-policy mixture for a run; it does not automatically refresh checkpoints
or create a self-play pool.

Run the exact host-only helper tests with Node and a C++17 compiler:

```sh
node ocean/rek_g1/native5/frozen_mix.test.cjs
```

## Pinned trainer warm start

The archived native trainer pin previously read `base.load_model_path` only
for evaluation/match, despite the mutable checkout containing a newer training
load path. `pufferlib5_initial_model.patch`, applied after the action-mask patch,
adds the load to the archived `run_train` implementation:

```sh
./puffer-rek-native5 train --headless \
  --base.load_model_path=/private/previous-stage.bin
```

The file must have exactly the expected FP32 parameter byte count. All ranks
load policy 0 before first-use graph capture and rollout; asynchronous actor
parameters are refreshed too. This is **weights-only initialization**.
Optimizer momentum, recurrent state, RNG, global step, and learning-rate
schedule are newly initialized. It is not a full training-state resume.

The artifact owner saves a step-zero readback under the new run's checkpoint
directory as `0000000000000000.bin` and prints `Loaded initial policy`,
`step=0`, and `fresh_optimizer=1`. Before accepting a continued stage, compare
that readback's SHA-256 to the intended source checkpoint. Equality verifies
all initial FP32 parameters, not merely a sample. Keep both files private.
Use a fresh run directory and an explicit verified path; `latest` searches
checkpoint modification metadata and is less precise for audited curricula.

## Credit across canned actions

At 50 Hz, horizon 16 contains 0.32 s, while the longest current canned move
contains 158 ticks, 3.16 s. The previous gamma 0.99 and GAE lambda 0.95 give
`(gamma * lambda)^50 = 0.04655` and `^158 = 0.00006176`. A short horizon still
bootstraps a value estimate; it does not provide a 3.16-second direct advantage
or recurrent-gradient path. Recurrent state carry alone does not extend
truncated backpropagation.

For comparison, horizon 128 covers 2.56 s. Gamma 0.999 and lambda 0.995 give
`(gamma * lambda)^128 = 0.46317` and `^158 = 0.38672` before horizon truncation.
Gamma's discount half-life increases from 1.38 s to 13.86 s. These settings
retain more delayed reward signal but can increase variance and compute; they
require measured training/evaluation. They still do not span every move or a
300-second round directly, so value bootstrapping remains important.
