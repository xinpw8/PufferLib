# Native PPO on captured authentic trajectories

`authentic_ppo.cu` reuses the prepared PufferLib5 PPO loss, encoder/MinGRU/
decoder backward kernels, and Muon optimizer. Its inputs are complete recorded
policy observation streams and a separately frozen behavior replay, not a
simulated environment or demonstration label corpus. It performs no game
connection, environment stepping, or Python/Torch training.

```sh
bash ocean/rek_g1/native5/build_authentic_ppo.sh PREPARED_BC_BUILD NEW_BUILD
NEW_BUILD/authentic-ppo DATA.bin BEHAVIOR.bin INITIAL.bin INITIAL_SHA256 NEW_OUTPUT.bin EPOCHS LEARNING_RATE HORIZON CLIP VF_CLIP VF_COEF ENT_COEF [--allow-bounded-bf16-batch] [--targets=complete-mc-zero-baseline]
```

Epoch count is restricted to 0, 1, or 2. Zero epochs performs target/parity
validation and an exact checkpoint round trip. Output and all epoch snapshot
paths must be fresh. Each epoch also writes `NEW_OUTPUT.bin.epoch-N.bin`.
Live evaluation and GPU scheduling are separate, explicitly coordinated steps.

## Immutable behavior and chronology

The `REKRL001` dataset and `REKBR001` frozen replay formats are defined by
`authentic_trajectory.h`. Replay is bound to the exact dataset SHA256, initial
checkpoint SHA256, seed, chosen actions, and all 34 original native logits.
Both old log probabilities and old values remain FP32 and immutable across
updates. They are never recomputed using updated weights. The initial policy
must be the same checkpoint that generated the captured behavior.

Each full recorded round is one sequence. Every original worker observation,
native legality mask, and reset is retained, including terminal-race requests
that did not apply. Those rows receive actor weight 0, including zero entropy
gradient, and value weight 1. They are not silently dropped from recurrent
history. Padding receives zero actor and value gradient. Full preceding
sequence replay uses current weights as detached burn-in before each chunk's
recurrent backpropagation. This retains complete state chronology without
using stale recurrent state from before an optimizer update.

Before any update, the trainer compares its forward output against every
frozen behavior row. Maximum absolute logit error, value error, and deviation
of the importance ratio from 1 must each be at most 0.0001. A mismatch is
reported and stops optimization in the strict default mode.
Source reward/terminal validation and action replay are necessary but do not
replace this teacher-versus-training-forward test.

The explicit `--allow-bounded-bf16-batch` option accepts a measured numerical
approximation only when the existing Puffer one-step forward matches every
teacher logit and value exactly, the maximum chosen-action ratio deviation is
at most `0.1 * CLIP`, and the initial clipping fraction is zero. This is not
exact batch parity. The mode and acceptance decision are recorded alongside
full legal-distribution KL, ratio quantiles, clipping fraction, and value
diagnostics. Strict acceptance remains the default. No activation-cache or
backward-kernel replacement is introduced.

## Native loss and CUDA targets

The build adapter extracts the prepared native `PPOGraphArgs` and
`ppo_loss_compute` definitions. It makes a named overload with FP32 pointers
and scalar reads for frozen old log probabilities and old values. Reversing
those substitutions must reproduce the original definitions byte for byte.
Clipping, value loss, entropy, reductions, and derivatives are unchanged.
The current log-probability cache is already FP32. No BF16 subtraction
compensation is used. Advantages and returns are consumed in BF16 as in the
native loss; that conversion occurs on CUDA.

Advantages are not centered or normalized. The pinned prepared pipeline also
passes raw advantages directly from `puff_advantage` into `ppo_loss_fwd_bwd`.
Unlike its per-minibatch current-value/V-trace target path, this experiment
freezes complete-round targets from the recorded behavior values. This
difference is intentional and separate from reusing the native loss kernel.

Actual GAE/return targets are produced on CUDA, one sequential recurrence per
closed round, using recorded per-transition gamma/lambda and frozen old
values. Computation uses a FP64 recurrence and FP32 outputs. The host-only
`authentic_gae.h` is an independent verification reference, never the source
of training targets. All CUDA targets must match that reference within 1e-6.
For transition t:

`delta[t] = reward[t] + gamma[t] * next_old_value[t] - old_value[t]`

`advantage[t] = delta[t] + gamma[t] * lambda[t] * next_advantage[t]`

True terminal transitions set both next terms to zero. A closed recorded
terminal is required; truncated trajectories are not silently treated as
terminal. Observed awarded points and actual round outcomes define rewards
under the declared `round_outcome_v1` potential contract. The exporter owns
timestamp alignment and per-transition elapsed-time discounting.

The optional `--targets=complete-mc-zero-baseline` control requires
`VF_COEF=0`. CUDA computes complete shaped Monte Carlo returns backwards over
each closed round, `return[t] = reward[t] + gamma[t] * return[t+1]`, with zero
continuation at the true terminal, and sets `advantage[t] = return[t]`.
Neither target uses the learned old value. Merely setting GAE lambda to one
would still subtract that value and would not implement this control. The
potential-shaped return already includes the known state-potential baseline.
The baseline-invariance identity applies to the on-policy unclipped policy
gradient, not to arbitrary finite-epoch clipped PPO updates. Default target
mode remains frozen-value GAE.

## Numerical diagnostic on corrected recorded data

The corrected four-round dataset SHA256 is
`f2b67a599600bcca878085a942a6ae9b174e4d054aff0bd3cddf3d883c6d3b04`.
Its frozen replay SHA256 is
`06c14f853cba325fd276fa9e97102f3917a623c6c9969722d43aca2c161cb171`.
The base 50 Hz discounts are gamma `0.9998844821426083` and lambda
`0.9978673240629938`, exponentiated by each actual elapsed duration. Earlier
export r1 used incorrect discounts, remains preserved, and received no updates.

With unchanged weights, sequential Puffer logits and values matched the
native teacher exactly for all 22,585 rows. BF16 batch arithmetic gave mean
legal KL `4.5191e-8`, maximum KL `0.000156797`, chosen ratios from
`0.9842514` to `1.0061889`, median and 95th percentile ratio 1, and zero initial
clipping at 0.2. Maximum batch logit/value differences were 0.125/0.0625.
The original strict check rejected this run and no optimization occurred.
The bounded option was subsequently added explicitly on this evidence.

The frozen critic is miscalibrated on these authentic rounds: mean value
7.91780 versus complete shaped Monte Carlo return 0.17283. Default GAE targets
retain substantial bootstrap influence (mean 6.19228). GAE advantages have
mean -1.72551 and population standard deviation 3.99732. These are diagnostic
observations, not evidence of policy improvement.

The first dataset comprises four development rounds from the same original
policy. There is no heldout split in this fine-tuning dataset. PPO loss, value
error, clipping fraction, and approximate KL measure optimization behavior,
not an independently established win-rate improvement. Preserve raw captures,
dataset/replay hashes, native source/build provenance, parity reports, and
per-epoch checkpoints for subsequent authentic evaluation.
