# Authentic-trajectory native PPO controls

Two bounded one-epoch controls were trained from the same original native
checkpoint on four complete authentic development rounds. Both are ready for
authentic evaluation. Neither has an established live win or combat improvement
in this report. There is no heldout trajectory split.

## Inputs and implementation

- 22,585 original recurrent observations with actual legal masks and resets.
- 22,582 applied requests receive actor weight 1. Three terminal-race requests
  receive actor weight 0 and value weight 1; their recurrent inputs and terminal
  rewards remain in the sequence.
- Dataset SHA256:
  `f2b67a599600bcca878085a942a6ae9b174e4d054aff0bd3cddf3d883c6d3b04`.
- Frozen behavior replay SHA256:
  `06c14f853cba325fd276fa9e97102f3917a623c6c9969722d43aca2c161cb171`.
- Initial outcome checkpoint SHA256:
  `61f97b0b0a4504c6bdd0ee16d369ad4c1915e3cdf73d6358bab01ce64c8fde3f`.
- Native trainer executable SHA256:
  `b5c2c61932ad63c4dc40c137a6ef493a423826306aebf83859efdb5c0197da1a`.
- Training-source snapshot SHA256:
  `e8f8d86c2deee3bc39199219683fad630e11f4cb134fecb4a483d271149d7c4f`.

The dataset and frozen replay contracts are documented in
[AUTHENTIC_TRAJECTORIES.md](../../../AUTHENTIC_TRAJECTORIES.md).
[AUTHENTIC_PPO.md](../../AUTHENTIC_PPO.md) documents the native trainer and
loss adapter. It reuses the prepared Puffer5 encoder, MinGRU, decoder backward,
PPO loss and Muon. No game connection, simulator rollout, Python or Torch is
used during this training. Learning targets are computed on CUDA and checked
against an independent CPU reference. Full current-weight prefix replay
provides detached recurrent burn-in before each training chunk.

Frozen old log probabilities and old values remain FP32. The narrow native
loss overload changes their pointer types and reads while leaving the native
loss mathematics unchanged. Current BF16 network computation and native BF16
advantage/return inputs are retained. Advantages are unnormalized, as in the
pinned native pipeline.

## Numerical acceptance

Original native behavior replay reproduced all 22,585 sampled actions exactly.
With unchanged parameters, the existing Puffer one-step forward also matched
every frozen logit and value exactly. Batch BF16 forward had maximum
logit/value differences 0.125/0.0625 from arithmetic grouping. Its full legal
distribution mean KL was `4.51907388105e-8`, maximum KL `0.000156796977545`.
Chosen-action ratios ranged from `0.984251417480` to `1.00618886839`, with both
median and 95th percentile equal to 1. Initial clipping fraction at 0.2 was 0.

The strict default 1e-4 maximum-error check rejected the diagnostic and allowed
no optimizer updates. An explicit bounded BF16 option was then approved and
implemented: exact sequential logits/value equality, maximum chosen-ratio
deviation at most `0.1 * CLIP` (0.02 here), and zero initial clipping. The mode,
evidence and decision are logged. This is a measured batch approximation,
not exact batch parity. Default strict behavior is unchanged.

The original r1 export used incorrect base discounts 0.999/0.995. It received
no optimizer updates and remains preserved. Corrected r2 uses the original
task's base gamma `0.9998844821426083` and lambda `0.9978673240629938` per
0.02 s, exponentiated by each captured elapsed duration. Observation, action,
legality, timing and reset records are unchanged.

## Control definitions and measured updates

Both runs used one epoch, learning rate 0.00001, horizon 128, PPO clip 0.2,
value clip 0.2, entropy coefficient 0.001 and 178 optimizer updates. Each
started from the original checkpoint with fresh optimizer state, rather than
continuing from the other control.

| Control | CUDA advantage target | Value coefficient | Start/end UTC, 2026-09-20 | Wall time |
| --- | --- | --- | --- | --- |
| Frozen-value GAE | Variable-duration GAE using immutable original values | 0.5 | 01:18:47.863044834 / 01:18:50.872633776 | 3.009589 s |
| Complete MC, zero learned baseline | Complete shaped return directly | 0 | 01:18:50.877897391 / 01:18:53.714041386 | 2.836144 s |

Timing includes target validation, initial distribution checks, exact
sequential diagnostic, optimizer updates, final full-dataset diagnostics and
checkpoint writes. It excludes compilation. These are small development
updates, not a throughput benchmark.

The GAE critic is materially miscalibrated: frozen mean value 7.91780 versus
complete shaped Monte Carlo return 0.172831. Mean GAE return is 6.19228;
GAE advantage mean is -1.72551 and population standard deviation 3.99732.
The complete-MC control sets `advantage = shaped_return` and `VF_COEF=0`, so
neither actor targets nor an active critic loss use that learned baseline.
Setting lambda to one alone would still subtract the old value and would
not implement this control. Terminal-potential zero makes the shaped return
equivalent to discounted outcome minus the known current-state potential.
Baseline invariance applies to the unclipped on-policy gradient, not to
arbitrary finite-epoch clipped PPO updates.

Full-dataset diagnostics after the final optimizer update, using batch
forward and the immutable original behavior distribution:

| Metric | Frozen-value GAE | Complete MC, zero learned baseline |
| --- | ---: | ---: |
| Mean legal KL, old to current | 0.000328106563 | 0.000241301795 |
| Maximum legal KL | 0.0538424949 | 0.0411665206 |
| Chosen-action approximate KL | 0.000333408952 | 0.000245815801 |
| Chosen-ratio minimum / median / 95th percentile / maximum | 0.629163 / 1.001071 / 1.033980 / 1.465196 | 0.718190 / 0.999755 / 1.039939 / 1.279705 |
| Clipped fraction, all rows | 0.000619880452 | 0.000309940226 |
| Value MSE to this control's target, before / after | 18.9559265 / 16.9794798 | 116.547674 / 116.299499 |

MC value errors are diagnostic only because its critic loss coefficient is
zero. Actor updates still change the shared recurrent representation. These
full-dataset numbers differ from the progressively measured training losses;
neither establishes authentic fighting performance.

Checkpoint SHA256 values:

- `train-gae-r2/ppo-one-epoch.bin`:
  `3cab45333a414dd6f9b0bdf978e8b75d1f1c13e2a4fdf8213ecfe5f17efb35f8`.
- `train-mc-zero-r2/ppo-one-epoch.bin`:
  `a985d6c06b5dfab7198319caf5e99099b85da30eb40402758cc254e407e2059f`.

Each matching `.epoch-1.bin` snapshot has the same hash. The existing
223-observation, 33-action recurrent checkpoint layout is unchanged.

## Reproduction

Use the prepared native BC build described in [BC_TRAINING.md](../../BC_TRAINING.md).
Set these paths to the exact, hash-verified recorded artifacts above, and use
fresh output paths. GPU scheduling and any later game evaluation remain
separate operations.

```sh
PREPARED_BC_BUILD=/path/to/prepared-bc-build
NEW_BUILD=/path/to/fresh-authentic-ppo-build
DATA=/path/to/authentic-trajectories.bin
REPLAY=/path/to/behavior-replay-r2.bin
INITIAL=/path/to/0000000033554432.bin
INITIAL_SHA=61f97b0b0a4504c6bdd0ee16d369ad4c1915e3cdf73d6358bab01ce64c8fde3f
OUTPUT=/path/to/fresh-results

bash ocean/rek_g1/native5/build_authentic_ppo.sh "$PREPARED_BC_BUILD" "$NEW_BUILD"
sha256sum "$DATA" "$REPLAY" "$INITIAL" "$NEW_BUILD/authentic-ppo"
mkdir "$OUTPUT"

"$NEW_BUILD/authentic-ppo" "$DATA" "$REPLAY" "$INITIAL" "$INITIAL_SHA" \
  "$OUTPUT/gae.bin" 1 .00001 128 .2 .2 .5 .001 \
  --allow-bounded-bf16-batch >"$OUTPUT/gae.stdout.jsonl" 2>"$OUTPUT/gae.stderr.txt"

"$NEW_BUILD/authentic-ppo" "$DATA" "$REPLAY" "$INITIAL" "$INITIAL_SHA" \
  "$OUTPUT/mc-zero.bin" 1 .00001 128 .2 .2 0 .001 \
  --allow-bounded-bf16-batch --targets=complete-mc-zero-baseline \
  >"$OUTPUT/mc-zero.stdout.jsonl" 2>"$OUTPUT/mc-zero.stderr.txt"

sha256sum "$OUTPUT/"*.bin
```

For a diagnostic-only run, replace epoch count 1 with 0 and use a fresh output
path. Omitting the numerical option retains strict acceptance and reproduces
the initial batch-arithmetic rejection. Do not substitute the incorrect r1
discount export or an updated checkpoint for the frozen behavior checkpoint.

## Verification and retained evidence

CPU checks passed: 24 target-reference checks, 14 numerical-acceptance checks,
and two native-kernel adapter tests. On each actual 22,585-row control, CUDA
targets matched the CPU reference with maximum absolute error 0. Both exact
sequential teacher checks passed. No Compute Sanitizer run is claimed for
this authentic PPO executable.

Retained stage artifacts include `controls-r2.commands-and-stdout.txt`,
`controls-r2.stderr.txt`, `build-r5.stdout.txt`, `build-r5.stderr.txt`, build
provenance, ELF dependencies, `source-controls-r2.tar`, initial failed
diagnostics, each control's `stdout.jsonl`/`stderr.txt`, and both checkpoint
forms. Earlier failed runs were preserved. Live trial outcomes belong in
the authentic evaluation report and are not inferred from these losses.
