# One-epoch learning-rate sibling

CPU-prepared only. No GPU execution, live preparation, game control or production edit.

The sole training change versus the executed timing500 on-policy r2 update is learning rate: 1e-5 to 3e-5. Start again from the actual recorded behavior checkpoint `5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4`, with a fresh optimizer. Do not continue from c2c4987. Use the same eight complete rounds, all 24,010 rows, 24,002 applied actor rows, eight terminal rejects, actual seeds, full 223 observations, legal masks and immutable original FP32 behavior log probabilities. The existing verified replay is reused byte-for-byte.

One epoch, 192 updates, H128, clip0.2, VF clip0.2, VF0, entropy0.001, complete-MC zero baseline and five-second point-delta discount are unchanged. Final epoch1 is selected prospectively. There is no development-selected epoch, new reward, class balancing, attack gate, feature mask, cadence resampling or teacher migration. The native CUDA trainer binary is unchanged. `parent-run-plan.json` is an exact copy of the frozen r2 plan; `experiment.json` pins that plan, replay and replay receipt. Every underlying checkpoint, worker, native object, mask, dataset, identity, selection and executable is hashed again before execution.

The previous update had exact 24,010-action replay, zero sequential teacher error, zero CUDA/CPU return-reference error, and post-update mean legal KL0.000679761. The source correctly uses the true old denominator, full-episode returns and full-prefix current-weight recurrent burn-in. H128 limits backward propagation, not return duration. No identified objective or sequence bug motivates this change. A larger fixed step is a controlled learning-strength test; increased KL or better Bot1 results are not guaranteed. Eight episodes remain a small development dataset.

## CPU verification

```sh
node --test run_candidate.test.cjs prepare_live.test.cjs
node run_candidate.cjs train --check
node run_candidate.cjs diagnose --check
```

The checks read pinned artifacts without CUDA initialization. The live-preparer test verifies configuration preservation and prospective seeds1401..1420.

## Root-controlled execution after the live cohort closes

```sh
node /home/spark-advantage/rek-training/timing500-ppo-lr3e5-20260924-r1/run_candidate.cjs train --run
```

The runner exclusively creates `train-one-epoch/`, records exact argv, experiment configuration, UTC bounds, full native process wall time, stdout/stderr, exit status and checkpoint SHA. Existing output is refused. It retains the existing explicit BF16 distributional acceptance and exact sequential teacher requirement. It does not recompute or modify behavior probabilities. No additional acceptance framework is introduced.

Optional zero-update numerical diagnosis, if root needs it:

```sh
node /home/spark-advantage/rek-training/timing500-ppo-lr3e5-20260924-r1/run_candidate.cjs diagnose --run
```

This writes only `diagnose-zero-epoch/` and is not a prerequisite or a training run. Native GPU use remains under root control.

After training, root may pass the actual final checkpoint path and SHA to:

```sh
node /home/spark-advantage/rek-training/timing500-ppo-lr3e5-20260924-r1/prepare_live.cjs CHECKPOINT_PATH CHECKPOINT_SHA256
```

This derivative preserves the frozen timing500 driver/controller, encoder, worker, all-ones mask, 500/750ms bridge and all17 attacks. It prepares a new `timing500-ppo-lr3e5-live-20260924-r1` stage with labels `lr3e5-s1401` through `lr3e5-s1420`, target18/20 and stop-third-nonwin contract. Only output path, seeds, labels and hypothesis differ from the reviewed r1 preparer; checkpoint path/hash remain explicit runtime arguments. It never launches the controller. Do not execute preparation before training and review.

## Human-BC alternative

Existing native BC supports per-row weights and full33 support. Historical conditional attack fitting improved development CE3.79579 to2.68312, but category17 training recall stayed0/16. Five-epoch movement fitting remained worse than uniform14-class CE and forward recall stayed0/499 train,0/932 development. The prepared full33 event candidate is technically executable but still uses 50Hz history and partial observations with unavailable176..183,202,204,205 explicitly zero, versus approximately25Hz full-input deployment. It is not an input-equivalent alternative. New full-schema human demonstrations recorded at deployed cadence could support unchanged native BC; missing temporal/referee fields cannot be reconstructed merely by assigning zero or changing labels. This LR sibling avoids combining that domain shift with an objective change.
