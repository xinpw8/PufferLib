# Native command-intent behavior cloning

`bc_train.cu` fine-tunes the existing 223-observation, 33-action, hidden-256,
two-layer MinGRU checkpoint using the prepared PufferLib5 architecture, backward
kernels, and Muon optimizer. It has no simulator, game connection, Python, or
Torch dependency. Output remains flat FP32 encoder, decoder, recurrent weights
in the existing native checkpoint order. The value decoder row receives zero
loss gradient. BC changes the shared representation, so this does not preserve
value predictions.

Build from the source directory produced by `build_native.sh`, including its
temporal-credit patch. The build adapter copies that source into a fresh build
directory, extracts the existing tensor/model/optimizer boundary, and records
the hashes of every included source and the executable. It does not rewrite
kernel bodies or compile the environment/runner.

```sh
bash ocean/rek_g1/native5/build_bc_train.sh PREPARED_BUILD/trainer/src NEW_BC_BUILD
NEW_BC_BUILD/bc-train --self-test INITIAL.bin INITIAL_SHA256
NEW_BC_BUILD/bc-train demonstrations.bin INITIAL.bin INITIAL_SHA256 NEW_OUTPUT.bin 1 0.0001 128
```

The final three training arguments are epochs, learning rate, and horizon.
Epochs are bounded to 0 through 100, learning rate to (0, 0.01], and horizon to
multiples of 4 between 4 and 256. Existing output files are refused. Preserve
stdout/stderr, source provenance, dataset/mask hashes, and checkpoint hashes.
Zero epochs verifies the checkpoint round trip without optimization. Each epoch
also writes `NEW_OUTPUT.bin.epoch-N.bin` with its SHA256 in stdout. All output
paths must be fresh. A failed write can leave a partial file that must never be
loaded without size and hash verification.

## Sequence and input contract

The dataset exporter owns causal observation construction. Every observation
must precede its command-intent target. Unknown, unsupported, or colliding
commands are unlabeled and retained in the separate command ledger. They must
never become inferred no-op labels. Whole recorded rounds form disjoint train
and heldout sequences. Their ordered, unlabeled timesteps remain in recurrent
history. Native action completion is not asserted by a command-intent label.
If missing observations force a new contiguous segment inside a round, the
exporter must mark and count that truncation explicitly. Its zero-state start
is an offline-history limitation, not evidence of a native round reset. All
segments from one round remain in the same split.

Each training chunk replays its entire preceding sequence under the current
weights with no loss before truncated recurrent backpropagation in the chunk.
This is full-prefix burn-in, with detached state at the chunk boundary. It is
more costly than cached state, but avoids stale state after an optimizer step.
Reset is applied before the first observation of each true sequence. Padding
is reset and has zero loss. There is one update per labeled training chunk,
with fixed denominator `(total training label weight / all training rows) *
horizon` for every chunk, including a short final chunk. Thus every row retains
the same weight coefficient throughout an epoch. Heldout statistics never
enter this denominator. Metadata identifies this normalization as
`global_training_weight_over_all_training_rows_times_horizon_v2`. Earlier v1
runs divided by each chunk's weight and do not implement this global objective.
Heldout rows never update parameters
or choose an epoch automatically. Reported cross entropy and accuracy measure
command-intent classification, not combat performance. Train and heldout diagnostics
include each action's CE/recall and category CE/argmax-category recall. Category
CE sums probability within hold, neutral command, movement, kick/knee (actions
16 through 19), punch (20 through 31), or emote (32). Missing classes report
null metrics, not zero performance. These breakouts expose neutral-class
dominance; neither aggregate accuracy nor attack recall establishes gameplay
quality.

For the explicit three-group weighting experiment, `balance_bc_data.cjs`
produces a separate dataset, modifying only positive training-label weights.
It derives counts from training labels alone and assigns equal total mass to
neutral command (action 1), movement (2 through 15), and the requested attack
group (16 through 32, including the emote class). The average labeled training
weight remains 1. It refuses preweighted training labels or missing groups.
Heldout rows, observations, labels, masks, sequence markers, and timestamps are
byte-identical. Its manifest records source/output hashes, counts, weights,
and invariance checks. With the v2 fixed denominator, relative row coefficients
preserve the specified global weighting even in neutral-only chunks. The
optimizer still makes sequential updates and recomputes recurrent burn-in with
the current weights.

```sh
node ocean/rek_g1/native5/balance_bc_data.cjs ORIGINAL.bin ORIGINAL_SHA256 NEW_BALANCED.bin
node --test ocean/rek_g1/native5/balance_bc_data.test.cjs
```

A single fixed 223-byte binary feature mask is applied before BF16 conversion.
The exact same mask and observation/cadence conventions must be applied at live
and PPO policy-input boundaries. The trainer emits the raw mask SHA256 and
observed minimum/maximum timestep intervals. Missing native busy, transition,
settlement, or action-route state must remain masked. Do not silently fill such
fields from current/future targets. A class-support mask describes known action
vocabulary support; it does not fabricate native runtime legality. The exporter
may explicitly use all 33 vocabulary classes. Live execution still requires
the real native legality mask.

## REKBC001 binary format

All numbers are little-endian. The fixed header is 256 bytes and each row is
1056 bytes. Reserved values must be zero. The reader rejects nonfinite values,
unsupported labels, malformed length, repeated noncontiguous sequence IDs,
mixed-split sequences, nonincreasing timestamps, and misplaced resets.

| Header offset | Value |
| --- | --- |
| 0 | 8-byte ASCII `REKBC001` |
| 8, 12, 16 | uint32 version 1, observation count 223, action count 33 |
| 20, 24, 28 | uint32 row count, row bytes 1056, reserved 0 |
| 32 | 223 uint8 feature-mask values, each 0 or 1 |
| 255 | zero padding byte |

| Row offset | Value |
| --- | --- |
| 0, 4, 8 | uint32 split (0 train, 1 heldout), sequence ID, reset-before-step |
| 12 | int32 action; -1 means unlabeled |
| 16, 20 | float32 label weight; uint32 reserved 0 |
| 24 | float64 sequence-relative timestamp, seconds |
| 32 | 223 float32 observations |
| 924 | 33 float32 class-support values, each 0 or 1 |

Action -1 requires weight 0. A labeled action requires positive weight and
support 1. The feature mask, reset, label, weight, support, split, and timestamp
are control/metadata fields, never additional learned input features.
