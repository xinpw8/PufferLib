# Human command data migrated to observable balance

The separate offline C++ adapter `human_observable_data.cpp` regenerated all
11,985 observations from the two pinned original human protocol-v5 captures
using the unchanged `observable_balance.h`. It produced a fresh private
`REKBC001` dataset for `rek.native5.observable_balance.v1`. Existing exporters,
datasets, runtime, shared projection, BC trainer and live encoder are unchanged.
No GPU, physics, training or game connection was used.

## Preserved evidence and changed feature schema

Every row's split, sequence, reset, action, weight, timestamp, reserved bytes
and 33-class support mask are byte-identical to the original dataset. Both
original ledgers are copied byte-identically. Only the 223 observation floats
and global feature-mask header change. A separate projection ledger identifies
the current and preceding raw source lines used for each new observation;
the original row ledger retains its historical feature-provenance fields.

| Split | Rows | Grounded labels | LEFT_FRONT category 17 | Segments |
| --- | ---: | ---: | ---: | ---: |
| First round, training | 5,993 | 4,943 | 16 | 2 |
| Second round, development holdout | 5,992 | 4,990 | 2 | 2 |

The raw captures are pinned to SHA256
`547cec42f700e97b9594f8d2c6df88f966052c0e0e5ce7395c4862b619fb6d7f`
and `ec55c32a6e2272a8e7656d73260ca35876be2cd6c8d8d6fe8c9dfe77cd25a309`.
The adapter also pins the original dataset, manifest and both ledgers before
reading their labels. Source paths are retained in the private manifest.

| Artifact | SHA256 |
| --- | --- |
| Original dataset | `71238da150db6e926170efdc27662f4426145aac7f69a7c93f619ece54d6cf1d` |
| New observable dataset | `228b5696fdcb0663c15dce32b6e8bed171ed995661c3acd5d050f95402aa7c8f` |
| Original 215-column feature mask | `7e5a991e79b495133a92571beb1530b534580d7b55c755603df4a6668ec31eaa` |
| New shared 166-column feature mask | `53055f6a763c528cdc101171a87d85ebcb3e8541425fc3eb9fa8979922c436e5` |
| Unchanged row ledger | `3f24a13f09a145d41f6c3c33d1da1fec0ca17b72d9c3c370cad668b55938ac83` |
| Unchanged command ledger | `8a1f6f625bbb1ed50452c705ad7dba65642357428120a0ecf3c5cbb6f1694797` |
| New projection ledger | `e18a175f50446dff0864aed6f357201c14c503102bde5d3aea707def5f458057` |

Roots come from sampled rendered root positions/quaternions, transformed by the
shared Unity conversion. Time comes from recorded `sample.unity_unscaled_time`;
there is no invented QPC clock. Derivatives and point changes use the preceding
retained observation within the same preserved segment. All 11,981 noninitial
rows have valid history. Four segment starts explicitly lack history. Historical
midround segment breaks remain offline truncations, not claimed round resets.

Both joint-pose availability flags are zero, matching current physical/live
adapters. Referee availability and count padding are zero: the old capture has
received referee packets but cannot prove the newer QPC/lifecycle freshness
contract. The adapter does not infer falls from height/tilt, use old native
velocity fields, or relabel point changes as attacks. Shared projection fills
excluded columns with zero padding. The maximum observed root tilt fraction is
0.645988107; this is a quaternion-derived feature, not a fall classification.

## Verification

The native C++ unit suite passed 1,221 checks, including row metadata/support
invariance, explicit unavailable fields, tilt, vertical rates, point deltas,
history reset, ledger disagreement, source identity, gaps and counter regression.
Replacing every old observation with different sentinel values leaves the new
dataset identical, demonstrating that old features do not enter the projection.

The independent Node integration check reread and hashed both original captures,
checked every nonobservation row byte and both complete ledgers, verified split
counts, all excluded zeros, joint/referee availability, and raw-derived tilt,
vertical derivatives and point deltas. All 11,985 observation rows changed.
Maximum independent tilt error was `6.099323524022537e-8`; maximum vertical-rate
error was zero. Existing-output rejection preserved the output hash; swapped
raw inputs failed hash validation before creating an output directory.

The first test build stopped on a `-Werror` range-loop-copy warning in the new
unit fixture. That test-only loop was corrected; fresh `build-r2` and `run-r1`
completed successfully. The build links no CUDA, MuJoCo, Python or Torch library.

## Reproduction and private outputs

Private Windows stage:
`C:\rekagent\work\human-observable-bc-20260921-r1`.

New dataset:
`run-r1\dataset\human-observable.bin` (12,656,416 bytes).
The same directory contains its manifest, both masks, unchanged original ledgers
and new projection ledger. `build-r2` holds native executables and source hashes;
`run-r1` holds commands, integration/CLI receipts and output hashes.

Using Linux or WSL paths:

```bash
bash ocean/rek_g1/native5/build_human_observable_data.sh NEW_BUILD
bash ocean/rek_g1/native5/run_human_observable_data.sh \
  NEW_BUILD ORIGINAL_DATASET_DIR TRAIN_RAW HELDOUT_RAW NEW_RUN_DIRECTORY
```

The exact executed private command is saved in `run-r1/command.txt`. Existing
`bc_train.cu` can consume this binary without changes, using a fresh or already
observable-balance checkpoint. Old scaled-polar checkpoints are incompatible.
This task did not run BC or create a policy checkpoint.

## Separately prepared group-balanced BC input

After the unweighted migration passed, the unchanged `balance_bc_data.cjs`
created a separate `balanced-r1/human-observable-balanced.bin`, SHA256
`1ec2befa572bba51781b43c44e5f9d10aeb1dadfb2a9b699ff0b09f7736c8523`.
Only 4,943 positive training weights changed. Every held-out byte and every
nonweight byte, including all new observations and masks, remains unchanged.
The neutral/movement/attack weights are 0.4692870080471039,
1.2159901857376099 and 21.39826774597168, respectively. This gives equal total
weight to those three groups; it does not balance individual kick classes.

The unweighted dataset, balanced variant, manifests and prepared launch script
were copied to the fresh Spark stage
`/home/spark-advantage/rek-training/human-observable-bc-20260921-r1`.
Both dataset hashes were independently verified after transfer.

The existing corrected native BC executable can be reused without rebuilding:
`/home/spark-advantage/rek-training/imitation-20260919-r1-native-bc/build-r5/bc-train`.
Its current binary SHA256 is
`febdf12d528f815ad737385208daed27d8676287b8335802da43a5442537d42d`.
Build provenance and the current staged sources match the repository exactly:
`bc_train.cu` SHA `35d50624e2003d0b5594fa2b730097e42d097fd3ac37235ea9c3f0c04bec5eec`,
`bc_dataset.h` SHA `fa22e627083bad7c1118b4779342ef2612b372d6d4740cad69afeb5e8ff3265f`.
The prior corrected BC tests/results are documented in
`validation/native-bc-20260919/README.md`; this task did not rerun GPU tests.

Prepared script `run-human-observable-bc.sh`, SHA256
`7676a235fe3a9c5847d62b625b91478f04d514f34a1ae9f5224b3aaab49e7d22`,
requires a fresh `train-...` label and an explicitly chosen 1 through 20 epochs.
It pins the executable, balanced dataset and fresh initialization hashes; uses
learning rate 0.0001, horizon 128 and a 120 s timeout; records schema/data/build
manifests and preserves per-epoch checkpoints. Its schema receipt is external:
the unchanged BC binary validates shape and feature mask, not the schema name.

The pinned fresh initial checkpoint is
`/home/spark-advantage/rek-training/physical-observable-balance-20260921-r1/ppo-smoke-r2/checkpoints/rek_native5/ppo-smoke-r2/0000000000000000.bin`,
SHA256 `d8dc477e1eb33380f7d2cb260b704f88c4f2498cd9d6ccf832d44542b462593f`,
reverified directly on Spark. A one-epoch command, prepared but not executed:

```bash
bash /home/spark-advantage/rek-training/human-observable-bc-20260921-r1/run-human-observable-bc.sh train-r1 1
```

No BC GPU work was started while the separate physical benchmark ran.

The labels remain observed outgoing interface requests. They prove neither
server acceptance nor successful execution, contact, scoring, or key intent.
Class support remains vocabulary support, not measured game legality. These
two rounds come from one human session, and round two was previously examined
in BC development. There are only two held-out left-front labels. Prior BC's
zero held-out kick-category recall remains a generalization warning; schema
migration alone is not a demonstrated improvement in fighting.

## Private archive

Completed CPU builds, source snapshots, unweighted and separately balanced
datasets, manifests, command/test receipts and the unexecuted BC launch script
are preserved at:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\human-observable-bc-r1\human-observable-bc-20260921-r1-evidence-r2.tar.gz`

Size: 4,935,009 bytes. SHA256, verified on local and NAS copies:
`69afe2d46c6f0cb32f151c57945edecd60a13fd4b4d6aac1ac6d471bf0385b37`.
Original raw captures remain unchanged at the pinned source paths; this archive
contains their derived private datasets and hashes, not duplicate raw captures.
No private dataset, ledger, capture or binary is added to the repository.

## Completed native BC runs

After the separate physical benchmark released the GPU, the coordinator ran the
prepared eight-epoch experiment. A separately authorized 100-epoch run then
started from the **same fresh initialization**, not the eight-epoch checkpoint.
Both used the unchanged native BC binary, unchanged group-balanced dataset,
learning rate 0.0001, horizon 128 and a 120 s timeout. Both exited 0 with empty
stderr. Whole-process elapsed times were 4.48 s and 45.97 s. There were 47
optimizer updates per epoch: 376 and 4,700 updates, respectively. No physics or
game connection was used by these supervised runs.

The unchanged executable caps requested epochs at 100. The separate
`run-human-observable-bc-100.sh` respects that cap and preserves every epoch's
checkpoint and class metrics. Its SHA256 is
`eeb97507fcac2af28bb1a569aaf8213f0fda0431e2f035047c7e0f6f1a6752f7`.
The original runner and eight-epoch outputs were preserved unchanged.

### Eight-epoch result

| Epoch | Weighted training CE | Held-out CE | Held-out exact-action accuracy |
| --- | ---: | ---: | ---: |
| Initial | 3.64086974 | 3.71215682 | 0 |
| 1 | 3.61079559 | 3.67526845 | 0 |
| 2 | 3.56760909 | 3.63098288 | 0 |
| 3 | 3.52482972 | 3.58751771 | 0 |
| 4 | 3.48306596 | 3.54526267 | 0 |
| 5 | 3.44201509 | 3.50303638 | 0.000400801603 |
| 6 | 3.40145343 | 3.46112729 | 0.00641282565 |
| 7 | 3.36158584 | 3.41903331 | 0.116032064 |
| 8 | 3.3221892 | 3.37782501 | 0.228657315 |

Accuracy is a fraction. Training metrics use the balanced weights; held-out
metrics retain natural label frequencies, so their aggregate losses/accuracies
use different weighting. Epoch 8 minimized held-out CE among these eight saved
epochs and was identical to `bc-final.bin`, SHA256
`95e1f7c5366265665c5b763d79e5f5c8997e7bad748a6b6d882055276ee1cae3`.
Its LEFT_FRONT17 recall was 0/16 training and 0/2 held out. Held-out class-17 CE
was 3.66634488. Every supported held-out exact-action recall except neutral
action 1 was zero. This checkpoint was not evaluated in a game.

### 100-epoch result and frozen development selection

Selection uses minimum held-out CE only, with the earliest epoch breaking exact
ties. It does not use kick recall or any game outcome. **Epoch 97** is the selected
checkpoint. The development holdout was already examined and selected upon;
these figures are not untouched-test generalization estimates.

| Measurement | Selected epoch 97 | Final epoch 100 |
| --- | ---: | ---: |
| Optimizer updates | 4,559 | 4,700 |
| Weighted training CE | 1.59645007 | 1.56951799 |
| Weighted training exact-action accuracy | 0.407450428 | 0.406073453 |
| Held-out CE | 1.50055312 | 1.50380332 |
| Held-out exact-action accuracy | 0.652104208 | 0.651903808 |

Selected epoch-97 held-out diagnostics:

| Command group | Held-out labels | Group CE | Argmax group recall |
| --- | ---: | ---: | ---: |
| Neutral command | 3,256 | 0.534520286 | 0.999385749 |
| Movement | 1,625 | 1.76612364 | 0 |
| Kick or knee | 6 | 2.39355854 | 0 |
| Punch | 103 | 2.16429815 | 0 |

Group CE uses summed probability within the target group; group recall does not
require the correct individual action. Hold and emote have no labels, so their
metrics are unavailable. LEFT_FRONT17 still has only 16 training and two
held-out labels. Its exact-action recall remains **0/16 and 0/2**; respective
class CE values are 2.76686848 and 3.09802032.

All 3,254 correctly predicted held-out labels at epoch 97 are neutral commands.
Always predicting neutral would yield 3,256/4,990, or 65.2505%, slightly above
the selected checkpoint's 65.2104%. Lower aggregate CE therefore does not
establish useful movement/attack imitation or stronger fighting. All 100 epochs'
per-action and per-group metrics remain in the private stdout. The summary keeps
every epoch's aggregate metrics and the selected epoch's class/group details.

Spark run directory:
`/home/spark-advantage/rek-training/human-observable-bc-20260921-r1/train-100-r1`.

| Checkpoint | SHA256 |
| --- | --- |
| `bc-final.bin.epoch-97.bin`, selected | `ec640746b48cae9348e2508d0dd9678db3e250bdfb6dc8cadffc14d515dc81ba` |
| `bc-final.bin`, identical to epoch 100 | `68347fb07646595a9d9eb1747fe49558a2ec048bda167346a1296120acf166d0` |

### Explicit pure-BC deployment availability mask

Historical demonstrations have no established fresh referee/count availability.
For pure-BC deployment, a separate 223-byte mask retains 163 fields and excludes
only columns 202 (referee availability), 204 (actor count active) and 205 (opponent
count active) beyond the original 166-field structural mask. The structural
schema itself, all other shared fields, training inputs and labels are unchanged.
This prevents live received-count inputs from activating columns that were zero
throughout these demonstrations. It is not a mask change for the physical PPO
candidate.

Mask path on Spark:
`/home/spark-advantage/rek-training/human-observable-bc-20260921-r1/deployment-mask-r1/bc-availability-feature-mask.bin`.
SHA256: `656e3470bf6b5c1634a6c3b14544752129938a1b2fe28d1a4a4e54053fe1ce75`.
The original 166-field mask is preserved separately, unchanged.

CPU verification compared the resulting feature bytes on all 11,985 rows of
both unweighted and balanced datasets. Every old/new masked observation was
byte-identical; the concatenated feature-matrix SHA256 was
`fc17586c74d7cd2e73f7b400d8dfe99db71f08b25ca43bd2346b297ce181b664`.
Both source dataset hashes were unchanged. This is exact input equivalence on
the recorded sequences, without additional GPU policy calls or live behavior
claims.

### Completed-run preservation

Fresh NAS folders under the existing `human-observable-bc-r1` project archive:

- `train-eight-r1`: all eight epoch checkpoints, final checkpoint, commands,
  metrics, manifests, native binary, initial checkpoint and hash receipts.
  Archive: 16,116,093 bytes, SHA256
  `7558dd7eb0e5b8e087a3ba2024694b34d5cecc9d3db7fcb490fb397b3108ca1d`.
  NAS receipt: `4fa7c9fd61b34bb2e792f3d0531d679bce2a1ada4927a543df8bfd134b710d77`.
- `train-100-r1`: all 100 epoch checkpoints, final checkpoint, commands, metrics,
  manifests, explicit deployment mask and its verification, and hash receipts.
  Archive: 163,307,814 bytes, SHA256
  `9f94e8e8291d0e299c3369fea54a7e63a1c746339ec0885f94898b390fb872e0`.
  NAS receipt: `0eae0946674c93b9722c1ba1dbae8d6bf43e72a4dcab166b16707f9ac55a44c2`.

All 22 eight-epoch and 115 hundred-epoch input/build/checkpoint hash entries were
recomputed successfully. Packaging verified source hashes before/after and tar
contents. Spark/local/NAS archive hashes matched. No existing destination file
was overwritten. The prior dataset archive was also reverified.
