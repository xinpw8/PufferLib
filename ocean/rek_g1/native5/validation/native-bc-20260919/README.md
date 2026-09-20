# Native recurrent BC, 2026-09-19

This experiment implements command-intent fine-tuning of the existing native
223-observation, 33-action, hidden-256, two-layer MinGRU policy. Training uses
the prepared PufferLib5 CUDA architecture/backward kernels and Muon optimizer,
with BF16 parameters/activations and FP32 master weights. No Python, Torch,
simulator stepping, or game connection is part of BC training. See
[BC_TRAINING.md](../../BC_TRAINING.md) for the reproducible build, CLI, data
layout, recurrent chronology, and loss-normalization contract.

The initial checkpoint is the 33,554,432-step `round_outcome_v1` checkpoint,
SHA256 `61f97b0b0a4504c6bdd0ee16d369ad4c1915e3cdf73d6358bab01ce64c8fde3f`.
All versions start from that checkpoint. None continues a preceding BC run.

## Data and limitations

The evidence is approximately four minutes from one human session: one round
for training and one heldout round. This is not a broad demonstration corpus
or an independently sampled test population. The exporter retains 11,985
ordered 50 Hz rows: 5,993 training rows with 4,943 grounded labels, and 5,992
heldout rows with 4,990 labels. Training labels comprise 3,511 neutral commands,
1,355 movement commands, and 77 one-shot moves. Heldout has 3,256 neutral,
1,625 movement, and 109 one-shot labels. The full ledger preserves all 186
one-shot requests and 14,300 outgoing movement messages, including unsupported
continuous commands that are not quantized into labels.

Observations precede command-intent targets. Missing native busy/settle/route
fields 176 through 183 are masked identically in the declared policy-input
contract. Vocabulary support is not inferred native action legality. Each
round has one midround truncated-observation reset after missing-pose guards,
yielding four contiguous sequences. Those resets are a missing-history
limitation, not native round transitions. Full preceding sequence replay
provides burn-in under current weights, with detached state at chunk starts.
Recorded requests establish intent, not acceptance, execution, or success.

Original dataset SHA256:
`71238da150db6e926170efdc27662f4426145aac7f69a7c93f619ece54d6cf1d`.
Raw 223-byte feature-mask SHA256:
`7e5a991e79b495133a92571beb1530b534580d7b55c755603df4a6668ec31eaa`.

The balanced variant changes only positive training-label weights. Neutral,
movement, and one-shot groups receive equal total weight from training counts
alone; weights are approximately 0.469287, 1.215990, and 21.398268. The mean
labeled training weight remains 1. Every heldout byte and every nonweight byte
is unchanged. Balanced dataset SHA256:
`434595754c2e22a3e008329cb07f4036c61698661d81f98f3bc039af3da5e7ca`.

## Versions and observed classification results

All runs use horizon 128. CE is heldout action cross entropy. Punch recall
below means the fraction of punch-labeled rows whose argmax action is in the
punch category, not exact-move recall or combat success. Heldout kick-category
recall is zero for every listed endpoint.

| Version | Epoch | Learning rate | Heldout CE | Heldout punch-category recall |
| --- | ---: | ---: | ---: | ---: |
| Initial checkpoint, same masked observations | 0 | none | 5.24483 | 73.79% |
| Unweighted, v1 per-chunk normalization | 5 | 0.0001 | 3.43242 | 30.10% |
| Balanced, v1 per-chunk normalization | 5 | 0.0003 | 2.32865 | 19.42% |
| Balanced, v1 per-chunk normalization | 10 | 0.0003 | 2.11373 | 14.56% |
| Balanced, v1 per-chunk normalization | 20 | 0.0003 | 2.45196 | 11.65% |
| Balanced, corrected v2 global normalization | 5 | 0.0003 | 2.32839 | 18.45% |
| Balanced, corrected v2 global normalization | 10 | 0.0003 | 2.20418 | 20.39% |
| Balanced, corrected v2 global normalization | 20 | 0.0003 | 2.36223 | 10.68% |

The v1 denominator was each chunk's label-weight sum. This cancels relative
group scaling in single-group chunks and distorts the requested globally
weighted objective. Those runs remain preserved as historical evidence. V2
uses the fixed denominator `training_weight / training_rows * horizon`, which
is 105.57383728 here, derived from training data alone. Every row retains a
consistent pre-optimizer coefficient, including the short final chunk. Muon's
own update normalization is unchanged.

The corrected run shows overfitting: at epoch 20, training punch-category
recall is 84.75% versus heldout 10.68%; movement is 81.33% versus 5.29%; kick
is 83.33% versus 0%. Heldout neutral recall reaches 91.09%. Lower aggregate CE
and higher aggregate accuracy therefore do not establish a better fighter.
Epoch checkpoints are candidates for independent live evaluation. This report
makes no native-game win-rate or promotion claim.

## Runtime and verification

Unweighted training ran 2026-09-20 00:37:08 to 00:37:11 UTC. Balanced v1 ran
00:40:19 to 00:40:50 UTC and overlapped a masked PPO benchmark running
00:40:06.732 to 00:40:44.344 UTC. The PPO benchmark's approximately 905,299 SPS
was GPU-shared and must not be reported as isolated throughput. Corrected v2
ran in the reserved window 00:45:50.938223136 to 00:45:59.447320768 UTC, with
940 updates in 8.51 seconds of wrapper time. These timings are not general
training-throughput benchmarks.

Verification includes 4,496 CPU dataset checks; four balancing tests; native
CUDA full-network updates; bit-identical masked-feature/heldout update
invariance; zero optimizer updates for unlabeled sequences; unchanged value
decoder weights; exact checkpoint round trip; full-prefix chronology and
reset-before-observation checks; and zero unsupported-class gradients. The
CUDA loss gradient matches its CPU reference within 1.85e-9. The v2 two-chunk
aggregate, including a short final chunk, matches the globally weighted CPU
objective within 3.04e-8. Compute Sanitizer reported zero memory errors for
the v1 native self-test. No additional v2 sanitizer run was performed during
the reserved live-evaluation window.

## Corrected checkpoints and retained artifacts

Corrected checkpoints are beneath the Spark stage
`/home/spark-advantage/rek-training/imitation-20260919-r1-native-bc/train-balanced-global-r1/`.
Each is `bc-final.bin.epoch-N.bin`, 1,836,032 bytes in the existing flat format.

| Epoch | SHA256 |
| --- | --- |
| 5 | `bf9a737df0f60445c174eef7d051462b60f020fe2ff6975a983dc34ef50262a1` |
| 10 | `e18d741ecdd43bf294f85d8cbc2c19933085c54b3089e8bd13a27ca11366fbf4` |
| 20 | `676013566360bf1d02dae3dedbab67a049bdc49547609f4e7147fdd3fa1e71f5` |

The same Spark stage retains `train-r1`, `train-balanced-r1`,
`train-balanced-global-r1`, all epoch checkpoints, datasets, source archives,
build provenance, command traces, stdout/stderr, and self-test reports.
Corrected executable `build-r5/bc-train` SHA256 is
`febdf12d528f815ad737385208daed27d8676287b8335802da43a5442537d42d`.
Local copies of logs, scripts, and source snapshots are under
`C:\rekagent\work\imitation-20260919-r1\native-bc-r1`, with corrected results in
`balanced-global` and checks in `checks-v2`. Earlier artifacts were not
overwritten or deleted.
