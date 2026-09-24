# One conditional human attack BC candidate

This is the pre-training design record. The fixed native run and full-input drift diagnostic have since executed; see `RESULTS.md` and the execution receipts for measured results. The design below retains the original pre-training decisions.

Start from current scorecredit5s checkpoint `0daf90a96442d541d38d9dd4f8fe765917c3c766ca80019855cfd7d358d9fe12`. Use original human round1 for training and round2 for previously inspected development holdout. Preserve all11985 chronological rows and4 historical segments. This is supervised command-intent data, never an on-policy trajectory with invented old log probabilities.

## Existing native objective is sufficient

The unchanged `bc_dataset.h` accepts per-row weight and33-class support. The unchanged `bc_train.cu` masked softmax normalizes over each row's supported classes. No CUDA loss change is required. Their current SHA256 values are `fa22e627083bad7c1118b4779342ef2612b372d6d4740cad69afeb5e8ff3265f` and `35d50624e2003d0b5594fa2b730097e42d097fd3ac37235ea9c3f0c04bec5eec`. Existing Spark native executable `imitation-20260919-r1-native-bc/build-r5/bc-train` has documented SHA256 `febdf12d528f815ad737385208daed27d8676287b8335802da43a5442537d42d`.

Use support1 only for policy categories16..32 and weight1 only at recorded one-shot request rows:77 training labels and109 development labels. Derived nonattack labels become-1 with weight0; original datasets/ledgers remain unchanged. The objective is `-log(exp(z_target)/sum(exp(z_a), a=16..32))`. It does not train neutral rows as attacks and does not reward firing attacks. Categories0..15 receive zero direct loss-logit gradient. Shared representation changes can nevertheless change total attack mass, movement and timing during ordinary33-class live selection. No timing-invariance claim is justified.

Existing full-prefix current-weight burn-in retains unlabeled observations. Existing global normalization becomes `77/5993*128` atH128, with one update per chunk containing a training attack. Existing reports already provide conditional heldout per-action CE/recall. Category17 support is16 train/2 development; unseen classes remain in the17-class denominator. Natural event weights are used; class balancing would be another change.

## Missing-feature decision

The original scaled-polar human dataset masks unknown columns176..183 (held intent, route, settlement, busy). The current live policy uses allones and acknowledged-dispatched-request duration projection. Old outgoing human request records have no dispatch-return acknowledgement. They cannot honestly reproduce these inputs.

The validated Sept21 human observable dataset can supply the same8 balance columns `[9,95,72,158,202,203,204,205]` at byte-aligned original rows. Rates/tilts/history are measured from recorded roots and sample time;202/204/205 remain unavailable0 because protocol-v5 lacks the later QPC/lifecycle freshness proof. Overlaying exactly these8 leaves the other215 original observation cells unchanged. Old network-packet joints versus current rendered local bones and50Hz versus current variable cadence remain transfer limitations.

Approved supervised-input mask is0 at176..183,202,204,205 and1 elsewhere. Root explicitly keeps current live allones with the missing-input training/deployment mismatch. Referee availability0 is a schema-supported unknown, not a measured no-count assertion. Current/future targets never fill observation cells. The conditional action support is used only by the training loss, never by the live worker.

## Minimal tests and later GPU review

CPU checks cover retained rows/splits/resets/times/features, exact17-class conditional support, original immutability, rejected preweighted inputs, and conditional CE gradient reference. The native dataset reader parses the final export unchanged. Before any update, root reviews the exact export. Root selected one fixed five-epoch candidate withLR1e-4/H128, not an epoch sweep or development-selected checkpoint. There are31 labeled training chunks per epoch and155 expected updates. GPU kernel/recurrent behavior and pre/post full33-action mass on frozen live observations can be measured after root grants GPU access. These are diagnostics, not exact-timing guarantees.

Dataset SHA256: `b07f599e8f38ae0cfe59c3e7e0e10fa5aee00955f0483a9f763cfb662dbbfdef`.
Partial training mask SHA256: `f0f9c515a7f9a122780a6efef970f673bc3f52852384a83a311c004a01240a76`.
Native binary checks checkpoint SHA/1836032-byte shape and REKBC001223x33 shape. It does not read an observation-schema string; the isolated runner pins that externally in the dataset manifest. This is a declared partial balance8 view, not input equivalence to current full223 live observations.

GPU command, prepared only:

    node /home/spark-advantage/rek-training/scorecredit-human-attackbc-20260924-r1/run_bc.cjs --run

The same command with `--check` validates hashes and emits the exact argv without loadingCUDA.

## Actual full-input drift diagnostic

`diagnose_live_drift.cu` includes the frozen balance8 worker implementation unchanged with its original main renamed, and calls the same parser/Engine/native object. It adds offline FP32 readback of the34 native diagnostic logits after each sampled decision. No policy kernels or checkpoint are changed. Compile-only build succeeded; binary SHA256 `61e8dc64fd2bafec4084d470f0c048ee4e0a7bbebf17b66b0628fed7f260d8b2`.

`compare_live_drift.cjs` reuses all1702 original worker requests from closed `credit5s-s1001-retry3`, with actual seed1001, original legal masks, reset chronology and current full223 allones inputs. It requires baseline0daf sampled actions to match the recorded actual worker before reporting candidate drift. It reports legal-distribution KL, sampled-action changes, legal and unmasked attack mass, category17 mass, and sampled attack counts. This is a fixed recorded observation stream, not a counterfactual game trajectory or proof of preserved timing. Candidate outputs do not control any game.

After root authorizes GPU and the five-epoch checkpoint is produced:

    node compare_live_drift.cjs train-five-epochs/policy.bin FINAL_SHA256 --run

The `--check` variant reads/hashes only. Final SHA256 comes from native BC stdout and an independent file hash. No old behavior probabilities are replaced or invented. Baseline/each epoch's conditional CE and per-action recall are already emitted by the unchanged BC executable; use them rather than aggregate neutral accuracy. Fixed epoch5 remains the candidate irrespective of development metrics.
