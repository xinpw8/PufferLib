# Previous sampled-action contract and CPU migration

Status: helper and CPU migration verified. No GPU equivalence, policy training,
game launch, or strength result is claimed by this report.

## Contract

The explicit opt-in schema is
`rek.native5.observable_balance_prev_action.v1`. It retains the 223-column
architecture and every existing `observable_balance.v1` feature. It adds only
the policy's previous successful sampled categorical action. This says nothing
about delivery, local acceptance, pending moves, or server execution. A sampled
request subsequently rejected still remains the policy's previous sample.

Categories 0 through 32 map in order to columns:

```
176..183, 186..187, 192..201, 206..216, 219, 220
```

Column 221 is availability. Unknown history is 34 zeros. Sampled category 0 is
one-hot column 176 with availability 1; it is distinct from unknown and from
category 1. All 34 columns were structurally excluded metadata padding in base
v1. No joint feature, raw joint coordinate, or shared `Snapshot` is repurposed.
The global structural mask now contains 200 available columns, versus 166.

`observable_prev_action.h` is a host/device helper. `write` changes only the 34
assigned columns; invalid history leaves the observation untouched. `record`
accepts finite integral categories in [0,32] and does not mutate history on
invalid input. The caller writes history before forward and records the new
sample only afterward. Repeated observations and value-only bootstrap do not
record another action. History carries through rollout horizons and minibatches.

Clear history at the same genuine episode or explicit reset boundary as the
policy recurrent state. Do not clear it for a counted-fall body reset. The
existing physical wrapper exposes terminal observations, then resets the native
episode on its next step. The trainer clears recurrent state before forwarding
that terminal observation; the action sampled there is applied across the native
reset and appears in the following observation's history. The compact autoreset
path exposes reset pose with terminal 1. A post-step wrapper overlay should clear
terminal outputs and otherwise use the unchanged sampler-owned learner action
buffer. This preserves the existing policy boundary in both cases.

The live worker can own the same history locally, clear it with its existing
round-change/reset/terminal handling, write it before forward, and record each
successful sample before delivery. Protocol rejections that never invoke the
sampler do not advance history. These integrations are separate owned work;
this report covers the helper and migration only.

## Source evidence and layout

The pinned trainer copies environment observations to rollout storage before
`arch_forward`, then samples into the separate rollout and environment action
buffers (`pufferl.cu:825,888,897`). `arch_forward` receives only observations and
recurrent state. Native live inference has the same order: prepare observation,
encoder, MinGRU state update, decoder, sample (`native_policy.cu:109`). The live
worker previously supplied only encoded observations and masks, without an own
sample-history feedback path. Masks affect categorical selection but are not a
separate recurrent input.

The migration utility derives layout by invoking the actual pinned trainer's
CPU metadata registration, without allocation on CUDA or executing kernels.
The pinned `algo.cu` SHA is
`8a514cb8dd12d49b79cbd5afe7298875b6f0ca0491270bb19a8696bd527f4d92`.
It equals the actual physical trainer source used for the source checkpoint.
Registration order is encoder [256,223], decoder [34,256], and two recurrent
matrices [768,256]. The flat checkpoint contains 459,008 float32 parameters,
1,836,032 bytes. The encoder starts at float offset 0, with element
`hidden_row * 223 + input_column`; this matches native policy GEMM loading.

Migration writes positive zero to the 34 new columns across 256 encoder rows:
8,704 float entries. All remaining 450,304 floats, including every decoder and
recurrent byte, must match the input exactly. The utility verifies size,
finiteness, input SHA, architecture, and the input checkpoint's hash-bound base-v1
sidecar. It exclusively creates new output files and reads them back. The new
sidecar binds the new schema and hash, original sidecar and checkpoint, migration
binary, zeroed columns, and complete original provenance.

This neutralizes the new inputs initially, rather than relabeling the old
checkpoint. CPU encoder equivalence passed; exact native GPU equivalence remains
pending and is explicitly false in the migration receipt.

## CPU evidence

Private stage on Spark:
`/home/spark-advantage/rek-training/observable-prev-action-20260921-r1`.
The matching local stage is
`C:\rekagent\work\observable-prev-action-20260921-r1`.

Build command, executed with no GPU work:

```sh
bash source/ocean/rek_g1/native5/build_observable_prev_action.sh \
  /home/spark-advantage/rek-training/authentic-ppo-20260919-r1/build-r5 \
  /home/spark-advantage/rek-training/observable-prev-action-20260921-r1/build-r2
```

- Helper: 11,462 native CPU checks passed. They cover all categories, one-hot
  uniqueness, unknown versus action 0, byte-preservation of base features,
  lifecycle carry/reset, repeated reads, and invalid input.
- Migration: registered layout, 8,704 zeroed entries, protected-byte rejection,
  signed-zero preservation outside the new columns, nonfinite rejection, and
  8,704 exact CPU encoder dot-product comparisons passed.
- Actual CLI: existing output, malformed SHA, and wrong SHA were rejected.
  Existing artifacts remained unchanged; rejected output files were absent.
- Both test programs ran with `CUDA_VISIBLE_DEVICES=`. No CUDA API is called by
  the migration main path or CPU metadata-registration tests.
- Initial `build-r1` failed compilation because the new utility omitted
  `<cstring>`. That attempt is preserved. Corrected fresh `build-r2` passed.

Migration command:

```sh
CUDA_VISIBLE_DEVICES= build-r2/observable-prev-action-migration --migrate \
  /home/spark-advantage/rek-training/physical-bot1-integration-20260921-r1/train-bot1-continue8m-r1/checkpoints/rek_native5/bot1-continue8m-r1/0000000008388608.bin \
  ef85a01b207d033417d8e1bc9d2b32b9eba610e784180dc7dd287a894acf2f4b \
  /home/spark-advantage/rek-training/observable-prev-action-20260921-r1/migrated-ef85-r1/ppo.bin
```

The source checkpoint was not modified. All 8,704 selected weights actually
changed, and the other 1,801,216 bytes were preserved.

| Artifact | SHA256 |
| --- | --- |
| Shared helper | `42d79bd6d453085b11058b5679b11aadec5175a2a060c534b71463282d4a28fa` |
| Migration source | `aea7c96bbbdf7f3f6f483867b7351d4dd700b1ee1318987616484e5d7a89de33` |
| Migration executable | `40d6518eff231802c0e288f01521cb067d57fb842ce7c4b99df1053b3cba45e7` |
| Migrated checkpoint | `ba4339bb40a82d91b19d926a0f2ce8e03386632dadf8e23b14b6ee164c40a632` |
| Migrated checkpoint sidecar | `9160b922be73aa7f59575df018fda7674314303c40f5084b6c8f894dc1a92959` |

Complete commands, build hashes, test output, negative-test receipts, and the
new checkpoint remain private in the stage. No checkpoint or captured gameplay
is added to the repository. Future native GPU replay must compare original
base-v1 inputs against migrated inputs carrying all categories, including reset
and horizon boundaries. This work alone does not establish a performance or
fighting-strength improvement.
