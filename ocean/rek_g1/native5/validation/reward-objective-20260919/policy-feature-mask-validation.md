# Native policy feature-mask validation

2026-09-19, Spark GB10, CUDA 13, `sm_121`. This change retains the 223-feature
layout and adds an opt-in `REK_POLICY_FEATURE_MASK` path for `semantic_cuda`.
The shared loader accepts exactly 223 raw bytes, each 0 or 1, with at least one
retained feature. An omitted path preserves the original output calculation.

The enabled mask uploads once during environment creation. Existing GPU
publication applies it to the learner's scaled observation and to both scaled
actor rows used by policy inference. Raw observations, physical state, action
legality, rewards, terminal flags and cumulative metrics remain unmasked.
There are no new per-step host copies or public API changes. Initialization
reports the enabled mask's SHA256 and retained-feature count.

The dataset mask excludes only indices 176..183 and retains 215 features:

- Raw file: `C:\rekagent\work\imitation-20260919-r1\dataset-r1\feature-mask.bin`.
- SHA256: `7e5a991e79b495133a92571beb1530b534580d7b55c755603df4a6668ec31eaa`.
- Exclusion means unavailable historical command/route/settle/busy inputs. It
  does not claim measured zero state or add balance dynamics.
- BC, subsequent PPO and live inference must use the same mask. Both actor rows
  receive it, so a frozen opponent must also be compatible with this input
  contract. Existing unmasked checkpoints retain their original default path.

## Tests and build

`test_policy_feature_mask.cu` passed 36,053 checks with the actual file-loaded
dataset mask and an alternating-byte stress mask. It compares omitted and
all-one masks, learner and both-actor outputs, unchanged diagnostics/state,
reset, step, 18 CUDA graph steps and six terminal auto-resets. The unchanged
`test_training_autoreset.cu` additionally passed all 592 checks, including its
original 48 checks and both reward modes.

Fresh private output root:
`/home/spark-advantage/rek-training/policy-feature-mask-20260919-r1`.

- `tests-dataset-r1/test.stdout.jsonl`: file-loaded mask result.
- `tests-dataset-r1/provenance.txt`: source, object, mask and test hashes.
- `autoreset-regression-r1/test.stdout.jsonl`: 592-check regression result.
- `build/fast-build.txt`: full trainer build and source hashes.
- `build/build-source-manifest.txt`: pinned Puffer trainer provenance.

Full native trainer built at `build/puffer-rek-native5`, SHA256
`b5e738508d546e71e02b236520bf760b764e749864619e933562b6c93431aca0`.
The previous reward-stage build remains untouched. The patched trainer source
hashes match that stage: `pufferl.cu`
`ae71826468701bf19691548555c1a2d354f8795065bb3bc7fb6e5e2e2b0eb378`,
`algo.cu` `8a514cb8dd12d49b79cbd5afe7298875b6f0ca0491270bb19a8696bd527f4d92`.
The build reports the existing unreachable-loop compiler warnings at trainer
lines 1358 and 1394 and has no Python or Torch dynamic dependency.

Source hashes for this change:

- `fast_runtime.cu`: `07c22ca1d4255db9940dc9e39a2ff0f1d1368491dfda4e3bbc17a09a61c11273`.
- Shared `policy_feature_mask.h`: `9c443b068fa76a369352564703a19bda948223beb8f81d61862153d8d15787d4`.
- `test_policy_feature_mask.cu`: `fb28a0bafc4f7789f6db2496e9b373dff3521d0b70a4fcc24ea0c5054c1e0842`.
- Test binary: `9b681aae0cc5ebf91d7a1f3c8d2c4c81acf347543d08a17e7a849475dca8cf56`.

To reuse the trainer, retain the existing reward-stage training configuration,
select this new executable and set `REK_POLICY_FEATURE_MASK` to the copied
`feature-mask.bin` under the fresh output root. The root task supplies the
BC checkpoint and schedules the training run. No long training, throughput
benchmark, deployment or game interaction was performed by this subtask.
