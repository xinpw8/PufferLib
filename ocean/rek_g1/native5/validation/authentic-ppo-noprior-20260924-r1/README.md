# Native PPO from five authentic Bot1 rounds

Date: 2026-09-24. This bundle records a completed native CUDA training experiment, with no live validation of the resulting checkpoint yet. No synthetic fall probability, inferred attack credit or extra fall bonus was introduced.

## Result

Exact frozen behavior replay reproduced **28,715 / 28,715** sampled actions from five completed no-prior rounds, using actual per-round seeds 601 through 605. Existing strict native score and referee validators passed for all five rounds: 66 native score receipts, final points 55:50, two wins and three losses. These five rounds are development training data, not held-out evaluation.

The selected one-epoch update completed with exit code zero in **4.58 s**, including the full native process, target calculation, numerical checks, 226 optimizer updates and checkpoint writes. It does not count new environment interactions and should not be described as simulator training SPS.

Output checkpoint SHA256:

`056818f3947c3e7efb1a8050521cfed72f6147c2444b21b02c7aa7ef0166b5d7`

Spark output:

`/home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1/distributional-candidate/epoch-1/policy.bin`

Post-update full legal-distribution KL versus frozen behavior was mean `0.000495983981592`, maximum `0.0246379835966`. The final sampled-action clipped fraction was `0.000800975100122`; maximum chosen ratio error was `0.438687022092`. Initial numerical-acceptance budgets are not post-update policy guarantees. Live performance remains unmeasured at publication.

## Data and objective

Included rounds: `noprior-s601`, `noprior-s602`, `noprior-s603-retry2`, `noprior-s604-retry4`, `noprior-s605-retry3`. All used behavior checkpoint SHA256 `7561859790d0ed55a55471d96442fe954a593c52d8617b51c143687ba9d98f96`, BF16 H256/L2 native worker, 223-feature `rek.native5.scaled_polar_xy.v1`, all-ones input-feature mask and the actual saved 33-action legality masks.

Every sampled decision is retained in recurrent order. Of 28,715 decisions, 28,710 have locally applied acknowledgements; five terminal-race rejected decisions retain recurrence and critic position with actor weight zero. Omitted source snapshots are not inserted. Acknowledgements establish local acceptance, not server execution or action causality.

Reward remains the established win objective with point-potential shaping:

`outcome + gamma_t * Phi(next_points) - Phi(current_points)`

`Phi = point_difference / (5 + abs(point_difference))`; terminal Phi is zero. The native +5 awards affect the observed counters. Discounts use actual QPC intervals: `pow(reference_per_20ms, dt/0.02)`, float32. Reference gamma is `0.9998844821426083`, lambda `0.9978673240629938`.

Training used complete-MC zero-baseline targets with value coefficient zero, avoiding reliance on the old critic's different reward scale. CUDA target recurrence matched its CPU reference exactly. Advantages were not normalized. Horizon 128 is truncated backpropagation length, not recurrent reset cadence; each chunk retains full-prefix current-weight reconstruction.

## Numerical acceptance

The original max-only bounded-BF16 mode rejected before any optimizer update: maximum chosen ratio error `0.0528736217504` exceeded its `0.02` budget. Exact sequential teacher logits and values were zero-error. At horizon 128, only ten rows exceeded 2% ratio error and none reached the 20% PPO clip boundary. Mean absolute advantage-weighted surrogate perturbation divided by mean absolute advantage was `3.86362154335e-5`, or 0.00386%.

Horizon 64 reproduced the same discrepancy. Horizon 256 was worse. No further horizon sweep was performed.

The explicit new `--allow-distributional-bf16-batch` mode requires:

- Exact sequential teacher logits and values.
- Zero initial sampled-action clipping and maximum chosen ratio error strictly below PPO clip.
- Mean full legal-distribution KL at most `1e-5`, maximum state KL at most `1e-3`.
- Absolute advantage-weighted surrogate perturbation divided by total absolute advantage at most `1e-3`.
- Finite, nonnegative evidence.

These engineering budgets were selected after examining the discrepancy and explicitly approved for this experiment. They are not a gradient-parity proof or a prospectively registered threshold. The original strict and max-only modes remain available. True frozen behavior log probabilities remain the denominator; no ratios were forced to one and no outliers were removed. PPO loss, gradients, reward, targets, native kernels and optimizer are unchanged.

Full evidence and caveats are in [DISTRIBUTIONAL-ACCEPTANCE.md](DISTRIBUTIONAL-ACCEPTANCE.md). The zero-epoch candidate passed and saved the unchanged input checkpoint before the separately approved one-epoch run.

## Source and reproduction

This directory is a selective publication bundle. The tested adapter is also integrated into the ten corresponding `ocean/rek_g1/native5` source paths on the current branch: four modified files and six added files. All ten integrated hashes match `source-manifest.json`. `source-adapter.patch` remains a reproducible patch against the exact four baseline files recorded in that manifest; do not reapply it to the already integrated source. `source/` contains final source copies for inspection. The patch passed `git apply --check` against an isolated baseline copy. Existing unrelated dirty files were preserved; this integration performed no commit or push.

Baseline source tree:

`/home/spark-advantage/rek-training/f7-action-id-fix-20260924-r1/source/ocean/rek_g1/native5`

The v3 adapter extends the binary header from 256 to 384 bytes, binding the identity JSON, worker executable, checkpoint and original native-policy object. Former row-reserved words hold each round's seed. Replay recreates a policy per recorded seed and must reproduce every sampled action before publishing. The legacy aggregate replay seed field is zero for v3, meaning per-sequence seeds, not seed-zero sampling.

Existing Spark reproduction assets are pinned by `export-config.json`, `run_native.sh` and `run_distributional.sh`. Existing outputs intentionally fail closed instead of being overwritten. Exact commands used:

```bash
stage=/home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1
node "$stage/source/authentic_trajectory_v3.cjs" "$stage/export-config.json" "$stage/export"
bash "$stage/source/build_authentic_replay_v3.sh" \
  /home/spark-advantage/rek-training/semantic-fast-20260914-v1/build-v4/native_policy.o \
  "$stage/replay-build"
bash "$stage/run_native.sh" replay --run
bash "$stage/distributional-candidate/source/build_authentic_ppo.sh" \
  /home/spark-advantage/rek-training/imitation-20260919-r1-native-bc/build-r5 \
  "$stage/distributional-candidate/ppo-build"
bash "$stage/run_distributional.sh" --diagnose
bash "$stage/run_distributional.sh" --train
```

For a fresh reproduction, use new output/build directories and update the launcher paths deliberately. Do not rerun against a live game sharing the GPU. Private raw trajectories and checkpoints remain on Spark; this bundle includes compact verification and training receipts, not raw native captures or authentication data.

## Tests and hashes

Passed: 21 JavaScript export tests, 15 legacy native dataset checks, eight actual v3 dataset checks, 24 GAE checks, 14 original parity checks, two kernel preparation tests and 16 new distributional-gate checks. Actual sampled-action replay passed all 28,715 decisions. New tests include nonfinite evidence, exact-sequential failures, clipping, KL/surrogate budget breaches, old-gate rejection and horizon-256 rejection.

After integration, the same 23 combined JavaScript export/kernel tests and native CPU checks were rerun from the actual repository source, including the 28,715-row v3 dataset. All passed. Native test executables and the private dataset copy were kept outside the repository under `C:/rekagent/work/authentic-ppo-noprior-20260924-r1/integration-cpu-tests`.

- Dataset: `8ced592947fc1167f771d9480a0a56da3bab025d5bae292dc90308552f0bf83b`
- Behavior identity: `674009772cbdbb719ff5ec7e15b4022c123cd3e61370a578a6de99aed97eb537`
- Replay binary: `ff391c9898357023e2f4d9185a69d2aec3ae5a68feb5c3fe690ff8116b2ba0f8`
- Final trainer executable: `1ddff37719ad26670c8ebf8a19abddd83872882c893d4c120b6c863335bde680`

This tested route uses an all-ones feature mask. Optional masked v3 exports require the sequential parity probe to apply the mask too; the unchanged probe currently reads raw observations. Masked training is not validated by this result. Future live evaluation must preserve checkpoint/encoder/worker identity and use prospectively selected new seeds. Five training episodes and small numerical discrepancy do not establish policy improvement or true server-physics parity.
