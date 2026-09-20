# Frozen critic calibration experiment

Status: CPU and CUDA tests, fitting, value-only checkpoint production, and exact native replay passed on 2026-09-20. Scaling fixes the gross value magnitude, but its held-out MSE is worse than the train-mean constant baseline. After reviewing that result, one separately authorized GAE actor-control epoch completed. No game interaction occurred. No runtime, observation, training implementation, or existing checkpoint was changed.

## Representable correction

The discrete native5 decoder has a weight matrix and no bias. Pinned `algo.cu` registers encoder, decoder, then two MinGRU matrices (`weights_create`, line 964). `decoder_reg_params`, line 636, registers only `(34,256)` for this discrete policy, and `decoder_forward`, line 617, applies a matrix multiplication without an intercept. The native inference path agrees: `native_policy.cu`, lines 96, 104, and 121.

Consequently, an arbitrary `a*V+b` cannot be encoded by changing only existing value-output parameters. This experiment instead fits `a*V` through the origin. The unrestricted affine fit is diagnostic only. It is never written to a checkpoint or implemented through shared features.

The tool calls the pinned `build_arch` and `weights_create` on CPU and inspects their actual registrations. It does not infer offsets from checkpoint size. The result is 459,008 FP32 parameters; the final decoder row occupies indices `[65536,65792)`, or byte offsets `[262144,263168)`. All other bytes must remain unchanged. The saved checkpoint is reopened and compared with the intended output.

## Fixed data split and target

Use the unchanged MC checkpoint `a985d6c06b5dfab7198319caf5e99099b85da30eb40402758cc254e407e2059f` and its exact native behavior replay of the iteration2 dataset:

| Sequence | Round | Result | Rows | Calibration role |
|---|---|---:|---:|---|
| 0 | r11 | 11:9 win | 5,810 | fit |
| 1 | r13 | 20:12 win | 5,788 | held-out diagnostic |
| 2 | r15 | 13:21 loss | 5,762 | fit |

All 17,360 value rows are retained, including the three terminal-race requests with zero actor weight. The fit uniformly weights the 11,572 rows in sequences 0 and 2. This is a two-round fit and a one-round held-out diagnostic, not evidence of broad generalization. Later rounds must not change this fit.

CUDA computes complete shaped returns backwards within each closed sequence: `G[t] = reward[t] + (terminal[t] ? 0 : gamma[t]*G[t+1])`. It uses the serialized actual-time-scaled gamma and reward values. There is no learned-value bootstrap and no lambda term in the complete return. CUDA then fits `a = sum(V*G)/sum(V*V)` using only sequences 0 and 2. It also computes their mean return as the constant comparator and their unconstrained affine fit as a diagnostic. All fits and return recurrences use double-precision CUDA arithmetic; C++ handles input validation, file I/O, and byte comparisons.

Reported MSE and signed error compare unchanged V, ideal `a*V`, the train-mean constant, and diagnostic `a*V+b`. A new checkpoint scales only the 256 FP32 value weights. BF16 weight/output rounding means its realized native value need not equal `a` times the previously rounded value. The separate native replay verification reports that realized value's metrics as well.

## Observed result

The CUDA fit gives `a = 0.013353263756134198`; the train-mean constant is `0.25564011220967403`. Signed error below is prediction minus complete return.

| Predictor | Train MSE | Held-out r13 MSE | Held-out signed error |
|---|---:|---:|---:|
| Unchanged native V | 103.304024961 | 161.950817336 | +10.330454596 |
| Ideal scalar `a*V` | 0.334352878 | 0.306775776 | -0.456425566 |
| Actual scaled checkpoint, native BF16 | 0.334558141 | 0.307045417 | -0.456671917 |
| Train-mean constant | 0.287861841 | 0.183542070 | -0.346774945 |
| Diagnostic affine, not representable | 0.287688222 | 0.186137228 | -0.354594961 |

The diagnostic affine coefficients are `a = -0.001571674727778107`, `b = 0.26500301193257592`. Its train MSE is only slightly below the constant comparator, and its held-out MSE is higher. These results do not demonstrate that the frozen critic contains useful transferable return variation. They also do not assess the efficacy of a future TD/GAE actor update. In particular, fitting a baseline after collecting these trajectories would need to be declared explicitly in such an update.

All 17,360 recorded sampled actions, all 33 actor logits per row, and every chosen-action log-probability are bitwise identical with the new checkpoint. All 458,752 parameters outside the value row are bitwise unchanged, and the original checkpoint digest still matches. No tolerance was introduced or relaxed for these checks.

The bounded GPU sequence ran from `2026-09-20T02:16:02.914231126Z` to `2026-09-20T02:16:07.093287161Z`. Calibration took 0.38 s and exact native replay took 3.01 s according to `/usr/bin/time`; the enclosing 4.179056035 s also includes fixtures and verification. GPU work was exclusively authorized between live-round boundaries.

## Subsequent bounded GAE actor control

This is an empirical control, not a validated predictive critic. It starts from the new value-scaled checkpoint and uses its freshly verified replay. The behavior actor originally collected the data as checkpoint `a985...`; only its value baseline was fitted after collection. Actor logits and saved action log-probabilities were demonstrated identical before the update. This distinction is recorded in the command log, since the training binary binds its replay header to the value-scaled initial checkpoint.

The existing `build-r5/authentic-ppo` executed one epoch with a fresh optimizer, learning rate `1e-5`, horizon 128, policy clip 0.2, value clip 0.2, value coefficient 0.5, and entropy coefficient 0.001. It used `--allow-bounded-bf16-batch` and no complete-MC flag. The same rewards, observation/action masks, and serialized time-scaled discounts were retained; their base gamma and lambda are `0.9998844821426083` and `0.9978673240629938`. No later rounds were used. All three original sequences, including r13, participate in the actor update, so r13 remains held out only for the preceding calibration diagnostic, not for this updated actor.

CUDA-generated frozen-value GAE targets matched the independent CPU reference with maximum error zero. Advantages had mean `0.00745795279838`, standard deviation `0.206450266984`, minimum `-0.646270573139`, and maximum `0.924476623535`. There were 138 updates, 17,357 actor rows, and 17,360 value rows. Current-weight prefix burn-in and the existing whole-episode chunk order were unchanged. As in prior controls, advantages were not normalized.

The sequential Puffer teacher matched native replay logits and values exactly before optimization. Initial batch-forward mean legal KL was `1.34143118816e-7`, maximum KL `0.000126276976587`, maximum chosen-ratio error `0.0168326013282`, and clipped fraction zero. The existing explicitly approved 0.02 BF16 chosen-ratio bound accepted this initial batch approximation; no thresholds changed.

Post-update batch-forward diagnostics against the frozen actor were:

| Metric | Value |
|---|---:|
| Mean legal KL, old to current | 0.000244957995783 |
| Maximum legal KL | 0.00829439537165 |
| Chosen ratio minimum / median / 95th percentile / maximum | 0.838840649229 / 0.994493916894 / 1.04138833653 / 1.13713062175 |
| Clipped fraction | 0 |
| Frozen / updated value MSE to frozen GAE target | 0.0426773337592 / 0.0392791567844 |
| Mean frozen GAE target | 0.108842207571 |

These are batch-forward training diagnostics, not fighting outcomes. The last value MSE compares bootstrapped GAE targets and must not be compared directly with complete-MC MSE in the calibration table. This control changes the estimator and enables value-head/shared-network learning compared with the MC-zero-baseline control; it is not an isolated learning-rate comparison.

Execution ran from `2026-09-20T02:19:37.387751611Z` to `2026-09-20T02:19:39.655076884Z`, with `/usr/bin/time` reporting 2.23 s and exit status zero. GPU reservation was then released. The new actor checkpoint is `train-scaled-gae-r1/ppo-one-epoch.bin`, and the epoch copy has the same hash. The separate value-only checkpoint and original MC checkpoint were rehashed and remain unchanged.

## Verification and commands

Sources are `critic_calibration.cu` and `build_critic_calibration.sh`. The CPU test derives the layout and rejects mutations immediately outside either end of the value row, in the encoder, and at the end of the recurrent parameters. The CUDA fixtures check a closed-form return recurrence, held-out exclusion, scalar and affine fits, train-mean metrics, value-only weight changes, and rejection of a singular scalar fit. Both test modes passed.

Build against the existing prepared PPO source, without running a GPU test:

```sh
bash build_critic_calibration.sh PREPARED_PPO_BUILD NEW_BUILD_DIRECTORY
```

After GPU authorization, run in this order, using new output paths:

```sh
critic-calibration --gpu-self-test
critic-calibration DATA OLD_REPLAY ORIGINAL_CHECKPOINT ORIGINAL_SHA NEW_CHECKPOINT
replay-authentic-behavior DATA NEW_CHECKPOINT NEW_SHA NEW_REPLAY
critic-calibration --verify DATA OLD_REPLAY NEW_REPLAY
```

The existing replay utility uses the actual native BF16 policy, original unmasked observations, saved legal masks, and seed 73 per newly started worker. It must reproduce every recorded sampled action. The calibration verifier additionally requires every actor logit and chosen-action log-probability to remain bitwise identical across all rows. No checkpoint is approved for use before those checks. Improved held-out critic error, if observed, would not establish improved fighting or improved TD learning.

The current CLI deliberately supports exactly these three sequences. `--verify` recalculates the same fit from the original replay and split. It must not be pointed at a different three-round batch to evaluate the frozen calibration, because that would refit. Any later out-of-sample evaluation needs an explicit fixed-fit path or must retain the original fitting sequences unchanged while extending only the diagnostic set.

## Private artifact provenance

Spark stage: `/home/spark-advantage/rek-training/critic-calibration-20260919-r1`. CPU build: `build-r1/critic-calibration`. Prepared architecture: `/home/spark-advantage/rek-training/authentic-ppo-20260919-r1/build-r5/source`. Local command transcript and build logs: `C:\rekagent\work\consistent-fighter-20260919-r1\critic-calibration-r1`.

New checkpoint, replay, metrics, command script, timings, and full stdout/stderr are in Spark `fit-scalar-r1/`. The local mirror is `C:\rekagent\work\consistent-fighter-20260919-r1\critic-calibration-r1\spark-results\fit-scalar-r1`. All 13 files, totaling 4,494,759 bytes, were independently rehashed after copying and match Spark. The new checkpoint is `value-scaled.bin`; its refreshed value-bearing replay is `calibrated-replay.bin`.

The subsequent actor-control artifacts are in Spark `train-scaled-gae-r1/` and the corresponding sibling under local `spark-results/`. All nine files, totaling 3,687,614 bytes, were rehashed after copying and match Spark. Trainer stderr is empty; the separate traced-command stderr preserves the exact invocation.

Input dataset and replay are under `/home/spark-advantage/rek-training/authentic-trajectory-mc-iteration2-20260919-r1`: `data-r2/authentic-trajectories.bin` and `replay-r1/behavior-replay.bin`. Original checkpoint: `/home/spark-advantage/rek-training/authentic-ppo-20260919-r1/train-mc-zero-r2/ppo-one-epoch.bin`. Exact replay executable: `/home/spark-advantage/rek-training/authentic-trajectory-20260919-r1/build-r1/replay-authentic-behavior`.

| Artifact | SHA-256 |
|---|---|
| Dataset | `b5cbc0b8de82df0d02036d069e505f0ae60825a8c77c1e44eaa8ac15037705b3` |
| Original replay | `f583516f3dd7410a2b7a8522d307123f2b4af15b2fab0376d42d16b3aca3cca9` |
| Original checkpoint | `a985d6c06b5dfab7198319caf5e99099b85da30eb40402758cc254e407e2059f` |
| Native replay executable | `b44ee7d100bf9719eb2543f665b5ccbc592fced2427af374064542016b2b59ec` |
| Calibration executable | `eee4abbe3aec6926450f9c2e2918a8b00f085af06d654fabf6a57a4cfa8c1993` |
| Calibration source | `d1e647eb1cac4cfa89f0fdd9d0be071b8c0e7d8bddbd9df23671793855b7a999` |
| Build script | `3d45411ec1de4bbbf1de7c05000782be4e053d4ef719fed3dd92eb4602bccbec` |
| Pinned `algo.cu` | `8a514cb8dd12d49b79cbd5afe7298875b6f0ca0491270bb19a8696bd527f4d92` |
| Prepared `puffer5_bc_core.cuh` | `d3e07e6c5f376584543cdba457d582d6674183a0efb29c0755f236adcf284454` |
| Trajectory decoder header | `617c0d0b8d7b419f78d611a9472dceb7cff90b3bf8ca4419c67a0939145ef140` |
| New value-only checkpoint | `f11f02c3c7a3f9cb8cdcb5810ea6f39219aa71a302904b42456d180318467c8b` |
| New exact native replay | `40ee1edaa1deed5120a6d25145aa269162837b7b599ed915eb127a4298a89c66` |
| `calibration.json` | `1fb9a76f25b257ecf4ad806751676c510b9631da5fc61b19185907eeaab25637` |
| `verification.json` | `815d8423b3dea0675ccfeff9579fe552aa4320ecdd3a8a304ea8b74c376abd8d` |
| Exact GPU command script | `3a89d983f5ecab53244e5d9b795640ce3fa462349f43252aee47de8c0f8bbd9f` |
| Full commands/stdout | `960d6de40d9f4f920f7abdff0dc43010ad618b47e897fe82592813f408fd972c` |
| Full traced commands/stderr | `9c8401c40c8bbdf30cfbc82e2ac641eaf79521caae8aa1869dda67642548bf1a` |
| Existing authentic PPO trainer | `b5c2c61932ad63c4dc40c137a6ef493a423826306aebf83859efdb5c0197da1a` |
| New GAE actor checkpoint and epoch copy | `f8bcd3f3d6ef3d691209823d5a0d452ca16ff16b03c7ed1867715510986ea483` |
| GAE control `stdout.jsonl` | `7a64124f28570c18c10c04a66a8c2e65c5239e16eec4139ec18c5d6500d17aea` |
| GAE control exact command script | `811b66e3dd73606fbeaa46efae40894fc953ae5657f55c567c11fd8f19fe4b65` |
| GAE control full commands/stdout | `6434c803203c5e8f4288121a5ff5983098faaf6a892ae7a09f508ed8e3b0080e` |
| GAE control traced commands/stderr | `b49caf43e4875b2a715fd5442ef2079e9f58280b41f6dd089564de9dfaaa0b41` |

No raw captures, checkpoints, or proprietary assets are included in this source report.
