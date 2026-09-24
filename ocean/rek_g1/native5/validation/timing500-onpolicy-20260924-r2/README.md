# Timing500 on-policy native update

The complete parent cohort supplied 24,010 recorded decisions from eight completed native 120-second Bot1 rounds: 5 wins, 3 losses, 104:93 points. All eight rounds, including all three losses, entered this update. The predefined 18/20 evaluation criterion had failed on its third nonwin. This is a small development training batch.

| Seed / completed attempt | Points | Decisions | Undiscounted reward |
| --- | --- | ---: | ---: |
| 1201 / retry8 | 16:10 | 2821 | 1.2 |
| 1202 | 6:13 | 2964 | -1.4 |
| 1203 | 4:3 | 3064 | 0.2 |
| 1204 | 11:4 | 3100 | 1.4 |
| 1205 | 20:14 | 3102 | 1.2 |
| 1206 | 18:14 | 3022 | 0.8 |
| 1207 / retry2 | 17:22 | 3000 | -1.0 |
| 1208 | 12:13 | 2937 | -0.2 |

Every recorded observation, all 223 features, legal-action mask, action, seed, recurrent order/reset, timestamp and actor weight is retained. No resampling or teacher migration is applied. The 24,002 locally applied rows retain actor weight 1; each round's final rejected request retains actor weight 0 and remains in recurrent history and return propagation. Native/referee validation passed all eight rounds and reconciled the full 104:93 score counters. Recorded worker cadence was 23.781 to 26.128 decisions/s using complete source-to-terminal QPC spans.

## Controlled-start compatibility correction

The failed private `timing500-onpolicy-20260924-r1` export remains preserved. Its first round failed only the legacy analyzer's first-action coverage bound: 1.127513542 seconds from the earliest passive snapshot exceeded 1 second. The unmodified analyzer and its JSON outputs are retained. Across all eight rounds, its original `completed_policy_round` remains false for seven and true for s1205.

`controlled_policy_coverage.cjs` adds an explicit `rek.controlled_policy_coverage.v1` derivative. It binds the exact passive readiness snapshot to the actual first controlled snapshot under the frozen driver's 117.5-second readiness and 117-second controlled-start bounds. It requires both observed scores to remain zero through the first worker decision. It reapplies the original 1-second first-ACK, inter-ACK and terminal-gap limits from that measured controlled source. The exporter independently rederives the receipt from actual sources and acknowledgements and verifies hashes of the unchanged contact summary, derived trial summary, relay stream and worker input. It does not flip the legacy boolean or weaken packet/referee/terminal validation.

Coverage intervals use **UnityTime**, matching the original analyzer. Startup identity binding and reward discounts use **QPC**. These are distinct clocks; coverage is not a new wall-clock guarantee. Local applied acknowledgements do not prove server acceptance.

Actual control began with 118.64971 to 118.78309 seconds remaining, after 1.21691 to 1.35029 native timer seconds. Controlled-source to first-ACK intervals were 0.186081 to 0.259573 UnityTime seconds; the maximum inter-ACK gap was 0.130561 seconds. All eight derivative receipts pass. The 120-second description refers to native round duration, not 120 seconds of policy control. No unobserved startup actions, rewards or measurements are fabricated.

## Reward and optimization

The separately preserved baseline v3 export contains the original terminal-outcome plus discounted score-potential metadata. Derived v5 changes only reward and gamma cells:

```text
r_i = ((nextOwn - own) - (nextOpponent - opponent)) / 5
gamma_i = float32(2^(-actual_QPC_dt_i / 5))
G_i = r_i + (terminal_i ? 0 : gamma_i * G_(i+1))
advantage_i = return_i = G_i
```

There is no terminal win bonus, score potential, reward clipping, guessed fall hazard or reward for referee flags alone. Measured five-point awards enter through received score counters. Actor weight does not multiply the return recurrence. Terminal rejected-row rewards can therefore influence earlier decisions. Discounting is between reward-row source times, retaining the standard one-observation receipt-time quantization.

Training executed one native CUDA epoch, LR 1e-5, horizon 128, PPO clip 0.2, VF clip 0.2, VF coefficient 0, entropy 0.001, complete-MC zero baseline, and no advantage normalization. Original behavior FP32 log probabilities remain immutable. No environment stepping or Python training occurred.

Native replay reproduced **24,010/24,010 sampled actions**, zero mismatches, in 4.38 seconds full process wall time. Original logits were not recorded; the pinned original checkpoint, worker/native object, masks and actual seeds reconstruct them. Sequential teacher logit/value parity against this replay was exact.

Initial BF16 batched evaluation was explicitly accepted under the existing distributional mode, with mean legal KL 1.81612e-7, maximum 0.000478677, chosen ratio error maximum 0.0271792, relative absolute surrogate perturbation 3.35920e-5, and zero initial clipping. This does not establish exact batched parity or exact batched gradients. The true behavior denominator was retained.

CUDA complete-MC targets matched the CPU reference with maximum error 0. Training completed 192 updates, exit 0, in 2.58 seconds full native process wall time. Post-update mean legal KL was 0.000679761137, maximum 0.0102871179, and clipped fraction 0.000166597. This is offline optimization timing, not environment-training SPS or evidence of improved win rate.

New checkpoint SHA256:
`c2c4987b268996cd912fe35e6ba5f9b20a94d69fd93e4ef15b6c5fc71ee5e533`

Parent actual behavior SHA256:
`5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4`

## Source, receipts and reproduction

This folder is a private variant publication. Production native5 files and unrelated dirty work are untouched. The 19 source/config files are byte-identical to their private-stage originals; see `SOURCE-MANIFEST.json`. The unchanged `prepare_live.cjs` and test are copied from private r1. All local module dependencies needed by the CPU tests are included. Native CUDA binaries were reused without code changes from the prior scorecredit5s stage; no binary, dataset, checkpoint, raw capture or proprietary asset is committed here.

The native source/build reproduction remains documented in [the scorecredit5s package](../scorecredit5s-20260924-r2/README.md), using its `materialize_source.cjs`, native source overrides and pinned existing kernel dependencies. Exact executable hashes and commands are retained in `receipts/run-plan.json` and the replay/train receipt directories. Existing contact/referee validators remain at their hashed Spark paths recorded in `receipts/evidence-manifest.json`; this adapter does not replace them.

From this publication directory, CPU tests require only Node.js:

```sh
node --test onpolicy.test.cjs controlled_policy_coverage.test.cjs test_score_delta.cjs prepare_live.test.cjs source/authentic_trajectory_data.test.cjs source/authentic_trajectory_v3.test.cjs
```

All 40 tests passed from this exact repository copy. The 39 export/reward tests passed on both Windows and Spark before actual export. Test coverage includes adverse startup identity/round/QPC/slot/score/bounds, missing native evidence, ACK gaps, receipt hash changes, unchanged observations, rejected-row returns and current native command configuration. Recorded output is in `receipts/PUBLICATION-CPU-TESTS.txt`.

Actual private stage:
`/home/spark-advantage/rek-training/timing500-onpolicy-20260924-r2`

Executed preparation/export and native commands:

```sh
node select_closed_cohort.cjs
node prepare_export.cjs selection.json --run
node run_native.cjs run-plan.json replay --run
node run_native.cjs run-plan.json train --run
```

`select_closed_cohort.cjs` verifies the closed cohort's final ledger and all eight completed episodes, including losses, and explicitly sets minimum 8. The generic default remains 20. The actual selection is `receipts/selection.frozen.json`; `selection.example.json` is startup-check scaffolding, not the executed selection. Re-execution requires a new isolated stage and its explicit destination in the selection. Existing outputs are refused and must not be overwritten. Restore private datasets/checkpoints/raw captures from their archive dependencies before replay or export; they are intentionally absent from Git.

## Prospective live evaluation

The live cohort `/home/spark-advantage/rek-training/timing500-ppo-live-20260924-r1` launched at 2026-09-24T10:37:22Z, seeds 1301..1320, with the same 500/750 ms runtime contract, encoder, worker, all-ones mask, all 17 attacks and sampled actions. Target remains 18/20 with stopping on the third nonwin. `prepare_live.cjs` preserves the runtime configuration and changes only checkpoint, sampled seed and output paths. No live result is included at this publication cutoff. Replay and optimizer receipts do not establish authentic Bot1 improvement.

## Verified private archive

The completed native stage and eight required native captures were archived separately from Git at:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\timing500-onpolicy-native-20260924-r2`

- `evidence.tar.gz`: 35,856,692 bytes, SHA256 `337925d9f1e6a3bcad1e8db2caa120279725d8e90582e6532ec427f207723aff`.
- `native-captures.tar.gz`: 242,401,146 bytes, SHA256 `d6adde359c1e0ecb33987d31dd7999ad072192557821dc1e5dc64b7dd99e8142`.

NAS readback and all source/native before-after hashes passed. The stage archive preserves 88 symlinks into the separately archived parent cohort rather than duplicating raw streams. Exact parent archive identity, the eight native capture hashes, and restore instructions are in `receipts/closed-archive-receipt.json`. Restore that parent dependency before using the symlinks. No private stage or active live cohort was changed by archiving.
