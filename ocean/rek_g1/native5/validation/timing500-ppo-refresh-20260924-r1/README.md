# Fresh C2 on-policy PPO update

Native replay and training completed successfully using every completed round of the closed C2 cohort. The final checkpoint is `f147bdc358261e272c58e70167cd8b0b9e4953806c417d8891d94a3ad083ac84`. This package records training, not improved live performance. No completed live evaluation of this checkpoint is included.

## Data and objective

The actual behavior checkpoint was `c2c4987b268996cd912fe35e6ba5f9b20a94d69fd93e4ef15b6c5fc71ee5e533`, at `/home/spark-advantage/rek-training/timing500-onpolicy-20260924-r2/train-score-delta/policy.bin`. The source cohort closed at 2026-09-24T11:00:34.706Z with 2 wins, 3 losses and 49:55 points. All five completed rounds are included; nine incomplete attempts are excluded. Minimum five is an explicit development-batch override, not a passed evaluation criterion.

| Seed / completed attempt | Own:opponent points | Decisions |
| --- | --- | ---: |
| 1301 / retry7 | 2:6 | 3005 |
| 1302 / retry2 | 9:11 | 3090 |
| 1303 / retry3 | 20:14 | 2958 |
| 1304 | 12:10 | 2838 |
| 1305 | 6:14 | 3066 |
| Total | 49:55 | 14957 |

All 223 observations, action masks, samples, actual seeds, recurrent sequence order, reset boundaries and source QPC intervals are retained. The five terminal-race requests have actor weight zero, remain in the recurrent sequence and retain return propagation. There are 14,952 applied actor rows. No feature migration, resampling, invented observations, reward bonuses or attack gates are introduced. Original actor logits were not recorded; exact replay reconstructs them from the actual C2 worker and seeds.

The original v3 reward export is preserved privately. Derived v5 changes only reward and gamma cells:

```text
r = ((nextOwn-own) - (nextOpponent-opponent)) / 5
gamma = float32(2^(-actual_QPC_dt / 5))
G = r + (terminal ? 0 : gamma * nextG)
advantage = return = G
```

No terminal-win bonus or score potential is added. Undiscounted per-round rewards equal the point margin divided by five within float32 tolerance. The reward-bearing row's own interval is undiscounted under the existing recurrence. H128 limits recurrent backpropagation, not complete-episode return calculation. Natural measured cadence spans 23.890 to 26.039 decisions/s across these episodes.

All five exact native-capture SHA/size checks, native score-packet validation, received-referee validation and controlled-start coverage checks passed. Original legacy coverage results remain four false and one true. The separately recorded coverage contract anchors the existing one-second ACK bounds to measured first controlled source, using UnityTime; startup binding and reward discount use QPC. Initial controlled timers range from 118.59961 to 118.86636 seconds. A native 120-second round does not imply 120 seconds of policy control. All startup counters remain zero through the first decision.

## Executed native result

- Replay: 14,957/14,957 sampled actions exact, zero mismatches, 2.43 s full native process, exit 0.
- CUDA complete-MC target reference: maximum absolute error 0.
- Exact sequential teacher: logits and values have maximum error 0.
- One fresh epoch from C2: LR `3e-5`, H128, clip0.2, VF clip0.2, VF0, entropy0.001, no advantage normalization, 120 optimizer updates, 1.99 s full native process, exit 0.
- Final frozen-behavior mean legal KL `0.00239410671307`, maximum `0.0386989684803`, clipped fraction `0.0114996322792`.
- Final epoch1 was selected prospectively. No environment stepping or development-checkpoint selection occurred.

Initial batched BF16 computation uses the existing explicit distributional acceptance, with true old log probabilities unchanged. It is not exact batched parity: initial maximum ratio error was `0.0208234353093`, mean legal KL `9.84229831576e-8`, maximum legal KL `0.000120368802149`, relative absolute surrogate error `3.0482960779e-5`, and initial clipped fraction 0. The exact sequential check passed. The original numeric budgets and PPO kernels were unchanged.

The separate earlier LR-only checkpoint7001 cohort closed 2 wins, 3 losses, 47:60 points at 2026-09-24T11:25:21.524Z and failed its criterion. Its rounds are not this training batch. Five C2 episodes remain a small, correlated training sample; no efficacy claim follows from the optimizer metrics.

## Reproduction

Executed private stage: `/home/spark-advantage/rek-training/timing500-ppo-refresh-20260924-r1`.

Source copies are byte-identical to the executed private modules. They include the exact native-capture manifest, selector, strict sidecar derivation, reward exporter and runners. `SOURCE-MANIFEST.json` binds the copies. The frozen pre-execution README, CPU receipt and run plan are preserved under `receipts/`; their GPU-false fields describe the earlier preparation cutoff. Actual replay/training commands, stdout, times and exit receipts supersede that status.

The unchanged native reader, CUDA source and build dependencies are published in [scorecredit5s-20260924-r2](../scorecredit5s-20260924-r2/README.md). Controlled-start logic is inherited from [timing500-onpolicy-20260924-r2](../timing500-onpolicy-20260924-r2/README.md). No production source is modified here. Raw captures, datasets, replay tensors, checkpoints, executables and proprietary assets are excluded.

CPU tests from this publication directory:

```sh
node --test onpolicy.test.cjs controlled_policy_coverage.test.cjs test_score_delta.cjs source/authentic_trajectory_data.test.cjs source/authentic_trajectory_v3.test.cjs prepare_live.test.cjs
```

Private-stage commands, already executed, require the pinned raw evidence and assets. Existing outputs are refused rather than overwritten:

```sh
node select_closed_cohort.cjs
node prepare_export.cjs selection.json --run
node run_native.cjs run-plan.json replay --check
node run_native.cjs run-plan.json train --check
node run_native.cjs run-plan.json replay --run
node run_native.cjs run-plan.json train --run
```

## Prospective live configuration

`prepare_live.cjs` accepts an explicit checkpoint path and SHA. It preserves the reviewed driver `b57654...`, controller `160299...`, bridge `11fcfa...`, worker `52741...`, encoder `4c820e...`, all-ones mask `59158...`, balance8 schema and 500/750 ms runtime. All 17 attacks remain available. The frozen runtime template is5b19, while training-parent metadata is explicitly C2. Only the new checkpoint, output, labels/seeds and hypothesis change.

```sh
node prepare_live.cjs /home/spark-advantage/rek-training/timing500-ppo-refresh-20260924-r1/train-score-delta/policy.bin f147bdc358261e272c58e70167cd8b0b9e4953806c417d8891d94a3ad083ac84
```

This prepares `/home/spark-advantage/rek-training/timing500-ppo-refresh-live-20260924-r1`, labels `refresh-s1501` through `refresh-s1520`, with the existing 18/20 target and stop on the third nonwin. It does not launch the controller. Different policy seeds do not control server randomness. Live execution and results must be recorded separately.

Preparation subsequently completed with planned-rounds SHA256 `7c2761d015ee7f2e20742c72eee07645ff7eed1a283b428e1a31f3d64a8174c4`; the exact plan is copied to `receipts/planned-rounds.json`. The cohort controller launched at 2026-09-24T11:37:51Z with PID3051691. This is launch status only; no completed live result is included here. The plan's `controller_started:false` records its earlier preparation cutoff.

The closed native stage is recoverable from the verified NAS archive described in [ARCHIVE.md](ARCHIVE.md). The changing live stage is excluded.
