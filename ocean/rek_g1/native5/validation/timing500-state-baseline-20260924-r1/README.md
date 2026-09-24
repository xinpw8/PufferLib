# Fixed state-return baseline and isolated PPO variant

The fixed CUDA diagnostic reduced pooled held-out return MSE by **44.5%** against the zero predictor on five development episodes. Four episodes improved and one worsened substantially. This measures return prediction; it does not establish reduced policy-gradient variance or better live fighting.

The diagnostic fit executed on 2026-09-24 at 11:37:43 UTC into `fit-fixed-r1`. The consuming PPO update subsequently completed one epoch and 120 updates at 12:44:42–12:44:45 UTC, saving checkpoint `35263bb752e049c40e2d65ea80d58b72ba1bd8c98042d0a51ddbf175ce8b9680`. It is not deployed or live-validated; runtime control remains frozen f147. [TRAINING-RESULTS.md](TRAINING-RESULTS.md) records the executed checks and metrics.

## Fixed experiment and all five results

The unchanged [PROTOCOL.md](PROTOCOL.md) was frozen before CUDA fitting, SHA256 `32ee5bb1dd2cdb427cfeff7ae01f663a350858b3d35d80655d86987f64293dd9`. Inputs are the exact 14,957 pre-action 223-column rows from all five completed C2 rounds: 2 wins, 3 losses, 49:55 points. The dataset, actual C2 behavior and immutable replay are described in [the fresh-PPO publication](../timing500-ppo-refresh-20260924-r1/README.md).

Five leave-one-whole-episode-out linear ridge fits use coefficient 0.01, training-only standardization and an unpenalized intercept. Only actor-eligible rows from the other four episodes enter each fit. The five terminal-rejected rows remain in complete return propagation and artifact ordering, but not fitting/evaluation. No action, current reward, outcome, fold ID or future measurement is appended to the recorded state vector. Constant columns map to zero. No hyperparameter sweep, clipping, post-test refit or best-fold selection occurred.

| Held-out seed | Eligible rows | Zero MSE | Training-mean MSE | Ridge MSE |
| --- | ---: | ---: | ---: | ---: |
| 1301 | 3004 | 0.01587685 | 0.01523078 | 0.03217266 |
| 1302 | 3089 | 0.06882370 | 0.06841798 | 0.02631658 |
| 1303 | 2957 | 0.11332972 | 0.12008929 | 0.05022870 |
| 1304 | 2837 | 0.05992555 | 0.06089449 | 0.02444631 |
| 1305 | 3065 | 0.04677643 | 0.04751365 | 0.03536617 |
| Pooled | 14952 | 0.06078016 | 0.06223832 | 0.03372233 |

Seed 1301 is worse with the fitted predictor. Pooled centered residual variance is 0.06054573 for zero, 0.06223824 for the training mean and 0.03371878 for ridge; ridge residual mean is -0.0018849601. Reported centered variance is distinct from MSE. Complete CUDA returns match the CPU reference exactly. Training-scaler relative errors are below 6.2e-14 and normal-equation relative residuals below 1.4e-16. The full all-fold metrics are retained in `receipts/report.json`.

The frozen row-baseline artifact SHA256 is `8d4a24282f834a687a7740c14729974b87757a3b80779cb7fe004ba0936b268a`; it is private, contains 14,957 original-order rows and is not committed. The diagnostic changes no policy, replay, reward or environment.

## PPO variant

`ppo-variant/authentic_ppo.cu` is an isolated copy of the original source SHA256 `173bfaf5c8e511ab84f55efbfee956bd54fb1bc780542667269d5c9c742dde11`. The new explicit target mode loads the pinned baseline before GPU initialization and changes only:

```text
advantage[i] = unchanged_complete_MC_return[i] - frozen_cross_fitted_prediction[i]
```

CUDA computes the original complete-MC recurrence, then pointwise subtraction writes only advantages. Returns are compared exactly before and after. Rewards remain observed score difference/5 with recorded QPC five-second half-life. Old FP32 behavior log probabilities, old values, recorded actions, legal masks, actor weights, feature mask and recurrent sequence/burn-in are unchanged. VF remains 0. No advantage normalization or value-head fit is added.

The loader rejects wrong artifact/protocol/dataset SHA, shape, row index, episode, source sequence, actor eligibility, nonfinite values, changed MC returns or inconsistent residuals. Default zero-baseline mode bypasses the new loader/subtraction. Static tests prove the original MC kernel and the replay/mask/parity/loss/optimizer regions are byte-identical. Existing BF16 distributional acceptance remains an explicit approximation, not exact batched parity.

The executed run starts from actual C2, not from f147, on the same dataset and true replay as the f147 zero-baseline control. The sole learning change is the baseline: one epoch, LR 3e-5, H128, clip 0.2, VF clip 0.2, VF 0, entropy 0.001. Final epoch 1 was selected prospectively. `ppo-variant/run-plan.json` and `run_once.sh` pin the inputs and refuse an existing fixed output. The automatic CUDA self-tests passed; real return/subtraction reference error was 0 and returns remained unchanged. Post-update mean legal KL was 0.0042795853 and clipped fraction 0.0430567627, measured on the recorded training states.

Five correlated episodes, one adverse fold, finite-sample cross-fitting and PPO clipping limit the interpretation. Lower prediction MSE alone does not guarantee an unbiased baseline, lower gradient variance or better gameplay. Rewards and the deployed observation/action contract remain unchanged to isolate this experiment.

## Source, checks and reproduction

Private stage: `/home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1`.

This package contains the fixed native diagnostic source/build/protocol, reader headers, small PPO variant, tests and native command plan. The unchanged native dependency source is in [scorecredit5s-20260924-r2](../scorecredit5s-20260924-r2/README.md). Existing private build scripts retain their exact dependency paths. No production source is changed.

Executed checks: six native diagnostic CPU checks, 29 native variant rejection/reference checks, exact validation of all 14,957 real MC returns/14,952 eligible rows, source-equality checks and four live-preparer tests. The publication's native CPU tests and JavaScript tests were rerun successfully. CUDA compilation exited 0. Diagnostic CUDA synthetic checks execute automatically before the completed fit. The PPO variant CUDA binary SHA256 is `4087ab38dd972673370af39261a2bcb5a36713f608b9db54566e5c61de6c5344`.

Portable JavaScript checks from this directory:

```sh
node --test ppo-variant/prepare_live.test.cjs
node ppo-variant/test_unchanged_zero.cjs ../scorecredit5s-20260924-r2/source/authentic_ppo.cu ppo-variant/authentic_ppo.cu
```

CPU native checks can be built with C++17 and `-lcrypto`, including `vendor/`: compile `state_baseline.cu` as C++ with `-DREK_BASELINE_CPU_ONLY`, then execute `--cpu-test`; compile `ppo-variant/test_crossfit_baseline.cpp` and execute without arguments. No CUDA execution occurs in either check.

The diagnostic command below was executed with the fixed output. Existing outputs must be preserved; it is not a rerun instruction:

```sh
/home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1/build/state-baseline --run \
  /home/spark-advantage/rek-training/timing500-ppo-refresh-20260924-r1/score-delta-5s/authentic-score-delta-v5.bin \
  eb6b1ae210b3b3911597f93a513ae32081f15e4871b29c1cb29e75829f718655 \
  /home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1/PROTOCOL.md \
  32ee5bb1dd2cdb427cfeff7ae01f663a350858b3d35d80655d86987f64293dd9 \
  /home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1/fit-fixed-r1
```

The actor update command `bash ppo-variant/run_once.sh` was executed from the private stage under separate GPU authorization and exited 0. Its fixed output now exists and must be preserved; this is an execution record, not a rerun instruction. No new archive was produced by this publication update.

Original build receipts, run-plan status and preparation README remain verbatim and describe their earlier pre-fit/pre-training cutoffs. `SOURCE-MANIFEST.json` records copies and the sole source adjustment: the live-preparer test uses local frozen fixtures rather than a machine-specific private path. Those three fixture files are exact copies. The later training result and scalar receipts are separate additions. The live preparer preserves the original runtime and all 17 attacks, with prospective seeds 1601..1620; it does not launch a controller.

No baseline weights, policy weights, binaries, raw datasets, replay logits or raw captures are published. The separate [G1 opponent-selection note](G1-OPPONENT-SELECTION.md) records a read-only source finding and does not change this experiment.
