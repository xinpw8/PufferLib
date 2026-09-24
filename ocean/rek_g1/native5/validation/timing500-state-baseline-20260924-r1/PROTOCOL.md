# Fixed cross-fitted state-return baseline diagnostic

Status: protocol specified before any CUDA fitting or heldout metric evaluation. This is a private five-episode development diagnostic. No policy, replay, reward, simulator hazard, action mask, or live controller is changed.

## Frozen inputs and targets

Dataset: `/home/spark-advantage/rek-training/timing500-ppo-refresh-20260924-r1/score-delta-5s/authentic-score-delta-v5.bin`.

- SHA256: `eb6b1ae210b3b3911597f93a513ae32081f15e4871b29c1cb29e75829f718655`.
- Behavior checkpoint SHA256: `c2c4987b268996cd912fe35e6ba5f9b20a94d69fd93e4ef15b6c5fc71ee5e533`.
- Behavior identity JSON SHA256: `5e65d45f54e444a47d5f87e10691715ac1b0bc82a44685dd015e0e9efdb333c0`.
- Five complete episodes in original order: `timingppo-s1301-retry7`, `timingppo-s1302-retry2`, `timingppo-s1303-retry3`, `timingppo-s1304`, `timingppo-s1305`.
- Episode row counts: 3005, 3090, 2958, 2838, 3066. Total 14957, with 14952 actor-eligible rows and five actor-excluded terminal rows.

Use exactly the 223 pre-action observation floats already stored in each V5 row. Do not append action, action support, outcome, reward, source sequence, fold ID, time, or post-action features. Observation features that already encode state or past requests remain unchanged. This does not make the observation Markov or create a guarantee of unbiased advantages.

The unchanged target is complete-episode Monte Carlo return with reward `(own_point_delta - opponent_point_delta)/5` and recorded `gamma = float(2^(-elapsed_seconds/5))`. CUDA uses double recursive accumulation and stores float targets, matching the existing complete-MC-zero-baseline kernel. Include every row in this recurrence, including the five actor-zero terminal rows. Never bootstrap from the old simulator value head.

## Fixed fit

Five leave-one-whole-episode-out fits. Each fit uses only actor-eligible rows from the other four episodes for scaling, target mean and fitting. Evaluate only actor-eligible rows in the heldout episode. Preserve original order for the artifact, including explicitly flagged excluded rows.

For each training fold, compute feature means and population standard deviations with CUDA double Welford moments. A constant column has standard deviation zero and is mapped to zero for both training and heldout predictions. There is no clipping, heldout normalization, feature selection, current-action conditioning or missing-value imputation. Nonfinite data or results abort.

Fit one scalar linear head with intercept by minimizing `mean((G - b - z*w)^2) + 0.01 * sum(w^2)`. Therefore `A = X'X/n + 0.01*diag(1,...,1,0)`, with the intercept last and unpenalized. CUDA cuBLAS computes double normal equations, CUDA cuSOLVER uses double Cholesky. A nonzero solver info, nonfinite result, or normalized normal-equation residual above `1e-10` aborts. The fixed coefficient is 0.01 for all five folds. No lambda sweep, test-driven feature changes, best-fold selection, or post-test refit.

## Evaluation and verification

CUDA reports count, MSE, residual mean and centered residual population variance for all three predictors: zero, that fold's training-return mean, and the heldout ridge prediction. Report every episode and all eligible rows pooled. Centered variance is reported separately so removal of an offset is not confused with lower MSE. No metric threshold changes implementation or enables a baseline.

CPU reference computation is used only for tests and numerical verification. Tests include excluded-terminal reward propagation, training-only scalers, constant columns, actor-excluded fitting/evaluation, synthetic Cholesky comparison, and CUDA-versus-CPU return/metric checks. CUDA synthetic tests and actual fits require a separate explicitly authorized GPU run. Compiling and `cpu-check` never use CUDA.

The diagnostic answers whether this fixed linear state predictor reduces heldout return error on these five development episodes. Five correlated episodes provide limited evidence. This does not establish PPO variance reduction, transfer, better gameplay, causal contact labels or independent final-test performance. Report failures and adverse metrics without refitting.

## Output and immutability

The executable requires this protocol's SHA256, the fixed dataset SHA256 and a nonexistent output directory. It records protocol, binary, input and behavior pins before any GPU fit. It writes five fold scaler/weight JSON files, per-fold and pooled metrics, and `row-baselines.bin`. It rechecks the source dataset digest after completion. Existing input/replay/policy files are read-only.

Binary artifact uses little endian. Header: magic `REKSB001` (8 bytes), six uint32 values (version 1, row count, 223 features, five folds, row bytes 32, reserved 0), ASCII dataset SHA256 (64 bytes), ASCII protocol SHA256 (64 bytes). Header totals 160 bytes. Each 32-byte row contains four uint32 values (original row index, original episode ID, source sequence, actor eligibility), then four float32 values (MC return, heldout ridge prediction, training-fold return mean, return minus prediction). Excluded rows receive a heldout prediction for order preservation and remain excluded from metrics/fitting.

No consuming PPO implementation is included. The actor's recorded actions, behavior log-probabilities, old values, support and recurrent history remain untouched.
