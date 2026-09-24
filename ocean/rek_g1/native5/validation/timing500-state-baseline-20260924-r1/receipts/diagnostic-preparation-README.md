# Cross-fitted state-return baseline diagnostic

Private staged implementation. Native CPU checks and exact input validation passed. The CUDA executable compiled. CUDA reference tests, fitting and heldout metrics have not run. No policy/trainer/replay/controller files were changed.

The frozen experiment is in `PROTOCOL.md`; build pins and executed checks are in `BUILD-RECEIPT.json`. `state_baseline.cu` contains the native diagnostic and focused tests. `build.sh` compiles and executes only a CPU-only binary. Its second binary is compiled for CUDA but is not executed by the build script.

The exact pinned reader is copied to Spark `vendor/`. It independently validates finite observations, V5 score-delta rewards, five-second discount, action legality, actor eligibility, transition linkage and complete episodes. A separate full source review checked fold isolation and mean-scaled ridge normalization. Native CUDA synthetic checks remain mandatory before actual fitting and are automatically invoked by `--run`.

## Optional run after separate authorization

Run from a scheduler slot explicitly authorized for this diagnostic. This command has not been executed:

```sh
/home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1/build/state-baseline --run \
  /home/spark-advantage/rek-training/timing500-ppo-refresh-20260924-r1/score-delta-5s/authentic-score-delta-v5.bin \
  eb6b1ae210b3b3911597f93a513ae32081f15e4871b29c1cb29e75829f718655 \
  /home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1/PROTOCOL.md \
  32ee5bb1dd2cdb427cfeff7ae01f663a350858b3d35d80655d86987f64293dd9 \
  /home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1/run-r1
```

The new output directory must not exist. A failure may leave partial diagnostic outputs there; preserve them and choose a new explicit attempt directory if repeating an unchanged experiment. Report failed numerical checks instead of silently weakening tolerances or tuning the model from heldout results.

Expected outputs: `run-provenance.json`, five `fold-N.json` files with training scalers/weights and heldout metrics, `report.json`, and 478784-byte `row-baselines.bin`. The latter contains all original rows, including the five flagged actor-excluded terminal rows. Predictions for those rows do not enter fit or metrics.

Use `report.json` to compare each fold and pooled MSE, residual mean and residual variance against zero and training-fold mean. A lower centered variance with worse MSE is not evidence of a better zero-centered advantage. Five development episodes do not establish robust transfer or better policy learning. This stage deliberately contains no PPO baseline integration.
