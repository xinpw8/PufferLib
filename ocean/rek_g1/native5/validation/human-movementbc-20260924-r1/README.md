# Human movement BC publication

Completed five-epoch negative experiment, not promoted. Read [RESULTS.md](RESULTS.md). No training, GPU job, game interaction or process control was performed during publication preparation.

Source: closed Spark stage `/home/spark-advantage/rek-training/human-movementbc-20260924-r1`. Verified archive: `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\human-movementbc-native-20260924-r1\evidence.tar.gz`, 15,280,641 bytes, SHA-256 `a31d14e6fcae2e20d766d7fbe6ffb0f9dba6116f027028a8fd804b54500c6b82`. Only explicitly selected source, manifests, metrics and receipts were extracted. See `receipts/source-identities.json` and the original archive receipt.

## Contents and reproduction

- `prepare_dataset.cjs` and tests: original movement labels 2–15, unit weights; all other rows retained with zero loss. Exact 223-float partial observations, mask and chronology preserved from pinned sources. `dataset/manifest.json` records counts and hashes, without dataset bytes.
- `run_bc.cjs`: original fixed-five-epoch runner, LR 1e-4, horizon 128, unchanged native Muon trainer. It pins trainer `febdf12d...`, initial checkpoint `5b19d628...` and dataset `b4f8261d...`; existing output is refused. `--check` does not initialize CUDA but requires the private pinned artifacts. Do not rerun `--run` against the archived completed stage.
- `training/`: original command/execution receipts and derived metrics. Native `heldout` names are retained numerically and explicitly interpreted as already examined development data. `summarize_metrics.cjs` regenerates the compact metrics from the SHA-pinned native stdout retained in the private archive.
- `compare_live_drift.cjs` and tests, `live-drift/`: original offline comparison and native execution/report receipts. The publication copy changes only its import to the existing repository neighbor `../human-attackbc-20260924-r1/compare_live_drift.cjs`. That helper is byte-identical to the archived dependency, SHA-256 `f2eb2f490515eb1353e51850dbadc8815f7be1559ca3384c97b803a9ac1ab07d`. Its native diagnostic source and pinned build script are in that neighboring directory. Native reruns require the archived private stream/checkpoints and unchanged pinned executable; they are not needed for CPU tests.
- `test_export.cpp`: original CPU dataset-reader check. It uses the existing `../../bc_dataset.h`; compile with `c++ -std=c++17 -I../.. test_export.cpp -o <new-private-output>` from this directory, then supply the private pinned dataset. No native reference assembly is required.

Run source/dependency checks without private data or native execution:

```sh
node --test prepare_dataset.test.cjs compare_live_drift.test.cjs publication_repro.test.cjs ../human-attackbc-20260924-r1/compare_live_drift.test.cjs
```

Publication verification passed all 10 Node tests and a C++ syntax/dependency compile check. The copied execution, drift and dataset-manifest hashes match the archive receipt. See `receipts/publication-checks.json`.

The original archived files are unchanged. The dataset and CPU manifests describe the earlier preparation phase; their `training_performed:false` fields are historical, superseded by `training/execution.json`. The separate movement-25 preparation was never executed and is not published as a result. Repository production files, including the existing dirty native sources, were not edited. No datasets, checkpoint binaries, private observations or raw runtime logs are included.
