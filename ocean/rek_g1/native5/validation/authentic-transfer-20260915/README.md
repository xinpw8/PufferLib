# Authentic REK transfer trial

Trial r4 ran the frozen V4 mixed checkpoint in the authentic REK Steam client
on `spark-4ae3`, under the isolated Wine 11.13/X11 `:98` runtime. It completed
the native 120-second round against private Sparring Bot 1.

- Native result: `WonByPoints`, winner slot 1 (Bot 1). Policy was slot 0.
- Replicated clean hits: policy 15, Bot 1 32. Replicated falls: 0:0.
- 3,686 CUDA decisions; 81 confirmed native `SendMoveEvent` returns.
- 3,689 telemetry records in 3,689 distinct rendered frames.
- Zero stale-action rejections or watchdog stops in this trial.
- Live decision frequency 30.30 Hz; median worker latency 0.287 ms, p95 0.435 ms.
- Process exit 0; the stream ended at the observed inactive round boundary.

This verifies actual client control and observed combat. It does not establish
winning strength, server action acceptance for every request, or simulator
parity. Requested move timing and the pose-to-policy feature mapping are
explicit projections. Candidate-environment win rates are not live REK results.

`measured-report.json` preserves the raw counter result, categorical action
counts, rejection reasons, latency and hashes of the complete private traces.
`executables.sha256` pins the adapter binaries used for this run. The checkpoint
SHA is `0d612521839298ffbe5783ed4fa449286a940b62a3a6768da7cc5ab248eb843b`.

Private full logs remain at:

```
/home/spark-advantage/codexrook-runtime/live-transfer-20260915/trial-r4/
```

Execution commands on Spark:

```
node /home/spark-advantage/codexrook-runtime/live-transfer-20260915/live_transfer_run.cjs /home/spark-advantage/codexrook-runtime/live-transfer-20260915/trial-r4.config.json
node /home/spark-advantage/codexrook-runtime/live-transfer-20260915/summarize_live_transfer.cjs /home/spark-advantage/codexrook-runtime/live-transfer-20260915/trial-r4
```

Spark shell access was through Windows PowerShell, WSL and `ssh spark`.
No Windows keyboard, mouse, focus or gamepad injection was used. No proprietary
game binaries, model assets, authentication material or raw account state is
included here.
