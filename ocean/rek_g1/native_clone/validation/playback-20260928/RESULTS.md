# Playback timing and deployment evidence

Separating rendering from simulation improved the measured private viewer pace to **0.772844x** real time. The prior human-play observation was **0.550308x**. These workloads used different commands and presentation cadence, so this comparison is descriptive. It does not establish an isolated causal speedup, real-time execution or official-game parity.

| Measurement | Numerator and denominator | Result |
| --- | --- | --- |
| Prior viewer, aggregate active timing only | 6,516 intervals × 0.02 s / 236.812719 s | 0.550308x |
| Private app with separate 20 Hz renderer | 775 intervals × 0.02 s / 20.055788155 s | 0.772844x |
| Unrestricted candidate worker | 150 control ticks / measured worker loop wall time | 44.882153 ticks/s |
| Prior worker, two separate processes | 150 ticks each | 44.191157 and 44.941384 ticks/s |
| Candidate restricted to fast CPU cores | 150 ticks / 3.325824287 s | 45.101601 ticks/s |

The app completed 776 steps over a 20.093148518 s measurement envelope; its pace statistic uses the 775 intervals between steps. One control tick advances 20 ms with ten 2 ms physics substeps. These are control-step rates, not new training throughput or measured official-server physics ticks. The 0.489% affinity gain did not justify restricting the deployed processes.

The workload retained other GPU activity, including the idle official client. Each condition was a short bounded trial without a confidence interval. The candidate completed all five cache checks: exact immediate snapshot, one attack edge through a three-step request, rejected commands without advancing, exact reset snapshot, and rendering without advancing simulation. Old-versus-old and old-versus-candidate discrete fields matched, but floating-point trajectories differed from tick 1 in both comparisons. No numerical trajectory parity pass is claimed. Aggregate qpos errors mix positions and angles and are not distances in metres.

The [closed profile](profile/REPORT.md) measured 256 ticks using a separate instrumented executable. Its CUDA timeline averaged 20.856 ms for runtime and 1.200 ms for subsequent validation; snapshot and feedback took 0.354 ms on the host. Timelines overlap and include possible host enqueue gaps. The 19.436 ms host status scope mainly waits for outstanding runtime work. Removing safety checks is not supported by these measurements. The profiled 42.372 ticks/s is separate from the uninstrumented rate above.

## Deployed revision

The [prepared identity](deployment/VIEWER-PREPARED.json) and [initial local verification](deployment/VIEWER-LOCAL-VERIFIED.json) bind the same app revision 4 and candidate binary to fresh run-r5 at `http://127.0.0.1:18772/`. The latter receipt was recorded at 2026-09-28 21:08:01 UTC: paused, tick 0, no renderer failure, with a decoded 1280 × 720 PNG. CPU affinity was unrestricted. Port 18771 and its historical run were preserved. These two closed receipts contain no subsequent gameplay and make no claim about later runtime health or human-play pace.

App revision 4 keeps one render job in flight plus one replaceable latest snapshot. Rendering runs in a separate process and consumes qpos only. Image tick, reset generation, hash and age remain explicit. Simulation dt, physics substeps and CUDA-graph-disabled settings are unchanged. The app's [39 CPU tests](../../app/validation/tests-r4-final.txt), the launcher's [7 mock tests](../../tools/playback/launcher-tests-r2.txt), and the native renderer parser's [97 checks](candidate/CPU-TESTS.json) cover their respective contracts.

## Files and reproducibility

- [RESULTS.json](RESULTS.json) is the unchanged aggregate source report. The three `closed/*/summary.json` files preserve native, affinity and private-app results. Raw human timing and guard logs are excluded.
- `tools/` contains the byte-preserved benchmark, CPU tests, affinity helper and original command/staging receipts. `python -B -m unittest -v test_benchmark.py` in that folder passed [11 tests](CPU-TESTS.txt). Execution defaults to plan-only. The historical live-process IDs, paths and guards must be reviewed and rebound for a future experiment; the archived commands are not a current launch instruction.
- [The candidate archive](../../artifacts/rek-native-clone-playback-candidate-r1.tar.gz) is 2,779,759 bytes, SHA256 `a781889f84609bd3f2007e3d2d0ec0d2c8dbe62ab465fbdd4ad7083947a0e69a`. It contains 69 files including its manifest. All 68 listed payload hashes passed verification; its 51 source files match the repository. The ARM64/SM121 executable SHA256 is `ef519db4c8b3b3a6696ebc8dfd7b686bffc789a7b91f1ef8d60dfab3494e0975`. It uses the existing inventoried Spark dependency closure.
- [COPY-VERIFICATION.json](COPY-VERIFICATION.json) pins this selection to the 462-file closed source package. This compact publication omits bulk synthetic trajectories and PNGs. The candidate archive's original manifest says runtime testing was pending at its compilation stage; the later benchmark and deployment receipts above document subsequent work without rewriting that historical record.

The root historical `PAYLOAD-MANIFEST.json` and frozen release-r1 archive remain unchanged. They describe their own earlier publication, not the current playback source closure.
