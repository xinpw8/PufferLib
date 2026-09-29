# Native viewer slowdown audit

Revision 9 with whole-control CUDA graphs sustained 0.999887 times real time in a four-minute complete-app test, with zero discarded scheduling time. It is approved for an experimental interactive preview. Numerical equivalence and full Unity parity remain unverified.

The human run exposed a real throughput failure. Across 11,449 recorded control steps, the app reported 0.88463 times real time and discarded 29.860 seconds of accumulated scheduling debt. Two stretches had 40 to 47 ms native request/reply latency against a 20 ms control interval. The earlier 60-second synthetic test did not establish reliable real-time performance under human play.

The fixed 152,246,196-byte trace prefix has SHA256 `2b3f8425754320452a764abd3e5c8eaa305661d78bfa8cc2d648731eca440cc7`. It contains 11,449 step replies, 5,209 recorded frames and zero worker-error events. Raw commands, states and images remain on the private physical-server evidence store.

| Measurement | Result |
|---|---:|
| Step request/reply median | 19.580 ms |
| Step request/reply 99th percentile | 50.239 ms |
| Renderer request/reply median | 19.224 ms |
| Renderer request/reply 99th percentile | 24.868 ms |
| Reply to next step request, mean | 0.729 ms |
| Reply to next step request, maximum | 10.160 ms |
| Paused old-viewer tunnel traffic, observed sample | 7.68 MB/s |

Request/reply latency includes native work, transport and host wakeup/parsing. These measurements alone do not prove GPU compute time, solver iterations, rendering contention or recording I/O caused the long steps. Step and renderer costs overlap and must not be added.

The first isolated 1,500-command replay sustained 51.941 steps/s without a renderer, but it was not an exact physical replay: GPU arithmetic already diverged at tick 1 and later trajectories differed substantially. This is not evidence that removing rendering resolves the slowdown.

## Viewer changes, app revision 9

- Matching frame validators return HTTP 304 with zero image payload. Hidden pages stop frame requests and abort pending image requests.
- A recent five-second measurement uses actual completed control intervals and is displayed separately from the session average. Pauses are excluded, resets clear the window, and actual coverage is explicit.
- Bounded timings separate JSON parsing, native request/reply, recording serialization/write, frame decode/hash and frame-write elapsed time. Event-loop delay and memory statistics are retained. Aggregates are cached for 250 ms with their age exposed.

The changes preserve the native binary, model weights, 20 ms control interval, ten 2 ms physics substeps, input commands, recorded states and recorded rendered frames. They do not establish that the native slowdown is fixed. At this audit checkpoint, app revision 9 is frozen and staged on Spark; existing human viewers retain their previous processes and sessions.

The CPU app suite passed 65 tests, skipped one unchanged genuine-model preparation test and failed none on both Windows and Spark. HTTP tests cover unchanged/new/reset frame identities; browser-script tests cover hidden and aborted image requests while preserving the state heartbeat; pace tests cover fast/slow completions and pauses. No live Windows input was sent.

App source manifest: `9138485e02c0c6ec69a8f84aad87f3469b046170e9db720433b2508a674481a2`.

## Native execution comparison

The existing binary can capture the entire control step in a CUDA graph. Its eager path already captures individual physics substeps; whole-control capture additionally reduces host launch overhead. This uses the same motor weights, solver settings, ten 2 ms substeps, commands and referee functions. It does not reduce physical fidelity settings to reach the target speed.

| Isolated workload | Control steps/s |
|---|---:|
| 1,500 human commands, eager, no renderer | 51.941 |
| Same commands, eager, renderer polled at 20 Hz | 50.033 |
| Same commands, whole-control graph, renderer polled at 20 Hz | 55.104 |

These are separate trajectories, because repeated GPU physics executions are numerically nonidentical. The eager rendered case has almost no throughput margin above the required 50 steps/s. Graph execution adds margin, but the original 40 to 47 ms episodes were not reproduced. Their precise cause remains unknown. The original trace lacks solver-iteration and contemporaneous GPU-resource measurements, so blaming contact complexity, recording or thermal throttling would exceed the evidence.

## Four-minute complete-app trial

The frozen revision-9 app and the original `ef519db4` native binary ran on Spark with whole-control graphs enabled. The test used automatic 20 ms stepping, varied inputs at approximately 50 Hz, state reads at 10 Hz, conditional image requests at 20 Hz, native state recording and PNG recording. All three pre-existing human viewers were guarded by exact process identity and paused state; only the fresh private test app received input. The official REK process remained running.

| Measurement | Result |
|---|---:|
| Actual completed control steps | 11,999 |
| Outer elapsed time | 240.007009 s |
| Simulated time / outer elapsed time | 0.999887465 |
| Active completion-interval pace | 0.999998196 |
| Pace after first 5 s | 0.999996718 |
| Scheduler rebases / discarded wall time | 0 / 0 ms |
| Native request/reply mean / 99th percentile / maximum | 17.025 / 23.240 / 29.298 ms |
| Input packets | 11,999, with one sender deadline miss |
| Image polls / recorded frames | 4,800 / 4,844 |
| Worker, renderer, recorder or guard failures | 0 |

All 24 approximately ten-second active windows were between 0.99989285 and 1.00006601 times real time. Independent outer-wall windows ranged from 0.99799765 to 1.00025699; none fell below 0.99. Sampled server-side frame age was 50 ms at the 99th percentile and 53 ms maximum. That age excludes Windows tunnel transfer and browser display latency. Image requests returned 4,673 full PNG responses and 127 zero-body 304 responses. Combat included four opponent fall-counter increments across rounds, one completed round won and an ongoing second round; no completed match or in-play reset occurred. Recovery behavior was not separately verified.

The independent closed-capture analysis verified every one of the 4,844 PNG files against its recorded byte count and SHA-256, the exact frame-file set, all 11,999 step replies, 12,000 full native states and 2,400 polled states. Both owned native workers exited cleanly. The analysis receipt is `713d6ea3ba5e9f978e1c824825bc0212e7d723b07fbfd4bb6b3504a10e9bae14`.

The renderer's initial 299.7 ms startup occurred before play. JSON serialization averaged 0.0206 ms per log event, synchronous log writes 0.0143 ms, simulation JSON parsing 0.0629 ms, and asynchronous frame-write elapsed time 0.7825 ms. Node event-loop utilization was 10.14%. These spans overlap and cannot be added. This trial provides no evidence that recording dominated runtime cost.

## Numerical and acceptance limits

Reset states, commands, ticks, masks and discrete outputs matched in the short graph/eager gates. Floating-point states did not. Three additional eager repeats reached maximum differences of 0.0001369 in generalized position and 0.0023095 in velocity. One graph process was an outlier after a cold reset and a three-step batch: maximum differences reached 0.0008511 and 0.0246126. The intermediate two states were not saved, so the precise cause is unknown. Other graph repeats overlapped the observed eager scatter. No hard graph implementation bug was demonstrated, but no numerical-equivalence tolerance passed either.

The independently reviewed decision is to allow an experimental interactive performance viewer, preserving the old process/configuration and the numerical evidence. This does not approve a parity-validated training backend. The four-minute workload establishes real-time behavior for that workload; it does not establish universal performance under human play, identical trajectories, or full official Unity gameplay parity.

## Recording and resource diagnostics

The staged mirror revision refreshes recorder health and active logs between bounded frame batches, and checkpoints its inventory in batches instead of rewriting it after every image. Twenty-three CPU/fake-SFTP tests cover copying, corruption rejection, interruption, resume and checkpoint failures. In a 512-record fixture, local inventory writes fell from 512 to 16. This is a serialization result, not a measured NAS throughput gain.

A separate passive resource watcher records process CPU, resident memory and I/O once per second, plus bounded GPU utilization, clocks, power, temperature and per-process activity samples every five seconds. Eleven fake-only tests cover process identity, missing data, timing, STOP and query timeouts. It cannot send input or signal a viewer. Missing counters and sample age remain explicit.

## Deployed experimental viewer

The fresh viewer is available at `http://127.0.0.1:18774/`. At the 02:57:05 UTC handoff verification on 29 September 2026, it was healthy and paused at tick 0. The 1280 by 720 startup frame passed SHA-256 verification, and a repeated conditional request returned HTTP 304. Node PID 3702006/start 133846865 owns the new run; the exact prior viewer identities remained preserved. The deployment uses the frozen revision-9 app, original native binary, verified batch-2 models, graph mode enabled and the same saved controls. No old viewer was restarted or changed.

The new tunnel, batched mirror and passive resource watcher were verified running. The NAS resource sample was 1.75 seconds old with GPU counters available and no query errors. Source recordings are in Spark `run-r8`; the physical-server mirror is `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-29\rek-performance-audit-r1\viewer-r8`. Older browser tabs continue to run their older builds.

Machine-readable evidence is in [public-validation.json](public-validation.json) and [viewer-startup.json](viewer-startup.json). The exact guarded deployment sources and seven passing CPU checks are preserved under [deployment-source-r1](deployment-source-r1/SOURCE.json). The complete raw synthetic trial and original human evidence remain private on Spark and the physical evidence store; backup completion is recorded separately from source-integrity verification.
