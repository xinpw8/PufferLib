# Native viewer slowdown audit

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

## Remaining acceptance work

Native profiling and controlled rendering/graph comparisons must identify the costly spans. A replacement must sustain the actual 50 Hz control rate through representative combat, falls, recovery and round boundaries with recording and viewing enabled. Report recent windows, tail latency and frame freshness as well as the whole-run average. Keep timestep, solver settings and scoring fixed; preserve failed trials and numerical differences. A short mean-throughput result does not establish full Unity parity or universal real-time behavior.
