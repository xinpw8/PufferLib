# Real-time playback result

The selected one-arena GPU app sustained **0.999985x after the first5s**, with50Hz synthetic controls,20Hz PNG requests and recording active. This is approximately1x steady-state playback. The complete cold-start trial remained below literal1x.

| Measurement | Result |
|---|---:|
| First5s excluded | 2,750 control intervals /55.000839s |
| Steady control rate | 49.999237 steps/s |
| Steady simulation/wall ratio | 0.999984739x |
| Entire trial, actual steps /outer wall | 2,991 /60.007963s |
| Entire-trial simulation/outer-wall ratio | 0.996867692x |
| App active-interval ratio, including startup | 0.997277157x |
| Recorded frames /PNG requests /input packets | 1,207 /1,200 /3,000 |

All2,991 returned states had contiguous ticks. Both owned workers exited normally, with no recorder, renderer, worker or guard failures. The scheduler reported one wall-debt rebase of162.620ms and a maximum task duration of181.915ms. The first10s window was0.9793x; the later10s windows ranged0.9971–1.0043x. Startup and short scheduling variation remain visible in the record.

The prior r6 full-app run was0.955883x. Absolute20ms deadlines improved the unrestricted r7 run to0.993805x overall and0.996341x after5s. The selected run additionally confined its own Node, simulation and renderer processes to CPU cores5–9 and15–19. This measured comparison is one run of each configuration, not a statistical performance guarantee.

The final runtime uses one physical arena, two controller rows, the existing batch2 weights, separate rendering and the r7 scheduler. Control dt remains20ms, physics dt2ms with ten substeps, and solver/motor weights are unchanged. CUDA graph mode is off. Static tensors match the batch8 export; all4,096 sampled tokens matched exactly, while connected decoder outputs differed by at most9.54e-6. That supports bounded numerical compatibility, with the failed exact comparison preserved under `batch2`.

The four-arena phase profile identified physics step/refresh as14.917ms of20.350ms mean runtime. Measurement/referee/reset used2.764ms and preparation1.118ms. Host and CUDA timing intervals overlap and must not be added. The r6 timing analysis also found1,132 of2,867 steady RPCs above20ms: completion-relative scheduling retained those delays instead of recovering during shorter steps. The included SCHEDULER.json states the limits of this causal timing model.

Native executable SHA: `ef519db4c8b3b3a6696ebc8dfd7b686bffc789a7b91f1ef8d60dfab3494e0975`.
App source manifest SHA: `6be9fa74673eca108da8c07cd414d27d55b6cc9f625db1ab70e2f5c3e6f02cca`.

The parent acceptance gate of at least0.99x passed. This report does not claim a literal whole-trial result of at least1.000x, full Unity/server physics parity, improved fighting ability or a universal50Hz guarantee. No model weights or raw human-session traces are included. The linked strict-reference alternative remains an unexecuted fallback.
