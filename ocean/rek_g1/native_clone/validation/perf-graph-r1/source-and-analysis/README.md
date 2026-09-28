# Optional one-control-step CUDA graph

This isolated worker delta adds strict boolean `cuda_graph_step`, default false. The ordinary worker remains preserved as `eval_worker.original.cpp`; `worker.diff` shows the scoped change. The controller, physics, commands, rendering, and source worktree are unchanged.

The constructor binds persistent device action/override/direct-command buffers before reset. Graph mode then records only `rek_native5_step`. Recording executes no warm-up step. Each real step copies current commands and runs any high-level policies before graph launch. Host tick increments, status checks, snapshots, terminal collection, and command-result collection remain outside capture. Explicit reset discards and rebuilds the capture. No automatic fallback or tolerance relaxation is implemented.

Capture support already exists in `native5/runtime_api.h`; `mujoco_gpu/native_step.cpp` inserts the physics schedule directly into an active capture to avoid unsupported nested conditional graphs, and `conditional_if.cpp` inserts the masked reset branch. Existing `physics_probe.cu` and `test_sonic_controller.cu` exercise capture. These source facts do not replace the new actual runtime equivalence test.

CPU mock tests cover capture without simulation advancement, changing persistent input contents, recapture, unavailable launch, enqueue/capture/instantiate/launch errors, and cleanup. Static worker tests confirm command copies/policies and reporting code remain unchanged. These tests make no GPU throughput claim.

Root runs `compare_workers.py TASK_ROOT` after compiling the worker/header into `build-r4/rek-native-clone`. It uses `run-r2` configuration/environment, produces a fresh `perf-graph-comparison-r1` directory, and runs two fresh processes sequentially. A fixed command sequence covers neutral, both signs of all axes, combined magnitudes, attack/cancel edges, cold resets, side changes after reset, and short-round terminals. Every functional per-step reply is compared exactly, including qpos/qvel/observations/masks/outcomes/events. A separate 512-step batch measures whole-worker wall time without frames, and its final state and all emitted events are compared. Intermediate benchmark poses are not returned.

Graph mode must remain disabled until that actual comparison passes. Any mismatch is preserved and fails the comparison. Match lifecycle changes require another comparison of the same graph feature with the new runtime.

The actual first comparison failed:356 of365 replies had trajectory differences, despite identical discrete outcomes/events. The512-step benchmark improved1.1138x. Graph remains disabled. `closed-analysis-r1` corrects the executed driver's misleading scope sentence while preserving its original failure output and trace. The future driver now uses conditional pass/fail wording.
