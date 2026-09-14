# Headless native training profile

`profile_training.sh BUILD NEW_OUTPUT CATALOG [65536]` wraps the existing
native training runner with Nsight Systems node-level CUDA graph tracing.
Set the same `REK_TRAINING_BACKEND`, opponent, and native MuJoCo kernel/PTX
variables as the unprofiled run. `REK_TRAINING_RUNNER` can select the exact
staged runner. No Python interpreter, Nsight Python reports, or `nsys stats`
are used. Export is the native `nsys export --type=sqlite`; analysis uses the
native SQLite CLI and Node builtins.

The training batch remains512 agents with horizon16. The default64k profile
contains eight PPO epochs. It excludes the first complete epoch and capture
setup from steady-state measurements, then counts actual learner sampling
kernels to validate the transition denominator. Incomplete timestamp groups
or an unexpected action count produce warnings or unavailable training SPS.

## Attribution

- Physics: exact cached CUDA kernel symbols and native physics-wrapper names.
- SONIC robot controller: same-stream GPU window from `pack_encoder` through
  `apply_actions(RobotState,...)`, including all intervening cuBLAS GEMMs;
  controller and servo functions are also recognized by their source signatures.
- PPO: the three actual `puf_stamp` GPU kernels bracket each training epoch;
  training-only kernels identify auxiliary gradient streams.
- Combat, motion scheduling, observations, and contact measurement: native
  environment function signatures.
- Shared GEMMs without a verified phase boundary remain unclassified.

Category percentages divide by summed kernel duration. GPU kernel-busy time
uses the interval union and is reported separately. Concurrent kernels can
overlap; summed duration is not wall time. CUDA API duration is reported
separately because a synchronizing API can be waiting for GPU execution rather
than doing CPU computation.

`profiledSteadyStateTrainingSps` includes environment, policy inference, PPO,
and between-epoch gaps in the measured interval. Node tracing adds overhead;
the unprofiled run remains the throughput result. Device kernel timings do not
establish gameplay parity or policy strength. Raw Nsight traces and model
artifacts stay private. Aggregate JSON, commands, and hashes can be shared.
