# Native physics backend experiments, 2026-09-14

## Explicit backend selection

The default remains CUDA Puffysics with unchanged solver warm starting.
`REK_PUFFYSICS_STABILIZATION=joint_cold_start` selects a parity-relaxed
Puffysics experiment. Before every 2 ms physics step, all joint cached impulses
are set to zero. Current constraints are solved by the existing Puffysics
solver. Contact warm starting, force laws, controller, geometry, timestep,
solver iteration counts, scoring, and failure detection are unchanged.

This deliberately changes the integrator's approximation. It prevents cached
joint impulses from accumulating across steps, removing the immediate failure
mechanism in the recorded baseline. It does not establish accurate articulated
motion, long-run stability, or REK parity. No body states or failure flags are
replaced, cleared, or ignored. An unrecognized profile is rejected.

`REK_PHYSICS_BACKEND=mujoco_cpu_eval` plus `REK_ALLOW_CPU_EVALUATION=1` selects
an explicitly CPU-only native MuJoCo evaluation backend, restricted to at most
eight viewer arenas. It calls the installed MuJoCo 3.7.0 C API's `mj_step` and
refreshes derived fields with `mj_forward`. Original XML collision geometry is
used, including the cylinders replaced by capsules in the Puffysics diagnostic.
Controller inference, combat scheduling, observation construction and hit logic
continue through the existing CUDA modules. There is no Python interpreter.

CPU evaluation requires explicit opt-in, prints `cpu_physics=1` and
`training_backend=0`, and rejects CUDA graph capture. The native trainer must
also reject this selector at its entry point, including when graphs are
disabled. This CPU viewer is not the previously benchmarked MuJoCo Warp GPU
training path. Its timing cannot be presented as GPU training SPS.

`REK_PHYSICS_BACKEND=puffysics_cpu_eval` with the same explicit CPU-evaluation
opt-in selects the actual checked-in Puffysics engine compiled for the host.
It uses the same `b3_step`, torque law, configured cold-start setting, contact
export, and masked reset helpers. There is no animation surrogate, teleport
controller, or alternate collision model. This viewer mode keeps its transfer
buffers pinned and persistent, while controller and combat kernels remain on
CUDA. CPU and GPU floating-point execution need not be bitwise identical.

The final four-arena CPU Puffysics probe advanced 200 decision ticks in
1.247543 s, equivalent to **160.32 Hz per arena**. Thus the measured simulation
work fits a 50 Hz interactive budget, before rendering and HTTP overhead.
These are viewer/control-loop timings with zero PPO updates, not training SPS.
Large-batch headless training remains CUDA, not this CPU-only viewer path.

At the initial investigation, the installed MuJoCo Warp implementation had
cached generated CUDA source and
PTX under `/home/spark-advantage/.cache/warp/1.12.0`, but its existing model/data
initialization and launch scheduling are Python code. No reusable native
launch/data manifest was found in the targeted existing REK experiment search.
The cached kernels alone do not provide a native equivalent of `mujoco_warp.step`.
Native GPU rehosting required implementation and validation of the model/data
ABI and ordered launch schedule. That work is now complete for the pinned REK
model; see the native CUDA section below. The earlier CPU viewer results remain
historical evidence and are not the selected GPU training path.

## Executed tests

All tests ran on `spark-4ae3` through Windows PowerShell, WSL, and `ssh spark`.
Existing GPU applications were preserved. The pre-existing evaluator was not
modified or stopped. Model weights and checkpoints remain private on Spark.

| Test | Executed result |
| --- | --- |
| Native PPO, 512 arenas, cold joint cache | 163,840 transitions; 20 PPO epochs; 320 ticks / 6.4 simulated seconds per arena; exit 0; status checks and printed losses finite |
| Native-controller long probe, 8 arenas, cold joint cache | Valid through 1,300 ticks / 26 simulated seconds per arena; 120 s wall timeout, exit 124; no completed episodes; no failure reported at checked boundaries |
| Native MuJoCo CPU viewer smoke, 4 arenas | 10 ticks / 0.2 simulated seconds per arena; 0.253698 s wall; exit 0; zero PPO updates |
| Native Puffysics CPU viewer, 4 arenas, final helper build | 200 ticks / 4 simulated seconds per arena; 1.247543 s wall; exit 0; zero PPO updates |
| Native CUDA physics regression, final helper build | 41 physics steps, isolated masked reset, graph capture, nonfinite rejection, persistent failure flags; exit 0 |

The native PPO test took 34.605 s native uptime and finished around 4.8k
training SPS. This is a shared-GPU stability experiment, not a maximum-SPS
result. The old baseline failed by its fifth rollout, around tick 80. The new
result extends beyond that failure but does not demonstrate completed fights
or reliable learned behavior. Completed-episode counts are zero in all reported
rollout tests, so there is no win-rate result.

The first host/device-helper build passed its CPU viewer probe but failed the
GPU regression with an invalid local-memory read in the masked-forward kernel.
Forcing the shared large helpers inline fixed that regression; the final build
passed the complete existing GPU adapter test above. Both failed and passing
test outputs are retained. Training equations and cold-start configuration were
not changed by this execution-layout fix.

The long probe's episode, win, loss, draw, score, and fall counters describe
completed episodes only. Their zero values do not imply no contact or falls
occurred during its unfinished rounds. The native runtime's older hardcoded
Puffysics banner appears in the CPU smoke log because the probe linked the
previous runtime object; the adjacent explicit physics backend banner identifies
the actual executed CPU path.

Sanitized commands, host identity, model hashes, stdout/stderr, metrics, and exit
codes are in `validation/physics-stability-20260914/`. No checkpoints are included.

## Reproduction

Build the ordinary native executable with `build_native.sh`, then use:

```sh
REK_PUFFYSICS_STABILIZATION=joint_cold_start bash run_spark_probe.sh \
  /absolute/native/build /absolute/new/results 163840
```

`physics_stability_probe.cu` is a separate executable linked against the same
native runtime objects, excluding `pufferl.o`. It chooses legal deterministic
actions on the GPU, captures a full decision tick, and checks statuses every
100 ticks. It performs no PPO updates. Its positional interface is:

```text
physics_stability_probe ARENAS TICKS XML EXPORT ASSETS FEATURES ENCODER DECODER
```

For the CPU viewer smoke, use the explicit CPU selectors above. The probe
disables graph capture for that named backend. The tested batch-eight models
were in `codexrook-runtime/generated/gear-sonic-g1batch8-mode0-20260909T0203Z`;
the eight-arena GPU probe used
`rek-training/gpu-runtime-20260910/controller-batch16-aa4b4d5c-20260910`.

The test binaries and original evidence remain under
`/home/spark-advantage/rek-training/native5-stability-20260914/`.
The native trainer binary SHA-256 was
`69301ba0eb2cf12cec8351e2dd039e17375c53e5ceceaf485e782cf58a636e6a`;
the controller-only probe was
`9d739bf3d9cf49caaeb5cdf141e610c789f0863b763f8fc4644788e5f8717952`.

## Full rounds and articulated candidate: subsequent results

The 20-second native evaluator checks completed with explicitly CPU-only
physics and CUDA controller/combat, without PPO updates:

| Physics | Final tick | Score | Falls | Process wall time | Result |
| --- | ---: | --- | --- | ---: | --- |
| MuJoCo CPU | 1,000 | 2-0 | 0-0 | 17.63 s | Completed, finite |
| Puffysics CPU, independent contacts, joint cold start | 1,009 | 20-20 | 4-4 | 9.00 s | Completed, finite, repeated falls |

Both used a scripted player against the existing internal candidate opponent.
The Puffysics points are four five-point falls per fighter, not twenty landed
attacks. These are evaluator timings, not training SPS. Raw records are at
`/home/spark-advantage/rek-training/native5-league-20260914/integration-rounds-v2/`.
At tick 512, Puffysics pelvis heights were 0.142/0.164 m and maximum joint
speeds were 293.6/390.9 rad/s. MuJoCo heights were 0.712/0.666 m and joint-speed
maxima were 4.17/7.15 rad/s. A completed finite round does not imply sound
motion dynamics. A subsequent frozen-opponent native training run also failed
around 8.96 simulated seconds in arena 274 despite joint cold start.

The independent-contact path applies the existing affine position PD actuator
as explicit body torque and omits exported rotor armature in free integration.
All 58 hinges have nonzero armature. For example, the left ankle pitch's
isolated free-body-pair axial inertia is 1.89e-5 kg m^2, while its exported
rotor armature is 7.22e-3 kg m^2. This is a local stiffness diagnostic, not the
coupled robot's full generalized inertia or a proof of the sole failure cause.
The eight cylinder-to-capsule substitutions affect shoulder shapes, not feet;
source sliding friction is 1.0. There is no observed frictionless-floor setting.

`physics_articulated_candidate.cu` is an isolated opt-in translation unit. It
selects the existing `rp_art_step` mode 1, `B3_ART_CONTACTS=1`, and previously
tested fixed-pose articulated factor cache. Ordinary builds continue to
compile `physics.cu` with mode 0. The candidate reads all 58 existing armatures,
keeps the same torque law, assets, controller, clocks, combat, and failure
checks, and reports `puffysics_cuda_articulated_candidate` or the explicit CPU
viewer equivalent. No new physics engine equations were introduced.

**The articulated candidate failed its bounded test.** In the four-arena CPU
viewer, arena 1 became nonfinite at physics time 2.41000056 s. Its fighter 1
root and all joint states became NaN while fighter 0 remained finite. Status
was `nonfinite=1`, `solver_status=0`, `max_contacts=34`, with zero collision
failure metadata. The last completed arena-0 snapshot at 2.40 s was upright,
had no falls, and joint-speed maxima of 11.5/10.5 rad/s. Armature-aware early
motion improved, but this is not a successful 20-second round or training
backend. No 512-arena training was launched with this candidate.

The first failing operator inside articulated free integration, contact
response, or the predictive joint-limit pass is unknown. The concrete next
diagnostic is stage-level nonfinite capture within `rp_art_step` on this exact
reproducer. It must identify the bad operator and retain its inputs before a
numerical correction is justified. Lowering gains or resetting NaNs would
conceal the problem rather than establish a working backend.

Source, compiler output, dense per-tick observations, and raw failure operands
are retained under
`/home/spark-advantage/rek-training/native5-articulated-20260914-v1/`.
`eval-round-r1` is an invocation-only failure from an omitted `--config` flag;
`eval-round-r2` is the first executed failed round; `eval-first-failure-r3`
narrows the failure to decision tick 121; `eval-round-r4-diag` contains the
exact arena/time/status and qpos/qvel failure dump. The diagnostic evaluator
SHA-256 is
`a62b7d2bd79b41b51dc2cc0d4fac2cac03c5303a5616919c01c7d2f05ed27f39`.
The bounded command harness is `physics_articulated_eval_probe.sh`; it performs
zero PPO updates and treats protocol errors as a failed round independently
of the worker process exit code.

## Native MuJoCo-Warp CUDA opt-in

`REK_NATIVE5_ENABLE_MUJOCO_GPU=1` builds the additional native backend;
`REK_PHYSICS_BACKEND=mujoco_cuda` selects it. Catalog, conditional PTX, and
PTX hash paths are explicit inputs. Legacy Puffysics and CPU-only viewer
selectors retain their existing paths.

The native backend packs the compiled XML's original model constants and
existing prepared actuator parameters into persistent GPU arrays. Its step
and Newton convergence loop execute original cached MuJoCo-Warp CUDA kernels
through the CUDA Driver API and GPU conditional graphs. Python, Torch,
`libwarp`, and CPU dynamics are absent from this execution path. XML parsing
and metadata preparation remain native host setup work. Unlike the legacy
Puffysics frame-map path, native MuJoCo startup does not call CPU kinematics.

The `PhysicsDescriptor` aliases those GPU arrays, including qpos, qvel,
body/geometry transforms, controls, and the pooled contacts. Per-step state
transfers to the host are unnecessary. CUDA kernels maintain persistent
nonfinite and capacity flags. Selected resets clear solver warmstarts on
the GPU and regenerate current contacts without advancing time. A GPU
mask-any conditional skips that reset work for empty masks.

The four-arena adapter probe passed ten 0.002 s substeps and selected-reset
isolation with zero linker-wrapped CPU step/forward/kinematics calls. See
[`mujoco_gpu/validation/adapter-20260914`](mujoco_gpu/validation/adapter-20260914).
Its 0.020 s run is a routing and reset check, not a training throughput or
gameplay-parity result. The independent 1,000-step native CUDA diagnostic
exercised active contacts and remained finite. The subsequent native trainer
completed 1,048,576 transitions and 946 rounds at 6,020 mean training SPS,
with zero reported runtime failures. See
[complete training evidence](mujoco_gpu/validation/training-20260914/README.md).
This establishes an executed native GPU training path, without establishing
maximum throughput, final-policy strength or gameplay parity.
