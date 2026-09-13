# Actual PPO training profile on DGX Spark

Measured on `spark-4ae3`, NVIDIA GB10, 2026-09-13 UTC. The paired training
comparison is **516 learner SPS for the experimental REK Puffysics adapter
versus 4,097 for CUDA MuJoCo**, a **7.93x difference**. Both ran native PPO
updates, produced finite losses, and changed the policy weights. The longer
Puffysics run failed its physics validity checks. This is the locally modified
REK prototype, not a benchmark of unmodified public Puffysics.

After explicitly relaxing parity, capsule geometry plus independent contacts
reaches **5,085 learner SPS at 512 arenas**, with further batch scaling below.
At **4,096 arenas it reaches 10,610 SPS** across three actual PPO trials.
The earlier 516 SPS result is specific to the expensive experimental articulated
configuration. Longer-run numerical instability remains unresolved.

The earlier 870 versus 10,062 arena-SPS comparison was a fixed-reference
controller/physics rollout. It contained no PPO training. These new results
come from `profile_training.py`, which runs the full semantic combat task.

## Setup, including the answer about Raylib

Training is headless. Raylib is used for human evaluation/rendering and is
absent from the training loop. There is no Raylib-trained policy being ported
to another engine in this test.

- Two G1 robots per arena: 60 massive bodies, 91 shapes and 58 hinges.
- The fixed Sonic encoder/decoder and motion assets drive the articulated
  robots. The trainable policy chooses one of 33 semantic action categories.
- The existing scheduler, held-input semantics, contact scoring, fall/reset
  logic, observations and deterministic candidate opponent run in both arms.
- Observations use the same `scaled_polar_xy_v1` encoding. Both arms load the
  same checkpoint and use native PufferLib MinGRU, 256 hidden units, two layers,
  PPO, prioritized replay and Muon. The `[torch] network=MLP` entry does not
  select the network in this native learner.
- Learning rate 0.003, gamma 0.999, GAE lambda 0.995, replay ratio 4,
  minibatch 4,096. Opponent samples do not enter PPO.
- Agent/controller tick 20 ms, target filtering 4 ms, physics step 2 ms.
  Model parsing and upload occur before timing. Physics, controllers, learned
  policy inference and optimization run on CUDA. Native CPU env count is zero.

The actual runtime already existed at
`/home/spark-advantage/rek-training/gpu-runtime-20260910`, using
`/home/spark-advantage/.venv/bin/python`: Torch 2.9.1a0+gitd38164a, CUDA 13.0,
MuJoCo 3.7.0, Warp 1.12.0. Checking only system Python or its default PATH
misses this installation. No checkout reset or system installation was needed.

## Training measurements

One SPS unit is **one learner transition**, with both robots physically
simulated. It includes rollout, learned-policy inference and PPO updates.
It does not count optimizer replay samples or multiply by physics substeps.

| Workload | Run 1 | Run 2 | Run 3 | Total transitions / total timed seconds |
| --- | ---: | ---: | ---: | ---: |
| Puffysics, matched 512 arenas, horizon 16 | 518.28 | 515.96 | 514.92 | **516.38 SPS** |
| CUDA MuJoCo, same matched workload | 4,035.98 | 4,111.30 | 4,144.30 | **4,096.69 SPS** |
| Existing optimized MuJoCo config, horizon 256 | 6,114.67 | 6,061.92 | 6,185.16 | **6,120.17 SPS** |
| Relaxed Puffysics, capsules and independent contacts, 512 arenas, horizon 16 | 5,120.96 | 5,078.02 | 5,057.89 | **5,085.49 SPS** |
| Relaxed Puffysics, same configuration with 4,096 arenas | 10,635.71 | 10,582.67 | 10,610.65 | **10,609.63 SPS** |

The matched trials each execute one real warmup update and two timed updates:
8,192 warmup transitions plus 16,384 timed transitions. Environment state
continues across horizons. That is 0.32 s of warmup and 0.64 s of measured
simulation per arena. Setup, status reporting and checkpoint I/O are outside
the primary timer. The JSON also reports their cost separately.

Both matched arms retain fused combat and deferred observation packing.
MuJoCo's conditional reset-forward optimization is disabled in the pair
because it depends on Warp internals. The separate existing-production row
retains it, uses horizon 256, and times 262,144 transitions per trial including
periodic reporting. Its ratio to the short Puffysics trial is therefore not
a measurement of the physics engine alone. None of these rows is a claim
that maximum achievable training throughput has been found.

The larger-batch check used the same configuration with an 8,192-robot
controller export, giving 4,096 arenas. MuJoCo completed at **3,991.69 SPS**.
Puffysics hit three solver failures during its first 16-tick warmup, so it
has no valid training-SPS result at that size. Increasing batch size did not
produce a usable training run in this test.

## Where the time goes in the experimental articulated solver

Separate event-instrumented training trials measured these fractions of the
CUDA stream timeline. The environment category includes the fixed controller,
opponent, physics, combat, observations and metrics.

| Category | Puffysics | MuJoCo |
| --- | ---: | ---: |
| Environment | 99.852% | 98.771% |
| Learned-policy inference and rollout storage | 0.013% | 0.101% |
| PPO update | 0.132% | 1.111% |

Separate Nsight captures traced every CUDA graph node during actual training.
For Puffysics, **97.554% of summed kernel duration is `rp_step_kernel`**:
320 calls consume 31.062 s. The new adapter's kernels collectively consume
**0.955%**. The measured slowdown is concentrated in the physics kernel.

That kernel launches only **16 blocks of 32 threads** for 512 arenas, on a
GPU with 48 multiprocessors. It uses 196 registers and a compiler-reported
186,736-byte stack per thread. Each thread handles both articulated robots
and their contacts. These are concrete investigation targets for the engine
author. The trace does not isolate how much of the kernel's duration is
caused by stack traffic, contact response solves or low parallelism.

For MuJoCo, the largest measured kernels were sparse JTCJ gradient assembly
(24.84%) and blocked Cholesky gradient work (11.82%). Its work is distributed
across many more blocks. Kernel-duration percentages and overlapping host API
durations must not be added together. Node tracing perturbs throughput; the
SPS table uses unprofiled runs.

## Sustained failure in the experimental articulated solver

At the normal 256-tick horizon, Puffysics completed an attempted rollout and
PPO call, then failed validation before entering the measured phase. All
512 arenas had a failure: 486 recorded nonfinite states, 27 recorded collision
solver failures, and one arena had both. Contact capacity did not overflow.
The attempted warmup took 210.234 s. No valid sustained training SPS is
assigned to that run, and its diagnostic policy must not be used for evaluation.

The failure report includes per-arena flags and the first failing collision's
operands. Both collision failure codes 2 and 16 occurred. The short successful
training measurements do not make this backend suitable for a full training run.

## Standard step function and relaxed geometry diagnostics

The follow-up explicitly drops the engine-parity requirement. Mode 0 invokes
`b3_step` instead of the prototype's `rp_art_step`. With the existing cylinder
model, one eight-tick native PPO update completes at 2,426.98, 2,439.41 and
2,426.44 learner SPS in three runs. These are **cold-start diagnostics** spanning
only 0.16 simulated seconds, with finite losses and changed weights. They are
not a sustained training benchmark or a matched ratio against the warmed table.
Nsight attributes 88.931% of kernel time to `rp_step_kernel` in that interval.

A normal 16-tick warmup fails with nine collision failures, no nonfinite states
and no contact-capacity overflow. Six have status 16 (degenerate), three have
status 2 (EPA seed failure). The first recorded box/cylinder query reproduces
exactly on both CPU and CUDA: status 16, zero contacts, six GJK iterations and
zero EPA iterations. `reproduce_standard_collision.cu` requires only the engine
headers and that single pair's numerical operands, with no game model or policy.

Crucial attribution: finite-cylinder collision and `b3_convex.cuh` were added
locally. Public Puffysics commit `4f6653cc52da92c3bb4972c6f00b6c733f5c2dc9`
has no `B3_CYLINDER` path. This failure is not evidence that fbr's unmodified
public collision code is broken. Also, `b3_step` defaults to articulated contact
response; selecting mode 0 alone does not remove that computation.

`make_capsule_diagnostic.py` replaces cylinders with enclosing capsules, retaining
radius and centre-segment half-length. This adds hemispherical ends, keeps mass
and inertia unchanged, and deliberately changes collision geometry. The converter
records every changed geometry ID and the original export hash. That removes
the local GJK/EPA failures, but the 256-tick warmup with articulated contacts
still produces 235 nonfinite arenas. Zero collision failures and zero capacity
overflows were reported. Its 38.416 s attempted warmup is not valid training SPS.

These interventions separate the local cylinder extension from the articulated
contact path. They do not support attributing all previous failures or the full
7.93x difference to unmodified public Puffysics.

### Independent contacts, `B3_ART_CONTACTS=0`

This public compile-time option uses the independent contact solver. Together
with the eight capsule substitutions, it removes both local cylinder GJK/EPA
work and articulated contact response. The native wrapper was guarded so this
configuration compiles; no engine header was changed. Build and standalone
collision instructions are in [STANDARD_COLLISION_REPRO.md](STANDARD_COLLISION_REPRO.md).

At 512 arenas, the same one-warmup/two-measured-update, horizon-16 workload
completes three times at **5,085.49 pooled learner SPS**. Every run has finite
state and losses, actual native PPO updates, and changed weights. They still
span only 0.96 simulated seconds including warmup. The normal 256-tick warmup
fails with 84 nonfinite arenas after 26.453 s of attempted work. There are zero
collision failure bits and zero capacity overflows. This does not demonstrate
a stable fighter environment or successful learning.

Nsight on the 512-arena short run shows 75.207% of kernel time in physics,
7.379% in body-state export, and 1.896% in contact export. The physics kernel
uses 128 registers and a 28,304-byte stack per thread, down from 196 registers
and 186,736 bytes in the experimental build. It still launches only 16 blocks
of 32 threads for 512 worlds. This is an engine/model-adapter parallelism issue
to investigate, with a now-visible secondary cost in state export.

The relaxed geometry and integration change the task distribution seen by the
loaded policy. PPO loss finiteness only proves optimizer execution: KL and value
losses are large, and no successful-learning claim follows from these timings.

Increasing this relaxed configuration to 4,096 arenas completes three times,
giving **10,609.63 pooled learner SPS** over 393,216 measured transitions in
37.062 s. The earlier original-geometry MuJoCo trial at that batch size yielded
3,991.69 SPS; its optimized 512-arena production configuration yielded 6,120.17.
Geometry and solver fidelity differ. These are throughput comparisons of those
stated configurations, not proof that a replacement learns the same behavior.
The 4,096-arena run uses the same 8,192-robot controller export already tested
with MuJoCo. No larger-batch optimum or stable long-run maximum is established.

## What fbr can act on

- Training uses native PufferLib CUDA PPO and a headless physics/controller loop.
  Raylib does not participate. The learner chooses moves; a fixed Sonic controller
  drives two 29-actuator humanoids per arena at 500 Hz physics / 50 Hz decisions.
- The 516 SPS result belongs to our experimental articulated fork. The local
  cylinder extension has a standalone, asset-free failure reproducer. It should
  not be presented as an upstream engine defect.
- Supported independent contacts and approximate capsule shapes materially
  improve throughput. Their longer-run nonfinite states are an unresolved
  integration problem. The imported force law, controller/solver mismatch,
  constraints and timestep need investigation; this profile does not identify
  the first cause of those NaNs.
- The traces identify physics as the largest cost. More intra-world parallelism,
  reduced per-thread solver storage, and cheaper body-state export are testable
  optimization targets. Their individual speedups have not been measured here.
- Full training reproduction requires the privately provisioned model, motion,
  controller and checkpoint files. The public collision reproducer has none of
  those dependencies. No game binaries or policy/controller weights are published.

## Reproduce and inspect

The source baseline is `73c918b6`; the additional adapter is described and
verified in [TRAINING_ADAPTER.md](TRAINING_ADAPTER.md). It preserves original
v8 stepping bit for bit in the differential fixture. Existing production
files and engine headers were not modified. Wrapper compile guards add the
independent-contact build without changing the default stepping paths.

On the existing Spark runtime, from this directory:

```sh
bash run_paired_training.sh mujoco mujoco-new-h16 16 2
bash run_paired_training.sh puffysics puffysics-new-h16 16 2
bash run_paired_training.sh puffysics puffysics-new-trace 16 2 nsys
bash run_paired_training.sh puffysics puffysics-new-sustained 256 2
REK_PUFFYSICS_SOLVER_MODE=0 bash run_paired_training.sh puffysics standard-new 16 2
REK_PUFFYSICS_SOLVER_MODE=0 REK_BENCH_WARMUP_UPDATES=0 \
  bash run_paired_training.sh puffysics standard-cold-new 8 1
```

Run arms sequentially. `run_paired_training.sh` records commands, stdout,
stderr and exit codes. Its environment variables select an existing runtime,
configuration and checkpoint on another machine. `profile_training.py --help`
exposes the direct configurable interface. No renderer, Steam client or display
is required. The private model, motion/controller assets and policy checkpoint
must already be provisioned; they are not included in this publication.

For the relaxed configuration, compile the independent library using the command
in `STANDARD_COLLISION_REPRO.md`, then select it explicitly:

```sh
task_root=/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z
task_python=/home/spark-advantage/.venv/bin/python
"$task_python" make_capsule_diagnostic.py \
  --source /home/spark-advantage/rek-training/rek-puffysics-prototype-20260911/model-export.json \
  --out "$task_root/model-export-capsule-new.json"
export REK_PUFFYSICS_SOLVER_MODE=0
export REK_PUFFYSICS_LIBRARY="$task_root/source/ocean/rek_g1/puffysics_prototype/librek_puffysics_independent_v1.so"
export REK_PUFFYSICS_MODEL_EXPORT="$task_root/model-export-capsule-new.json"
bash run_paired_training.sh puffysics independent-new-512 16 2
REK_DUEL_CONFIG="$task_root/paired-4096.json" \
  bash run_paired_training.sh puffysics independent-new-4096 16 2
```

Use a new output name on each invocation. The converter refuses to overwrite
an existing export. `paired-4096.json` changes only the controller manifest to
the provisioned 8,192-robot export; its full content is in the original raw archive.

The committed [results directory](training-results/20260913/) contains kernel
summaries and an archive of training reports, losses, checkpoint hashes,
commands, logs, validation results and kernel summaries. Extract
`raw-reports.tar.gz` there to inspect individual records. Start with
[training-summary.json](training-results/20260913/training-summary.json).
The relaxed solver follow-up is in
[solver-diagnostics-summary-v2.json](training-results/20260913/solver-diagnostics-summary-v2.json)
and `solver-diagnostics.tar.gz`, with two additional Nsight kernel summaries. Its summary
checks shared inputs within each group of three repetitions. Source metadata
identifies the later cold-start option and contact-setting additions; the six
original primary measurements retain their original benchmark hashes.
Raw Nsight traces remain private because their process environment metadata
can contain credentials. The public summaries contain kernel names, timings,
launch dimensions and API aggregates, without that metadata.
The aggregator checks that all six primary trials have identical shared
configurations, controller/checkpoint/extension hashes and benchmark code.

Complete working evidence and private checkpoints remain at
`/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/`.
The Windows evidence copy is
`C:/rekagent/evidence/paired-physics-training-20260913T0423Z/`.

These measurements compare two backends for the current candidate. They do
not establish authentic REK trajectory parity, winning ability or superhuman
performance. The fixed opponent is the human evaluator's candidate script,
not authenticated REK Bot 1.
