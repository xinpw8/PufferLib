# Native PufferLib 5.0 REK/Puffysics training

This is the standalone C++/CUDA training path. Build, model loading, fixed
controller inference, environment stepping, policy rollout, and PPO/Muon updates
execute without Python or Torch. Native MuJoCo loads the XML and computes startup
kinematics; all simulated physics steps use the CUDA Puffysics adapter. The host
still performs native configuration, allocation, graph launch, reporting, and
checkpoint I/O. Training is headless. Raylib draws no frames.

Short training runs execute successfully. Sustained training remains unsuitable:
the independent-contact capsule diagnostic produces nonfinite physics states.
The short SPS results below do not establish learning quality, REK parity, or a
usable fighter policy. No completed-round win rate was measured.

## Executed on Spark

Host: `spark-4ae3`, aarch64, NVIDIA GB10. Measurements: 2026-09-13 UTC.
Every Spark invocation used Windows PowerShell, WSL, then `ssh spark`.
Existing unrelated GPU processes remained running. These measurements are not
isolated maximum-throughput results; reported GPU utilization/memory are global.

One SPS unit is one learner transition with two physical G1 robots. It includes
environment rollout, policy inference, PPO/Muon, and native reporting boundaries.
It does not multiply by the opponent, replay samples, or physics substeps.

| Arenas | Physical robots | Warm training SPS | Measured transitions | Timed interval |
| --- | ---: | ---: | ---: | ---: |
| 512 | 1,024 | 5,585.17 | 16,384 | 2.933483 s |
| 4,096 | 8,192 | 12,130.13 | 131,072 | 10.805495 s |

Each row is one unprofiled three-epoch run. The table pools epochs two and three,
using transition deltas divided by native uptime deltas. The first epoch performs
graph capture and is excluded. Each arena advances 48 decision ticks total:
16 warmup ticks plus 32 measured ticks, equivalent to 0.96 simulated seconds.
The raw native INI downsamples three records into five entries and repeats the
last one; those duplicate entries are not additional trials.

Configuration: horizon 16; minibatch = arenas * horizon; replay ratio 1;
MinGRU 256 hidden units, two layers, approximately 459,000 parameters; fresh policy; learning
rate 0.0003; gamma 0.99; GAE lambda 0.95; 33 categorical actions; 223 observations.
The fixed exported SONIC encoder/decoder controls both robots. One 20 ms decision
tick executes ten 2 ms physics steps. The internal opponent is the existing
deterministic semantic-candidate approach dummy, not authenticated REK Bot 1.

The earlier [training comparison](../puffysics_prototype/TRAINING_PROFILE.md)
used a Python-driven integration with different replay/hyperparameter settings
and an existing policy checkpoint. Ratios between its results and this table
cannot isolate the effect of removing Python. The earlier physics and native
PPO kernels were already CUDA; this path also removes Python orchestration,
Torch tensor operations, and the old learner bindings.

## Profile of actual native training

A separate Nsight capture traced every graph node during three real PPO epochs
at 512 arenas. Native `sqlite3` produced the aggregate report; no Python analysis
script ran. Profiled throughput is excluded from the unprofiled table above.

| Kernel | Calls | Total GPU duration | Share of summed kernel duration |
| --- | ---: | ---: | ---: |
| `rp_step_kernel` | 480 | 3,283.848 ms | 79.28% |
| `rps_body_kernel` | 1,011 | 338.426 ms | 8.17% |
| `rps_contacts_kernel` | 1,011 | 68.294 ms | 1.65% |
| Motion `pre_kernel` | 48 | 58.456 ms | 1.41% |

The trace contains 26,569 kernels with 4.142174 s summed GPU duration. Physics
launches 16 blocks of 32 threads for 512 arenas, with 128 registers per thread.
Each thread processes one complete two-robot world. Eight times more arenas
improves measured warm training throughput by 2.17 times. Low world-count
parallelism is one concrete engineering target; this test does not quantify its
contribution separately from memory traffic and solver work.

Native PPO updates take approximately 1.6 ms per 8,192-transition batch, while a
warm rollout takes 1.42 to 1.48 s. Optimizing PPO or removing another host wrapper
cannot recover most of this measured time. The dominant work is inside the
physics kernel, followed by reconstruction of body state. Host API wait durations
overlap GPU work and must not be added to kernel durations.

The graph-enabled dashboard reports zero for `perf/eval_env` and
`perf/eval_model`; the combined work is in `perf/rollout`. Those zeros do not mean
the environment or policy were skipped.

## Failure and validation limits

An extended 512-arena run reached an invalid rollout by decision tick 80. Arena 1
reported sticky failure bits 15: nonfinite observations, combat state, scheduler,
and physics. The initial wrapper checked at the next rollout boundary, allowing
that invalid batch to reach PPO. The final wrapper checks immediately after
rollout completion, before PPO, including the graph-capture path.

This type of physics divergence predates the native5 port: the previous
independent-contact capsule diagnostic reported 84/512 nonfinite arenas at
horizon 256. Packed initial body, shape, joint, and root arrays were verified
byte-for-byte against that baseline. No default replacement or NaN suppression
was added.

Two additional adapter/solver discrepancies have native arithmetic reproducers:

- Eight exported hinge intervals cross the signed-pi boundary. Controller/state
  extraction unwraps the angle; the solver's limit comparison uses the wrapped
  angle. A legal 3.45958567 rad position becomes -2.82359958 rad and is classified
  outside its [-1.63292122, 3.77757883] rad interval.
- For 20/58 hinges, an isolated explicit passive-damping stability proxy exceeds
  2; its maximum is 15.63679. The mode-0 adapter does not include rotor armature.
  This scalar calculation is not a stability proof for the full constrained tree.

Neither discrepancy alone proves the cause of the observed training failure.
`physics_adapter_audit.cpp` reproduces the arithmetic without initializing CUDA
or executing CPU physics steps.

A separate [native first-failure replay](PHYSICS_FAILURE_DIAGNOSTIC.md) reproduces
a failure twice with real controller feedback and zero PPO updates. At physics
call 417, arena 12 has finite controls but a cached joint-limit impulse of
6.47818579e28. Joint warm starting drives an angular velocity component to
3.13031634e34 rad/s; the subsequent biased joint solve creates the first NaN.
This identifies the immediate failure independently of the learner. The cause
of the preceding impulse accumulation remains unresolved.

The [native controller tests](CONTROLLER_VALIDATION.md) also expose a strict
batch-1024 encoder mismatch: one of 65,536 tokens differs by one quantization bin.
Decoder error is at most 1.34e-5. Graph/eager execution and robot-state/drive tests
pass. These results are incompatible with claiming bitwise controller or REK
parity. The capsule geometry and independent-contact model are intentionally
approximate diagnostics authorized for the performance experiment.

## Build and reproduce

See [architecture and build details](NATIVE5_ARCHITECTURE.md). The trainer is
pinned to the public PufferLib commit
`773f923d80e73bdc255a2ba730c918b28e416aa1`. Build a fresh native executable:

```sh
bash ocean/rek_g1/native5/build_native.sh /absolute/new/build --build-runtime
bash ocean/rek_g1/native5/run_spark_probe.sh /absolute/new/build /absolute/new/result-512 24576
REK_NATIVE5_ARENAS=4096 bash ocean/rek_g1/native5/run_spark_probe.sh \
  /absolute/new/build /absolute/new/result-4096 196608
```

The probe script uses the existing private Spark asset paths. Its six explicit
environment arguments identify the XML, compiled physics export, semantic assets,
foot features, and two controller models. They must exist locally. No game
binaries, model weights, or checkpoints are included here. `REK_NATIVE5_PROFILE=1`
adds native Nsight capture with environment inheritance disabled. Raw profiler
files remain private; `profile_summary.sql` exports only kernel/API aggregates.

`validation/` contains sanitized command lines, exit codes, metrics, and test
outputs. The canonical build and final rollout-check results are recorded in
`validation/FINAL_RUNS.md`.
