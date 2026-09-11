# Spark backend experiment, 2026-09-11 UTC

The implemented Puffysics prototype runs the current candidate model and
Sonic controller on CUDA. It has not demonstrated either a performance
advantage over the existing CUDA MuJoCo backend or authentic REK parity.

## Public example reproduction

The public robot-arm template was compiled and executed on the same Spark
GB10. At 4,096 worlds its three constant-action trials achieved 480,406,
483,033 and 479,952 environment decisions/s. At 512 worlds they achieved
85,562, 86,122 and 86,514 decisions/s. Each decision used eight physics
substeps; each trial contained 32 decisions/world. Outputs remained finite.

This is a seven-joint robot-arm kernel workload without policy inference or
training. Its result does not establish equivalent speed for the present
two-humanoid model. Raw results and source hashes are in command
`robot-arm-kernel-benchmark-01`.

## Paired throughput measurement

Host `spark-4ae3`, NVIDIA GB10; 512 arenas, 1,024 robot rows; three independent
runs per backend, each with 32 timed control ticks after three warmup ticks
and a full reset. CUDA graph replay enabled. Physics timestep 0.002 s,
controller period 0.02 s, target-filter period 0.004 s. Both sides use the same
candidate model, motion assets, explicit-batch Sonic controller, target filter
and fixed idle reference. Setup, model parsing, graph capture and result
serialization are excluded. Four device-to-device trace copies per tick are
included. The two backends run sequentially, without overlapping GPU tests.

| Backend | Run 1 arena SPS | Run 2 | Run 3 | Pooled arena SPS |
| --- | ---: | ---: | ---: | ---: |
| CUDA MuJoCo candidate | 10,257.2 | 10,089.4 | 10,136.4 | 10,160.5 |
| Puffysics prototype v4, cached ABA | 869.9 | 871.3 | 867.7 | 869.7 |

After the collision corrections, the final v8 build was measured again under
the same conditions. All six short idle benchmark runs passed:

| Backend | Run 1 arena SPS | Run 2 | Run 3 | Pooled arena SPS |
| --- | ---: | ---: | ---: | ---: |
| CUDA MuJoCo candidate, final paired runs | 10,219.9 | 10,058.3 | 9,912.2 | 10,061.9 |
| Puffysics prototype v8, cached ABA | 867.3 | 873.9 | 869.7 | 870.3 |

The final prototype is 11.56 times slower. The native shared-library SHA256 is
`165adaee16bfff14b3113580f3ec1635382a4b091acbb36dde1c907f57986434`.
The passing short idle benchmark does not override the failed longer motion
tests described below.

Pooled SPS is total arena control steps divided by total timed wall seconds.
One arena control step advances two robots and ten physics substeps. Robot
decision SPS would be twice these values. These are fixed-reference rollout
measurements, not training SPS. They exclude combat scheduling, scoring,
opponent AI, reward computation and policy updates. No superhuman-policy claim
is supported by this experiment.

The v4 native prototype is 11.68 times slower in this paired test.
The earlier fast public robot-arm example is a different model and workload;
its throughput does not predict this two-humanoid articulated-contact workload.

The v4 Puffysics timing is GPU-dominated: run 1 used 18.8331 CUDA-stream
seconds, 18.8335 wall seconds and 0.0095 host CPU-seconds. Both backend reports
record zero CPU physics steps and zero CPU controller inferences. Source
inspection shows one thread per arena, so 512 arenas launch 16 blocks of
32 threads. The v8 compiler reports 186,736 bytes of stack per thread and
196 registers, with zero explicit register spill stores or loads. Limited
parallelism, scratch-memory traffic and repeated articulated contact-response
solves are profiling targets. Their individual time percentages have not been
measured, so this experiment does not assign a proven percentage to any one
cause.

## Accuracy and engineering changes

Both engines were initialized from the same candidate idle pose. Native
initial qpos reconstruction differs by at most 1.31e-7. The first controller
outputs differ by at most 7.16e-7, before the physics trajectories separate.

Three independently reproduced ABA equation defects were corrected. The
corrected free-body test has zero spurious COM acceleration. The planar
floating-hinge test matches the independent generalized mass-matrix solution
within 4.77e-7. Armature and contact-response tests pass.

For the four-arena, two-second idle rollout, the maximum root-position error
versus MuJoCo decreased from 0.226 m to 0.111 m after those corrections. The
corrected RMS root error is 0.0774 m. All eight robots remained above the
explicit diagnostic fall proxy: root height below 0.45 m or root-up component
below 0.5. This proxy is not REK's knockout rule.

Fixed-pose contact mass-factor caching preserved all recorded CUDA qpos,
qvel, controller actions, position targets and simulation-clock values bit for
bit over the two-second idle trace. Timed wall duration fell from 120.508 s to
53.765 s, a 2.24 times improvement. The corresponding CPU operator fixture
matched 342,120 values bitwise while reducing mass-factor rebuilds from 5,317
to 48 without reducing RHS solves or solver iterations.

The v4 walking test exposed a GJK iteration-limit failure at approximately
1.210 simulated seconds. All four arenas stopped advancing, with finite
states and an explicit failure bit. The failed run is retained. This failure
is not counted as a completed rollout or included in the throughput table.
Exact-pair CPU/CUDA reproduction exposed floating-point cancellation. Promoting
simplex operands before subtraction resolved that query. A subsequent walking
trace exposed near-contact EPA initialization failure. Retaining certified
support-plane lower bounds resolved the captured query and exact-touch cases.
Ten CUDA queries, including local perturbations and genuine penetrating cases,
passed without changing iteration budgets or tolerances.

The complete v8 motion tests still failed:

- Walking: all four arenas produced nonfinite states near 1.940 s of the
  requested 2.000 s. Collision failure bits were zero, contact capacity was
  not reached, and the numerical failure remains unresolved.
- Left-front kick: all four arenas stopped near 2.470 s of the requested
  3.000 s. A box-cylinder query failed EPA seed initialization. Its exact
  operands were captured; states remained finite. This failure remains
  unresolved.

Both runs returned exit code 2. Their recorded requested-step/wall-time
quotients are not valid completed-rollout SPS measurements. Failed traces
must not be mistaken for successful movement demonstrations. The renderer
labels halted time and invalid states explicitly.

## What remains unproved

MuJoCo integration, contact compliance, static joint friction and soft joint
limits are not reproduced. The single-point cylinder manifold differs from
MuJoCo's manifold. These differences matter independently of performance.
The current prototype is unsuitable as a replacement training backend.

The MuJoCo candidate itself is not an authentic REK recording. Matching this
candidate would still leave the project's held-out authentic-game trajectory
and event acceptance tests outstanding. Idle stability, visually similar
motion, and finite values are all weaker than that acceptance criterion.

## Evidence locations

Command text, raw stdout/stderr, exit codes and hashes:

`C:/rekagent/evidence/motion-parity-D21-20260903T071000Z/commands/`

Reports, state/action traces and rendered playback:

`C:/rekagent/evidence/motion-parity-D21-20260903T071000Z/puffysics-prototype/`

Spark source, built libraries and original run artifacts:

`/home/spark-advantage/rek-training/rek-puffysics-prototype-20260911/`

Relevant command IDs include `puffysics-native-build-04`,
`puffysics-cuda-graph-paired-benchmark-02`, `puffysics-native-idle-v3-01`,
`puffysics-native-idle-v4-01`, `puffysics-cuda-cache-equivalence-01`,
`puffysics-compare-idle-v4-01`, and `puffysics-motion-trials-01`.

Final-build command IDs: `puffysics-native-build-08`,
`puffysics-package-cpu-v8-01`, `puffysics-motion-trials-03`,
`puffysics-kick-v8-01`, `puffysics-cuda-graph-paired-benchmark-03`,
`puffysics-benchmark-summary-v8-01`, `puffysics-render-final-v8-01`.
Final aggregate: `puffysics-prototype/benchmark-v8-summary.json`.
Recorded playbacks: `puffysics-prototype/render-v8-walk/comparison.gif` and
`puffysics-prototype/render-v8-kick_left_front/comparison.gif`.

The initial v2 two-second simulation finished, but its shell artifact-copy
step failed after the launcher script was edited while executing. The
unchanged simulation outputs were subsequently recovered and compared under
`puffysics-compare-idle-v2-01`. This wrapper failure is preserved in the logs.

Private model exports, game binaries, controller binaries and runtime state
archives are not included in the source commit.
