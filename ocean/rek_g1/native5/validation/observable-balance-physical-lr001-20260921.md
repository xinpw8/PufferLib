# Physical observable-balance learning-rate arm: 0.001

The controlled 0.001 arm completed **4,194,304 learner transitions and 16 updates**
on 2026-09-21, with runtime and wrapper exit codes 0. Final checkpoint:
`d34f3fefc71b3ca4977590be142100beca58872fd2051813a56bc57536bae60f`.
The results below are changing-policy simulator training against
`CandidateApproachDummy`, not authentic REK or frozen-policy evaluation.

## Controlled configuration

Both this arm and the [0.0001 continuation](observable-balance-physical-continuation-20260921.md)
start from the same original physical checkpoint:
`390007e256574d2fc5e1100eab4da048fccfec2532bd5e5a59bdcc19cf326310`.
This arm does not initialize from the preceding `a11ace...` continuation output.
The copied input and native step-zero readback match the original hash exactly;
the final checkpoint differs. Both arms use weights-only initialization with
fresh optimizer, RNG, recurrent state and step count. Neither uses a BC
checkpoint or critic transformation.

The recorded environment and expanded command were compared mechanically with
the 0.0001 arm. They match except for initial learning rate, run ID and
initial/log/checkpoint output paths. The 0.001 value was verified in the command,
initial schema binding and every produced checkpoint sidecar. The clean trainer
and runtime were unchanged.

Configuration: `mujoco_cuda`, native SONIC inference, no CPU physics or Python
runtime; `rek.native5.observable_balance.v1`; `normalized_points_falls_v1`;
512 arenas / 1,024 fighters; horizon 512; minibatch 8,192; replay ratio 1;
120 s rounds; 50 Hz action stride 1; two 256-wide recurrent layers;
223 observations / 33 actions; base and environment seeds 419;
initial learning rate 0.001 with cosine annealing; entropy coefficient 0.001;
gamma 0.9998844821426083; GAE lambda 0.9978673240629938.
Each arena executed 8,192 learner steps, or 163.84 simulated seconds.
The 1,200 s process timeout was not reached.

`CandidateApproachDummy` retains its existing approach/backoff/turn and cycling
attacks using raw inspection observations. It is not the recovered Bot 1
controller or self-play. No reward, loss, controller, opponent or physics changes
were made for this arm. Joint pose/rate correspondence remains unavailable in
the observable schema; the native count-state field remains available.

## Complete-run comparison

| Measurement | Initial LR 0.0001 | Initial LR 0.001 |
| --- | ---: | ---: |
| Transitions / updates | 4,194,304 / 16 | 4,194,304 / 16 |
| Completed rounds | 512 | 512 |
| Learner wins / losses / draws | 232 / 242 / 38 | 220 / 263 / 29 |
| Completed-round points, learner : opponent | 5,410 : 5,538 | 5,383 : 5,646 |
| Lifetime confirmed falls, learner : opponent | 956 : 1,120 | 1,006 : 1,114 |
| Lifetime awarded points, learner : opponent | 6,817 : 7,352 | 7,046 : 7,589 |
| Runtime failure bits / reward saturations | 0 / 0 | 0 / 0 |
| Native trainer uptime | 700.117 s | 681.997 s |
| Complete-training native SPS | 5,990.86 | 6,150.03 |
| Whole-process elapsed time | 709.57 s | 691.54 s |
| Whole-process SPS | 5,911.05 | 6,065.16 |

The 0.001 arm has zero redos and zero unclassified results. It recorded fewer
training-round wins and more learner falls than the 0.0001 arm. These single-seed,
changing-policy training cohorts do not establish the relative authentic strength
of the final checkpoints. Authentic evaluation is recorded separately.

Completed-round points exclude unfinished rounds. Lifetime counters cover all
executed runtime transitions, including unfinished rounds. Confirmed falls are
native `BECAME_FALLEN` events. Reward scaling remained fixed at 0.01, own confirmed
fall penalty -0.01, bounds [-1,1], no terminal bonus and zero saturation events.

Native uptime includes all rollouts, PPO updates and checkpoint writes; its clock
excludes initial construction and CUDA graph-capture construction. Whole-process
time includes initialization and shutdown. Display resolution is 1 ms for native
uptime and 0.01 s for process time. Neither SPS value is physics-only throughput
or a selected dashboard window.

## Reproduction and preservation

Spark stage:
`/home/spark-advantage/rek-training/physical-observable-balance-20260921-r1`.

Executed wrapper command:

```sh
bash /home/spark-advantage/rek-training/physical-observable-balance-20260921-r1/run-balance-physical-lr-followup.sh .001 lr001-r1
```

Output directory: `train-lr-lr001-r1`.
Final checkpoint relative path:
`train-lr-lr001-r1/checkpoints/rek_native5/lr-lr001-r1/0000000004194304.bin`.
The exact native arguments/environment are retained in `command.txt`.

| Artifact | SHA256 |
| --- | --- |
| Clean trainer | `bd696ecc152c8b326bd6891bdf60ee97f7b0fa523f39e98696fe21708165185d` |
| Executed LR runner | `17e90c448e3ba3230aaaa2f813bf25ea3b0a74824887cb4ffa5ad2698b6e50b2` |
| Final checkpoint | `d34f3fefc71b3ca4977590be142100beca58872fd2051813a56bc57536bae60f` |
| Training archive | `bb7812050b3907d59fdcba96234b03314b54424a0b9c0ace8b7b3eaf5bfac545` |
| Aggregate result | `39ac711f80b8a86d86bf62d4bb4de013f2c58ceb8ceae4aed99438e2e7470323` |
| NAS receipt | `d190089a752c1b95bdd1ba5ed18a0a9177c10d48fad50c4323dff2a94841944e` |

Fresh private NAS directory:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-observable-balance-r1\lr001-r1`.

The 32,811,893-byte archive preserves 73 selected source/build/run files plus
aggregate and inventory sidecars. It includes input weights, step-zero and all
16 update checkpoints, hash-bound schema/LR sidecars, exact scripts, raw logs,
metrics, timing, clean binary and build provenance. All referenced provenance
hashes and checkpoint sidecars were verified. Source hashes before/after
packaging, tar comparison and Spark/local/NAS archive hashes matched. The
original training archive was reverified. All destinations were fresh and no
existing evidence was overwritten.

No additional trainer, inference or game process was launched by the monitoring
and collection work. The 0.015 arm was not launched by this work. No proprietary
assets, binaries, checkpoints or raw captures are published here.
