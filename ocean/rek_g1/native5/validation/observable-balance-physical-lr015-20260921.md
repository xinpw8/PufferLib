# Physical observable-balance learning-rate arm: 0.015

The controlled 0.015 arm completed **4,194,304 learner transitions and 16
rollout/training iterations** on 2026-09-21, with runtime and wrapper exit codes 0.
Final checkpoint:
`a2481a82dc2c116b88e94c2461ccaef970240e2821ee68edae57cd8984ad1a6b`.
These are changing-policy physical simulator training results against
`CandidateApproachDummy`, not frozen-policy or authentic REK evaluation.

## Controlled configuration

This arm, the [0.0001 continuation](observable-balance-physical-continuation-20260921.md)
and the [0.001 arm](observable-balance-physical-lr001-20260921.md) all initialize
from the same original physical checkpoint:
`390007e256574d2fc5e1100eab4da048fccfec2532bd5e5a59bdcc19cf326310`.
The higher-rate arms are separate branches from that checkpoint, not sequential
fine-tunes of one another. Each uses weights-only initialization with fresh
optimizer state, RNG, recurrent state and step count. No BC checkpoint or
critic transformation was used.

The copied input and native step-zero readback have the exact original hash
and are byte-identical. The final checkpoint differs. The recorded environment
and command were compared mechanically with the 0.0001 arm: only initial
learning rate, run ID and initial/log/checkpoint output paths differ. The 0.015
rate was verified in the command, initial schema binding and every output
checkpoint sidecar. The clean trainer and physical runtime were unchanged.

Configuration: `mujoco_cuda`, native SONIC inference, CPU physics disabled,
no Python runtime; `rek.native5.observable_balance.v1`;
`normalized_points_falls_v1`; 512 arenas / 1,024 fighters; horizon 512;
minibatch 8,192; replay ratio 1; 120 s rounds; 50 Hz actions with stride 1;
223 observations / 33 actions; two 256-wide recurrent layers;
base/environment seeds 419; initial learning rate 0.015 with cosine annealing;
entropy coefficient 0.001; gamma 0.9998844821426083;
GAE lambda 0.9978673240629938. Every arena executes 8,192 learner steps,
or 163.84 simulated seconds. The 1,200 s timeout was not reached.

The opponent is the unchanged deterministic approach/backoff/turn and
cycling-attack dummy using raw inspection observations. It is not the recovered
Bot 1 controller or a self-play policy. No loss, reward, controller, opponent or
physics changes were introduced for this arm. Shared joint pose/rate mapping
remains unavailable; native count-state observation remains available.

## Complete-run comparison

All arms completed 4,194,304 new transitions and 512 rounds. Learner values precede
opponent values in paired counters.

| Measurement | LR 0.0001 | LR 0.001 | LR 0.015 |
| --- | ---: | ---: | ---: |
| Wins / losses / draws | 232 / 242 / 38 | 220 / 263 / 29 | 259 / 226 / 27 |
| Completed-round points | 5,410 : 5,538 | 5,383 : 5,646 | 5,891 : 5,638 |
| Lifetime confirmed falls | 956 : 1,120 | 1,006 : 1,114 | 982 : 1,165 |
| Lifetime awarded points | 6,817 : 7,352 | 7,046 : 7,589 | 7,529 : 7,571 |
| Runtime failure bits / reward saturations | 0 / 0 | 0 / 0 | 0 / 0 |
| Native trainer uptime | 700.117 s | 681.997 s | 709.207 s |
| Complete-training native SPS | 5,990.86 | 6,150.03 | 5,914.08 |
| Whole-process elapsed time | 709.57 s | 691.54 s | 718.23 s |
| Whole-process SPS | 5,911.05 | 6,065.16 | 5,839.78 |

The 0.015 arm has zero redos and zero unclassified outcomes. It has the highest
training-round win total and completed-round point difference of these three
arms. Its learner fall count is above the 0.0001 arm and below the 0.001 arm.
This single-seed changing-policy comparison does not establish that its final
checkpoint is stronger in authentic REK. Authentic evaluation is recorded
separately.

Completed-round points exclude unfinished rounds. Lifetime counters cover all
executed transitions, including unfinished rounds. Confirmed falls are native
`BECAME_FALLEN` events, not inferred labels. Rewards retained fixed scale 0.01,
own confirmed-fall penalty -0.01, bounds [-1,1], no terminal bonus and zero
saturation events.

Native uptime covers all rollouts, PPO updates and periodic checkpoint writes,
but excludes initial construction and CUDA graph-capture construction.
Whole-process time includes initialization and shutdown. Display resolution is
1 ms and 0.01 s respectively. Neither SPS value is a physics-only benchmark or
a selected dashboard window.

## Frozen authentic evaluation

The final checkpoint subsequently completed authentic development round r113
against private Sparring Bot 1 and lost **2 : 16**, with ordinary awarded points
**2 : 1** and five-point awards **0 : 3**. All six evidence checks passed and the
owned client closed. This is a separate one-round frozen-checkpoint cohort;
the changing-policy training wins above are not pooled with it. No promotion
is justified by this result. Full validation and preservation details are in
[the authentic evaluation report](observable-balance-authentic-evaluation-20260921.md#lr-0015-round-r113-loss).

## Reproduction and preservation

Spark stage:
`/home/spark-advantage/rek-training/physical-observable-balance-20260921-r1`.

Executed wrapper command:

```sh
bash /home/spark-advantage/rek-training/physical-observable-balance-20260921-r1/run-balance-physical-lr-followup.sh .015 lr015-r1
```

Output: `train-lr-lr015-r1`.
Final checkpoint relative path:
`train-lr-lr015-r1/checkpoints/rek_native5/lr-lr015-r1/0000000004194304.bin`.
Exact native arguments/environment are retained in the output's `command.txt`.

| Artifact | SHA256 |
| --- | --- |
| Clean trainer | `bd696ecc152c8b326bd6891bdf60ee97f7b0fa523f39e98696fe21708165185d` |
| Executed LR runner | `17e90c448e3ba3230aaaa2f813bf25ea3b0a74824887cb4ffa5ad2698b6e50b2` |
| Final checkpoint | `a2481a82dc2c116b88e94c2461ccaef970240e2821ee68edae57cd8984ad1a6b` |
| Training archive | `78acaf2405faeead3acde84ef680e92fed2ddd334832d3545823b0cf5d793acf` |
| Aggregate result | `18ae609849759f8341704e0ce0beedbd502288f0e6171fb2ee1baad51ad3eefc` |
| NAS receipt | `714ef7547fa846515015247bc8dbc0731d1e0ed3b52812837062f5de8df3c1d8` |

Fresh private NAS directory:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-observable-balance-r1\lr015-r1`.

The 32,890,793-byte archive contains 73 selected source/build/run files plus
aggregate and inventory sidecars: initial weights, step-zero and all 16 update
checkpoints, hash-bound schema/LR sidecars, exact scripts, raw logs, metrics,
timing, clean binary and build provenance. All referenced hashes and checkpoint
sidecars were verified. Source hashes before/after packaging, tar comparison,
and Spark/local/NAS archive hashes matched. The original training archive was
reverified. All destinations were fresh; existing evidence was not overwritten.

No additional trainer, inference or game process was launched by the monitoring
and collection work. No proprietary assets, binaries, checkpoints or raw
captures are published here.
