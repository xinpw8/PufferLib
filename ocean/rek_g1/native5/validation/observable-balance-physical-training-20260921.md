# Physical observable-balance training, 2026-09-21

The clean native trainer completed **4,194,304 learner transitions and 16 updates**
with exit code 0. This is changing-policy physical simulator training against
`CandidateApproachDummy`. It is not frozen-policy evaluation or authentic REK
evaluation, and it does not establish improved fighting strength.

## Executed configuration

The run used `mujoco_cuda`, native SONIC controller inference,
`rek.native5.observable_balance.v1` and `normalized_points_falls_v1`.
Startup receipts confirm both opt-in contracts, CPU physics disabled and no
Python runtime. Joint pose/rate correspondence remains unavailable in both
observation adapters. See the separate [adapter report](observable-balance-physical-integration-20260921.md).

Fresh random policy weights, no warm start, no self-play and no frozen opponent:
512 arenas / 1,024 fighters; horizon 512; minibatch 8,192; replay ratio 1;
two 256-wide recurrent layers; base/environment seed 419; 120 s rounds;
50 Hz actions with stride 1; initial learning rate 0.0001 with cosine annealing;
entropy coefficient 0.001; gamma 0.9998844821426083;
lambda 0.9978673240629938. Each arena executed 8,192 learner steps, or 163.84
simulated seconds. The process timeout was 1,200 s and was not reached.

`CandidateApproachDummy` uses the existing deterministic approach/backoff/turn
logic and cycling attacks with raw runtime inspection observations. It is not
the recovered Bot 1 controller. The learner receives the new shared observable
projection; the opponent's existing input and logic are unchanged.

## Completed results and throughput

| Measurement | Actual result |
| --- | ---: |
| Complete native trainer uptime | 680.744 s |
| Transitions / native trainer uptime | 6,161.35 SPS |
| Whole-process elapsed time | 690.19 s |
| Transitions / whole-process elapsed time | 6,077.03 SPS |
| Completed rounds | 512 |
| Learner wins / losses / draws | 204 / 274 / 34 |
| Completed-round points, learner : opponent | 5,149 : 5,610 |
| Redos / unclassified results | 0 / 0 |
| Runtime failure bits | 0 |
| Reward saturations | 0 |

The native uptime spans all completed rollouts, PPO updates and periodic
checkpoint writes. The pinned trainer starts its clock after initialization and
shifts that clock to exclude CUDA graph construction. Whole-process timing
includes those costs and final shutdown. Native uptime is displayed to 1 ms;
`/usr/bin/time` elapsed time is displayed to 0.01 s. Neither figure is a
physics-only throughput benchmark.

The INI's last completed-round metrics row is at step 3,407,872 / epoch 13,
uptime 553.050930 s, and contains only four newly reported rounds. Its win value
0.5 and SPS value 6,242.26 are not the complete-run results. The table uses final
epoch-16 console uptime, whole-process timing and final native round counters.

Lifetime counters cover **all executed runtime transitions**, including
unfinished rounds: confirmed `BECAME_FALLEN` events were
980 : 1,099 and awarded points were 6,568 : 7,430. These are distinct from the
completed-round point totals. They establish exposure to actual physical fall
events in this run; they do not establish a beneficial learned response to falls.
Normalization remained fixed at 0.01 with bounds [-1,1], own confirmed-fall
penalty -0.01, no terminal bonus and zero clipping events.

## Reproduction, provenance and preservation

Spark stage:
`/home/spark-advantage/rek-training/physical-observable-balance-20260921-r1`.
Executed command is preserved in `train-balance-physical-r1/command.txt` and its
exact environment/arguments in `launch-script.sh`. The clean production binary
was `trainer-build-r1/puffer-rek-native5`, without the earlier smoke-test guard
or initial-checkpoint instrumentation. All 16 update checkpoints are preserved;
the clean trainer does not emit a fresh step-zero checkpoint.

| Artifact | SHA256 |
| --- | --- |
| Clean trainer | `bd696ecc152c8b326bd6891bdf60ee97f7b0fa523f39e98696fe21708165185d` |
| Final step-4,194,304 checkpoint | `390007e256574d2fc5e1100eab4da048fccfec2532bd5e5a59bdcc19cf326310` |
| Training archive | `00513323c312615e3d11e3db82fe3843e33e4c0cbad26615c0b7ec1e7a0e997b` |
| Aggregate result | `de90df1dde3a527b67c9620519e98e33eaee431f0f43215280ac5c4c038d5113` |
| NAS receipt | `c8ee05c5306adcd8e0d2a90e523799ef7d68185e4fd6bbbd35e2933d9c6fe470` |

Final checkpoint relative path:
`train-balance-physical-r1/checkpoints/rek_native5/balance-physical-r1/0000000004194304.bin`.

Private NAS directory:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-observable-balance-r1\training-r1`.
The 29,531,409-byte archive contains 36 selected source/build/run files plus
summary and inventory sidecars. Every referenced build/provenance hash was
recomputed successfully. Source hashes before/after packaging, tar comparison,
and Spark/local/NAS archive hashes passed. The existing adapter archive was also
reverified. All destination files were new; no existing evidence was overwritten.
No proprietary assets, binaries, checkpoints or raw captures are published here.
