# Native Puffer5 end-to-end training throughput

One authorized, headless, native-only training measurement completed on Spark GB10 on 2026-09-20. This measures the entire rollout-plus-learning workload, not a pooled arena-step kernel. The generated checkpoint is performance-only and has not been promoted or submitted for authentic evaluation. The original live candidate remains unchanged.

## Measured result

| Measurement | Result |
|---|---:|
| Learner transitions | 33,554,432 |
| Arenas / horizon / minibatch | 512 / 512 / 8,192 |
| Training epochs | 128 |
| Trainer-loop elapsed time | 36.16706991195679 s |
| Whole-training throughput | 927,761.969152689 transitions/s |
| Process wall time | 36.86 s |
| Startup-inclusive throughput | 910,320.9983722193 transitions/s |
| Process CPU user / system time | 0.28 / 0.55 s |
| Maximum process resident memory | 739,560 KiB |
| Exit status / runtime failure bits | 0 / 0 |

The wrapper started at `2026-09-20T02:34:56.895700618Z` and ended at `2026-09-20T02:35:33.912520368Z`. The GPU reservation was released after completion. An unrelated existing process, PID 2325, was observed in the read-only ownership snapshot and was not modified or stopped. No live client was running during this reservation.

`summarize_fast.cjs` computes throughput from the final completed learner-transition count divided by the trainer's final `uptime`. Each transition is one learner arena tick at 0.02 s. Opponent updates and the eight contact substeps do not multiply that count. The trainer excludes CUDA graph capture/instantiation time from its `uptime`; `/usr/bin/time` includes process startup and graph setup. The process-minus-loop difference is 0.692930088 s and is not a separately isolated graph-capture measurement.

The earlier reserved BC-plus-PPO run measured 929,730.498102775 transitions/s over 36.0904929638 s, with 36.78 s process wall time. This run is 0.211731% lower. There is no repeated-run uncertainty estimate, and the input policy and mask differ, so this single comparison does not establish a throughput regression.

## Existing profiler breakdown

No profiler, runtime, trainer, or benchmark implementation was changed. Existing logged CUDA component times give:

| Recorded component | Time |
|---|---:|
| Rollout graph, including policy inference and compact environment | 27.8651343435 s |
| Training model work | 7.5631758645 s |
| Training miscellaneous work | 0.1595876787 s |
| Training subtotal | 7.7227635346 s |
| Recorded CUDA-component total | 35.5878978781 s |
| Trainer-loop time outside these recorded components | 0.5791720338 s |

Rollout and training account for 78.2995% and 21.7005% of the recorded component total, respectively. The residual includes work outside these timers and any skipped initial profiling; it is not identified as exclusively CPU work, copies, or synchronization.

The 64 saved metric entries are bin means with repeated fallback/final entries, not 64 independent timing intervals. The full dashboard records 44 actual logging epochs: 1, 4, 7, then every third epoch through 127, followed by 128. Inspection of the pinned `log_history_bin_mean` and final artifact writer establishes that the first metric bin averages epochs 1 and 4, later genuine bins each hold one interval, and the final history sample is appended again when writing the artifact. The totals above restore the first bin's weight of two and count each subsequent genuine interval once, excluding that final duplicate and padded fallback entries. The reconstructed epoch means were checked against every corresponding saved metric bin.

In graph mode the rollout profiler records only the whole rollout graph. Its separate `perf/eval_model`, `perf/eval_env`, and `perf/eval_copy` counters are zero because that finer split is not recorded. They do not mean those operations have zero cost. Relevant pinned `pufferl.cu` locations are `rollout_finish` near line 1367, training timestamp accumulation near line 1714, `log_history_bin_mean` at line 2523, periodic accumulator export/reset near line 3424, and final history append near line 3532. The final one-epoch interval recorded 0.2284483165 s rollout and 0.0597694702 s training; it is not the whole-run timing.

## Exact reused setup

The executable and runtime source are unchanged from the prior 929,730.5 SPS run. The warm start is the authentic scaled-GAE actor `f8bcd3...`, with its original 223 unmasked observation features. The prior run instead started from the balanced BC epoch-5 checkpoint and masked features 176 through 183. This run explicitly unsets `REK_POLICY_FEATURE_MASK`.

Unchanged settings include 120 s simulated rounds, seed 419, learning rate 0.0001 with the existing learning-rate annealing, entropy coefficient 0.01, gamma 0.9998844821426083, lambda 0.9978673240629938, value coefficient 0.5, policy/value clipping 0.2, replay ratio 1, hidden size 256, two MinGRU layers, BF16 training, CUDA graphs enabled, synchronous rollout/training, one buffer, and recurrent carry across horizons. It uses the existing scripted recovered-Bot1 approximation, randomized initial gaps of 0.55 to 2.5 m, heading spread pi radians, and no extra shaping or frozen opponent.

The exact prepared command script is preserved privately as `performance-only-command.sh`. Its substantive invocation, using the existing wrapper and config without edits, is:

```sh
reference=/home/spark-advantage/rek-training/policy-feature-mask-20260919-r1
stage=/home/spark-advantage/rek-training/critic-calibration-20260919-r1
native="$reference/source/ocean/rek_g1/native5"
initial="$stage/train-scaled-gae-r1/ppo-one-epoch.bin"
output="$stage/train-performance-only-f8-r1"

unset REK_POLICY_FEATURE_MASK REK_FAST_CONTACT_POTENTIAL
unset REK_TRAIN_OPPONENT_CHECKPOINT REK_FROZEN_OPPONENT_FRACTION
unset REK_FAST_SHAPING_GAMMA REK_FAST_SHAPING_TARGET
unset REK_FAST_SHAPING_BEARING_WEIGHT REK_FAST_REWARD_GAMMA
export REK_FAST_REWARD=round_outcome_v1
export REK_FAST_SCORING=recovered_hit_rules_v2
export REK_FAST_GEOMETRY=primitive_samples_v1 REK_FAST_CONTACT_SUBSTEPS=8
export REK_FAST_OPPONENT=recovered_bot1_v1
export REK_FAST_OBSERVATION=rendered_pose_v1 REK_FAST_OPPONENT_MODE=scripted
export REK_FAST_RANDOM_RESETS=1
export REK_FAST_RESET_GAP_MIN=.55 REK_FAST_RESET_GAP_MAX=2.5
export REK_FAST_RESET_HEADING_SPREAD_RAD=3.14159265 REK_FAST_SHAPING_WEIGHT=0
export REK_TRAIN_SEED=419 REK_TRAIN_LEARNING_RATE=.0001
export REK_TRAIN_MINIBATCH=8192 REK_TRAIN_ENTROPY=.01 REK_TRAIN_TIMEOUT=300
export REK_TRAIN_GAMMA=.9998844821426083
export REK_TRAIN_GAE_LAMBDA=.9978673240629938

bash "$native/run_diverse_training.sh" "$reference/build" "$output" \
    33554432 512 512 120 "$initial"
node "$native/summarize_fast.cjs" "$output"
```

The saved `command.txt` contains the fully expanded native executable invocation, including all private asset paths, logger/checkpoint paths, and warm-start argument. The wrapper requires a fresh output directory, copies the exact INI configs, records hashes and process timing, verifies successful completion, and compares the step-zero checkpoint byte-for-byte against the requested warm start. That comparison passed. The live-candidate input digest was rechecked after the run and remains unchanged.

## Scope and compact-runtime limitations

The runtime still reports `authentic_parity=false`, canned pose routes, compact planar slider dynamics, approximate collision contacts, constant upright state, and unmodeled knockdowns. This is not native MuJoCo balance dynamics or authentic match execution. Training produced 5,120 completed compact rounds: 4,869 learner wins, 224 opponent wins, and 27 ties, with awarded totals 376,073:211,858. Those changing-policy training counts are not held-out strength or evidence of improved fighting. The new checkpoint is quarantined under a `train-performance-only` path.

## Artifact preservation and hashes

Spark run directory:
`/home/spark-advantage/rek-training/critic-calibration-20260919-r1/train-performance-only-f8-r1`.

Local mirror:
`C:\rekagent\work\consistent-fighter-20260919-r1\critic-calibration-r1\spark-results\train-performance-only-f8-r1`.

All 18 run files, totaling 5,610,833 bytes, were copied and independently rehashed against Spark. The two outer wrapper logs, totaling 14,278 bytes, were separately copied and hash-verified under sibling `benchmark-wrapper/`. Existing archives and checkpoints were not overwritten.

| Artifact | SHA-256 |
|---|---|
| Native trainer | `b5e738508d546e71e02b236520bf760b764e749864619e933562b6c93431aca0` |
| Built/current `fast_runtime.cu` | `07c22ca1d4255db9940dc9e39a2ff0f1d1368491dfda4e3bbc17a09a61c11273` |
| Pinned patched `pufferl.cu` | `ae71826468701bf19691548555c1a2d354f8795065bb3bc7fb6e5e2e2b0eb378` |
| Reused `run_diverse_training.sh` | `602371fd8d5955ee169536b08f5d7e6abc17ac09effd38b53aab228a1444ef77` |
| Reused `summarize_fast.cjs` | `0a96b50a4e5b72fb476b28faabf88df1610b9c424e7fc320d3ba727f647c3217` |
| Warm start and verified step-zero readback | `f8bcd3f3d6ef3d691209823d5a0d452ca16ff16b03c7ed1867715510986ea483` |
| Performance-only 16,777,216-transition checkpoint | `d119c78420dc91793a2e16466fb2fcdaa6d4441cbf58042bef3a8d9d434a3484` |
| Performance-only 33,554,432-transition checkpoint | `ace4daee2e481835ef7aa2f88a32194708bcbe2fc7a0fa30c74f2792cfde5bef` |
| `summary.json` | `1f810c23ae3707ae6c0d5eea6fddeffc1561472da52565a7dde61ac10b701e42` |
| Trainer `stdout.txt` | `e65d5dc0772b1ce5800aab61cfeb78a10ad733cd553618edd5cdda3f67c912cd` |
| Trainer `stderr.txt` | `c5e25e89c6101417c73c57f6c8669c197b4fcf64de178c209fd9aac330d22dea` |
| Fully expanded `command.txt` | `0b58d35993010e441ee97c95a12162fa3c539c2b1b0455c1d57550665cbdf6f9` |
| `performance-only-command.sh` | `8081c0eadc4b0e13defcc449ed2110b7938ce8e2e731c4fa1d1a81475d717ba6` |
| `process-timing.txt` | `2186a677c0f7c68b0a513d2df02cba75d050926f70d6281de2269937ddcb2d33` |
| Full trainer metrics INI | `684fc4e3b91a89708fb5f88a095559bbd44093c179450efce0c4d9fe1d3745d2` |
| Outer wrapper stdout | `78d69aaa13a05e87415b3f145a4ef5bc01d4807266c3385e0c7232ac42e76fff` |
| Outer traced-command stderr | `4644237fe41f356a4d0de22779f13e9b41a62d48411e7b161804e03f1acbab62` |

The final private checkpoint filename is `checkpoints/rek_native5/train-performance-only-f8-r1/0000000033554432.bin`. No binary checkpoint, raw recording, or proprietary model payload is included in this report.
