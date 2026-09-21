# Warp-cooperative geometry-pair contacts

The production CUDA runtime measured 934,428.86 full-training transitions/s
with legacy sphere-proxy velocity and 916,844.42 with body-cvel velocity.
Those are +25.01% and +26.55% against their respective serial references.
Both complete 33,554,432-transition runs produced byte-identical final
checkpoints to their serial counterparts. This is a performance improvement
within the compact simulator, with no new fighting-strength result.

## Measured bottleneck and exact change

The preceding [kernel profile](../contact-kernel-profile-20260920/README.md)
found that geometry-pair mode added 1,156.41 ms to `fast_step` in its measured
ROI, while learner and inference kernel totals stayed approximately constant.
The environment still assigned a warp to each arena, but lane 0 performed all
108 pair histories per attacking side serially.

`geom_pair_contacts_warp` now assigns one limb to each of lanes 0 through 5.
Each lane preserves the original zone, striker and temporal-sample order,
shape arithmetic, broad-phase predicate and apex shortcut. Each owns disjoint
pair bits, computes its history XOR delta against the initial words, and
writes it to per-warp shared storage. Lane 0 merges the six disjoint deltas.
There are no atomic updates and no reordered floating-point reductions.

Scorer mutations remain serial in their original order. Legacy velocity keeps
the original limb-speed maximum and zone aggregation. Body-cvel stores each
entered pair's own relative speed, then lane 0 replays limb, zone and striker
order. The first qualifying entered pair wins the same-apex dedup exactly as
before; slow rejections cannot lend speed or consume another pair's cooldown.
Side 0 finishes before side 1. The body-velocity bake and optional allocations
from `d1057422` remain unchanged.

`advance_arena_warp` retains lane-0 action selection, movement, round updates
and rewards, with uniform warp synchronization around contacts. Its input-error
early return remains at the original point; an error arising later in bot
motion still follows the original finalization path. Geometry-pair mode selects
the cooperative kernel. Default legacy limb-contact mode retains `fast_step`.
No time step, contact sampling rate, action cadence, observation or reward was
changed.

## Equivalence tests

All executed production tests passed:

- Full-warp contact oracle: 18,470 checks, 256 transitions, 385 idle calls,
  127 active calls, 63 invalid-apex calls, 21 resets and 240 awarded points.
  The oracle remains the original full-sampling function from `2981e7c8`.
- Whole-arena flow: 10 scenarios and 10,364 comparisons covering normal ticks,
  reset waits, terminal/reset transitions, invalid input, existing errors,
  bot recovery errors, held yaw and new attacks. Arena, results, logs and action
  arrays match the serial implementation byte-for-byte.
- Full-warp body-cvel: the original 16 scoring checks, four additional
  ordered-pair/same-apex assertions and 11,249 reference comparisons. Every
  contact invocation compares the entire Arena with the serial scorer.
- Existing body composition: 1,080 cases; existing body runtime/allocation
  checks, contact-entry 16, apex-history 18,470, autoreset 592 and recovered
  scoring 6 host / 12 production / 36 isolated cases all passed.
- Default, geometry-pair, explicit legacy-velocity and body-velocity real-asset
  snapshots match the corresponding archived references. All snapshot files
  hash to `d23177ac2344ddaf476cc369bcfd36d03b33da0a0e6666ccd7267eb91e8341a0`.

These checks establish the sampled optimization equivalence. They do not prove
universal CUDA determinism or authentic REK physical/contact parity. Synthetic
contact-rich fixtures cover behavior absent from the short real-asset snapshots.

## Headless training measurements

All rows use 33,554,432 learner transitions on `spark-4ae3`, 512 arenas,
horizon 512, minibatch 8192, 128 epochs, 50 Hz, stride 1, eight contact substeps,
keyboard-reset yaw, recovered Bot1, rendered-pose observations and identical
round-outcome reward/PPO settings. Initial checkpoint SHA256 is
`f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4`.
The native C++/CUDA learner and simulation use no Python runtime. Build flags
include `-std=c++17 -O3 -arch=sm_121`. These are single ordered measurements,
not a repeated statistical benchmark or environment-only throughput.

| Build and velocity mode | Training SPS | Loop s | Process s | Startup-inclusive SPS |
| --- | ---: | ---: | ---: | ---: |
| Serial apex, sphere proxy | 747,475.67 | 44.890334 | 45.57 | 736,327.23 |
| Private warp prototype, sphere proxy | 956,306.33 | 35.087535 | 35.79 | 937,536.52 |
| Integrated warp, sphere proxy | 934,428.86 | 35.909028 | 36.63 | 916,036.91 |
| Serial body-cvel | 724,505.09 | 46.313591 | 47.01 | 713,772.22 |
| Integrated warp, body-cvel | 916,844.42 | 36.597738 | 37.32 | 899,100.54 |

Sphere-proxy runs match checkpoint
`07fdce5f1d14833103189ad424ff2c6f25d37166e259a9e3ba9230680cb61766`,
4493 wins / 578 losses / 49 ties and 355849:215160 training points.
Body-cvel runs match checkpoint
`08f717ff7945ea0a75779ca5e48ed99bbbf244956c5753e1ffa3136f04435249`,
4585 wins / 479 losses / 56 ties and 337613:174580 training points.
All exits and failure bits are zero. These are changing-policy training counts;
they are not held-out authentic fighting results. No new actor needs promotion
from this equivalence-only optimization.

The final logged interval, which is not an accumulated kernel profile, changed
as follows. Rollout includes inference and environment work. The separate
`eval_model`, `eval_env` and `eval_copy` fields remain uninstrumented zeros.

| Final interval ms | Serial sphere | Warp sphere | Serial body | Warp body |
| --- | ---: | ---: | ---: | ---: |
| Rollout | 292.813 | 223.332 | 302.914 | 227.987 |
| Learner model | 58.464 | 59.251 | 61.199 | 59.449 |
| Learner miscellaneous | 1.257 | 1.266 | 1.249 | 1.251 |

Static integrated `fast_step_warp` resources are 211 registers, 416 stack bytes
and 5,056 shared bytes per 128-thread block. The retained serial `fast_step`
has 219 registers, 400 stack bytes and no shared bytes. The earlier standalone
warp prototype had 209 registers, 416 stack bytes and 1,600 shared bytes.
These static counts do not measure achieved occupancy. Learner kernels remain
48 registers for PPO loss and 40/39 for minGRU forward/backward.

Integrated sphere training ran from 2026-09-21 01:27:09.402846 UTC through
01:27:46.181465 UTC. The following body run finished at 01:28:34.833275 UTC.
GPU ownership was released immediately afterward. No live REK overlapped.

## Reproduction and retained artifacts

Production source and the three new fixtures were frozen together with matching
body-cvel asset source in
`/home/spark-advantage/rek-training/contact-warp-body-20260920-r1`.
The complete stage contains `build-runtime.sh`, `verify-runtime.sh`,
`run-training.sh`, all exact commands, native executables, stdout/stderr,
configurations, checkpoints, hashes and timing/resource output. The run script
takes `legacy_sphere_proxy_v1` or `body_cvel_v1`; it retains separate outputs.
`timing-measurements.json` includes the exact input hashes for the table above.

Executed on Spark through Windows WSL SSH, with each invocation's stdout/stderr
redirected to its separately retained stage logs:

```bash
stage=/home/spark-advantage/rek-training/contact-warp-body-20260920-r1
bash "$stage/build-runtime.sh"
bash "$stage/verify-runtime.sh"
bash "$stage/run-training.sh" legacy_sphere_proxy_v1
bash "$stage/run-training.sh" body_cvel_v1
```

The frozen training script sets learning rate 0.0001, entropy 0.01,
gamma 0.9998844821426083, lambda 0.9978673240629938, training seed 419 and the
existing base seed 73; randomized gaps 0.55 to 2.5 and headings plus/minus pi;
120 s rounds, zero shaping, no feature mask and no frozen-policy opponent.
Geometry gaps are the existing simulator's units, without a new physical
calibration claim. A preservation guard prevents silently overwriting a run.

The original prototype is independently preserved at
`/home/spark-advantage/rek-training/contact-warp-20260920-r1`.
Its README records CPU-only preparation and predates execution; the retained
run outputs and this report provide the completed result.

Private Windows mirrors are under
`C:\rekagent\work\consistent-fighter-20260919-r1\contact-warp-r1` and
`C:\rekagent\work\consistent-fighter-20260919-r1\contact-warp-body-r1`.
Complete archives include per-file SHA256 manifests, verified after transfer:

- Prototype archive: `cd000f48d2d1fb6c8534d41d680e5556fc849fcbcd2aff8688533bcd4f7f8a74`.
- Integrated archive: `838c065ae29a814b90cb80f2fdb34cf9a3d3400408f1a596e16424a0bb44b144`.
- Integrated trainer: `74e321f81b3c219a0b05a6a55b9d04fdd99a17128e413d7b0f332451c507df51`.
- Runtime source: `1f06197dc9bd333232ed6f09c168e1c6350e502038d82961c5b50efc25a6d589`.
- Contact fixture: `3b4676888e027ef2942b60840533f3679c0973a678b7937a9451c12dac9d3082`.
- Flow fixture: `9a89ad44937440aad95810303a28d88eff1330c5e61228af1ce139202bb756c1`.
- Body fixture: `084ff75e4ecfd2f99752f2ff5a7d5f5e8eac662437af21d2c8b6d02ad8bedfb4`.

All four source hashes match the shared checkout and the exact compiled source
snapshot. Raw private assets and checkpoint files are excluded from Git.

Both complete private result directories and the archival scripts were copied
to `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\native-contact-warp-r1`.
All 21 source files, totaling 56,607,913 bytes, passed source-before,
NAS-readback and source-after SHA256 comparison. Manifest SHA256:
`c95a92a794620e597e52a1b351265394dedac72cd217a97f742cf443002f6e03`.
The initial copy completed before a receipt-writing parameter error; its script
and transcript remain preserved. A separate finalizer reverified every file
and created the missing receipt exclusively, without overwriting any artifact.
