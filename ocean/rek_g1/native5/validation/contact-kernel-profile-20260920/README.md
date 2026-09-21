# Matched compact contact-mode GPU profile

The added cost is inside the fused environment kernel. In the same executable,
geometry-pair handling increased measured `fast_step` duration by 43.4%, while
learner inference and PPO kernel times were nearly unchanged. Its extra
1,156.407 ms accounts numerically for 96.8% of the 1,194.237 ms difference between
the two profiled training intervals. This localizes the regression to environment
execution; it does not isolate an individual contact instruction or establish
achieved occupancy.

## Controlled workload

Host `spark-4ae3`, GB10, native PufferLib 5.0 headless C++/CUDA training. Both arms
use the complete frozen apex executable and matching assets from
`/home/spark-advantage/rek-training/contact-apex-20260920-r1/fast-build-r1`, SHA256
`5c8ab4a26bf8563a52a8b8f8dc5ffad3bf059b772b31d9ac4d3f5a338aaaba5f`.
No runtime, assets, trainer or global system settings were changed.

Only `REK_FAST_CONTACT_ENTRY` differs: `legacy_limb_union_v1` versus
`geom_pair_v1`. Each run starts from f3 checkpoint
`f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4`, with fresh
optimizer, 4,194,304 learner transitions, 512 arenas, horizon512, minibatch8192,
120 s rounds, environment seed419/base seed73, keyboard-reset yaw, stride1,
eight contact substeps, recovered Bot1, rendered-pose observations, round-outcome
reward, no feature mask or shaping, learning rate0.0001, entropy0.01,
gamma0.9998844821426083 and lambda0.9978673240629938. The full settings and expanded
commands are retained privately. No Python executes in training or analysis.

The workload preserves the current horizon; the older `fast_profile.sh` recipe
hardcodes horizon16 and is not used. The private analyzer changes only the
import path and workload metadata of `fast_profile.mjs`; its exact-function,
stream-attribution and region-of-interest formulas remain unchanged.

## Full learner throughput versus tracing overhead

| Measurement | Legacy limb entry | Geometry-pair entry |
| --- | ---: | ---: |
| Unprofiled whole-training SPS | 990,304.56 | 770,657.08 |
| Unprofiled training-loop seconds | 4.235368 | 5.442504 |
| Unprofiled process seconds | 4.93 | 6.13 |
| Unprofiled startup-inclusive SPS | 850,771.60 | 684,225.77 |
| Traced whole-training SPS | 961,858.29 | 744,334.51 |
| Traced steady-state ROI SPS | 959,567.93 | 743,027.13 |

These are short single runs, not a repeated-run maximum-throughput estimate.
The same-binary geometry-pair arm is 22.18% lower in unprofiled learner SPS.
The earlier 33.55M results, approximately 929k and 747k, have longer workloads
and are not interchangeable with these 4.19M measurements. Geometry-pair and
legacy contacts deliberately differ semantically; faster legacy results are not
an argument to discard the contact correction.

Unprofiled runs executed legacy then geometry-pair; traces executed geometry-pair
then legacy. In each arm the profiled final checkpoint equals its unprofiled
counterpart byte-for-byte. The arm checkpoints differ, as expected when contact
semantics change. The identical initial state and controls do not force later
trajectories or learned actions to remain identical across arms.

## Actual GPU timeline

Nsight Systems 2025.3.2 used CUDA graph node tracing. Analysis excludes the first
complete PPO epoch and ends after the last complete epoch. Both traces contain
16 complete epochs, 15 measured epochs, exactly 7,680 learner sampling kernels
and 7,680 `fast_step` calls, representing 3,932,160 learner transitions. There
are no missing-sample warnings. Opponent updates and contact samples do not
multiply the SPS denominator.

| Summed GPU kernel time in ROI | Legacy milliseconds | Geometry-pair milliseconds |
| --- | ---: | ---: |
| Exact fused `fast_step` | 2,664.288 | 3,820.695 |
| Verified PPO streams | 505.968 | 504.124 |
| Learner inference and rollout I/O | 367.967 | 368.833 |
| Unclassified streams | 348.869 | 379.571 |
| ROI wall time | 4,097.844 | 5,292.081 |

The fused step averages 346.913 versus 497.486 microseconds per batch of512
arenas. Its share of summed kernel time is 68.54% versus 75.31%. Unclassified
time remains explicitly unclassified: 8.98% versus 7.48%. Exact kernel naming
isolates `fast_step`; the existing disjoint-stream checks separate learner
inference and PPO without attributing every unknown GEMM by guesswork.

Kernel interval union covers 94.27% versus 95.40% of ROI wall time. That is
timeline coverage, not SM utilization. Kernel durations can overlap; summed
time and GPU-uncovered wall time must not be interpreted as exclusive CPU
compute or added to synchronizing CUDA API durations.

Both arms launch `fast_step` with grid128, block128, 211 registers per thread,
zero static/dynamic shared memory and zero reported local-memory bytes in the
CUPTI launch record. Static object inspection reports a400-byte stack in the
same executable. The reported local-memory field does not prove zero stack
traffic. Same executable and register allocation exclude a changed register
allocation as the reason for this particular two-arm timing difference.

## Occupancy limitation and bounded run accounting

Nsight Compute 2025.3.1 is installed. The driver reports
`RmProfilingAdminOnly=1`; noninteractive sudo is unavailable. One attempt used
`LaunchStats` and `Occupancy`, filtered `fast_step`, skipping1,024 matching
launches and requesting8, with clock/cache control disabled. It timed out at60 s
with exit124 before yielding a report. The timeout cause is not established.
Achieved occupancy and hardware-counter bottlenecks remain unknown. No retry,
permission change, driver setting change or clock adjustment was made.

All four training runs succeeded. The fifth, counter-read attempt is retained
as incomplete and supplies no throughput result. The allocation ran from
2026-09-21 00:52:26.480205 UTC to00:53:55.770663 UTC; the GPU was then released
before CPU-only export/analysis. No policy from this performance test is promoted
and no authentic fighting-strength claim follows from its compact training score.

## Reproduction and preservation

Private stage: `/home/spark-advantage/rek-training/contact-kernel-profile-20260920-r1`.
`run-arm.sh` pins the exact executable, f3 and training controls. `capture.sh`
contains the four-run order and bounded NCU command. The Nsight invocation is:

```sh
nsys profile --trace=cuda,nvtx,cublas --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none --backtrace=none \
  --cuda-memory-usage=false --stats=false --force-overwrite=false \
  --output="$output/training" \
  bash "$stage/run-arm.sh" "$mode" "$output/run"
nsys export --type=sqlite --force-overwrite=false \
  --output="$output/training.sqlite" "$output/training.nsys-rep"
node "$stage/profile.mjs" "$output/training.sqlite" "$output/summary.json"
```

Complete private Windows mirror:
`C:\rekagent\work\consistent-fighter-20260919-r1\contact-kernel-profile-r1`.
`comparison.json` retains exact metrics, category attribution, launch dimensions,
within-arm checkpoint comparisons and source trace/command hashes. The complete
stage archive includes commands, stdout/stderr, traces, SQLite, configurations,
checkpoints, all exit statuses and a per-file manifest. Its transfer was verified:
`contact-kernel-profile-20260920-r1-results.tar.gz` SHA256
`5c1fdb8484a8864c4df34bcdf3e0a9608bf94a71b97bf697baab6825c56e2568`.
Raw traces, assets and policies are not published in this repository.

The complete private mirror is also preserved under
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\native-contact-kernel-profile-r1`.
All nine files, totaling 32,776,272 bytes, passed source-before, source-after
and NAS readback SHA256 comparison. Archive manifest SHA256:
`c931eadb6df6011e9949c9cacc831af6b3166615807c7e06b2ff1a32f6ed1063`.
