# Apex-ineligible endpoint contact sampling

The exact apex shortcut measured 747,475.67 full-training transitions/s versus
737,571.91 for the original geometry-pair build. This single ordered comparison
is +1.34%; it does not establish a repeatable speed gain or recover the earlier
approximately 929,000 transitions/s. The final checkpoint remained byte-identical.

This is the second independent experiment, after the
[idle-only shortcut](../contact-idle-20260920/README.md), which measured
733,452.16 transitions/s and no demonstrated improvement. Both source snapshots
and all measurements are retained. No other optimization was combined here.

## Exact change and equivalence boundary

For each limb, construct the same `RekG1StrikeIntent`, clip cursor, body part,
side and minimum ramp used by the existing scorer. When there is no strike
intent, or `embedded_strike_intent_apex` returns false, temporal entry flags and
relative speed cannot produce points or mutate scorer state. After the unchanged
broad-phase predicate, evaluate only endpoint overlap and retain that pair bit.
`interpolate_shape(..., 1)` returns the endpoint directly, so this preserves the
original sampler's final contact history without interpolation rounding changes.

An apex-valid limb still uses the original complete temporal sampling and
velocity aggregation. Pair identity, all 108 histories, broad-phase floating
point arithmetic, score rules, cooldown/dedup behavior and resets are unchanged.
There is no cooldown shortcut. Legacy contact mode and default selection are
unchanged. This proves an optimization boundary in the existing compact model;
it does not establish authentic REK contact or physics parity.

The CPU fixture passed 83,831 checks across 108 pairs and 256 ticks. It includes
3,240 apex-ineligible cases, all primitive type combinations, invalid-apex
continuous overlap followed by valid apex without fabricated entry,
invalid-apex transient crossing, and later valid separation/reentry scoring.
All pair bits and score/cooldown/dedup state match the full-sampling reference.

The CUDA fixture passed 18,470 checks with zero failures across 256 transitions,
385 idle calls, 127 active calls, 63 invalid-apex active calls, 21 resets and
240 scored points. Its reference is the original production function from
`2981e7c8`, with only the function name changed. It compares pair histories,
awarded points, cooldown/dedup and last-hit fields after every call, with route
changes and later valid contact. The existing 16-case contact fixture passed.

Default and geometry-pair real-asset fixtures each matched the archived build
across 666 snapshots. All four snapshot files have SHA256
`d23177ac2344ddaf476cc369bcfd36d03b33da0a0e6666ccd7267eb91e8341a0`.
The contact-rich synthetic fixture supplies coverage that these snapshots alone
do not establish.

## Full training and existing timing data

All runs use 33,554,432 learner transitions, the same f3 initial checkpoint,
512 arenas, horizon512, minibatch8192, 50 Hz, 120 s rounds, keyboard-reset yaw,
stride1, eight contact substeps, recovered Bot1 and identical trainer settings.
The native C++/CUDA training path uses no Python runtime. CPU fixture compilation
used GCC 13.3.0 with `-std=c++17 -O3`; CUDA used `-std=c++17 -O3 -arch=sm_121`.

| Measurement | Original | Idle-only | Apex shortcut |
| --- | ---: | ---: | ---: |
| Full-training transitions/s | 737,571.91 | 733,452.16 | 747,475.67 |
| Training-loop seconds | 45.493099 | 45.748631 | 44.890334 |
| Process wall seconds | 46.20 | 46.45 | 45.57 |
| Startup-inclusive transitions/s | 726,286.41 | 722,377.44 | 736,327.23 |
| `fast_step` registers | 223 | 207 | 211 |
| `fast_step` stack bytes | 400 | 400 | 400 |

Static resource counts for learner kernels stayed identical: `ppo_loss_compute`
48 registers, `puff_advantage`40, row-scan minGRU forward40/backward39. Register
counts alone do not establish occupancy or explain the throughput result.

The following is the final recorded training interval from each existing log,
not an accumulated full-run profile. Rollout includes inference and environment
work; the logged `eval_model`, `eval_env` and `eval_copy` fields are all zero and
cannot separate those costs in these runs. Those zeros do not mean zero work.

| Final logged interval | Original | Idle-only | Apex shortcut |
| --- | ---: | ---: | ---: |
| Rollout milliseconds | 299.861 | 301.728 | 292.813 |
| Learner model milliseconds | 58.082 | 58.207 | 58.464 |
| Learner miscellaneous milliseconds | 1.255 | 1.280 | 1.257 |
| Rollout share of logged total | 83.48% | 83.53% | 83.06% |

All three runs produced 4493 wins / 578 losses / 49 ties and 355849:215160 points
while the policy changed during training, with failure bits0 and exit0. These
are training outcomes, not held-out fighting strength. All final checkpoint
bytes match SHA256
`07fdce5f1d14833103189ad424ff2c6f25d37166e259a9e3ba9230680cb61766`.
This establishes equivalence on the measured complete run, not universal CUDA
determinism. There is no new policy to promote or evaluate from this experiment.

The apex run began 2026-09-21 00:38:06.254256 UTC and ended 00:38:51.961014 UTC
on `spark-4ae3`. No further performance experiment was run in this allocation.

## Retained artifacts and hashes

Frozen source, native executable, fixtures, scripts, commands, stdout/stderr,
timings, configurations, source manifests and checkpoints are in
`/home/spark-advantage/rek-training/contact-apex-20260920-r1`.
The unchanged original source was copied before the separate body-velocity
work modified shared asset-bake files. Only the four runtime/helper/fixture
files were overlaid for this build.

Private Windows artifacts are in
`C:\rekagent\work\consistent-fighter-20260919-r1\contact-apex-r1`.
`contact-apex-20260920-r1-results.tar.gz` is a complete frozen stage mirror,
verified after transfer with SHA256
`93ec732cd8a8d119020633ed170be4b8cf6ab3d83b1bf06b298d5b69f80b440f`.
It contains a per-file SHA256 manifest and `timing-resources-summary.json` with
the precise existing-log fields and their input hashes.

- Original trainer: `58ebcd3d0447dae0b1ca9c2a9cec9c1f9c68b43f88dc8700e8950e3ac19cf979`.
- Apex trainer: `5c8ab4a26bf8563a52a8b8f8dc5ffad3bf059b772b31d9ac4d3f5a338aaaba5f`.
- Runtime source: `b3f8d9b7834be0762c5a70e261f18f5698132956e76f9f8acc41989b10ac0bca`.
- Contact helper: `5efe79df0b635a395879b68fee96346de1c3935557f54ba082ee2eac4fe92716`.
- CPU fixture executable: `2a45459f19bc407557d2dc4f6844512326535649729edbf88b18f2b7581a464e`.
- CUDA equivalence executable: `57e79d6cfb3a528cf100624033136a3bc61744d2b29f2d280b54bebbee519483`.

The complete idle/apex private mirrors and their scripts were copied to
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\native-contact-sampling-r1`.
All 27 files, totaling 36,818,663 bytes, passed source-before, source-after
and NAS readback SHA256 comparison. Archive manifest SHA256:
`fe9b34889c3ba44f2b802787575386741f51b562bca22e43ef07bdc939cac286`.
