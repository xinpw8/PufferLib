# Idle endpoint contact sampling experiment

The idle shortcut preserved the measured behavior exactly but produced no
demonstrated full-training speed gain. It is not promoted as a standalone
performance improvement.

The change applies only inside `geom_pair_v1`, after the unchanged broad-phase
predicate. With `strike_active == false`, entry/intersection flags and their
velocity proxy are discarded. The original sampler retains only the overlap at
`t=1`, so one endpoint overlap replaces nine temporal samples while preserving
all 108 geometry-pair histories. Active full sampling, score rules, interpolation,
pair identity and reset behavior remain unchanged.

The independent source snapshot is retained in
`/home/spark-advantage/rek-training/contact-idle-20260920-r1/source`.
It contains the 12-line production change and its fixtures, based on commit
`2981e7c8`. Later optimization experiments must use a separate stage.

## Equivalence and measured performance

CPU checks ran on `spark-4ae3`, aarch64, with GCC 13.3.0 and
`g++ -std=c++17 -O3`. All 74,638 checks passed across 108 pairs and 256 ticks,
including all primitive type combinations, 3,265 transient idle crossings,
2,512 broad rejects and 6,162 later active entries. Recovered scoring produced
168 points with identical cooldown/dedup state. Compile wall time was 0.45 s;
test wall time was 0.03 s. Both exited zero.

The CUDA production comparison passed 18,470 checks across 256 transitions,
385 idle calls, 127 active calls, 474 scored points and 21 resets. Its reference
is the full-sampling function copied from `2981e7c8`, with only its name changed.
Both pair-history words, per-side points, cooldown/dedup state and last-hit
fields matched after every call, including route changes and subsequent active
contacts. The existing 16-case contact-entry fixture also passed.

Both default and geometry-pair real-asset fixtures matched their archived
reference byte-for-byte across 666 snapshots per invocation. All four snapshot
files have SHA256
`d23177ac2344ddaf476cc369bcfd36d03b33da0a0e6666ccd7267eb91e8341a0`.
The synthetic production fixture supplies the contact-rich cases that the
snapshot fixture alone does not establish.

| Full-training measurement | Original geometry-pair sampling | Idle endpoint shortcut |
| --- | ---: | ---: |
| Learner transitions | 33,554,432 | 33,554,432 |
| Training-loop seconds | 45.493099 | 45.748631 |
| Full-training transitions/s | 737,571.91 | 733,452.16 |
| Process wall seconds | 46.20 | 46.45 |
| Startup-inclusive transitions/s | 726,286.41 | 722,377.44 |
| Changing-policy W/L/T | 4493 / 578 / 49 | 4493 / 578 / 49 |
| Changing-policy awarded points | 355849:215160 | 355849:215160 |
| Failure bits / exit | 0 / 0 | 0 / 0 |

The ordered comparison measured -0.56% SPS. A single pair does not establish a
slowdown; it provides no evidence that this shortcut recovered throughput.
Static CUDA resource inspection reports `fast_step` register usage of 223 in
the original build and 207 in the idle-shortcut build, with stack size 400 bytes
in both builds. This did not yield a measured
training-speed improvement.

The final checkpoints are byte-identical, SHA256
`07fdce5f1d14833103189ad424ff2c6f25d37166e259a9e3ba9230680cb61766`.
This is evidence for equivalence on this complete training run, not a claim that
every CUDA training run is deterministic or that the environment matches REK.

Both runs use the same f3 initial checkpoint, 512 arenas, 512-step horizon,
minibatch8192, 50 Hz, 120 s rounds, keyboard-reset yaw, stride1, eight contact
substeps, recovered Bot1 and identical native trainer settings. The optimized
run began 2026-09-21 00:26:24.770664 UTC and ended 00:27:11.357519 UTC.
No policy promotion or new authentic evaluation is justified by this unchanged
checkpoint. Native training and testing used no Python runtime.

## Reproduction and retained artifacts

Private Windows directory:
`C:\rekagent\work\consistent-fighter-20260919-r1\contact-idle-r1`.
The same-named Spark stage retains `build-cpu.sh`, `build-gpu-fixtures.sh`,
`verify-runtime.sh`, `run-training.sh`, all stdout/stderr, compiler flags, elapsed
times, verification snapshots, build manifests and checkpoint comparisons.
Builds use the existing native scripts and `nvcc -std=c++17 -O3 -arch=sm_121`.
The complete frozen stage was mirrored into the Windows private directory as
`contact-idle-20260920-r1-results.tar.gz`, verified after transfer with SHA256
`05b20db6402c83a53546355d833d5ea4bc5132df3b1aa4885e72ad1950a3319c`.
It includes a per-file SHA256 manifest. The later
[apex shortcut report](../contact-apex-20260920/README.md) compares the existing
timing and register outputs across all three independent builds.

Important SHA256 values:

- Reference trainer: `58ebcd3d0447dae0b1ca9c2a9cec9c1f9c68b43f88dc8700e8950e3ac19cf979`.
- Idle trainer: `91e4d59644dcf4d3775f6d981f12488ea82ea16188e3167d3e5d9eb7fa94962b`.
- Idle runtime source: `5fa1895b104977c03ad25efb15212855b8c54cd36369ca3791e9c3dfa8dd9e3c`.
- Idle contact helper: `fc61c1b5122b21edc92f86a49bf0ae628bfb2046d0587890754bf4b0e94917a8`.
- CPU executable: `0be0cc981363012ef94a7b16d64c8900eefe030d9e54d1bd80b89f14fc9e4553`.
- CUDA equivalence executable: `1c2c33b579768fbd5e2ee777554e8e948b0e0391fdcf6fcaab9952c3150a012a`.
