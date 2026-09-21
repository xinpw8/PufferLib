# Normalized points and confirmed falls reward validation

Implemented helper: `../normalized_reward.h`.

Mode: `normalized_points_falls_v1`.

For one step, `raw = own_awarded_points - opponent_awarded_points - own_confirmed_fall`.
The returned reward is `clamp(raw / 100, -1, 1)`. A confirmed fall means the recovered
`REK_G1_FALL_EVENT_BECAME_FALLEN` bit. The pure helper does not retain event history:
callers must consume each step's event once, and supply awarded deltas before
round counters reset. `FALLING_STARTED`, `FALLING_CLEARED`, and
`RESET_TIMEOUT_DUE` do not contribute. There is no opponent-fall bonus, repeated
penalty while down, contact reward, or extra terminal reward.

Examples: one point gives +0.01; five points give +0.05; conceding five gives
-0.05; one's newly confirmed fall gives -0.01. Conceding five and becoming fallen
in the same step gives -0.06. If these events occur on different steps, each
contributes only on its own step.

`Result` exposes `reward`, `score_reward`, `fall_reward`, `raw_total`, and
`saturated`. The component rewards are before clipping. `raw_total` is int64,
and normalization uses double before converting to float, avoiding signed-int
subtraction overflow even at integer extremes. Exact raw totals of -100 or +100
do not report saturation. Values outside those limits do.

The generic recovered-combat interface accepts caller-sized contact batches and
does not enforce a common small reward bound at this helper boundary. The scale
of 100 therefore has a safety clamp with an observable flag. This validation
does not claim that clipping is reachable during correctly configured play,
or prove a tighter legal bound across all runtime modes and aggregation windows.

## Executed validation

2026-09-21: WSL Ubuntu 22.04 and Spark CPU runs each passed 20,683,663 checks,
compiled with `-fsanitize=undefined -fno-sanitize-recover=all`. Spark CUDA 13.0
on NVIDIA GB10 (`sm_121`) passed 1,451,552 cases per replay over four CUDA graph
replays. CUDA rewards, components, raw totals and saturation flags matched the
independently calculated expectations exactly as float values. Build and test
stderr were empty.

Coverage includes:

- Every pair of integer deltas from -150 through +150 and all 16 native event
  masks, plus the same masks with all unknown high bits set.
- Integer extremes, clipping thresholds, monotonicity and score antisymmetry.
- Actual recovered detector transitions through two 10 s sample sequences with
  sustained fallen states, repeating timeout events, spawn reset and a second
  fall, plus transient falling followed
  by upright recovery. Each confirmed fall is charged once.
- A 6000-step score/fall sequence with delayed conceded points, opponent fall
  events, idle terminal and reset steps, and a new round's first score.

The helper tests did not launch a trainer or live client. These are reward
arithmetic and event contract tests, not evidence of physical fall dynamics or
authentic-play parity. Training and client evaluation were performed separately.

Reproduction from `native5`:

```sh
bash test_normalized_reward.sh /tmp/new-normalized-reward-cpu-results
REK_NORMALIZED_REWARD_TEST_CUDA=1 bash test_normalized_reward.sh /tmp/new-normalized-reward-cuda-results
```

Each output directory must be new. The runner records compiler information,
source and binary hashes, command lines, build stderr and JSONL results.
Raw executed artifacts remain at:

- WSL: `/tmp/rek-normalized-reward-cpu-20260921-a`.
- Spark: `/tmp/rek-normalized-reward-20260921.MSkagc/results`.

## Preserved evidence

On 2026-09-21, 36 files were preserved in the verified existing NAS project.
The source files, CPU/CUDA binaries, compiler output, test output, and provenance
were copied into a fresh destination without overwriting existing files. Each
source was hashed before and after copying, and the destination hash was checked.
The prior project folder `pufferlib/rek-evidence/2026-09-19` was verified first.

- [NAS helper test archive](<//192.168.0.19/MyShare/pufferlib/rek-evidence/2026-09-21/normalized-falls-r1/helper-tests>)
- [NAS archive receipt](<//192.168.0.19/MyShare/pufferlib/rek-evidence/2026-09-21/normalized-falls-r1/helper-tests/archive-receipt.json>)
- [Local archive receipt](<C:/rekagent/work/normalized-falls-20260921-r1/helper-tests/archive-receipt.json>)

Archive receipt SHA-256:
`cf3c71069a7f72b66ae7eac76bcb6c4b993c7349ce63dcae4aa33f32dc987813`.
The receipt lists the other 35 preserved files and their verified hashes. Its
own hash was verified separately against the NAS copy.

Validated source SHA-256 hashes:

```text
14cc963f8b07230da3c0390d0c0ff9e6d671bba3d762ee157056211d95745981  normalized_reward.h
d8adceae922eaf239e6cf38143e8a76bfd8d1084cab24f22033ed643d9b162b3  test_normalized_reward.cpp
923675fedf43a0b676c4cfdd1f9b352370374dceb987aae6b54bb4ef45f28a5e  test_normalized_reward.cu
0f77a74ec27e82fd8ad3a52b6cbfccdce9c33995eac09476881d6c9bd8df3c2c  test_normalized_reward.sh
```
