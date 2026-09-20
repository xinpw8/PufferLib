# Authentic PPO: second MC iteration

On 2026-09-20 UTC, two independent one-epoch candidates were trained from the
first authentic MC checkpoint on three subsequent completed development
rounds: r11, r13 and r15. They supplied 17,360 ordered decisions and two wins
plus one loss. This report records training and numerical diagnostics only.
Neither candidate's fighting efficacy is established here. No holdout split
or win-rate improvement is claimed.

## Chronological data and behavior binding

| Training round | Points, policy : bot | Outcome | Decisions |
|---|---:|---|---:|
| r11 | 11:9 | Win | 5,810 |
| r13 | 20:12 | Win | 5,788 |
| r15 | 13:21 | Loss | 5,762 |

All three rounds used the same first-iteration MC checkpoint, native BF16
H256/L2 worker, sampler seed 73, and unmasked 223-feature input. Exact native
replay reproduced all 17,360 saved sampled actions, with zero mismatches.
Frozen behavior probabilities are FP32 native sampler outputs. Of the
17,360 rows, 17,357 have actor weight one; three final ownership-race requests
have actor weight zero but retain their recurrent input and terminal reward.
The original four training rounds came from a different behavior checkpoint.
They were not silently mixed into this new on-policy batch. GAE-checkpoint
rounds and future evaluation rounds were also excluded.

The chronology is original simulator checkpoint, first authentic MC update,
three new rollouts from that MC policy, then this second update. New rollout
data includes the 13:21 loss rather than selecting only wins. These remain
development episodes; each decision is not an independent evaluation sample.

Reference discounts are gamma `0.9998844821426083` and lambda
`0.9978673240629938` per 20 ms, exponentiated using each actual source-to-next
QPC duration. Complete-MC targets do not use lambda or the learned critic.
Reward remains terminal outcome plus the discounted score-potential change,
with terminal potential zero. Advantages equal complete shaped returns;
value-loss coefficient is zero and advantages are unnormalized.

An initial PowerShell invocation serialized numeric command-line arguments to
15 decimal digits. The wrapper's exact manifest check caught this before any
GPU work. That export was preserved; a fresh `-r2` export passed the exact
decimal strings as quoted arguments. The two resulting float32 dataset
binaries are byte-identical, both with the dataset hash below. Training used
the corrected r2 manifest; no acceptance threshold was loosened.

## Configuration and safeguards

Both candidates independently loaded the same unchanged MC checkpoint with
fresh optimizer state. They did not continue from each other. The only
configuration difference was learning rate: `1e-5` versus `3e-5`. Both used
one epoch, horizon 128, PPO clip 0.2, value clip 0.2, value coefficient zero,
entropy coefficient 0.001 and 138 optimizer updates.

The existing training order completes each round before the next: win, win,
loss here. Each chunk independently reconstructs its current-weight recurrent
prefix. This preserves sequence context but can make the optimizer path
episode-order-sensitive. A future round-robin chunk-order control could reduce
long single-episode update stretches without mixing recurrent states. It was
not implemented or tested here, and this is not identified as a Puffer shuffle
bug. The present comparison changes learning rate only.

The existing `build-r5/authentic-ppo` was reused without source changes or a
new build. Both runs used the already-supported options
`--allow-bounded-bf16-batch --targets=complete-mc-zero-baseline`.
CUDA complete-return targets matched the CPU reference with maximum error
zero. Target mean/std were `0.371258394723` / `0.489604927003`; range was
`[-0.721674799919, 1.21195423603]`.

Initial existing Puffer sequential forward matched every frozen teacher logit
and value exactly. Initial batch-forward mean/max legal KL were
`1.34143118816e-7` / `0.000126276976587`; maximum chosen-ratio deviation was
`0.0168326013282`, below the unchanged explicit `0.1 * clip` limit of 0.02.
Initial clipped fraction was zero. This is accepted bounded BF16 batch
arithmetic, not exact batch parity. Frozen behavior log probabilities remained
unchanged throughout optimization. All replay/training processes exited zero,
and their three stderr files were empty. The initial checkpoint hash remained
unchanged after both runs.

## Measured final updates

These are full-training-batch forward diagnostics against frozen behavior,
after the final update. They are not game results or sequential live-policy
efficacy measurements.

| Metric | LR 1e-5 | LR 3e-5 |
|---|---:|---:|
| Mean legal KL, old to current | 0.000724616954528 | 0.00538277385856 |
| Maximum legal KL | 0.0167804696572 | 0.284779432673 |
| Chosen-action approximate KL | 0.000729870371602 | 0.00543271359584 |
| Chosen ratio minimum | 0.782867369807 | 0.261191194466 |
| Chosen ratio median | 0.994337913497 | 0.984411001817 |
| Chosen ratio p95 | 1.08105674842 | 1.23361869383 |
| Chosen ratio maximum | 1.31723848826 | 2.77905764245 |
| Clipped fraction, all rows | 0.000288018433180 | 0.0773617511521 |
| Value MSE to MC target | 122.845265926 | 123.887484690 |

The higher learning rate caused a materially larger distribution change.
Value MSE is diagnostic only because value loss is disabled; actor updates
still alter the shared representation. Neither training loss nor KL selects
the better fighter without subsequent authentic evaluation.

| Process | Start UTC, 2026-09-20 | End UTC | Timestamp window |
|---|---|---|---:|
| Exact behavior replay | 01:53:23.794808884 | 01:53:27.000421295 | 3.205612411 s |
| LR 1e-5 | 01:53:27.003607778 | 01:53:29.300044906 | 2.296437128 s |
| LR 3e-5 | 01:53:29.305053009 | 01:53:32.603022672 | 3.297969663 s |

These process windows include validation, diagnostics and checkpoint writes.
`/usr/bin/time` reported 3.20, 2.29 and 3.29 s, respectively. This small update
is not a full-training throughput benchmark. The reserved GPU work ended at
01:53:32.609774747 UTC and the reservation was released. No game was controlled
during training, and the unrelated persistent GPU process was untouched.

## Exact artifacts

| Artifact | SHA256 |
|---|---|
| Exported dataset | `b5cbc0b8de82df0d02036d069e505f0ae60825a8c77c1e44eaa8ac15037705b3` |
| Frozen behavior replay | `f583516f3dd7410a2b7a8522d307123f2b4af15b2fab0376d42d16b3aca3cca9` |
| Initial first-iteration MC checkpoint | `a985d6c06b5dfab7198319caf5e99099b85da30eb40402758cc254e407e2059f` |
| LR 1e-5 final checkpoint | `bc091d47cf7097b22ccd25091e5748e0726326df5a7436b7c649c845f7f3eb59` |
| LR 3e-5 final checkpoint | `70db296884aa7fb1da195df8b1432b40fd0de5f2e089e40aabff90d86d1c286c` |
| Native PPO trainer | `b5c2c61932ad63c4dc40c137a6ef493a423826306aebf83859efdb5c0197da1a` |
| Exact native replay executable | `b44ee7d100bf9719eb2543f665b5ccbc592fced2427af374064542016b2b59ec` |
| Exporter source | `8d4ff8d3862a137b9900157a33a3688256733d937865481b88ebb8b8bd638838` |
| Corrected export manifest | `64a4605ffdfa718c8a73327faf9fdf5a62c27b691428abf10115e3df1992fbe2` |

Private Spark stage:
`/home/spark-advantage/rek-training/authentic-trajectory-mc-iteration2-20260919-r1/`.
Final candidates are `train-mc-lr1e-5-r1/ppo-one-epoch.bin` and
`train-mc-lr3e-5-r1/ppo-one-epoch.bin`. Each `.epoch-1.bin` has the same hash as
its final checkpoint. Replay is `replay-r1/behavior-replay.bin`; dataset and
manifest are under `data-r2/`.

Exact commands are preserved in `run-spark-mc-iteration2.sh` and expanded in
`commands-and-stderr.txt`. Full stdout is `commands-and-stdout.txt`; each stage
also retains separate stdout, stderr and `/usr/bin/time` output. Combined
stdout/stderr SHA256 values are respectively
`c17d5f23bb00c881b0baa9585dd4602dc7b70bb3e3634b28cfeee01315d7ee30`
and `89b7316b271404189ef26bbb6c9c0a40f96c28c9e32110dc7dd89a5e3f51a3b5`.

The complete Spark stage was copied to
`C:/rekagent/work/consistent-fighter-20260919-r1/authentic-trajectory-mc-iteration2-r2/spark-results/`.
All 22 remote files, totaling 36,483,952 bytes, matched their local copies by
size and SHA256. The sorted path/size/hash JSON manifest digest was
`1f93abe1e5cc8b073736a87404d587e84f13b7f80ec80b21616d1ec8f12d26d2`.
Both local export scripts and their transcripts are preserved alongside the
export directories. No binary checkpoint or raw trajectory is included in
this public report.
