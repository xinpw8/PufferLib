# Why the high scripted win rate failed the AFK test

The exact final V2 and V3 checkpoints were reevaluated on `spark-4ae3`.
Each matrix case contains 1,024 matches: 128 arenas, four rounds per arena,
both fighter sides, BF16 sampled inference, and policy RNG seed 10001.
The runtime objects and checkpoints were preserved. Only the opponent input
source and round duration were varied. All eight runs exited 0 with zero
failure bits. There was no live browser interaction or retraining.

| Checkpoint | Round duration | Opponent | Wins | Losses | Draws |
| --- | ---: | --- | ---: | ---: | ---: |
| V2 33.6M | 20 s | Original approaching script | 1007 | 4 | 13 |
| V2 33.6M | 20 s | Neutral, stationary | 0 | 0 | 1024 |
| V3 33.6M | 20 s | Original approaching script | 1021 | 0 | 3 |
| V3 33.6M | 20 s | Neutral, stationary | 0 | 0 | 1024 |
| V2 33.6M | 300 s | Original approaching script | 1024 | 0 | 0 |
| V2 33.6M | 300 s | Neutral, stationary | 7 | 0 | 1017 |
| V3 33.6M | 300 s | Original approaching script | 1023 | 1 | 0 |
| V3 33.6M | 300 s | Neutral, stationary | 21 | 0 | 1003 |

Both 20-second stationary tests produced exactly zero points across all
matches. The original 20-second scripted results reproduced exactly. These
are sampled action trajectories at fixed starting geometry, not a randomized
opponent or initial-state suite. The stationary opponent receives external
neutral category 1 every tick, verified on the GPU; policy rows are preserved.

An independent trace used the actual native human-evaluation worker, policy
seed 73, the trained policy on blue, and neutral orange. V2 started eight
attacks in 20 seconds, spent 966 of 1,000 ticks busy, and traveled only
0.017455 m while the gap remained at least 1.797572 m. V3 started nine attacks,
spent 984 ticks busy, and traveled 0.003972 m. Both scored zero. Independent
300-second traces also ended 0:0 despite 108 V2 or 109 V3 attack entries.

The source explains why this can succeed against the benchmark: the scripted
opponent always approaches when the gap exceeds 1.05 m. It brings itself into
range of an opponent that attacks near its starting position. Training rewards
net points against this same script. A legal attack is not conditioned on its
target being within contact range. These observations support overfitting to
the pursuing opponent and demonstrate failure to approach an idle target.

The 300-second viewer also exposes timer observations outside the 20-second
training range. That is a separate distribution change. It cannot explain
away the AFK failure because the failure already occurs at 20 seconds. The
matrix changes both episode duration and timer observations, so it does not
isolate their individual effects.

The 98.34% V2 and 99.71% V3 numbers remain exact results against the original
script. They do not establish general fighting competence. Corrective training
needs varied opponent behavior and initial states, with held-out AFK pursuit,
retreating opponents, and checkpoint-versus-checkpoint results. No replacement
policy was trained or deployed by this diagnostic task.
