# Authentic evaluation status

As of 2026-09-24 07:03 UTC, the updated checkpoint completed two authentic G1-versus-G1 Bot 1 rounds: **0 wins, 2 losses, 16:26 points**. The 20-round campaign is incomplete and the checkpoint is not promoted. Eleven incomplete attempts are retained separately.

- Checkpoint SHA256: `056818f3947c3e7efb1a8050521cfed72f6147c2444b21b02c7aa7ef0166b5d7`.
- Prepared stage: `/home/spark-advantage/rek-training/authentic-ppo-live-20260924-r1`.
- Frozen policy seeds: 801 through 820.
- Opponent: authentic private sparring Bot 1, difficulty 0.
- Observed-round target: 18 wins out of 20; stop if three nonwins make that target impossible. This finite-sample target does not prove a 90% population win rate.
- Incomplete attempts are reported separately; completed losses cannot be discarded.
- All 223 observation features retained. No forced attacks, range gates, attack restrictions, or cooldown overrides.
- Fresh isolated Spark X98 client after each counted round. Initially `STRONGMEM=2` with default weak barriers; later game launches and then relay processes used `WEAKBARRIER=0`. These runtime cohorts remain recorded separately; no causal stability improvement is established.
- Windows desktop input remains untouched. MP4 delivery files must be below 20 MB each.

The user completed normal moogleod authentication. Workshop selection was changed from H100 Heavyweight to L100 Lightweight, REK's G1 robot. Both counted rounds verified the actual G1 runtime pairing. Windows desktop input was untouched.

Round s801-retry9 lost 3:12. Round s802-retry2 lost 13:14, including policy +10 and opponent +5 from received referee count-out awards. The remaining observed point increments were 3:9. [Detailed measurements](../authentic-live-20260924-r1/README.md) distinguish scores, count-outs, dispatches, and unknown server playback.

Incomplete causes: three worker startup OOMs, two Wine/CoreCLR client crashes, two malformed relay JSON exits, two cold-start action watchdog expirations, and two unsupported G1-versus-T800 opponent pairings. The last two were initially reported as generic readiness timeouts. The learner remained G1; the server supplied a T800 opponent. No policy input was sent in those attempts.

The controller was stopped between attempts and the isolated Spark client closed. Completed losses and all incomplete records were archived, not reset. The next separately versioned candidate adds measured balance/count inputs and uses the same five original training rounds. Its training completion does not establish live improvement.

The real-data PPO update includes measured point changes, including +5 count-out awards, and terminal outcomes. It does not repair or establish compact-simulator contact or fall parity. Five developmental rounds also do not establish generalization. Fresh authentic results are required before promoting this checkpoint or using it to justify a larger sweep.
