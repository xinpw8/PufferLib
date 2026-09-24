# Authentic evaluation status

As of 2026-09-24 05:35 UTC, the updated checkpoint has not been evaluated in REK. No improvement in authentic win rate is established.

- Checkpoint SHA256: `056818f3947c3e7efb1a8050521cfed72f6147c2444b21b02c7aa7ef0166b5d7`.
- Prepared stage: `/home/spark-advantage/rek-training/authentic-ppo-live-20260924-r1`.
- Frozen policy seeds: 801 through 820.
- Opponent: authentic private sparring Bot 1, difficulty 0.
- Observed-round target: 18 wins out of 20; stop if three nonwins make that target impossible. This finite-sample target does not prove a 90% population win rate.
- Incomplete attempts are reported separately; completed losses cannot be discarded.
- All 223 observation features retained. No forced attacks, range gates, attack restrictions, or cooldown overrides.
- Fresh isolated Spark X98 client after each counted round, baseline `STRONGMEM=2`, default weak-barrier setting.
- Windows desktop input remains untouched. MP4 delivery files must be below 20 MB each.

The preceding client attempted automatic cached authentication but its refresh was rejected with HTTP 400. Its normal recovery path clears that session. The subsequent private-room entry failures report a missing bearer token. A visible Home or Free Play screen does not establish authentication. The rejection's underlying cause is unknown.

A fresh isolated client and loopback-only noVNC tunnel are ready for normal account login. Email-link confirmation is required before the campaign can start. No credential is included here; no other client's credentials will be transplanted.

The real-data PPO update includes measured point changes, including +5 count-out awards, and terminal outcomes. It does not repair or establish compact-simulator contact or fall parity. Five developmental rounds also do not establish generalization. Fresh authentic results are required before promoting this checkpoint or using it to justify a larger sweep.
