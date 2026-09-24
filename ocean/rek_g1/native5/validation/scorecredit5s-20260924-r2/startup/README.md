# Live startup variant

Deployed only to `/home/spark-advantage/rek-training/scorecredit5s-live-20260924-r2` at 2026-09-24 08:42:43 UTC. Production source and the earlier cohort were left unchanged.

`live_transfer_run_prewarm_r1.cjs` derives from `../../balance8-authentic-20260924-r1/live-template/live_transfer_run_pairing_r1.cjs` (SHA256 `0666e0567173fb389ff5cce6a50cc93bab12560fd5325418969691b54d5196a1`). The deployed variant SHA256 is `b57654d54a76c80b38c9e44f59f53efb2502959a70dc568835ec5a8f6d238214`.

The driver awaits the encoder manifest and performs bounded read-only source warmup before starting control. It requires two fresh, distinct Unity-frame intervals within 200 ms, neutral input, private AI identity, unchanged round/slot, 0:0 score, and enough time for the existing fair-start criterion. Warmup does not advance the encoder, worker, recurrent state, or RNG. The existing 250 ms freshness requirement and watchdog remain unchanged. The estimated busy-duration action mask also remains unchanged; it is not authoritative server action state.

Six focused tests pass: `node --test startup-prewarm.test.cjs`.

Live result: s1001-retry2 reached 16 locally accepted commands, then failed. Its final action took 1.400 ms of worker time but was consumed on the next Unity frame after 252.6318 ms, exceeding the 250 ms source-age limit. The 18-source segment ran at 4.91 Hz. These data identify delayed Unity consumption, but do not identify its underlying cause. This incomplete attempt was excluded. Warmup has not eliminated cold-client stalls.

The subsequent warm-round attempt, s1001-retry3, completed and won 15:7. This is one result, not proof of consistent superiority.

`../prepare_live.cjs` also selects a staged campaign-controller change: sleep 250 ms, rather than 5000 ms, after an incomplete attempt with an observed terminal round; retain 5000 ms otherwise. The earlier campaign controller is untouched. The selected controller baseline SHA256 is pinned in that preparer. No pairing, score, fairness, or completion criterion is relaxed.
