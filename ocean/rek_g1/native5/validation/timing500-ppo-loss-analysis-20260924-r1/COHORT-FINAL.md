# Closed C2 policy analysis

The C2 cohort closed at 2026-09-24 11:00:34.706 UTC with **2W3L, 49:55 points**. It failed the 18/20 criterion at its third nonwin. This report covers all five completed C2 rounds and compares them with all eight completed parent rounds. It makes no consistent-winning claim. Different seeds and exclusion of incomplete attempts prevent a controlled checkpoint-performance comparison; runtime failure selection remains a caveat.

C2 checkpoint: `c2c4987b268996cd912fe35e6ba5f9b20a94d69fd93e4ef15b6c5fc71ee5e533`.
Parent checkpoint: `5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4`.

## Native scores and request transport

Every selected round passed the existing received-referee audit and independent raw native score decoding, exact scorer/counter-to-relay matching, and terminal score total check.

| Completed C2 attempt | Final points | Ordinary points excluding +5 | +5 award counts | Own armed requests | Recorded source cadence |
| --- | --- | --- | --- | --- | --- |
| s1301-retry7 | 2:6 | 2:6 | 0:0 | 85 | 25.32 Hz |
| s1302-retry2 | 9:11 | 4:6 | 1:1 | 92 | 26.02 Hz |
| s1303-retry3 | 20:14 | 5:9 | 3:1 | 73 | 24.88 Hz |
| s1304 | 12:10 | 2:10 | 2:0 | 78 | 23.88 Hz |
| s1305 | 6:14 | 6:9 | 0:1 | 65 | 25.86 Hz |
| C2 total | 49:55 | 19:40 | 6:3 | 393 | 25.19 Hz pooled |
| Parent eight | 104:93 | 54:58 | 10:7 | 576 | 25.28 Hz pooled |

C2 averaged 78.6 armed requests and 3.8 ordinary points per round; parent averaged 72 requests and 6.75 ordinary points. Both C2 wins lost the ordinary-point comparison and won through +5 awards. C2 made 14,957 predictions and received 14,952 locally applied ACKs. Source-controlled intervals are approximately 118.6 to 118.9 seconds within 120-second rounds; cadence is computed from source QPC, not assumed from worker target frequency.

All 393 C2 armed action ACKs and all 576 parent armed ACKs have a unique same-move native `REK_Move` outbound projection within 100 ms. Missing local move dispatch is therefore not the observed failure mechanism for those requests.

That projection remains `request_only:true`, with `server_acceptance:null`. It does not prove server execution. Own visual-only composer playback/punching fields remain false; no authoritative opponent move-start marker or opponent outbound command stream exists in these records. Actual accepted attack counts, executed durations, and misses are unknown. Inter-request intervals and projected busy periods cannot supply those labels.

## Exact categories and legal availability

The semantic action registry and bridge agree on move order `6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16` for categories 16 through 32. The parent native dispatches additionally confirm category 17 to native move 7.

| Category | Native move | Recovered identity | Documented keyboard mapping | Parent requests | C2 requests |
| --- | --- | --- | --- | --- | --- |
| 17 | 7 | Left front kick | Double H | 3 | 0 |
| 21 | 1 | Left jab | H | 222 | 145 |
| 23 | 3 | Right hook | U | 186 | 157 |
| 26 | 10 | Six-punch | Space+Y | 87 | 39 |

Source: `g1_semantic_action_table.c:25`, `native_motion_routes.c:56,81,92,109`, `G1PolicyStreamContract.MoveOrder`, and `human_eval_server.py:83,85,89,92`. The action-table SHA256 is `bedaa79fb970cc60707e6aabe1415ea7a2c0743ed41ffeeea63cbd4bbcf2a03f`; motion-route SHA256 is `ce198ba2b781006293a5c07da1680fe0c005fd640f3ad0f84dba7d40341bb6d7`.

Category 17 was legal at every attack-legal decision: 816/14,957 C2 decisions and 1,431/24,010 parent decisions. It was never selected by C2. Thus it is not selectively missing from the legal mask. Three parent selections are insufficient to establish its value or superiority.

C2 categories 21 and 23 comprise 302/393 requests (76.8%), versus parent 408/576 (70.8%). Category 26 is 39/393 (9.9%), versus 87/576 (15.1%). All five C2 rounds are included; the first two losses alone exaggerated the persistence of the category-26 decrease.

Encoder projected busy applies to 10,165/14,957 C2 decisions (68.0%), versus 15,911/24,010 parent decisions (66.3%). Held-translation blocking applies to 3,583/14,957 (24.0%), versus 6,092/24,010 (25.4%). These conditions can overlap. Neither is a direct measurement of server attack playback.

## Geometry and descriptive score associations

Distances are Unity numeric horizontal root distances, without claimed metre calibration. These bins describe observations; they are not measured hitboxes or proposed runtime gates.

| Request-time geometry | C2 | Parent |
| --- | --- | --- |
| Facing error <=45 degrees | 189/393 (48.1%) | 297/576 (51.6%) |
| Facing error >90 degrees | 93/393 (23.7%) | 116/576 (20.1%) |
| Root distance >1 | 72/393 (18.3%) | 125/576 (21.7%) |
| Distance <=0.8 and facing <=45 degrees | 106/393 (27.0%) | 167/576 (29.0%) |

Own ordinary score within two seconds followed 8/27 close/facing C2 right-hook requests and 0/62 behind-facing requests. Parent counts were 12/41 and 1/47. For six-punch, counts were 4/17 versus 0/2 in C2, and 7/27 versus 1/9 in parent. Requests can share a later score receipt, another request can intervene, and actual server execution is unknown. These are temporal associations, not per-action success rates or causal values.

The most directly supported policy-learning issue is poor conditional ordinary scoring despite working local dispatch. Geometry remains relevant in both datasets. The evidence does not identify an opponent whiff-response rule, establish a hard range threshold, or justify labeling every request without a subsequent score a miss.

## Next controlled comparison

The sibling learning rate `3e-5` is **larger** than C2's `1e-5`. It is an update-rate comparison, not a smaller-update experiment. These five unmatched rounds do not predict that sibling's outcome or justify changing an active cohort.

Retain parent ordinary-scoring trajectories when testing policy updates and evaluate ordinary points separately from +5 awards. Existing completed trajectories support conditional attack learning from actual returns, without fabricating attack-causal labels. Category-17 behavioral coverage could use existing verified human examples in a separate candidate, but the current live selections cannot establish its expected score benefit.

## Reproduction and publication boundary

`summary.json` contains whitelisted aggregates and native-source hashes/byte counts. It excludes raw captures, participant identities, filesystem source paths, wire bodies and per-request ledgers. `summarize.cjs` derives it from three private reports and rejects duplicate, incomplete or unvalidated rounds; its two focused tests passed. The private strict analysis adapter's two tests also passed.

Private report hashes, in input order:

1. Initial two C2 losses plus parent eight: `1801e1c7fef431dadb37eb0ff007557b9e7a261009ed2bb9d9300cf38500288b`.
2. Subsequent two C2 wins: `3a321054319c83bae325e58080e30041b1242679322ad61a859384cb06209617`.
3. Final C2 loss: `d6d65133645be30e03b437282d7b617b478962e85a3fdf410879518554aa31de`.

The reused native/geometry analyzer SHA256 is `d399c6f3490982a7389e99c2bc2b69b805d985f10deb9440bd18f75780a5f2b8`. Exact source paths and receipts remain in the private analysis stage and NAS archive manifest. The earlier four-round cutoff is preserved privately. No game, bridge, GPU, policy, active-trial, controller, or production-source changes were made by this analysis.

## Preserved evidence

The closed cohort and its five native captures were archived under `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\timing500-ppo-live-20260924-r1-closed-20260924T110034Z`.

- `evidence.tar.gz`: 171,319,050 bytes; SHA256 `2393472f31fd7b1060cf48674faa7da0699d6ea551bc9284adc072821fe68f30`.
- `native-captures.tar.gz`: 151,572,892 bytes; SHA256 `ac60c7db18bf634b4e1db59064578802035e452ea6e124dfd453c70719567c7d`.

All 367 cohort files, including the MP4s, and exactly five native captures are preserved. Source hashes stayed unchanged and both archives passed NAS readback verification. `archive-receipt.json` records the commands and dependencies. No source files were removed.
