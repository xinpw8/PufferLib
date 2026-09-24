# Native point ledger: first three completed no-kick-prior rounds

Created 2026-09-24T05:09:55.277Z. This is a read-only descriptive analysis. No policy, simulator, bridge, or live control was changed. Scope is exactly noprior-s601, noprior-s602, and noprior-s603-retry2. Incomplete attempts, s604, and later rounds are excluded.

## Result and authority

| Round | Final learner:Bot1 | Ordinary +1/+2 points | Native five-point award | Accepted local requests | Encoded observations |
| --- | --- | --- | --- | --- | --- |
| noprior-s601 | 7:9 | 7:4 | Bot1 +5 | 44 | 5728 |
| noprior-s602 | 12:7 | 7:7 | learner +5 | 92 | 5790 |
| noprior-s603-retry2 | 8:14 | 8:9 | Bot1 +5 | 66 | 5724 |

All three used checkpoint `7561859790d0ed55a55471d96442fe954a593c52d8617b51c143687ba9d98f96`: three completed rounds, one win and two losses, final aggregate 27:30, ordinary points 22:20, count-out points 5:10. These three rounds cannot identify the causal effect of removing the kick-fall prior. The winner issued zero category16 requests.

All 38 native score packets reconcile exactly to the bridge scorer and new counter. All 3,580 unique bridge fight-state receipts match native wire bytes/SHA, frame, clock, and decoded data; no selected call is left- or history-censored. All 202 locally accepted requests match distinct native REK_Move projected requests, with no extra native REK_Move requests. The two-byte projected bodies are independently checked. This proves local method invocation, not server acceptance, executed action, hit, or completion. Actual composer action identity is unavailable.

Each round has one received Slip followed by a received Knockout call and +5 to the other fighter, with approximately 3 s of counting. The native call name Knockout is not a terminal match KO here: all three summaries say WonByPoints, knockout=false. Ordinary +1/+2 receipts have a same-frame hit packet consistent with punch/kick; those packets lack causal attacker/executed-action IDs.

## Request-time geometry

| Round | Accepted | abs(bearing) >0.5 rad | abs(bearing) >1 rad | Distance >1 Unity unit | Both >1 |
| --- | --- | --- | --- | --- | --- |
| noprior-s601 | 44 | 29 | 19 | 13 | 8 |
| noprior-s602 | 92 | 65 | 50 | 18 | 8 |
| noprior-s603-retry2 | 66 | 42 | 30 | 10 | 6 |
| TOTAL | 202 | 136 | 99 | 41 | 22 |

Angles use the recovered body-forward convention from root rotation. Distances are horizontal numeric Unity units. One unit is not claimed to be a calibrated metre or a hitbox radius. Request geometry is from the exact policy observation named by the ACK; source-to-ACK delay spans 12.65 to 40.44 ms.

Separately, at the native score-receipt pose, all 20 learner ordinary score events had |own bearing| <= 0.414030 rad. Ten of 15 Bot1 ordinary score events occurred with learner |bearing| >1 rad. These are score-time observations, not request-time thresholds. Motion between request and score and unobserved execution prevent causal acceptance or hitbox inference.

Category23/native3 contributed 61 requests, including 36 with |bearing| >pi/2. The 32 category23 requests in s603 included 22 with the opponent behind the learner. Bot1 +2 at 40.977 s and 52.685 s in that loss followed category23 requests by about 0.967 s and 0.442 s; score-time own bearings were 1.949 and -1.717 rad. These are concrete repeated facing-risk sequences, not proof that those requests caused the points.

## Attempts and temporal score-following proxies

Zero-score windows are not proven misses. A request may fail to execute, execute later, overlap another request, or be interrupted. Windows can overlap; one score can follow multiple requests. This is deliberately not an attack-accuracy estimate.

| Round | Policy category / native move | Requests | Own ordinary score within 0.5 s | Within 1 s | Within 2 s | Request bearing >pi/2 |
| --- | --- | --- | --- | --- | --- | --- |
| noprior-s601 | 16 / 6 | 3 | 0 | 0 | 2 | 1 |
| noprior-s601 | 21 / 1 | 3 | 0 | 0 | 0 | 1 |
| noprior-s601 | 23 / 3 | 8 | 0 | 0 | 0 | 6 |
| noprior-s601 | 25 / 5 | 3 | 0 | 0 | 1 | 0 |
| noprior-s601 | 26 / 10 | 25 | 0 | 0 | 2 | 4 |
| noprior-s601 | 30 / 14 | 2 | 0 | 0 | 0 | 0 |
| noprior-s602 | 21 / 1 | 38 | 1 | 1 | 1 | 17 |
| noprior-s602 | 23 / 3 | 21 | 0 | 1 | 2 | 8 |
| noprior-s602 | 24 / 4 | 1 | 0 | 0 | 0 | 0 |
| noprior-s602 | 25 / 5 | 9 | 0 | 0 | 0 | 3 |
| noprior-s602 | 26 / 10 | 21 | 2 | 2 | 3 | 1 |
| noprior-s602 | 30 / 14 | 2 | 0 | 0 | 0 | 1 |
| noprior-s603-retry2 | 16 / 6 | 2 | 0 | 0 | 0 | 1 |
| noprior-s603-retry2 | 21 / 1 | 3 | 0 | 0 | 0 | 0 |
| noprior-s603-retry2 | 23 / 3 | 32 | 0 | 0 | 1 | 22 |
| noprior-s603-retry2 | 24 / 4 | 3 | 1 | 1 | 1 | 0 |
| noprior-s603-retry2 | 25 / 5 | 4 | 0 | 2 | 2 | 0 |
| noprior-s603-retry2 | 26 / 10 | 21 | 2 | 3 | 3 | 1 |
| noprior-s603-retry2 | 30 / 14 | 1 | 0 | 0 | 0 | 1 |

Preferred category16 maps to native6, established from ACK and native request, not category-minus16. Counts were 3, 0, 2. In s601 two +2 receipts followed category16 by 1.093 and 1.066 s; score-time distance was 0.661 and 1.012 units. In s603 both category16 requests had large request bearings (1.355 and 2.150 rad) and neither was followed by an own ordinary score within 2 s. The cohort is too small and uncontrolled to decide whether prior removal improved this attack.

## Own count-out timelines

Call elapsed time is the first policy observation carrying the new verified received call, not the physical fall onset. Request age below uses the two samples' Unity clocks. Tilt can rise before a Slip receipt.

### noprior-s601


Slip received at 72.209 s. Last three local requests:

| ID | Category/native | Elapsed s | Age s | Distance units | Own bearing rad | Own tilt deg | Own root height units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 26 | 30/14 | 66.908 | 5.203 | 1.218 | 1.357 | 6.307 | 0.730 |
| 27 | 25/5 | 68.308 | 3.800 | 1.527 | 0.877 | 12.286 | 0.665 |
| 28 | 16/6 | 69.308 | 2.872 | 1.657 | 0.404 | 10.769 | 0.660 |

Selected samples from the preserved quarter-second history:

| Elapsed s | Distance units | Own bearing rad | Own tilt deg | Own root height | Received count mask |
| --- | --- | --- | --- | --- | --- |
| 66.307 | 0.910 | 1.807 | 13.376 | 0.747 | 0 |
| 67.208 | 1.347 | 1.548 | 8.497 | 0.727 | 0 |
| 68.108 | 1.470 | 0.677 | 7.904 | 0.644 | 0 |
| 69.008 | 1.668 | 1.475 | 10.436 | 0.655 | 0 |
| 69.909 | 1.418 | 0.385 | 4.709 | 0.721 | 0 |
| 70.809 | 0.780 | 1.490 | 40.812 | 0.809 | 0 |
| 71.709 | 0.668 | 0.207 | 107.325 | 0.219 | 0 |

Knockout received at 75.214 s. Last three local requests:

| ID | Category/native | Elapsed s | Age s | Distance units | Own bearing rad | Own tilt deg | Own root height units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | 30/14 | 72.409 | 2.694 | 0.785 | 0.328 | 80.382 | 0.127 |
| 30 | 25/5 | 73.813 | 1.302 | 0.938 | 0.296 | 82.381 | 0.125 |
| 31 | 26/10 | 74.813 | 0.374 | 0.999 | 0.292 | 83.333 | 0.124 |

Selected samples from the preserved quarter-second history:

| Elapsed s | Distance units | Own bearing rad | Own tilt deg | Own root height | Received count mask |
| --- | --- | --- | --- | --- | --- |
| 69.308 | 1.657 | 0.404 | 10.769 | 0.660 | 0 |
| 70.209 | 1.162 | -0.496 | 4.465 | 0.701 | 0 |
| 71.109 | 0.787 | 0.901 | 32.680 | 0.763 | 0 |
| 72.009 | 0.683 | 0.640 | 98.266 | 0.145 | 0 |
| 72.909 | 0.745 | -0.087 | 75.129 | 0.122 | 1 |
| 73.813 | 0.936 | 0.301 | 82.317 | 0.126 | 1 |
| 74.713 | 0.989 | 0.286 | 83.234 | 0.124 | 1 |

### noprior-s603-retry2


Slip received at 102.715 s. Last three local requests:

| ID | Category/native | Elapsed s | Age s | Distance units | Own bearing rad | Own tilt deg | Own root height units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 58 | 26/10 | 96.312 | 6.344 | 0.952 | -0.291 | 6.761 | 0.701 |
| 59 | 26/10 | 99.013 | 3.641 | 0.906 | -0.246 | 10.839 | 0.699 |
| 60 | 23/3 | 101.714 | 0.913 | 0.391 | -2.879 | 87.086 | 0.632 |

Selected samples from the preserved quarter-second history:

| Elapsed s | Distance units | Own bearing rad | Own tilt deg | Own root height | Received count mask |
| --- | --- | --- | --- | --- | --- |
| 96.812 | 1.196 | -0.394 | 7.196 | 0.744 | 0 |
| 97.712 | 1.044 | -0.676 | 13.444 | 0.701 | 0 |
| 98.613 | 0.623 | -0.673 | 8.282 | 0.695 | 0 |
| 99.513 | 1.052 | -0.441 | 4.291 | 0.716 | 0 |
| 100.414 | 0.761 | -0.996 | 12.910 | 0.685 | 0 |
| 101.314 | 0.483 | -1.093 | 22.091 | 0.786 | 0 |
| 102.215 | 0.576 | -3.076 | 90.529 | 0.263 | 0 |

Knockout received at 105.721 s. Last three local requests:

| ID | Category/native | Elapsed s | Age s | Distance units | Own bearing rad | Own tilt deg | Own root height units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 58 | 26/10 | 96.312 | 9.357 | 0.952 | -0.291 | 6.761 | 0.701 |
| 59 | 26/10 | 99.013 | 6.654 | 0.906 | -0.246 | 10.839 | 0.699 |
| 60 | 23/3 | 101.714 | 3.926 | 0.391 | -2.879 | 87.086 | 0.632 |

Selected samples from the preserved quarter-second history:

| Elapsed s | Distance units | Own bearing rad | Own tilt deg | Own root height | Received count mask |
| --- | --- | --- | --- | --- | --- |
| 99.813 | 1.038 | -0.564 | 10.442 | 0.721 | 0 |
| 100.714 | 0.473 | -1.147 | 8.850 | 0.628 | 0 |
| 101.614 | 0.339 | -1.960 | 65.371 | 0.815 | 0 |
| 102.515 | 0.656 | -3.121 | 86.254 | 0.269 | 0 |
| 103.415 | 1.155 | 2.537 | 94.531 | 0.122 | 1 |
| 104.316 | 1.130 | 2.572 | 94.603 | 0.122 | 1 |
| 105.216 | 1.156 | 2.605 | 94.490 | 0.122 | 1 |

In s601 the last pre-Slip category16 request was 2.872 s before the received Slip and was followed by an own +2 before falling. Three further local attack requests were dispatched during the own count, while own tilt was approximately 80 to 83 degrees and root height approximately 0.12 units. In s603 the last pre-Slip request was category23 at approximately 87 degrees tilt, 0.913 s before the Slip receipt; severe loss of upright posture was already observed at that request. There is no supported rule assigning either fall to the most recent attack.

## Observation availability and training comparison

| Round | Raw source observations | Opponent move available | Opponent motion-name available | Opponent nonzero frame | Own actual-busy known | Count-active source observations |
| --- | --- | --- | --- | --- | --- | --- |
| noprior-s601 | 5731 | 0 | 0 | 0 | 0 | 144 |
| noprior-s602 | 5793 | 0 | 0 | 0 | 0 | 145 |
| noprior-s603-retry2 | 5727 | 0 | 0 | 0 | 0 | 146 |

The captured schema is rek.native5.scaled_polar_xy.v1 with dispatched_request_v4_duration busy projection. Across 17,242 encoded observations, own/opponent down and synthetic tilt fields 71/72 and 157/158, fall counters192/193, down-state202–205, and reset timers remain literal zero;77/163 remain2,209 remains0.5. Verified referee counting is available in raw source but is not represented in these legacy channels. Own root height, root rotation and rendered joint pose remain measured.

The local encoder source at ocean/rek_g1/native5/live_transfer/encode_live.cpp:198–204 derives own busy from the last dispatched request and fixed move duration when actual playback is unavailable. Lines211–216 set the legacy constants;230 sets own route/busy;237–238 leaves fall/down/reset channels structural. Captured projection manifests and column extrema independently confirm the behavior for these rounds; the currently dirty local source is not asserted to be byte-identical to their executable.

Training does not provide a direct opponent move index or phase input that live simply zeros. The preserved F7 handoff fast_runtime.cu:866–884 repeats entity pose/joint/down features for both fighters;886–897 makes controller columns172–183 actor-only, including own route179 and attacking182/183. Current dirty source has the same structure at810–841. Opponent route/phase is used internally to choose its rendered pose, not exposed as direct controller columns. A joint/rate mask removes an indirect opponent telegraph; this report does not predict a benefit. Own busy differs semantically: compact actual attack state versus live request-duration projection.

The logs justify investigating facing behavior, loss-of-upright state and count-aware action handling. They do not justify a fixed angle/range hit threshold, a new stochastic fall hazard, or claiming the joint mask resolves the dominant blindspot. Any change still requires an authentic live comparison.

## Reproducibility and point ledger

ledger.json contains every point's native line/frame/time, body hash, exact bridge counter match, preceding poses, same-frame hit packet metadata, and all prior requests within3s. augmentation.json adds native request joins, per-request geometry, call histories and encoder manifests/extrema. point-ledger.tsv and accepted-requests.tsv are derived convenience tables. Source paths, bytes and hashes are retained in ledger.json; raw captures are archived directly to NAS only.

| Round | Native score line | Elapsed s | Scorer | Award | New counter | Distance units | Own bearing rad | Same-frame is_kick | Prior request IDs within3s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noprior-s601 | 7688 | 10.473 | Bot1 | +1 | 1 | 0.286 | -2.461 | 0 | none |
| noprior-s601 | 11775 | 15.977 | learner | +2 | 2 | 0.661 | -0.219 | 1 | 2:16/6@-1.093 |
| noprior-s601 | 19045 | 25.685 | learner | +1 | 3 | 0.679 | -0.185 | 0 | 5:26/10@-1.187 |
| noprior-s601 | 35857 | 48.299 | learner | +1 | 4 | 0.681 | 0.389 | 0 | 15:23/3@-2.408,16:26/10@-1.472 |
| noprior-s601 | 52359 | 70.309 | learner | +2 | 6 | 1.012 | -0.025 | 1 | 27:25/5@-1.994,28:16/6@-1.066 |
| noprior-s601 | 55888 | 75.113 | Bot1 | +5 | 6 | 1.027 | 0.330 | none | 29:30/14@-2.621,30:25/5@-1.230,31:26/10@-0.301 |
| noprior-s601 | 68700 | 92.322 | learner | +1 | 7 | 0.685 | -0.153 | 0 | 32:26/10@-2.182 |
| noprior-s601 | 68909 | 92.522 | Bot1 | +2 | 8 | 0.573 | -0.546 | 1 | 32:26/10@-2.459 |
| noprior-s601 | 71909 | 96.624 | Bot1 | +1 | 9 | 0.578 | -0.671 | 0 | 34:26/10@-0.777 |
| noprior-s602 | 8933 | 12.076 | Bot1 | +1 | 1 | 0.674 | 0.377 | 0 | 4:26/10@-0.349 |
| noprior-s602 | 8979 | 12.076 | learner | +1 | 1 | 0.647 | 0.308 | 0 | 4:26/10@-0.414 |
| noprior-s602 | 17032 | 22.882 | learner | +1 | 2 | 0.613 | 0.362 | 0 | 8:26/10@-0.339 |
| noprior-s602 | 17687 | 23.783 | learner | +1 | 3 | 0.696 | 0.057 | 0 | 8:26/10@-1.224 |
| noprior-s602 | 17984 | 24.184 | learner | +1 | 4 | 0.603 | 0.049 | 0 | 8:26/10@-1.609 |
| noprior-s602 | 19749 | 26.485 | learner | +1 | 5 | 0.627 | -0.062 | 0 | 9:26/10@-1.257 |
| noprior-s602 | 39201 | 52.517 | learner | +5 | 10 | 0.620 | -2.876 | none | 27:23/3@-2.171,28:25/5@-1.236,29:21/1@-0.219 |
| noprior-s602 | 43364 | 58.122 | learner | +1 | 11 | 0.616 | 0.334 | 0 | 31:23/3@-1.136,32:21/1@-0.191 |
| noprior-s602 | 46188 | 61.925 | learner | +1 | 12 | 0.590 | 0.012 | 0 | 34:23/3@-0.608 |
| noprior-s602 | 73219 | 98.176 | Bot1 | +1 | 2 | 0.420 | -2.705 | 0 | 67:21/1@-0.741,68:21/1@-0.180 |
| noprior-s602 | 75608 | 101.377 | Bot1 | +1 | 3 | 0.566 | -2.505 | 0 | 69:21/1@-2.809,70:23/3@-2.160,71:21/1@-0.567 |
| noprior-s602 | 78400 | 105.079 | Bot1 | +1 | 4 | 0.470 | -2.045 | 0 | 74:23/3@-2.304,75:21/1@-1.360,76:23/3@-0.203 |
| noprior-s602 | 81318 | 108.981 | Bot1 | +1 | 5 | 0.553 | -2.445 | 0 | 78:21/1@-2.392,79:21/1@-1.608,80:23/3@-1.031,81:21/1@-0.034 |
| noprior-s602 | 88776 | 118.986 | Bot1 | +2 | 7 | 0.511 | -0.703 | 1 | 89:25/5@-1.886,90:21/1@-0.642 |
| noprior-s603-retry2 | 16402 | 22.167 | Bot1 | +1 | 1 | 0.409 | -2.350 | 0 | none |
| noprior-s603-retry2 | 18772 | 25.368 | learner | +1 | 1 | 0.636 | -0.134 | 0 | 5:25/5@-0.718 |
| noprior-s603-retry2 | 19173 | 25.869 | learner | +1 | 2 | 0.604 | 0.135 | 0 | 5:25/5@-1.255,6:26/10@-0.331 |
| noprior-s603-retry2 | 30448 | 40.977 | Bot1 | +2 | 3 | 0.391 | 1.949 | 1 | 14:23/3@-0.967 |
| noprior-s603-retry2 | 35692 | 47.982 | Bot1 | +1 | 4 | 0.485 | -1.794 | 0 | 19:23/3@-2.321,20:23/3@-1.374,21:23/3@-0.449 |
| noprior-s603-retry2 | 39151 | 52.685 | Bot1 | +2 | 6 | 0.541 | -1.717 | 1 | 24:23/3@-2.290,25:23/3@-1.372,26:23/3@-0.442 |
| noprior-s603-retry2 | 56469 | 75.800 | learner | +1 | 3 | 0.709 | -0.400 | 0 | 46:25/5@-2.648,47:23/3@-1.677,48:25/5@-0.732 |
| noprior-s603-retry2 | 56808 | 76.300 | learner | +1 | 4 | 0.652 | -0.301 | 0 | 47:23/3@-2.126,48:25/5@-1.181,49:24/4@-0.245 |
| noprior-s603-retry2 | 62271 | 83.604 | learner | +1 | 5 | 0.709 | 0.094 | 0 | 53:26/10@-0.811 |
| noprior-s603-retry2 | 62577 | 84.004 | learner | +1 | 6 | 0.568 | -0.042 | 0 | 53:26/10@-1.220 |
| noprior-s603-retry2 | 63485 | 85.205 | Bot1 | +1 | 7 | 0.786 | 0.043 | 0 | 53:26/10@-2.443 |
| noprior-s603-retry2 | 71309 | 95.711 | learner | +1 | 7 | 0.501 | -0.414 | 0 | 57:26/10@-2.117 |
| noprior-s603-retry2 | 78677 | 105.616 | Bot1 | +5 | 12 | 1.208 | 2.586 | none | none |
| noprior-s603-retry2 | 81596 | 109.523 | Bot1 | +2 | 14 | 0.431 | -1.462 | 1 | 62:26/10@-0.724 |
| noprior-s603-retry2 | 85385 | 114.626 | learner | +1 | 8 | 0.508 | -0.140 | 0 | 64:26/10@-0.368 |
