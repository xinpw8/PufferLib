# Calibration corrections and authentic Bot 1 evaluation

Status: development candidates, not accepted parity and not proven consistent Bot 1 wins.

## Two confirmed defects

The F7 calibration snapshot passed native move indices to a table indexed by policy action minus 16. The asset loader and live bridge use a different ordering:

| Policy action | Native move index |
| --- | --- |
| 16 | 6 |
| 20 | 0 |
| 21 | 1 |
| 23 | 3 |
| 25 | 5 |
| 26 | 10 |
| 30 | 14 |

Consequently several hit probabilities targeted the wrong moves. The intended action-16 kick penalty instead tested native move 0, which is action 20. The correction uses the accepted policy action for probability lookup and the loaded inverse mapping for the penalty target. Tests cover all 17 actions and every existing near/far/front/behind table value. No probability values were newly estimated.

A separate defect removed points after the scorer had already updated credited-hit counts, score cooldowns, deduplication and last-hit receipts. The correction gates those updates before scoring while preserving geometric contact history. This matters to observation compatibility: the actual F7 live encoder synthesizes last-hit history from score-counter changes, not every geometric overlap. The helper tests exercise the recovered scorer's real state transitions.

Each correction is opt-in. Independent calibration RNG is a separate option and remained disabled in these experiments. Existing source in the working branch was not overwritten by the older F7 snapshot.

## Headless native CUDA training

All three candidates start from the exact F7 checkpoint and use identical F7 training settings: 512 arenas, horizon 128, minibatch 8192, 16,777,216 transitions, seed 73, environment seed 419, learning rate 0.000055, entropy coefficient 0.00017, clip 0.13, value coefficient 1.02, gamma 0.9998844821426083 and GAE lambda 0.9978673240629938. Rollouts and PPO use the native CUDA executable. There is no Python training runtime or CPU physics stepping.

| Candidate | Full process time (s) | Full training SPS | Completed training rounds won |
| --- | ---: | ---: | ---: |
| Unchanged continuation control | 18.734739 | 895,514 | 2440/2560 |
| Coherent credited-hit receipts | 18.776690 | 893,513 | 2262/2560 |
| Receipts plus corrected action IDs | 18.505531 | 906,605 | 2288/2560 |

Full process SPS includes startup, graph capture, rollout, PPO, checkpoint writes and shutdown. These are individual measurements, not a statistically established throughput difference. Native dashboard SPS excludes some setup and graph-capture costs. Training win counts are on-policy training outcomes, not live or held-out policy strength. All runs exited zero and reported zero runtime failure bits.

In the last persisted training interval, both-side hit counts per round were 179.25 for control and 38.705 for receipt coherence. These intervals do not necessarily end at the final checkpoint. The difference is consistent with removing uncredited hit receipts; it does not prove better authentic combat.

## Authentic evaluation so far

All interaction uses the isolated Spark client, private sparring Bot 1 / difficulty 0, with no live attack gate, forced attack or cooldown. Windows input is untouched. MP4 recordings are limited to less than 20 MB each.

- Frozen original F7 pilot, seed 200: completed 120-second round, 16:14 win.
- Continuation control screening: 12:7 win, 1:14 loss; two additional client-crash attempts excluded and retained.
- Receipt-only screening: 18:14 win; one additional source-stream failure excluded and retained.
- The eight-label screening comparison was paused after these infrastructure failures and discovery of the shared action-ID defect. It is incomplete and cannot establish an A/B winner.
- The combined correction has a separate fixed prospective plan: seeds 401 through 420, target 18 wins in 20 full rounds, stop on the third nonwin. Its first two counted results are losses, 13:27 and 1:11. Two additional attempts are incomplete. This candidate is not promoted.

Counted rounds must start at zero score with 117 to 120 seconds remaining, have a 120-second non-redo duration, preserve round identity and checkpoint identity, and finish with an observed terminal score/result. Crashes and late starts remain separate attempts, not wins or inferred losses. Simulator wins are never substituted for authentic results.

The original F7 fixed campaign also retained a late-start attempt, a CUDA worker allocation failure and a client crash while leading 4:3 with 27.8 seconds left. None counts as a completed round.

## Remaining transfer gaps

Review of original F7 s160 through s163 found 10, 25 and 10 points from opponent +5 awards in the three wins, versus zero in the 1:14 loss. That loss contained four bot +2 score receipts at separations 0.436 to 0.576 Unity units, plus a separate later +5 award. A 1:1 mapping to candidate distance units remains an experiment assumption, not a measured metre calibration. Opponent move identity is unavailable in those records; the +2 receipts are not proof of a specific kick.

In the loss, learner tilt rose to about 98.8 degrees around 102 seconds; the +5 award arrived around 106 seconds. The preceding requests were action 23, not action 16. A kick-only fall approximation does not cover that event. Within-opponent-bearing 0.5 rad exposure was 20.4% in the loss versus 32.8%, 50.5% and 35.1% in the wins. Four rounds do not identify a causal angle threshold.

The existing kick-fall probability 0.2 remains an assumption. Opponent falls and contact-driven balance are still incomplete. Correct lookup and receipt semantics do not establish complete simulator parity.

## Artifacts and provenance

Private Spark roots:

- `/home/spark-advantage/rek-training/f7-control-20260924-r1`
- `/home/spark-advantage/rek-training/f7-coherent-contact-20260924-r1`
- `/home/spark-advantage/rek-training/f7-receipt-ab-20260924-r1`
- `/home/spark-advantage/rek-training/f7-action-id-fix-20260924-r1`
- `/home/spark-advantage/rek-training/f7-action-id-live-20260924-r1`

Checkpoint SHA256:

```text
F7:      6a5082750aee85183bdd23b42470775374c89740d3c8c00445ead2d6d2e39263
control: 4f61af65bb625063c86d8df89333042b3a92e50a08731f9b7c7f1a3c33a1ee06
receipt: 926f22fa0b43aee80f8bb32bdc6f4e1cf517a05d736d45cae93cddeac9e998e3
IDs:     947a283429a8661ceaab9adc5e6c14b3534cab43fafdc0c60c86cb33d51c686e
```

Patches are sequential candidate patches, not a declaration that they apply directly to this working branch. The first requires F7 snapshot `fast_runtime.cu` SHA256 `ee6b31be666b65f5aa116954727381b2c57d78571321993a3c17d72ac72e15d2`. The second applies after the first. Full staged native builds and CPU tests passed. Game binaries, private model assets, credentials, raw session logs and checkpoints are not committed here.

The live client repeatedly failed inside CoreCLR 6.0.7. A separate startup trial raises `BOX64_DYNAREC_STRONGMEM` from 1 to 2 based on the installed Box64 safest preset. It completed two rounds but subsequently also failed, so it has not resolved the crashes. The first combined-correction round applied 5729 predictions in 120 game seconds; this is not a full frame-timing comparison. Keep this runtime change distinct from policy changes when interpreting subsequent results.
