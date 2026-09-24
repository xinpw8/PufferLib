# Timing500/750: final closed cohort

Closed at **2026-09-24 10:22:51.442 UTC** with **5 wins, 3 losses and 104:93 points** from eight counted rounds. Eight incomplete attempts were excluded. The accepted target of 18 wins in 20 fixed seeds failed at the third completed nonwin; seeds 1209 through 1220 were not reached. This result does not establish consistent Bot1 wins and does not support promotion or a model-improvement claim.

Source stage: `/home/spark-advantage/rek-training/humanbc-timing500-live-20260924-r1`.
Unchanged checkpoint SHA256: `5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4`.
Timing-variant bridge DLL SHA256: `11fcfa2bea541f2829553c3833c301717c0182a59eebb0020f5f7b8cb066ca3d`.

The [two-round](LIVE-PROGRESS.md) and [four-round](FOUR-ROUND-ANALYSIS.md) reports remain preserved as historical cutoffs. This file gives the final cohort accounting.

## Counted rounds and actual control

All eight ledger entries have `complete`, `terminal`, `same_bot`, and `fair_start` true, initial 0:0 scores, non-redo 120 s rounds, and received terminal `WonByPoints` results at time zero. All summaries pin the same checkpoint and validate controlled startup. Scores are learner:Bot1.

| Seed / counted attempt | Result | Points | Predictions | Local applied ACKs | Initial controlled round timer, s |
| --- | --- | --- | ---: | ---: | ---: |
| 1201 / retry8 | Win | 16:10 | 2821 | 2820 | 118.766400 |
| 1202 / attempt1 | Loss | 6:13 | 2964 | 2963 | 118.749626 |
| 1203 / attempt1 | Win | 4:3 | 3064 | 3063 | 118.766320 |
| 1204 / attempt1 | Win | 11:4 | 3100 | 3099 | 118.766250 |
| 1205 / attempt1 | Win | 20:14 | 3102 | 3101 | 118.782750 |
| 1206 / attempt1 | Win | 18:14 | 3022 | 3021 | 118.783090 |
| 1207 / retry2 | Loss | 17:22 | 3000 | 2999 | 118.649710 |
| 1208 / attempt1 | Loss | 12:13 | 2937 | 2936 | 118.665710 |
| Total | 5W3L | 104:93 | 24010 | 24002 | |

These are 120 s game rounds, not 120 s of policy control. Control began about 1.22-1.35 s after the round timer started, while the required initial score was still 0:0 and at least 117 s remained. Readiness sampling was neutral and preceded the controlled stream. Its measured warmup duration is listed separately below; it is not the entire entry/startup delay.

| Seed | Readiness samples / elapsed ms | Timer at readiness, s | Controlled source QPC span, s | Source cadence, Hz |
| --- | ---: | ---: | ---: | ---: |
| 1201 | 4 / 935 | 118.866425 | 118.768196 | 23.7606 |
| 1202 | 4 / 869 | 118.949670 | 118.826651 | 24.9523 |
| 1203 | 4 / 927 | 118.866360 | 118.780361 | 25.8039 |
| 1204 | 4 / 865 | 118.966350 | 118.835350 | 26.0949 |
| 1205 | 4 / 884 | 118.882830 | 118.825735 | 26.1139 |
| 1206 | 4 / 873 | 118.983116 | 118.817827 | 25.4423 |
| 1207 | 5 / 1032 | 118.749730 | 118.639636 | 25.2951 |
| 1208 | 5 / 1018 | 118.765785 | 118.752547 | 24.7405 |

Cadence uses controlled source QPC intervals, rather than prediction count divided by nominal duration. Small differences between QPC span and round-timer span reflect distinct clocks and terminal observation timing. Local ACKs do not establish server-executed attacks or contact success.

## Exclusions and runtime result

The eight excluded attempts comprise one private-practice entry timeout, five `private_practice_reservation_already_in_progress` failures, and two unsupported local-G1/opponent-T800 pairings. The first seven precede seed 1201's counted attempt; the additional mixed pairing is seed 1207 attempt1. They are not wins, losses, or complete training episodes.

There were no midround watchdog failures in this cohort. Every counted round has exactly one rejected action ACK with reason `policy_stream_not_owned`, after a source state already reported inactive/time-zero `WonByPoints`. Those eight terminal rejections explain predictions minus applied ACKs; they are not stale-action watchdog failures. Controller PID 2898705 was absent when the closed ledger was inspected. The campaign log records scoped client recycling followed by `criterion_failed` and `campaign_end`.

The same checkpoint had earlier runtime-aborted attempts under the old timing contract. The timing variant permits 500 ms source age and a 750 ms established-action watchdog, but the completed cohort also ran at roughly 25 Hz rather than the earlier approximately 6.9 Hz sample. Client lifecycle and host workload differ. These observations do not isolate the timing change's causal effect, establish a slowdown cause, or demonstrate a model improvement. Retries and completion-conditioned evaluation remain limitations.

## Native scoring audit status

The first four rounds have the separately executed native score/referee analysis documented in [FOUR-ROUND-ANALYSIS.md](FOUR-ROUND-ANALYSIS.md): 37:30 total points, 22:25 ordinary +1/+2 points, and +5 award counts 3:1. That result must not be extrapolated to the last four rounds.

At preparation of this final report, the all-eight strict native sidecar/export job had not yet produced its final receipt. The complete 104:93 accounting here is verified against the closed ledger and all eight bridge summaries, not presented as an independently native-packet-validated all-eight decomposition. The first CPU export in `/home/spark-advantage/rek-training/timing500-onpolicy-20260924-r1` stopped on a legacy coverage rule: the first applied return was 1.1275 s after the first passive observation, exceeding its 1 s bound during the recorded warm-up. Its native packet/referee checks passed. Original outputs, including `completed_policy_round:false`, are preserved. A separately versioned controlled-start adapter is being prepared in `timing500-onpolicy-20260924-r2`; expected executed receipts are `evidence/manifest.json`, per-attempt `native-wire-validation.json`, and `referee-validation/live-referee-validation.json`. Prepared code and passing source tests are not those execution receipts.

## Verified temporal mismatch

F7 compact pretraining uses `DT=.02f` in the frozen `fast_runtime.cu:30` and `REK_POLICY_ACTION_STRIDE=1`, giving 50 Hz recurrence/decision steps. The human conditional-BC dataset preserves its recorded 50 Hz chronology without retiming. Intervening authentic-PPO updates used recorded trajectories, so this is not a claim that every update in the checkpoint's history used 50 Hz.

All eight live worker ready records specify hidden size 256 and two recurrent layers. The native architecture is the two-layer MinGRU; `native_policy.cu:109-123` advances its recurrent state once per observation step, and the worker protocol has no separate recurrent-update delta-time argument. At the observed 23.76-26.11 Hz, the same number of recurrent updates spans roughly twice the wall time of a 50 Hz sequence. Correcting a velocity feature's units does not rescale that recurrent transition.

The live encoder does use actual source QPC delta time for finite-difference velocity and angular-rate features (`encode_live.cpp:195,212-214`). Round countdown at index 189 and hit ages at 198/199 provide indirect elapsed-time information; busy flags at 182/183 are duration-based projections. Neither explicit delta time nor request age is a policy input. Thus the temporal mismatch is not a claim that all live rates or timing observations are wrong. It remains an unisolated transfer limitation, not an established explanation for the three losses. No resampling or recurrent timing change was introduced into this cohort.

## Next prepared work and provenance

The separately prepared event-based full-33-class human BC variant has not been run as part of this result. It did not produce the evaluated checkpoint. The next actual on-policy data stage is the separate all-eight CPU validation/export above; this report makes no claim that a new training run, checkpoint, or live evaluation has completed. Policy and runtime files were not changed for publication.

Authoritative closed source hashes:

- `root-campaign/ledger.json`: `dd9e0d6d3043cc79f1169234d28f8fc8a45f33f27ce9089e0b99192ddd7f3a63`.
- `root-campaign/campaign.log`: `5c668e43c51e3b8a21ed3806046cf8eaf231e9d58cc478c36d61206872c80116`.

Each counted attempt's `trial/summary.json`, `relay.stdout.jsonl`, and `worker.stdout.jsonl` provides its scores, startup timers, ACK classification, QPC cadence and architecture identity. Source hashes for the earlier native analyses remain in the preserved cutoff reports. The 50 Hz BC provenance is in `validation/human-attackbc-20260924-r1/dataset-manifest.json`; compiled compact legality and source hashes are documented in [ATTACK-READINESS.md](ATTACK-READINESS.md). All source evidence was retained. No commit or runtime action was performed by this publication task.
