# Conditional human-attack fine-tune, 2026-09-24

Native training actually executed on `spark-4ae3`, 09:27:36.809 to 09:28:07.265 UTC. Exit 0, 30.456 seconds, 155 updates, five fixed epochs, LR 1e-4, H128. This is a supervised training duration, not environment or PPO SPS. Unrelated GPU work was present and untouched.

Candidate checkpoint: `/home/spark-advantage/rek-training/scorecredit-human-attackbc-20260924-r1/train-five-epochs/policy.bin`, SHA-256 `5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4`. Starting checkpoint: `0daf90a96442d541d38d9dd4f8fe765917c3c766ca80019855cfd7d358d9fe12`.

## What changed

The existing CUDA BC trainer fine-tuned the current policy on 77 recorded attack requests from the first human round. The second round supplies 109 development labels. All 11,985 observation rows retain chronology; only attack requests contribute supervised loss. Conditional softmax support is categories 16 through 32 in training only. Live inference keeps the existing full 33-category selection and all-ones feature mask. No forced attacks, timing gate, cooldown, or deployment attack mask was added.

The two old human recordings lack 11 current input fields. Those are explicitly masked during this supervised update. This partial-input/full-input mismatch, old network versus rendered joint poses, and 50 Hz versus variable live sampling remain limitations. Requests are not proof of accepted or successful attacks. The development round was previously inspected and is not an untouched test.

## Measured training results

| Metric | Initial | Fixed epoch 5 |
| --- | ---: | ---: |
| Training conditional cross-entropy | 7.83683 | 3.14528 |
| Training exact attack-choice accuracy | 0% | 27.27% |
| Development conditional cross-entropy | 3.79579 | 2.68312 |
| Development exact attack-choice accuracy | 6.42% | 43.12% |
| Category 17 training argmax recall, 16 labels | 0% | 0% |
| Category 17 development argmax recall, 2 labels | 0% | 0% |

Most improved exact classification was category 21. The small development kick sample cannot establish learned kick competence. Epoch 5 was fixed before training; intermediate checkpoints were not selected using development performance.

## Full live-input drift

The unchanged native worker engine was replayed on all 1,702 original requests from closed `credit5s-s1001-retry3`, with original full 223 inputs, legal masks, reset history and seed 1001. The baseline reproduced all 1,702 recorded actions exactly.

- Candidate changed 149 sampled actions on that fixed observation stream.
- Mean legal-distribution KL: 0.03010; maximum: 1.70907.
- Sampled attacks: 47 baseline, 35 candidate.
- Mean legal attack probability: 0.02769 baseline, 0.02306 candidate.
- Mean unmasked attack probability: 0.49903 baseline, 0.33862 candidate.
- Category 17 sampled: zero for both; mean legal probability increased from 0.00001239 to 0.00003343.

This is an offline sensitivity diagnostic, not a counterfactual game rollout. It proves that conditional attack fine-tuning can alter overall attack frequency despite zero direct movement-logit loss. It does not establish improved live fighting. Authentic evaluation is required before promotion; no such promotion is made here.

Raw native metrics and execution receipt: `training-metrics.jsonl`, `training-execution.json`. Full drift receipt: `live-drift-report.json`. Eight CPU tests and the unchanged native dataset reader passed. No Python was used for training or inference.

## Preserved evidence

The closed native stage, including all five epoch checkpoints, dataset, source, native metrics and full-input drift, is archived at `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\scorecredit-human-attackbc-native-20260924-r1\evidence.tar.gz`. Size 16,163,800 bytes, 38 entries; SHA-256 `52f60141fa92a7019112385a999d30c86c36c518f8a8b3403491baf641f9a736`. Transfer, NAS readback and archive listing passed. Sources remain on Spark. See `archive-receipt.json`.

The subsequent authentic live cohort is separate at `/home/spark-advantage/rek-training/human-attackbc-live-20260924-r1`, prospective seeds 1101 through 1120, with the same 18-of-20 target and stop after the third completed nonwin. Preparation tests passed; preparation alone is not an evaluation result. A separately labeled seed-1199 runtime-crash diagnostic is excluded from training and counted evaluation.
