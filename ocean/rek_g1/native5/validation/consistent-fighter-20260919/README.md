# Consistent-fighter development experiments

Checkpoint/configuration comparisons against authentic private REK Sparring
Bot 1, difficulty 0. The game runs on D21's isolated Windows desktop; native
BF16 policy inference runs on Spark. No global keyboard or mouse input is used.
These are development results, not the separate 20-round frozen evaluation.

## Current authentic results

| Trial | Configuration | Awarded points, policy : bot | Result |
| --- | --- | --- | --- |
| r1 | original outcome checkpoint, sampled | 18:10 | Win |
| r2 | same | 8:21 | Loss |
| r3 | same | 17:15 | Win |
| r4 | same | 26:11 | Win |
| r5 | original outcome checkpoint, masked argmax | 5:19 | Loss |
| r6 | globally balanced BC epoch 5, sampled, 215 retained features | 1:2 at interruption | Incomplete |

The original checkpoint SHA256 is
`61f97b0b0a4504c6bdd0ee16d369ad4c1915e3cdf73d6358bab01ce64c8fde3f`.
The four additional sampled rounds are 3 wins and 1 loss, with total points
69:57. Their non-five-point awards total 29:42, while five-point awards total
40:15. Their advantage therefore does not demonstrate superior ordinary
striking. Two earlier outcome-checkpoint development wins remain a separate
cohort. They do not turn these results into a held-out consistency claim.

All five completed rounds passed the existing control-coverage, native packet
reconciliation, and referee checks. The argmax configuration issued only 17
attack requests, compared with 92 to 110 in the four sampled rounds. These
request counts do not assert executed or successful attacks.

In r6, Unity frame 4714 published observation 1756 at QPC 5463992929355;
frame 4715 handled its action at QPC 5463995439302. At 10 MHz the age was
0.2509947 s, exceeding the existing 0.250 s observation limit. The action was
rejected as stale and the watchdog ended control. That prediction's native
worker latency was 0.783744 ms. The partial 1:2 score is retained, with no
terminal outcome inferred. The game was closed after verifying ownership and
isolation. Its recorder remains a partial file with a truncated final JSON
record, and both strict native analyzers reject that input. It is preserved
unchanged. The same checkpoint is queued for an unchanged retry.

## Learning experiments

Human command export preserves all 186 recorded move requests and 14,300
movement requests across the two September 17 rounds. It provides 11,985
ordered observations, with 4,943 training labels and 4,990 held-out labels.
Ramped or ambiguous commands remain in the ledger but have no classification
target. Current and future commands never enter pre-action observations.
This is one human session, about four minutes, with limited move coverage.

See [native BC results](../native-bc-20260919/README.md) for the unweighted,
group-balanced and corrected global-normalization runs. Aggregate accuracy
improved mainly through neutral predictions; tactical held-out generalization
remains weak. BC success is not claimed. The corrected epoch-5 candidate is
`bf9a737df0f60445c174eef7d051462b60f020fe2ff6975a983dc34ef50262a1`.

The shared input mask excludes only historical fields 176 through 183 that
could not be recovered from the human recording. Its SHA256 is
`7e5a991e79b495133a92571beb1530b534580d7b55c755603df4a6668ec31eaa`.
BC, live inference and subsequent PPO use the same mask. This excludes features;
it does not claim they were measured as zero. GPU publication integrates the
mask into the existing kernels with no new per-step host copies.

A masked PPO control completed 33,554,432 transitions at 905,299 training SPS.
BC overlapped approximately 25.3 s of that 37.0645 s training loop, so this is
GPU-shared throughput. It must not be presented as an uncontended regression
measurement. The final control checkpoint is
`bf2f962dad618f6cfcd393f3beb6c00afcb69f733bbba3fb918662b2844f9cb9`.
It still requires authentic evaluation. Compact dynamics still omit balance
and countout dynamics; simulation win rate is not authentic fighting strength.

The corrected BC epoch-5 checkpoint then completed a 33,554,432-transition PPO
continuation in a separate reserved GPU window: 929,730.5 training SPS,
36.09049 s training loop, 36.78 s process wall time. The exact warm-start byte
comparison passed. This run used the same 512 arenas, horizon 512, minibatch
8192, learning rate 0.0001, entropy coefficient 0.01, reward/discount and
simulator settings as the masked control. Checkpoint SHA256:
`98d72685b73cf89641fc56dd0119ad73015b2ea6ede682fefe2b0a256d03f10e`.
Its authentic fighting strength is pending evaluation.

## Reproduction and evidence

- Run `aggregate_authentic_trials.cjs CURRENT_ROOT PREVIOUS_ROOT NEW_OUTPUT`
  to reproduce the public-field summary. Holdout membership requires an
  explicit cohort manifest; new attempts default to development.
- Current Windows root: `C:\rekagent\work\consistent-fighter-20260919-r1`.
- Previous root: `C:\rekagent\work\reward-objective-20260919-r1`.
- Raw human export: `C:\rekagent\work\imitation-20260919-r1\dataset-r1`.
- Spark BC: `/home/spark-advantage/rek-training/imitation-20260919-r1-native-bc`.
- Spark masked PPO: `/home/spark-advantage/rek-training/policy-feature-mask-20260919-r1`.

The first four complete Windows trials and native recordings were copied to
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\windows-development-r1-r4`.
All 164 files, 1,571,643,419 bytes, passed source/destination SHA256 comparison.
Archive manifest SHA256:
`2c99aa6211ca5baf2abc1df1078ce81e4d63b725423a74d54e92e4f9d9e09338`.
Private recordings, proprietary assets, and checkpoint binaries are not included
in this source report.
