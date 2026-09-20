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
| r7 | unchanged BC epoch-5 retry | 12:17 | Loss |
| r8 | masked PPO control, no human BC | 23:11 | Win |
| r9 | globally balanced BC epoch 5 plus PPO | 5:17 | Loss |
| r10 | authentic-trajectory PPO, frozen-value GAE | 10:6 | Win |
| r11 | authentic-trajectory PPO, complete MC, zero learned baseline | 11:9 | Win |
| r12 | unchanged authentic-trajectory GAE | 8:10 | Loss |
| r13 | unchanged authentic-trajectory complete MC | 20:12 | Win |
| r14 | unchanged authentic-trajectory GAE | 18:9 | Win |
| r15 | unchanged authentic-trajectory complete MC | 13:21 | Loss |

The original checkpoint SHA256 is
`61f97b0b0a4504c6bdd0ee16d369ad4c1915e3cdf73d6358bab01ce64c8fde3f`.
The four additional sampled rounds are 3 wins and 1 loss, with total points
69:57. Their non-five-point awards total 29:42, while five-point awards total
40:15. Their advantage therefore does not demonstrate superior ordinary
striking. Two earlier outcome-checkpoint development wins remain a separate
cohort. They do not turn these results into a held-out consistency claim.

All completed rounds through r15 passed the existing control-coverage, native packet
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
unchanged. The unchanged retry, r7, completed with a 12:17 loss and 95 native
attack requests. This does not support promoting BC as a fighting improvement.

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
Its first authentic round, r8, won 23:11 with 92 attack requests. Observed
non-five-point awards were 13:11; five-point awards were 10:0. A single win
does not establish a higher win rate. Compact dynamics still omit balance
and countout dynamics; simulation win rate is not authentic fighting strength.

The corrected BC epoch-5 checkpoint then completed a 33,554,432-transition PPO
continuation in a separate reserved GPU window: 929,730.5 training SPS,
36.09049 s training loop, 36.78 s process wall time. The exact warm-start byte
comparison passed. This run used the same 512 arenas, horizon 512, minibatch
8192, learning rate 0.0001, entropy coefficient 0.01, reward/discount and
simulator settings as the masked control. Checkpoint SHA256:
`98d72685b73cf89641fc56dd0119ad73015b2ea6ede682fefe2b0a256d03f10e`.
Its first authentic evaluation, r9, lost 5:17 with 60 attack requests. Both the
BC-only and BC-plus-PPO completed trials lost. Neither is promoted as an
improvement over the original policy or masked PPO control.

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

Completed Spark worker, native BC and masked PPO artifacts are archived in the
same NAS project folder under `native-completed-20260920T0057Z-r1`.
The archive contains 2,235 files and is 117,508,846 bytes compressed. Source
manifests before and after packing, archive comparison, and NAS copy hashes
passed. Archive SHA256:
`3c163d262fcd5a6e55938bb1cd49bf1300468880a93cc5a1b685724dfaa997c4`.

The completed native authentic-PPO subset is archived under
`native-authentic-ppo-controls-r2` in the same NAS folder. Its 21 payload
files total 7,483,144 bytes. Source-before, source-after and NAS readback
hashes all match. Manifest SHA256:
`cb877487e7a6a97f186403ab7deb4caf61e645e15d23ac11a0ba67668dd07ba3`.

## Learning from authentic trajectories

The four original sampled development rounds provide 22,585 ordered policy
decisions and 68 native score awards. Replaying the original unmasked BF16
checkpoint with the original seed and worker resets reproduced all 22,585
sampled actions exactly in 6.24 seconds. This recovers frozen behavior log
probabilities and values without inventing an executed-action label. Three
terminal-race rejected requests remain in history with zero actor loss weight.

An initial native PPO dry run computed full-round, actual-time-discounted GAE
on CUDA with zero error against its CPU reference. It did not optimize: native
batched BF16 train-forward differed from sequential inference by at most
0.0157486 in chosen-action probability ratio. Subsequent distribution-level
measurements, described below, justified a separately declared numerical
approximation. No authentic PPO checkpoint or improvement is claimed from
the initial dry run.

The first trajectory export used 0.999 / 0.995 reference gamma / lambda.
Review caught that unintended change before any optimizer update. A fresh
export restores the task-tailored 20 ms values 0.9998844821426083 and
0.9978673240629938. All observation, action, mask and timestamp bytes remain
unchanged. The corrected export also replayed all 22,585 actions exactly in
4.42 seconds. The earlier dataset is preserved as superseded.

Further numerical diagnosis established exact agreement between existing
Puffer one-step forward and the frozen native teacher across all 22,585 rows.
The batched path's mean legal-distribution KL was 4.5191e-8, maximum KL
0.000156797, and initial clipped fraction zero at clip 0.2. A separately
declared bounded-BF16 approximation is therefore being tested; it is not
called exact batched parity.

More materially, the original critic is miscalibrated on authentic states:
actor-weighted mean value 7.91857 versus actual Monte Carlo return 0.17282.
At round starts, predicted values are approximately 14 versus true returns
approximately +/-0.501. The task-time GAE targets retain substantial future
critic error. Both a complete-round Monte Carlo control with no learned-value
baseline and zero value loss and a frozen-value GAE control completed one
native PPO epoch. They took 2.84 and 3.01 seconds respectively, including
diagnostics, and each performed 178 small updates from the original policy.
Actual score rewards and terminal outcomes are unchanged. See the
[native PPO control results](../authentic-ppo-20260919/README.md).

The GAE checkpoint's first authentic trial, r10, won 10:6 with 119 attack
requests. All awards in that round were non-five-point awards. Native packet,
referee and continuous-control verification passed. This is a development
result, not evidence that four training episodes establish generalization.

The complete-MC checkpoint's first authentic trial, r11, won 11:9 with 104
attack requests. Its awards were 6:9 non-five-point points plus one local
five-point award. Existing native/referee/control checks passed. Both new
checkpoints initially had one new development win each. Their results remain
separate checkpoint outcomes and must not be pooled as one policy's win rate.

The unchanged GAE checkpoint then lost r12, 8:10, and won r14, 18:9. The
unchanged MC checkpoint won r13, 20:12, then lost r15, 13:21. Both finished
2 wins / 1 loss. GAE total points were 36:25; MC total points were 44:42.
Neither establishes a consistent winner. No checkpoint was promoted into
the separate 20-round acceptance evaluation on these results.

r12 led non-five-point awards 8:5 but conceded one five-point award. Native
referee receipts confirm an own count followed by a knockout. r13 awards
were 10:7 non-five-point points and 10:5 five-point points. r12 and r13 both
passed strict contact/referee checks, with maximum applied-control gaps
0.074061 s and 0.065359 s respectively. The count episode is analyzed in
[balance observability](balance-observability.md); it does not justify
inventing toppling dynamics in the compact simulator.

r14 passed the same existing checks. Its awards were 8:9 non-five-point
points and 10:0 five-point points, with 102 attack requests and maximum
applied-control gap 0.081777 s. These are award amounts, not inferred
move-specific hit labels.

r15 also passed strict verification, with maximum control gap 0.084469 s and
106 attack requests. Its non-five-point awards were 3:16 and five-point
awards 10:5. Thus this loss was not simply a missing own-knockdown signal.
Across its three trials, MC led five-point awards 25:10 but trailed ordinary
award amounts 19:32. The next learning batch uses all three actual MC-policy
rounds, including this loss, retaining real score consequences and measured
poses. Simulation wins cannot substitute for subsequent authentic evaluation.
