# Observable-balance policy: authentic REK evaluation

The fresh physical PPO checkpoint from
[the fall-aware training run](observable-balance-physical-training-20260921.md)
completed a three-round development screen against the actual Windows REK
client, private Sparring Bot 1
(client difficulty 0). Training ran on Spark. Inference is native CUDA/BF16 on
Spark; the owned game client runs on `D21`, isolated desktop
`WinSta0\RekPolicyEval`. No global keyboard, mouse or foreground input is used.

## Frozen physical candidate

- Schema: `rek.native5.observable_balance.v1`.
- Checkpoint SHA256: `390007e256574d2fc5e1100eab4da048fccfec2532bd5e5a59bdcc19cf326310`.
- Training: 4,194,304 fresh-start transitions, `normalized_points_falls_v1`,
  physical `mujoco_cuda`, against `CandidateApproachDummy`.
- Selection: sampled, worker seed 73, unchanged across the screen.
- Encoder: separate native observable-balance encoder, SHA256
  `6eb359ba5b356881a081da4cf853d83027cb0beb8fece1e7ad487044c5db8a07`.
- Shared projection uses observed roots, pose differences, awarded points and
  availability flags. Joint correspondence remains unavailable. This test does
  not establish full physical/client trajectory parity.

| Round | Result | Points, policy : bot | Ordinary points | Five-point awards |
| --- | --- | --- | --- | --- |
| r86 | Win | 13 : 12 | 8 : 2 | 1 : 2 |
| r87 | Win | 16 : 4 | 11 : 4 | 1 : 0 |
| r88 | Loss | 7 : 17 | 2 : 7 | 1 : 2 |

These are development-screen rounds, not the earlier frozen arm-10 acceptance
cohort. The result is **2 wins / 1 loss**, aggregate points **36 : 33**, ordinary
points **21 : 13**, and five-point awards **3 : 4**. This small screen does not
demonstrate reliable strength or fall avoidance. No unsuccessful round was
retried or discarded. The prepared 20-round acceptance script has not been run.
Received five-point awards are reported as awards, not inferred causal trips.

For all three rounds, all six existing evidence checks passed: owned/native PID
agreement,
completed policy round, complete native capture, consistent terminal evidence,
fully reconciled points and verified referee packets. The owned client closed
after each round. Each round's one stream-end rejection is retained in the raw
receipt; it did not replace or retry the round.

LEFT_FRONT category 17 was requested twice in r86, five times in r87 and once
in r88; all eight calls returned true from local execution. Server acceptance
and successful contact for those requests remain unknown. This is distinct
from the previous arm-10 cohort's zero
category-17 requests and does not prove the attack has been learned well.

## Imitation candidate, r89

The separately selected [100-epoch imitation checkpoint](human-observable-bc-20260921.md)
was also tested against private Sparring Bot 1 with sampled selection, seed 73,
the same encoder and the explicit pure-BC availability mask. This was one
development-screen round, not a statistically powered comparison.

- Selected checkpoint SHA256:
  `ec640746b48cae9348e2508d0dd9678db3e250bdfb6dc8cadffc14d515dc81ba`.
- Mask SHA256:
  `656e3470bf6b5c1634a6c3b14544752129938a1b2fe28d1a4a4e54053fe1ce75`.
- Result: **loss, 2 : 22**. Ordinary points were 2 : 12; five-point awards 0 : 2.
- All six evidence checks passed, and the owned client closed normally.
- LEFT_FRONT17 was requested once and returned true locally. Server acceptance
  and contact for that request remain unknown.

The low held-out command loss did not produce a useful fighter in this screen.
This result does not establish that behavioral cloning cannot help; it rejects
promoting this particular checkpoint on its aggregate supervised metric.

## Physical continuation and r110

After r89 and its evidence validation completed, an additional 4,194,304-transition
physical PPO run was launched from the **physical** checkpoint `390007...`, not
the imitation checkpoint. The weights-only restart uses fresh optimizer/RNG/RNN
state and the same normalized reward, schema, opponent and hyperparameters.
The run [completed successfully](observable-balance-physical-continuation-20260921.md)
at 5,990.86 complete-training-loop SPS. Its final checkpoint is
`a11ace1655cf1f3176dc0732e07fef47a6d3e8b32177a5da47cd075ac4a231bd`.

Frozen authentic round r110 used that checkpoint with the same schema, encoder,
sampled selection and seed 73. It won **19 : 17**, with ordinary points **4 : 7**
and five-point awards **3 : 2**. All six evidence checks passed and the owned
client closed. No LEFT_FRONT17 request occurred. This one narrow win depended
on the five-point award difference and does not establish improved or consistent
fighting. The earlier checkpoint's r86..r88 results are not pooled with r110.

Spark run directory:
`/home/spark-advantage/rek-training/physical-observable-balance-20260921-r1/train-warm-physical-continue-r1`.
Runner SHA256:
`b81c0a0dc343c38c66f764e0f5c44e840aefc5f72c3485b0b4834b62708141a8`.

After r110 closed and validated, the first controlled higher-learning-rate arm
was launched with initial LR 0.001. It starts from original checkpoint
`390007...`, exactly as the completed 0.0001 continuation did; it does not start
from `a11ace...`. Seeds, 4,194,304-transition budget, normalized rewards,
observations, opponent and remaining hyperparameters stay fixed. Its new output
is `train-lr-lr001-r1` under the same Spark stage. Runner SHA256 is
`17e90c448e3ba3230aaaa2f813bf25ea3b0a74824887cb4ffa5ad2698b6e50b2`.
This records an executed launch, not a completed result or improvement claim.

## Private evidence

Windows trial root:
`C:\rekagent\work\consistent-fighter-20260919-r1\live-point_difference_v1-r86`.
The legacy directory bucket `point_difference_v1` is not the training reward
mode; the frozen candidate manifest identifies normalized fall-aware training.

Validation root:
`C:\rekagent\work\observable-balance-live-eval-20260921-r1\r86-validation`.

NAS archive:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\normalized-sweep-r1\authentic-r86`.
All 49 files, including exact native capture and inference logs, were copied
with source-before/source-after/copy hash agreement. Archive receipt SHA256:
`86465edbd7320bfc0019e4b052accad972cb62991b63037412d447a1e2e32761`.
Round r87 is preserved in the adjacent `authentic-r87` folder, also 49 files;
its receipt SHA256 is
`35611a94901f85d3bc775794c3fa578e998af69c9f7210918c25e62ff63cab94`.
Round r88 is preserved in the adjacent `authentic-r88` folder, also 49 files;
its receipt SHA256 is
`1868865c6084c5cff6f5140686dd7236269c51c6923e85944a00068c9e160fcd`.
The BC round is preserved in adjacent `authentic-r89`, also 49 files, with
receipt SHA256
`e72cd218aed586bb7a2573cef538b44edec420921f62a94b677e282784331384`.
The continuation round is preserved in adjacent `authentic-r110`, also 49 files,
with receipt SHA256
`edad788b119aaae0801139c8c8cd123e7c4143898580595404f8320382391364`.
No checkpoints, proprietary binaries, raw game captures or account credentials
are published in this repository.
