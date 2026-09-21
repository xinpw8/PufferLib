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
The arm [completed successfully](observable-balance-physical-lr001-20260921.md)
at 6,150.03 complete-training native SPS (6,065.16 whole-process SPS). Its final
checkpoint is
`d34f3fefc71b3ca4977590be142100beca58872fd2051813a56bc57536bae60f`.

### LR 0.001 attempt r111: no round

The frozen checkpoint's first authentic attempt reached authenticated Home and
issued the private-arena entry request, but remained in FreePlay until the
existing entry timeout. The result is **incomplete**, with zero input sources,
predictions or policy actions and no round identity or opponent. It is neither
a win nor a loss, and supplies no evidence of fighting strength. The isolated
owned client closed normally. This failed attempt is retained rather than
silently replaced by a successful round.

The recorded request returned `private_practice_reservation_requested` with
`server_acceptance_observed=false`. All 305 relay snapshots remained in Lobby
and lacked a FightCoordinator. No explicit server/authentication refusal was
observed; the underlying reservation failure is unknown. No matching native
fight capture was produced. A fresh bounded attempt r112 was then launched
with the identical frozen checkpoint and configuration, without extending the
timeout or changing the bridge.

### LR 0.001 round r112: loss

The fresh attempt entered private Sparring Bot 1, completed the 120 s round
and lost **6 : 22**. Ordinary awarded points were **6 : 7**; five-point awards
were **0 : 3**. All six existing evidence checks passed, and the isolated owned
client closed. The one stream-end rejection remains recorded. LEFT_FRONT17 was
requested once and returned true locally; server execution and successful
contact remain unknown. This is one completed development round, with r111
retained separately as a startup failure. It does not demonstrate improvement
or consistent fighting.

After r112 completed and validated, the controlled LR 0.015 arm was launched
from the same original `390007...` weights. Its output is `train-lr-lr015-r1`
under the physical stage. The 4,194,304-transition budget, seeds, normalized
reward, observation schema and other parameters are unchanged. This is the
pinned trainer's default Muon learning-rate magnitude, not a demonstrated
optimum. The arm [completed successfully](observable-balance-physical-lr015-20260921.md)
with 4,194,304 transitions and 16 rollout/training iterations, runtime and
wrapper exit codes 0, and 5,914.08 complete-training native SPS
(5,839.78 whole-process SPS). Its final checkpoint is
`a2481a82dc2c116b88e94c2461ccaef970240e2821ee68edae57cd8984ad1a6b`.

### LR 0.015 round r113: loss

That frozen checkpoint completed a 120 s authentic round against private
Sparring Bot 1, difficulty 0, with sampled selection and worker seed 73.
It lost **2 : 16**. Ordinary awarded points were **2 : 1**; five-point awards
were **0 : 3**. All six existing evidence checks passed, and the isolated owned
client closed normally. The one stream-end rejection remains recorded.
LEFT_FRONT17 was requested five times and returned true locally on all five
requests; server acceptance, execution and successful contact remain unknown.

This checkpoint's development cohort consists of r113 alone. It is not pooled
with r86..r88, r89, r110 or r112, and r111 remains a separate incomplete entry
attempt. The higher training-round win count did not establish authentic
strength: this one completed round supplies no basis for promotion or a claim
of consistent fighting. No checkpoint is promoted from this comparison.

## Recovered Bot 1 training candidate, r114..r117

The [recovered G1 Bot 1 physical training arm](observable-balance-physical-bot1-training-20260921.md)
completed 4,194,304 transitions at LR 0.015 from original `390007...` weights.
It retained the reward, observations, seeds and transition budget of the prior
LR 0.015 arm, while replacing the approach dummy with source-derived Bot 1
tactics and continuous native commands. Candidate cadence/RNG and physical
parity limits remain explicit in that report. Its distinct final checkpoint is
`7c34eaa9f00c1ee97f8cbd8abf894d773d8a838547a10e496bcb5980de821c84`.

Frozen authentic round r114 used that checkpoint with
`rek.native5.observable_balance.v1`, sampled selection, worker seed 73 and no
feature mask. It completed a 120 s round against private Sparring Bot 1,
difficulty 0, and won **21 : 15**. Ordinary awarded points were **6 : 15**;
five-point awards were **3 : 0**. All six existing evidence checks passed,
and the isolated owned client closed normally. The one stream-end rejection
remains recorded. LEFT_FRONT17 was requested twice and returned true locally
on both requests; server acceptance, execution and successful contact remain
unknown.

This was the first completed development round recorded for checkpoint
`7c34eaa9...`. Its win depends on the five-point award difference while ordinary
points favor the opponent. It does not establish consistent fighting or
attribute awards to particular attacks. Results from the other checkpoints
above are not pooled into this checkpoint's cohort. No promotion or authentic
physics/controller parity claim follows from this single round.

### Same-checkpoint repeats, r115..r117

The three subsequent frozen rounds retained checkpoint `7c34eaa9...`, sampled
selection, worker seed 73, no feature mask and private Sparring Bot 1 at
difficulty 0. All completed their 120 s rounds.

| Round | Result | Points, policy : bot | Ordinary points | Five-point awards |
| --- | --- | --- | --- | --- |
| r115 | Loss | 9 : 10 | 9 : 5 | 0 : 1 |
| r116 | Win | 10 : 6 | 5 : 6 | 1 : 0 |
| r117 | Loss | 15 : 25 | 10 : 10 | 1 : 3 |

All six existing evidence checks passed for each round, and all three isolated
owned clients closed. Each round's one stream-end rejection remains recorded.
LEFT_FRONT17 was requested three times in each round, with all nine requests
returning true locally. Together with r114, that is eleven requests and eleven
local true returns; server acceptance, execution and successful contact remain
unknown. No round was discarded or replaced.

For the completed r114..r117 development cohort, the result is **2 wins / 2
losses**, total awarded points **55 : 56**, ordinary points **30 : 36**, and
five-point awards **5 : 4**. These totals include only this frozen checkpoint.
Both wins have an ordinary-point deficit and a favorable five-point award
difference; r115 has the reverse pattern, while r117 has tied ordinary points
and an unfavorable five-point award difference. The four-round sample does not
establish reliable strength, improved contact skill or authentic controller
parity. Awards are not assigned causally to particular requests. No checkpoint
is promoted from this cohort.

### Request-time behavior profile

The same four rounds contain 244 attack requests, all matched to an exact worker
round identity, observation sequence and policy action. At request time, 89 had
absolute root bearing at most 45 degrees, 64 were above 45 through 90 degrees,
and 91 were above 90 degrees. Median absolute bearing was 72.302 degrees.
Median horizontal root distance was 0.777 captured Unity units; 61 requests
occurred beyond one unit. Physical unit calibration remains unverified.

The worker produced 22,789 decisions: 21,394 with four legal actions, 899 with
16, and 496 with all 33. All 244 attack requests came from the last group.
These are descriptive observations of decision opportunities and requested
geometry. They do not classify requests as hits or misses, establish move reach,
or attribute any score or fall to an attack.

The CPU-only profile and per-action distributions are retained under
`C:\rekagent\work\observable-balance-live-eval-20260921-r1\request-profiles`.
Sixteen input files and the profiler are hash-bound in the aggregate JSON;
its SHA256 is
`7eb6a5f1be2e174a6636d086884296489d2f05a7a5d6bddef6a465621c266b36`.
Four synthetic tests passed. This analysis adds no acceptance condition or
reward-shaping term.

## Frozen 8M-continuation cohort

The subsequent physical continuation completed 8,388,608 new learner
transitions and produced checkpoint
`ef85a01b207d033417d8e1bc9d2b32b9eba610e784180dc7dd287a894acf2f4b`.
A prospective authentic cohort was registered at 2026-09-21T18:37:18.8968009Z,
before its first launch. Its requirement is at least 18 wins in 20 completed
120 s rounds against private Sparring Bot 1 / difficulty 0. The third non-win
stops the cohort because that threshold can no longer be reached. Draws count
as non-wins. An incomplete or invalid attempt stops execution and is retained
without being classified as a completed win or loss.

Checkpoint, sampled selection, seed 73, feature mask, encoder and evaluation
scripts remain fixed within this cohort. No result or promotion is inferred
from its changing-policy training metrics. The shared observation schema and
encoder are unchanged from r114..r117. The Windows harness uses its verified
owned isolated desktop and emits no global keyboard or mouse input.

Private plan directory:
`C:\rekagent\work\observable-balance-live-eval-20260921-r1\cohort-continue8m-r1`.
Candidate config SHA256:
`9e8a8f126ff21af1cd54498274aa8845684e3296991be525280ab12b22b81880`.
Wrapper SHA256:
`871a91a3506034a2a5ae9b6df26861a77d70c32a19c5996adb076ae5cbe325b5`.
Attempts start at r118. Results remain pending at registration.

Completed rounds so far, learner first:

| Attempt | Outcome | Total points | Ordinary points | Five-point awards |
| --- | --- | ---: | ---: | ---: |
| r118 | Loss | 0 : 12 | 0 : 7 | 0 : 1 |
| r119 | Win | 13 : 8 | 3 : 8 | 2 : 0 |

Both have all six existing evidence checks satisfied and verified owned-client
closure. These are interim results from the same frozen cohort, not a promotion
or an independent selection set. Five-point awards are not causally attributed
to individual move requests.

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
Incomplete attempt r111 is preserved separately under
`2026-09-21\physical-observable-balance-r1\authentic-r111-incomplete`, with
44 files and receipt SHA256
`a675bc2531a70196203b40354685ab68c919e0f696ff4428c24f372e2ffc2f3c`.
Its manifest explicitly leaves completed-round checks unclaimed and records
the missing fight capture.
Completed round r112 is preserved under adjacent normalized-sweep
`authentic-r112`, with 49 files and receipt SHA256
`d8600d2fa69f12899dadfc20243352ef29eb518841bcdeff17eaa9c95932590e`.
Completed round r113 is preserved under adjacent normalized-sweep
`authentic-r113`, with 49 files and receipt SHA256
`a131d2eedcd184a7d1c4a02ecec143b429f3cb822c842936fc48d102b197e7d2`.
Its validation root is
`C:\rekagent\work\observable-balance-live-eval-20260921-r1\r113-validation`.
The derived summary's five referenced input hashes were independently checked;
raw score events sum to 2 : 16, consistent with both terminal records and the
referee validation. The archive receipt hash was also checked directly; the
receipt records source-before/source-after/copy agreement for its 49 files.

Completed round r114 is preserved under adjacent normalized-sweep
`authentic-r114`, with 49 files and receipt SHA256
`306b51688eab2ffc8011e054e2eace28e6b497f574eecec9d3d509351a49c496`.
Its trial and validation roots are
`C:\rekagent\work\consistent-fighter-20260919-r1\live-point_difference_v1-r114`
and `C:\rekagent\work\observable-balance-live-eval-20260921-r1\r114-validation`.
The derived summary's five referenced input hashes were independently checked.
Raw score events sum to 21 : 15, with ordinary points 6 : 15 and five-point
awards 3 : 0, consistent with the recorded terminal result and referee
validation. The archive script verified source-before/source-after/copy hashes
for every file in the fresh destination; no existing evidence was overwritten.

Completed rounds r115 and r116 are preserved under adjacent normalized-sweep
`authentic-r115` and `authentic-r116`, with 49 files each. Their receipt SHA256
values are respectively
`bbe5cc6dc365aa8ced5cca2c277ef475f9342ceb86e0dd495e2280fc2b58feea`
and `4014b661cf9831be20f0afc42564c8da3c9761c655cae0c6973d30ebc878d251`.
Their trial and validation paths use the same roots above with suffixes
`r115` / `r115-validation` and `r116` / `r116-validation`.
Each derived summary's five input hashes were independently checked. Raw score
events independently reconcile to the table above and each terminal result.
The archive script checked source-before/source-after/copy hashes; a subsequent
read rehashed all 98 archived files, verified their sizes and both receipts,
and found no extra files beyond the two receipts. Both destinations were fresh.

Completed round r117 is preserved under adjacent normalized-sweep
`authentic-r117`, with 49 files and receipt SHA256
`ee6f70fba84eea7f8061df3cfc7c198ef82660677aa8148f70b6fc5c8f3d436b`.
Its trial and validation paths use the same roots above with suffixes
`r117` and `r117-validation`. All five derived-summary input hashes matched;
raw score events independently reconcile to 15 : 25, ordinary points 10 : 10
and five-point awards 1 : 3. The archive script verified source-before/source-after/
copy hashes. A subsequent read reverified all 49 NAS file hashes and sizes,
the receipt hash and exact file count. No existing evidence was overwritten.

Round r118 is preserved in adjacent `authentic-r118`, with 49 files and receipt
SHA256 `f23aecbfccc689e6df0300cf96c044eb0e0ee06268c66d80925943357a7b8a84`.
Every archived hash and size, each source hash, and the exact 50-file total
including the receipt were independently reverified.
Round r119 is preserved in adjacent `authentic-r119`, also 49 files plus
receipt. Its receipt SHA256 is
`5ee9a97a2ecd8f634fa6e4f8f74d890ea87e540657f1721b3023b6e07abe13a3`.
All archived and source hashes/sizes and the five derived-summary input hashes
matched; raw score packets independently reconcile to 13 : 8.

No checkpoints, proprietary binaries, raw game captures or account credentials
are published in this repository.
