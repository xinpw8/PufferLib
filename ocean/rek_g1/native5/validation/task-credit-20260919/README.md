# Temporal credit, episode handoff, and parity status

## Implemented corrections

The pinned native learner omitted the last action's reward at every rollout
boundary. The next batch stored that reward in slot zero and never credited it.
The CUDA regression reproduced a terminal five-point reward producing advantage
zero. The patched kernel produces five. Forty advantages/returns agree with the
reference within 8.42e-7. A separate reward/done/value tuple supplies the final
transition without another environment action.

The extra value forward operates on copied recurrent state. The actual native
network test, including four CUDA graph replays, verifies unchanged actor RNN
state, RNG, actions, observations, masks and parameters.

The compact training adapter also selected an action from a finished round's
observation, then executed it after resetting the arena. Its new training-only
autoreset publishes the next initial observation/mask while preserving the
previous transition's terminal reward/done. Forty-eight device checks cover
graph replay, cumulative results, nonterminal and failed arenas, and absence of
a second reset. Direct-runtime evaluation retains the final-state contract.
The trainer executable's `eval` command uses the same autoreset adapter as train.

The live evaluation driver now returns failure for `requested_duration_complete`.
That condition is an incomplete attempt, not an observed terminal result.

## Controlled native training experiment

All three runs used Spark GB10, the same frozen initial checkpoint
`dd335d696ab0f2ae891b448d174e3615d834cd5b8fd4cf8cffbae6c5126f4c46`,
seed 231, 512 arenas, 33,554,432 learner transitions, minibatch 8,192, replay
ratio 1, learning rate 0.0001, and entropy coefficient 0.01. Physics, inference
and learning ran in C++/CUDA. No Python interpreter or Torch runtime was used.

The reward was unchanged raw point differential. Contact-potential shaping was
disabled. No terminal win bonus or intrinsic exploration reward was added.
The simulated opponent was `recovered_bot1_v1`; the compact dynamics still lack
balance and knockdowns. Candidate training wins are not authentic REK wins.

| Run | Rollout | Gamma | Lambda | Full training SPS | Startup-inclusive SPS |
|---|---:|---:|---:|---:|---:|
| Existing learner | 128 | 0.999 | 0.995 | 934,986 | 918,293 |
| Boundary and autoreset fixes | 128 | 0.999 | 0.995 | 930,957 | 914,290 |
| Fixes plus task-time profile | 512 | 0.9998844821426083 | 0.9978673240629938 | 928,776 | 910,568 |

SPS counts one 20 ms learner transition once. It includes rollout and optimizer
work. Process-inclusive SPS also includes startup and checkpoint output. These
are three single runs in fixed order, not repeated performance confidence
intervals. The combined correctness fixes cost 0.43% in this measurement; the
task-time configuration was 0.66% below the existing learner.

The task profile covers 10.24 s per rollout, retains half of a reward across a
120 s round, and retains half of its combined gamma-lambda trace across 6.16 s.
The latter is the inspected longest move, 3.16 s, plus the recovered 3 s
no-recovery count. These are explicit hypotheses, not optimized settings or
evidence that the live client uses that count for every fall. Recurrent state
continues across rollout boundaries and resets at genuine episode endings.
PPO entropy regularization remains inherited and has not been exploration-tuned.

The full measurements are in `training-comparison.json`. Source, commands,
stdout/stderr, checkpoints and build hashes are retained privately under
`/home/spark-advantage/rek-training/task-credit-20260919-r1`.

## Parity and promotion

The existing visual recordings contain a material balance/root-motion gap.
The separate `balance-transition-20260919` audit uses session-held-out splits:
a root predictor improves one-step error but is worse than persistence at 1 s
in five of six groups. It is not integrated into the simulator.

The original recovered fall/countout rules now have a separate availability-
checked CUDA adapter and CPU/device tests. This adapter does not generate
physical contact, balance or root motion. It is not enabled on fabricated
upright/contact inputs in the compact trainer.

Neither the larger horizon nor passing software tests qualifies the candidate
as a faithful REK environment. Promotion requires completed authentic private-AI
trials plus targeted motion/contact/balance validation. Failed arena entry,
transport loss and timer expiry remain failed attempts, with their artifacts
retained separately from wins/losses.

## Authentic evaluation attempts

The task-time checkpoint SHA-256 is
`5bdad2893c5e97e682fdb33ab48a298e2cd2d9c03df724fa2e629cc1c9882246`.
All three attempts used the isolated Spark client and required private solo AI
proof. No global input or Windows client control was used.

| Attempt | Verified opponent | Actions applied | Last observed state | Qualification |
|---|---|---:|---|---|
| r1 | None yet | 0 | Private arena reservation exceeded entry timeout | Incomplete |
| r2 | Bot 1 / difficulty 0 | 28 | 0:0, 119.22 s remaining, pipe writer exception | Incomplete |
| r3 | Bot 1 / difficulty 0 | 3,803 | 5:10, 40.73 s remaining, REK process crashed with exit 5 | Incomplete |

The r3 trial ran after an isolated client restart with diagnostic-only writer
logging. Its process crash did not produce a writer-stage diagnostic. Neither
the diagnostic patch nor the temporal configuration can be claimed to have
fixed live reliability or improved fighting strength. No completed authentic
result exists for this checkpoint. The r3 MP4 is 7,123,223 bytes and validated,
but depicts an interrupted trial. Native raw captures are retained privately.

A separate native-packet export found 151 received hit packets and 169 score
packets in the ten earlier completed A/B trials. Their packet fields were absent
from the visual-only relay used by the initial balance audit. Receipt joins and
unknown execution/clock fields are documented in
`../balance-transition-20260919/native-hit-receipt/`.

The private training/test/attempt archive was copied to the physical file server:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\task-credit-20260919-r1\private-results.tar`.
Its SHA-256 is
`aaf80e7e72001b67ac2c76843b261e2501a02bb15732c1595e943223f6c44fe1`.
No proprietary game binary or raw gameplay teacher dataset is included in Git.
