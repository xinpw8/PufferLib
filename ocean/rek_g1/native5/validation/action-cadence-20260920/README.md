# Five-observation action cadence

The opt-in experiment restricts the learner to existing hold action 0 on four
of every five observations. Default stride 1 is unchanged. Recurrent inference,
value-learning samples, physics, canned-action durations, rewards and discounts retain
their 50 Hz clock. This is a nominal 100 ms decision interval, not fivefold
faster training. Authentic fighting improvement has not been measured.

## Why test it

The finalized r21-r28 recordings contain 12,659 completed nonzero owned-yaw
sign runs. Per-round medians are 0.03340 to 0.03441 s; 11,542 runs are below
0.1 s. None of the 9,863 runs entirely within saved projected-busy windows
lasts 0.5 s. These use retained `input.desired_action`, so hold0 does not
incorrectly count as release. The bridge's recovered keyboard-yaw ramp is
0.5 s, and sign changes or release restart it.

Exact native `REK_Input` request prefixes independently show the effect:

| Recorded quantity | r21 | r28 | Two human rounds |
| --- | ---: | ---: | ---: |
| Outgoing movement commands | 6,565 | 6,523 | 14,300 |
| Nonzero yaw commands | 4,036 | 3,788 | 3,999 |
| Nonzero absolute yaw, median | 0.070384 | 0.068008 | 0.702273 |
| Nonzero absolute yaw, p95 | 0.546014 | 0.237372 | 1.000000 |
| Absolute yaw >=0.9 | 152 | 0 | 1,712 |
| Completed nonzero sign-run median, s | 0.034044 | 0.033244 | Not compared |

Joining each prefix to its strictly preceding ready source within 0.05 s gives
2,545/4,810 and 3,094/5,552 nonzero-yaw commands during projected busy in r21
and r28. Their nonzero absolute-yaw p95 values are 0.20304 and 0.20597.
Two prefixes per round lack a qualifying source and are excluded from that join.

This supports frequent outgoing ramp interruption. It does not prove server
execution, physical angular harm, missed strikes, or loss causation. Busy is
the saved `dispatched_request_v4_duration` projection. A visual client's
transport-complete condition can clear bridge move-in-flight before that
projected duration ends, so outgoing yaw can resume during projected busy.
The instantaneous source velocity is often zero between callbacks and is not
evidence that all outgoing requests were zero. Human observations are from one
session and are not context-matched to policy play.

The [file-only measurement](../consistent-fighter-20260919/yaw-cadence-evidence.cjs)
joins ready source IDs, uses source/native QPC, ends runs at the first changed
sign, and excludes runs spanning >0.25 s gaps or the right-censored final run.
It preserves exact source hashes; quantiles use the lower order statistic
at index `floor((N-1)*p)`. Reproduce with:

```powershell
node ocean/rek_g1/native5/validation/consistent-fighter-20260919/yaw-cadence-evidence.cjs `
  C:\rekagent\work\consistent-fighter-20260919-r1 `
  C:\rekagent\evidence\runtime\rek-private-ai-protocol-v7 `
  C:\rekagent\work\imitation-20260919-r1\dataset-r1\command-ledger.jsonl `
  NEW_YAW_EVIDENCE.json
```

The preserved `yaw-evidence-r1.json` SHA256 is
`35b72e047022c7f1c8ed25b32285d80ab263cc22ff4ce0ea345bc888e753565b`.
Relevant bridge implementation is `G1HeldInputScheduleContract.AdvanceKeyboardYaw`
and `Plugin.G1PolicyStream.UpdateG1PolicyVelocity`; no bridge behavior was changed.

## Exact cadence contract

Set `REK_POLICY_ACTION_STRIDE=5` for compact training and add
`--action-stride 5` to the live encoder command. Unset or explicit 1 preserves
the default. Other values are rejected. The shared contract is
`rek.policy_action_cadence.ready_ordinal.v1`.

The initial compact export has tick 0 and full existing support. Subsequent
learner support is intersected with `{0}` unless episode tick modulo 5 is 0.
Recovered Bot1, including the explicit side-0 Bot1 diagnostic override, remains
unrestricted. Terminal exports bypass cadence. Autoreset exports the new
episode's tick-0 support while preserving the previous reward/done buffers.

Live phase counts successfully ready nonterminal encoder outputs, which the
existing driver forwards one-for-one to the worker. Warmup/unavailable rows,
skipped source sequences, Unity frames and wall-clock deadlines do not advance
it. A new worker round, terminal or explicit reset clears phase. Existing
feature-derivative resets preserve it, matching retained worker recurrent
history. The driver currently treats side/round identity changes as fatal.
No worker reset or existing feature-reset semantics were altered.

Busy frames cannot override hold-only support. Action0 retains the last owned
command without retriggering an attack; action1 remains explicit release at a
decision boundary. Every native/source legality restriction is retained, and
an unavailable source hold fails closed. All 223 observation features remain
unchanged. The periodic phase is controller state that the RNN must infer
from its full history; no hidden training-only input was added. A five-row
interval near the observed 48 Hz cadence exceeds 100 ms slightly, with jitter.
Opt-in encoder provenance records ordinal/phase alongside source QPC so the
actual interval can be measured. A stale prediction can still be discarded by
the existing driver after inference; it advances RNN history, not game execution.

Singleton support makes actor log-probability, entropy and actor gradient zero
on hold-only rows; value learning still uses those rows. There is no reward
aggregation, macro-action discount, altered loss normalization or busy-duration
rounding. Saved-mask trajectory exports remain compatible and still require
exact encoder/worker mask equality and native behavior replay.

## Verified implementation

The encoder passed 29,131 assertions plus 145 hinge checks. Checks include
serialized default/explicit1 equality, unchanged features, existing mask
intersection, busy/unavailable paths, skipped IDs, jitter, derivative gaps,
failed-output phase preservation, and terminal/new-round/explicit reset.

One bounded CUDA differential verification ran from 04:14:44.715469310 to
04:14:45.935038246 UTC on 2026-09-20, with no optimizer or live client.
The same ABI fixture was linked against the archived default runtime and the
new runtime. All 2,335,662 captured bytes match for default versus stride1.
Stride5 differs only in expected learner masks: 65,934 mask comparisons across
666 snapshots, including 352 restricted rows, 222 Bot1-override rows,
507 busy snapshots, three terminal snapshots and nine initial/reset snapshots.
All fixed-history nonmask observations, poses, actions, rewards and round
fields match byte-for-byte. This is software parity under the fixed history,
not authentic physical parity. No failed GPU attempt or relaxed-mask retry occurred.

Sources: [shared contract](../../action_cadence.h),
[CUDA fixture](../../test_action_cadence.cu),
[byte/mask comparison](../../test_action_cadence.cjs).
The exact compile/run commands are preserved in private `verify-runtime.sh`
and traced stdout/stderr. It links the fixture with each build's
`fast_runtime.o`, `fast_assets.o`, `cJSON.o`, MuJoCo 3.7 and OpenSSL,
then runs old/default, new/1 and new/5 under one 60 s timeout.

Final Spark build:
`/home/spark-advantage/rek-training/action-cadence-20260920-r1/fast-build-r2`.
Trainer SHA256: `7d0ea789b3d27d7175fd2bd9827123ab7b45a8926be4aeba6cb96ba94f381e51`.
Encoder: sibling `encoder-build-r1/encode-live`, SHA256
`dfac88ecde9a9825b80479ab60f6d9029a61f8f236783878b890d92765fa96d5`.
The complete source snapshot is sibling `source-r2`.

## Matched full-training pair

Both arms start independently from the same unmasked legacy-schema f3 actor
`f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4`.
They use the same new binary and byte-identical benchmark INIs. Settings are
512 arenas, horizon 512, minibatch 8,192, 33,554,432 learner transitions,
120 s rounds, hidden256/two MinGRU layers/BF16/CUDA graphs, learning rate
0.0001 with existing annealing, entropy 0.01, gamma 0.9998844821426083,
lambda 0.9978673240629938, vf 0.5, policy/value clip 0.2 and replay ratio 1.
Environment seed 419 differs from the unchanged native base seed 73.
Recovered Bot1, rendered-pose observations, primitive contact samples/eight
substeps, round-outcome reward, randomized gaps 0.55-2.5 and heading spread pi
are unchanged. There is no feature mask, frozen opponent or additional shaping.

| Measurement | Stride1 | Stride5 |
| --- | ---: | ---: |
| Native exit / failure bits | 0 / 0 | 0 / 0 |
| Learner transitions | 33,554,432 | 33,554,432 |
| Trainer-loop seconds | 36.18959713 | 35.64928198 |
| Full-training transitions/s | 927,184.46 | 941,237.25 |
| Process wall seconds | 36.90 | 36.35 |
| Startup-inclusive transitions/s | 909,334.20 | 923,093.04 |
| Rollout CUDA seconds | 27.9898190 | 27.5502885 |
| Training model / misc seconds | 7.5311246 / 0.1601545 | 7.4348002 / 0.1598524 |
| Unattributed loop residual seconds | 0.5084990 | 0.5043409 |

The single ordered pair differs by +1.516% SPS. It does not establish a speed
improvement or repeated-run uncertainty. All 50 Hz recurrent and learning
rows remain, so a fivefold throughput gain was never expected. Cadence5 has
one fifth as many unconstrained decision opportunities, without redefining SPS.
Changing-policy compact round counts were W/L/T 4856/238/26 versus 4815/269/36,
and awarded totals 379145:216258 versus 305991:190572. These are training
diagnostics, not held-out fighting evidence. Neither checkpoint has an
authentic evaluation or promotion claim.

The outer PowerShell-to-Bash wrapper exited127 after both successful native
runs and final summary because a trailing CR became an extra shell command.
Per-run exit0, completion counts, failure0 and checkpoint hashes were verified
independently. The wrapper error is preserved in `training-pair.stderr.txt`,
`training-wrapper-result.json` and `training-readback.txt`; no training was rerun.

Reproduce the pair with the exact preserved `run-training-pair.sh` settings
above and fresh outputs, whose substantive invocation is:

```sh
stage=/home/spark-advantage/rek-training/action-cadence-20260920-r1
native="$stage/source-r2/ocean/rek_g1/native5"
initial=/home/spark-advantage/rek-training/owned-yaw-migration-20260920-r1/matched-gae-r1/train-control-v1/ppo.bin
unset REK_POLICY_FEATURE_MASK REK_FAST_CONTACT_POTENTIAL REK_TRAIN_OPPONENT_CHECKPOINT REK_FROZEN_OPPONENT_FRACTION
unset REK_FAST_SHAPING_GAMMA REK_FAST_SHAPING_TARGET REK_FAST_SHAPING_BEARING_WEIGHT REK_FAST_REWARD_GAMMA
export REK_OBSERVATION_SCHEMA=rek.native5.scaled_polar_xy.v1
export REK_FAST_REWARD=round_outcome_v1 REK_FAST_SCORING=recovered_hit_rules_v2
export REK_FAST_GEOMETRY=primitive_samples_v1 REK_FAST_CONTACT_SUBSTEPS=8
export REK_FAST_OPPONENT=recovered_bot1_v1 REK_FAST_OBSERVATION=rendered_pose_v1 REK_FAST_OPPONENT_MODE=scripted
export REK_FAST_RANDOM_RESETS=1 REK_FAST_RESET_GAP_MIN=.55 REK_FAST_RESET_GAP_MAX=2.5
export REK_FAST_RESET_HEADING_SPREAD_RAD=3.14159265 REK_FAST_SHAPING_WEIGHT=0
export REK_TRAIN_SEED=419 REK_TRAIN_LEARNING_RATE=.0001 REK_TRAIN_MINIBATCH=8192
export REK_TRAIN_ENTROPY=.01 REK_TRAIN_TIMEOUT=180
export REK_TRAIN_GAMMA=.9998844821426083 REK_TRAIN_GAE_LAMBDA=.9978673240629938
REK_POLICY_ACTION_STRIDE=1 bash "$native/run_diverse_training.sh" \
  "$stage/fast-build-r2" NEW_STRIDE1_OUTPUT 33554432 512 512 120 "$initial"
REK_POLICY_ACTION_STRIDE=5 bash "$native/run_diverse_training.sh" \
  "$stage/fast-build-r2" NEW_STRIDE5_OUTPUT 33554432 512 512 120 "$initial"
```

Final checkpoint SHA256 values, under each run's
`checkpoints/rek_native5/RUN/0000000033554432.bin`:

- `train-cadence-1-r1`: `93180341b685c99549d9f55693a56b8cf42dfc45857790d77d43920d35219ba3`.
- `train-cadence-5-r1`: `53d8c4e867baf61ccaf30c114301193659c690047fd2c67855c1ade0c9d3a43b`.

The completed build/test/two-training subset was mirrored to
`C:\rekagent\work\consistent-fighter-20260919-r1\action-cadence-r1\spark-results`:
85 files, 30,154,672 bytes, all remote/local SHA256 equal, with unchanged
remote hashes before and after copy. Existing early build/source snapshots,
commands and wrapper diagnostics remain in the parent private stage.

The entire finalized private stage is archived in a fresh physical NAS child:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\action-cadence-r1`.
Its 98 source files total 47,531,593 bytes. Every destination byte count/SHA256
matches the source, with a full unchanged-source rehash after copying.
The archive also contains the verified archive command, manifest and proof.
Manifest SHA256:
`b1e07f287e8d86b828a5519b570780191ea92d29a950e0b4a4f085793cba753b`.
No source or pre-existing archive file was deleted or overwritten.
