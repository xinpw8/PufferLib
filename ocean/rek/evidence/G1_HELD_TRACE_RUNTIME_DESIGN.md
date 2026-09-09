# G1 held-input measurement runtime

## Implemented boundary

`RekUiBridgeAgent` exposes the hash-bound
`rek.g1_held_input_schedule.v2` contract through
`StartG1HeldInputSchedule`. `RekUiPipeClient g1-held` is its pinned caller.
The mode writes semantic `RobotInputController` state and calls recovered game
methods only. It has no keyboard, mouse, gamepad, window activation, or global
input path.

The pipe transcript is authoritative for client property writes, client method
returns, pending state, and intercepted client request methods. It does not
establish server acceptance or physical execution. The client fighters are
visual-only. Their `IsPunching`, `IsRecovering`, and
`SonicPolicyRunner.currentMotion` values are retained as local diagnostics, not
as server or motion-asset identity evidence. Physical response, input delay,
and entry-to-completion timing require correlation with the separately captured
`RekEvidenceRecorder rek.private_ai.protocol.v7` local-slot bone trajectory.

## Start and fail-closed gates

The client acquires one exclusive pipe lease and requires an inactive, proven
solo Sparring Bot 1 session with zero human occupancy. It arms and starts a
fresh round through the existing semantic `StartRound` bridge command. It waits
for `active_gameplay_proven` and a unique active round, then starts only if both
fighters have the exact 30-bone G1 runtime signature.

The bridge binds the coordinator, network session, round object, fight epoch,
round number, arena, endpoint, and local input controller. It requires clean
round hit and fall counters, the expected G1 move asset bindings, exact
`yawSpeed=1`, exact `keyboardYawRampTime=0.5`, and exact G1 transition thresholds
`transitionSettlePlanarSpeed=0.03` m/s and
`transitionSettleYawRate=0.03` rad/s. It also requires this isolation proof:

`wine_get_version=11.13;display=:98;prefix=/opt/codexrook/wineprefix;marker=spark-x98`

The schedule lasts 91.02 seconds and requires an additional 10-second safety
budget. `RoundState.RoundDuration` and `RoundState.TimeRemaining` must each prove
at least 101.02 seconds before start. Missing, non-finite, inconsistent, or
insufficient capacity fails before the schedule runs.

Every 500 Hz fixed substep rechecks active gameplay, the solo Bot 1 scope,
zero human occupancy, G1/G1 signatures, immutable runtime identity, and input
controller identity. A round end or changed identity stops the run with explicit
partial coverage.

## Deterministic schedule

The schedule changes desired held commands at 50 Hz on the exact 500 Hz Unity
fixed clock, with 10 fixed substeps per schedule tick. The canonical SHA-256 is
`0e28e089c73e603c7ce1d9cd5e6de4dd7f6f6017ce3bf4bcae2a90366b1adc2b`.
It has 4,551 ticks, ending at tick 4,550 after 91.02 seconds.

Each held condition lasts 100 ticks, or 2 seconds, with 50-tick neutral gaps:

1. W, S, A, D
2. Q, E
3. W+Q, W+E, S+Q, S+E
4. A+Q, A+E, D+Q, D+E

W, S, A, and D use pinned normalized speeds of 1. Q and E use the exact
recovered keyboard yaw ramp. Each rendered
`RobotInputController.LateUpdate` advances the ramp by
`Time.deltaTime / 0.5`, resets the ramp on sign changes, clamps it to 1, and
resets ramp and sign to zero on release. Transcript records separate desired
key masks, raw yaw targets, ramp state, and effective controller vectors. A yaw
kick edge explicitly resets the ramp and effective yaw before
`ExecuteMoveByIndex`.

Only G1 move indices 6, 7, 8, and 9 are used. F is excluded until binding
evidence exists. Each move has one translation-held probe and one yaw-preempted
probe. Every probe has one edge, no schedule-owned queue, and no retry.

Each translation probe holds W, S, A, or D for 50 pre-edge ticks and preserves
the held value at the kick edge. It releases the translation five ticks, or
100 ms, after the edge. Total translation hold is 55 ticks, or 1.1 seconds.
The remaining 195 ticks of the four-second observation window use an exact
neutral translation command. The release has its own fixed-substep and QPC
anchor. This bounds arena travel while preserving the held-at-edge condition.

Each kick observation spans 200 schedule ticks, or 2,000 fixed substeps. This
covers the maximum recovered clip length of 159 controller ticks plus a
40-tick dispatch and start margin budget. Probe windows do not overlap and have
a 50-tick gap. Before the next kick, the previous transport window must be
complete, no owned pending request may remain, and observable local action or
runner recovery flags must be idle. A prior translation probe must also have
returned true from the direction-specific `TransitionSettled` call after its
release, or the run aborts with partial coverage.

An accepted pending move remains intact across any number of fixed updates. It
can become terminal only after a matching rendered
`RobotInputController.LateUpdate` dispatch opportunity. The bridge records the
matching LateUpdate prefix and postfix. If that completed rendered opportunity
does not invoke `SendMoveEvent`, the run aborts as partial. Pending state is
never cleared at a 50 Hz boundary. An owned pending value can be cleared only
during terminal stop cleanup.

Kick lifecycle is sampled on every 500 Hz fixed substep. `SendMoveEvent` prefix
and postfix records include the exact fixed substep, schedule tick, Unity frame,
Unity fixed time, QPC ticks, and QPC frequency. Any local diagnostic delay is
computed from the send prefix, never from the kick arm edge.

After each translation release, every fixed observation directly calls the
public recovered `RobotInputController.TransitionSettled` with the original
direction: W maps to `Forward`, S to `Backward`, A to `StrafeLeft`, and D to
`StrafeRight`. The returned boolean is recorded with its exact substep.
`Robot.TryGetBaseVelocityLocal` linear and angular components and both
transition thresholds are recorded independently. Those components are not
treated as a reimplementation of the direction-specific native predicate.
The summary classifies a send as occurring while translation was held, after
release but before transition settling, after transition settling, or not
observed.

The requested `MocapClipConfig` pointer, runtime asset name, NPZ hash, and
recovered clip length are recorded. The visual-only runner exposes a generated
`MotionSequence`, not the requested `MocapClipConfig` identity. Therefore
requested-asset to `currentMotion` identity remains explicitly unknown, and the
pipe never certifies a runner duration.

## Results and coverage

Experiment coverage is independent of the measured behavior. Complete coverage
requires all 14 held conditions for exactly 100 ticks, all eight kick edges,
all eight local request terminal outcomes, all four translation releases, four
yaw preemptions, exactly 2,000 fixed observations for each probe, all eight
completed observation windows, and a returned final neutral velocity request.

Translation summaries report a local gate result and a physical behavior
result separately. An explicit false `ExecuteMoveByIndex` return with no
request supports the local gate. A true return or intercepted request
contradicts that local gate. Absence of a packet alone never supports a blocked
claim. Physical behavior remains `unknown` in the pipe for every outcome until
the recorder bone trajectory is correlated. A schedule can therefore complete
as an experiment even when the expected local gate is contradicted.

The translation probe has no second kick edge after its release because that
would violate the one-edge, no-retry contract and confound the original
request. The later yaw-preemption probe for the same move is neutral in its
effective controller vector at the edge and provides a separately measured
control if its local request succeeds. Server acceptance of either edge remains
unknown.

If Bot 1 ends the round early, identity changes, a matching rendered dispatch
opportunity fails, or a prior observable action remains active, the bridge emits
`experiment_coverage_complete:false` and `partial_coverage:true`. The client
releases the lease, publishes the validated transcript at the requested path,
writes a `client_result` status of `partial`, and returns nonzero.

## Machine invocation

From a fresh, inactive, proven solo Bot 1 G1/G1 session:

```powershell
RekUiPipeClient.exe g1-held C:\path\to\new-output.jsonl [timeout_seconds]
```

The default timeout is 150 seconds. One connection performs the semantic
fresh-round arm, waits for active gameplay, starts the schedule, validates the
stream, releases the lease, and publishes the requested JSONL path only after
terminal validation. During a failure before publication, the client preserves
`new-output.jsonl.partial-<pid>-<id>` and prints its exact path.
