# REK G1 fight semantics, current pinned build

## Status and scope

This document is an implementation contract recovered from the pinned Windows REK build. It covers accepted-hit flow, score updates, geometric fall detection, referee counting, round and fight termination, and physical reset. It does not supply guessed health, reward, timing, or recovery values.

Build inventory fingerprint:

`f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659`

Unity version:

`6000.5.8f1`

The serialized `FightCoordinator` and `HitDetector` values below are identical in the inspected `level1`, `level2`, and `level3` containers. The G1 fall values come from `g1_29dof_Prefab_SONIC`, not the T800 prefab.

Primary recovered sources:

- `controller-audit-isil/IsilDump/REKApp/REKApp/HitDetector.txt`
- `controller-audit-isil/IsilDump/REKApp/REKApp/PointTracker.txt`
- `controller-audit-isil/IsilDump/REKApp/REKApp/FightCoordinator.txt`
- `controller-audit-isil/IsilDump/REKApp/REKApp/FightCoordinator_NestedType__RefereeCountRoutine_d__204.txt`
- `controller-audit-isil/IsilDump/REKApp/REKApp/Robot.txt`
- `controller-audit-isil/IsilDump/REKApp/REKApp/Robot_NestedType__DeferredJointReset_d__334.txt`
- `controller-audit-isil/IsilDump/REKApp/REKApp/PointTracker.txt`
- `controller-audit-isil/IsilDump/REKApp/REKApp/SonicPolicyRunner.txt`
- `ocean/rek/evidence/evidence_out/fight_semantics_components.json`

These are derived metadata and native-code instruction recovery. No proprietary binary payload is reproduced here.

## Exact serialized values

### FightCoordinator

| Field | Value |
|---|---:|
| `fightFormat` | `BestOf3` (`0`) |
| `roundDuration` | `120.0 s` |
| `redoRoundDuration` | `30.0 s` |
| `betweenRoundDelay` | `5.0 s` |
| `fightOverDelay` | `5.0 s` |
| `autoFightStartDelay` | `3.0 s` |
| `knockdownWeakSpeed` | `1.75 m/s` |
| `knockdownStrongSpeed` | `6.0 m/s` |
| `knockdownWindowMin` | `1.5 s` |
| `knockdownWindowMax` | `5.0 s` |
| `koCountSeconds` | `10.0 s` |
| `doubleKnockdownCountSeconds` | `20.0 s` |
| `noRecoveryCountSeconds` | `3.0 s` |
| `koPoints` | `5.0` |
| `slipPointsToOpponent` | `3.0` |
| `allowHitsOutsideFight` | `false` |

The countdown Playable is an external `UnityEngine.Timeline.TimelineAsset` in `sharedassets0.assets` (`m_FileID=4`, `m_PathID=2759`) named `Countdown`. Its serialized fields include `m_FixedDuration=2.4722222222222223` and `m_DurationMode=0`. `StartRound` uses the asset's runtime duration when positive. The recovered fields do not establish that the runtime duration equals the serialized fixed-duration field, so the exact countdown duration remains unknown.

### HitDetector

| Field | Value |
|---|---:|
| `speedThreshold` | `1.75 m/s` |
| `knockdownStrikeApproach` | `2.0 m/s` |
| `perHandCooldown` | `0.30000001192092896 s` |
| `requireActiveRound` | `true` |
| `requireBothUpright` | `true` |
| `requireStrikeApex` | `true` |
| `scoreOncePerMove` | `true` |
| `apexMinRamp` | `0.20000000298023224` |
| `apexRequireLimbMatch` | `true` |
| `rejectBacksideHits` | `false` |
| `backsideRejectAngle` | `120.0 degrees`, inactive because rejection is disabled |

### G1 Robot fall detector

| Field | Value |
|---|---:|
| `fallingTiltThreshold` | `42.0 degrees` |
| `fallingHeightRatio` | `0.6` |
| `fallenTiltThreshold` | `69.3 degrees` |
| `fallenHeightRatio` | `0.4` |
| `fallenContactPoints` | `3` |
| `fallenHoldFast` | `0.15 s` |
| `fallenHoldSlow` | `0.5 s` |
| `fallenResetTimeout` | `3.0 s` |
| `floorNormalMaxAngle` | `50.0 degrees` |
| `autoDampenOnFallen` | `true` |
| `getUpCommandTimeout` | `10.0 s` |
| `recoverySettleMinTime` | `0.3 s` |
| `recoverySettleHold` | `0.15 s` |
| `recoverySettleLinearVelocity` | `0.15 m/s` |
| `recoverySettleAngularVelocity` | `0.5 rad/s` |
| `recoverySettleJointVelocity` | `1.0 rad/s` |

## Accepted hit flow

The authoritative sequence in the recovered client is:

1. `HitDetector.OnContactsProcessed` rebuilds its body caches when dirty. It handles contact-enter records only. Same-robot contacts are discarded. Each contact is tested in both A-to-B and B-to-A directions.
2. `CacheStrikerBodies` admits `BodyPartTag` values `Hand`, `Foot`, and `Shin` only. A side must be available. The cache key is the striker `MjBody` entity ID.
3. `TryScoreContact` requires cached striker identity, `ContactInfo.RelativeSpeed >= 1.75 m/s`, a resolvable target robot, and different striker and target robots.
4. Before the scoring-zone and scoring-qualifier gates, a standing target can receive `OnFighterStruck(targetIndex, speed)` when `IsAggressorStrike` passes. Fall attribution can therefore update from a contact that later receives no hit point.
5. The scoring target must classify as `Head`, `Torso`, `Pelvis`, `LeftHip`, or `RightHip`. All other `BodyZone` values are rejected for points.
6. Current qualifiers require an active round, both robots upright, an `IStrikeIntentSource`, a matching striker limb at strike apex, and apex ramp at least `0.2`.
7. The striker-body cooldown must have elapsed at least `0.30000001192092896 s`.
8. With once-per-move enabled, the current move ID and apex-event bit cannot already be recorded as scored. Apex indices are capped to bit index 30.
9. `PointTracker.RecordHit` must accept the event. Only then does `HitDetector` update the cooldown and scored-apex state and publish `OnHitDetected`.

`HitEvent.Impulse` is the magnitude of `ContactInfo.ImpulseSum`. `HitEvent.RelativeSpeed` retains the contact speed. Neither value scales the clean-hit award in this build.

`HitEvent.IsKick` is true for `Foot` and `Shin`. `RecordHit` awards `2.0` for a kick and `1.0` for every other accepted striker type. The float award is converted to `int` by truncation before addition to `RoundState.CleanHits`. `ScoringEvent.PointsAwarded` retains the float.

## Strike attribution is separate from scoring

`HitDetector.IsAggressorStrike` computes:

```text
d = normalize(victim_body_position - striker_body_position)
striker_approach = dot(striker_body_linear_velocity, d)
victim_approach = dot(victim_body_linear_velocity, -d)
```

Zero separation fails. Attribution passes when both conditions hold:

```text
striker_approach >= 2.0 m/s
striker_approach > victim_approach
```

`FightCoordinator.OnFighterStruck` stores `Time.time` and the reported contact relative speed for the victim. The knockdown attribution window is:

```text
t = clamp01((strike_speed - 1.75) / (6.0 - 1.75))
window = 1.5 + t * (5.0 - 1.5)
```

When a fighter first enters falling state during an active round, `OnFighterFalling` classifies the fall once:

- `elapsed_since_last_strike <= window`: knockdown
- `elapsed_since_last_strike > window`: slip
- active `EngineAIPolicyRunner` emergency-stop state: forced slip

Classification occurs at falling onset. The referee count starts only after the robot becomes fallen.

## Geometric fall state machine

`Robot.TiltAngle` is the angle between world up and `root_rotation * uprightUpLocal`.

`Robot.PelvisHeightRatio` is:

```text
(pelvis_world_y - floor_y) / standing_pelvis_height
```

The floor height comes from tracked contacts when available and otherwise from the recovered downward MuJoCo ray path.

Floor contacts include non-robot contacts whose floor-side normal is within `50.0 degrees` of vertical. Contact-enter and contact-exit records maintain pair membership and a per-body reference count. Foot bodies are tracked separately from non-foot grounded bodies.

An upright robot enters `isFalling` when either condition holds:

```text
TiltAngle > 42.0 degrees
BothFeetOffFloor && PelvisHeightRatio < 0.6
```

The falling state cancels when the fallen qualifiers are false, tilt returns to at most `42.0 degrees`, and at least one foot is on the floor.

Becoming fallen always requires tilt above `69.3 degrees` for a continuous hold. The hold logic is:

- Without usable floor tracking: use the slow hold, `0.5 s`.
- With usable floor tracking: pelvis ratio must be below `0.4`, and grounded support must qualify as `(no foot grounded and at least one non-foot body grounded)`, or as at least `3` non-foot bodies grounded.
- The fast hold, `0.15 s`, applies only when no foot is grounded and at least `3` non-foot bodies are grounded.
- Other qualifying low-pelvis support uses the slow hold, `0.5 s`.
- A failed qualifier clears the accumulated hold timer.

`BecomeFallen` sets fallen state, initializes its timers and recovery state, publishes `Robot.OnFallen`, and enters dampening because the current G1 prefab has `autoDampenOnFallen=true`.

## Exact fall, count, score, reset, and terminal flow

There is no recovered `FightCoordinator.OnFallTimeout` method. The equivalent fight-level deadline handler is `FightCoordinator.ResolveCountExpiry`, called by `RefereeCountRoutine` when `Time.time >= refCountDeadline`.

The current client also has a separate robot-local `fallenResetTimeout=3.0 s`. Once expired, `Robot.FixedUpdate` invokes `OnResetDue` when a subscriber exists and reloads that timer. `FightCoordinator.OnFighterResetDue` deliberately ignores this callback during an active live round. It resets both fighters only in `Sandbox`, or in `RoundActive` while `RoundState.IsActive` is false during the pre-round countdown. Consequently, the referee count controls active-round fall resolution.

### OnFighterFallen

During an active round:

1. Increment `RoundState.Falls[fallenFighterIndex]`.
2. Use the classification latched at falling onset. An unclassified direct fallen event defaults to slip.
3. Mark that fighter's count active and raise `Slip`, `SlipEStop`, or `Knockdown`.
4. For the first counted fighter, set deadline to `10.0 s` when `CanFighterGetUp` is true, otherwise `3.0 s`.
5. If the second fighter also falls, restart the shared timer at that instant. Set it to `20.0 s` when either fighter can get up, otherwise `3.0 s`. Raise `DoubleKnockdown`.

### Recovery before deadline

For each counted fighter, `RefereeCountRoutine` waits until the robot is no longer fallen and `IsRecovering` is false. It then clears that fighter's count and raises `BeatCount`.

- Recovered slip: award `3.0` points once to the opponent.
- Recovered knockdown: award no referee points.
- If all counts clear and round time is zero, end the round.
- `OnFighterRecoveryAbandoned` logs the abandoned attempt and leaves the count running.

### Deadline expiry

`ResolveCountExpiry` first clears the active count flags.

Single-fighter expiry:

1. Award `5.0` points to the opponent.
2. Raise `Knockout`.
3. If the fallen fighter's `CanGetUp` is true, call `RecordKnockout(opponent)`. This sets `KnockoutOccurred=true`, `RoundResult=WonByKO`, the winner, increments that winner's `FightState.RoundsWon`, and ends the round.
4. If `CanGetUp` is false, keep the five points. When `TimeRemaining > 0`, call `ResetBothToSpawn` and continue the same round. When time is zero, end the round.

Double-fighter expiry:

1. Award `5.0` points to each fighter.
2. Raise `DoubleKnockout`.
3. If either fighter's `CanGetUp` is true, call `RecordDoubleKnockout`. This sets `KnockoutOccurred=true`, `RoundResult=Tie`, `WinnerIndex=-1`, and ends the round.
4. If neither can get up, keep both awards. When `TimeRemaining > 0`, reset both and continue the same round. When time is zero, end the round.

The round clock clamps at zero but does not end the round while a referee count is active. Count recovery or expiry performs the deferred end.

### ResetBothToSpawn

`FightCoordinator.ResetBothToSpawn` invokes `Robot.ResetToSpawn` on both non-null fighters. `ResetToSpawn` sets a `2.0 s` post-reset fall-detection grace timer and immediately enters `TeleportAndResetJoints`.

`TeleportAndResetJoints`:

1. Stops any prior reset coroutine.
2. Clears falling, fallen, recovery, attribution, and floor-contact state.
3. Applies the saved root spawn position and rotation.
4. Starts `DeferredJointReset`.
5. After one `WaitForFixedUpdate`, restores motor gains, writes every joint position and velocity to zero, applies the root spawn pose again, measures standing height, clears resetting state, and publishes `OnResetComplete`.

`Robot.ResetAfterFall`, used when no `OnResetDue` subscriber handles the local timeout, sets a `0.5 s` post-reset grace timer and invokes the same teleport path immediately.

## Round and fight state machine

`StartRound(roundNumber, isRedo)` creates a new `RoundState`, appends it to `FightState.Rounds`, selects `120.0 s` for a normal round or `30.0 s` for a redo, clears referee state, resets both robots to spawn, and deactivates controls. A positive countdown Playable duration delays `ActivateRound`; otherwise activation is immediate.

`ActivateRound` sets `RoundState.IsActive=true`, starts `PointTracker`, activates controls, and publishes the round-start event.

At ordinary round end, `PointTracker.EvaluateRound` behaves as follows:

- `KnockoutOccurred=true`: preserve the result already written by the knockout path.
- Unequal `CleanHits`: higher score gets `WonByPoints`; increment that fighter's `RoundsWon`.
- Equal `CleanHits`: `Tie`, `WinnerIndex=-1`.

`PointTracker.EvaluateFight` requires two round wins for `BestOf3` and three for `BestOf5`. At the maximum regular round number, unequal `RoundsWon` also resolves `WonByRounds`. Equal cumulative round wins return false. `UpdateBetweenRounds` marks the next round as a `30.0 s` redo only when the immediately preceding `RoundResult` is `Tie`; an unresolved equal series after a non-tied round starts another normal `120.0 s` round. Round numbers continue increasing.

`EndRound` deactivates the current round and controls, ends and evaluates point tracking, sets phase `RoundEnd`, and publishes `OnRoundEnded`. If the fight is unresolved, it waits `5.0 s`, then starts the next round. If resolved, it sets `FightOver`, publishes `OnFightEnded`, waits `5.0 s`, exits fight mode, and returns to `Idle`.

`FightResult.WonByTKO` exists in the enum but is not assigned anywhere in the recovered point/referee pathway described here.

## State and enum values required by a state-based environment

`HitEvent` fields:

```text
AttackerFighterIndex, Hand, Zone, StrikerPart,
Impulse, RelativeSpeed, Timestamp
```

`ScoringEvent` fields:

```text
Timestamp, FighterIndex, Hit, PointsAwarded
```

`RoundState` fields:

```text
RoundNumber, RoundDuration, TimeRemaining, IsActive, IsRedo,
CleanHits[2], Falls[2], Result, WinnerIndex, KnockoutOccurred
```

`FightState` fields:

```text
Format, CurrentRoundNumber, Rounds, RoundsWon[2], Result, WinnerIndex
```

Relevant enum values:

```text
BodyPartType: None=0 Hand=1 Foot=2 Head=3 Torso=4 Pelvis=5
              Forearm=6 UpperArm=7 Shin=8 Thigh=9 Knee=10 Elbow=11

BodyZone: Unknown=0 Head=1 Torso=2 Pelvis=3 LeftShoulder=4
          RightShoulder=5 LeftElbow=6 RightElbow=7 LeftWrist=8
          RightWrist=9 LeftFist=10 RightFist=11 LeftHip=12
          RightHip=13 LeftKnee=14 RightKnee=15 LeftAnkle=16 RightAnkle=17

FightPhase: Idle=0 RoundActive=1 RoundEnd=2 BetweenRounds=3
            FightOver=4 Setup=5 Sandbox=6

FightFormat: BestOf3=0 BestOf5=1

RefereeCall: Slip=0 SlipEStop=1 Knockdown=2 BeatCount=3 Knockout=4
             DoubleKnockdown=5 DoubleKnockout=6

RoundResult: InProgress=0 WonByPoints=1 WonByKO=2 Tie=3 Redo=4
FightResult: InProgress=0 WonByRounds=1 WonByTKO=2
```

## Damage, health, and RL reward

No health field, damage field, hit-point field, or health-update method was found in the recovered `HitDetector`, `PointTracker`, `FightCoordinator`, `RoundState`, `FightState`, and related REKApp metadata search. In this inspected pathway, accepted hits change clean-hit score and event logs. Falls arise from physical state and contact history. No scalar health pool mediates either result.

This statement is bounded to the inspected fight pathway and recovered metadata. It does not assert that uninspected application code lacks every use of those words.

The client has no PufferLib reward signal. Its native measurable deltas are:

- accepted hit: scorer `CleanHits += 1` or `2`
- recovered slip: opponent `CleanHits += 3`
- count expiry: opponent `CleanHits += 5`, or both `CleanHits += 5`
- round win: winner `RoundsWon += 1`
- episode-relevant events: round end and fight end

Any transformation of these deltas into an RL reward is an environment design choice. It must be named separately and cannot be presented as recovered REK behavior.

## Remaining unknowns and required measurements

1. **Countdown duration.** The external Timeline asset has a serialized fixed-duration field of `2.4722222222222223 s` and duration mode `0`, but its runtime `PlayableAsset.duration` has not been measured or derived from the full Timeline contents.
2. **Runtime G1 get-up availability.** The inspected G1 prefab links `Robot.policyRunner` to `SonicPolicyRunner`, and its serialized `getUpProneClip` and `getUpSupineClip` references are null. `SonicPolicyRunner.CanGetUp` requires initialization, an unpaused runner, a motion composer, and both clips. No assignment to those fields was found in recovered REKApp native code. A catalog override or another uninspected runtime mechanism may still populate them. Recorder v0.7.3 now rejects capture unless direct `Robot.PolicyRunner.CanGetUp`, `FightCoordinator.CanFighterGetUp(slot)`, and G1 `SonicPolicyRunner.CanGetUp` readings are present and agree for both fighters. No v0.7.3 runtime capture exists yet, so the value remains unknown. It must be captured before selecting the 3, 10, or 20 second branch in parity claims.
3. **Contact impulse generation.** This report covers acceptance and referee semantics after MuJoCo supplies contacts. Solver-generated trajectories and impulses still require held-out replay validation.
4. **Policy-facing RL reward.** REK supplies scores and terminal events, not the training reward transformation.
5. **Runtime server authority.** These client-side methods establish local fight-state behavior. They do not establish whether a server can override results in every game mode.

Fail closed on these unknowns. Do not substitute defaults for them in a claimed current-build parity result.

## Evidence records

Every listed command record is under:

`<evidence-run>/commands/<id>/`

Each record contains `command.ps1`, `stdout.txt`, `stderr.txt`, and `result.json` with SHA-256 hashes.

Key IDs:

- `fight-semantics-component-probe-2`
- `fight-semantics-component-values-1`
- `fight-semantics-fight-coordinator-values-2`
- `fight-semantics-robot-fall-scalars-1`
- `fight-semantics-enum-values-2`
- `fight-semantics-result-enums-1`
- `fight-semantics-state-fields-1`
- `fight-semantics-hitdetector-bodies-1`
- `fight-semantics-hit-attribution-isil-1`
- `fight-semantics-aggressor-disasm-1`
- `fight-semantics-point-values-1`
- `fight-semantics-pointtracker-isil-2`
- `fight-semantics-referee-methods-isil-2`
- `fight-semantics-referee-coroutine-filtered-2`
- `fight-semantics-resolve-count-disasm-1`
- `fight-semantics-upright-falling-disasm-1`
- `fight-semantics-robot-fall-metrics-1`
- `fight-semantics-floor-contact-flow-1`
- `fight-semantics-reset-exact-1`
- `fight-semantics-reset-grace-use-1`
- `fight-semantics-deferred-reset-flow-1`
- `fight-semantics-startround-call-sequence-1`
- `fight-semantics-endround-isil-1`
- `fight-semantics-betweenround-disasm-1`
- `fight-semantics-health-damage-search-1`
- `fight-semantics-g1-policy-link-1`
- `fight-semantics-sonic-can-get-up-1`
- `fight-semantics-sonic-getup-assignments-1`
- `fight-semantics-getup-cross-type-search-1`
- `fight-semantics-countdown-pptr-2`
- `fight-semantics-report-source-index-1`
