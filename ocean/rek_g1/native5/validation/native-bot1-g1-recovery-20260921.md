# Current G1 Bot1 recovery and lifecycle

The current Sonic G1 configuration supports a source-grounded Bot1 fall path
without fabricated get-up motion. Native Straighten returns false for this
runner; both configured get-up clips are null. Bot1 repeatedly requests
Straighten after dampening while the existing fall countdown and bilateral
body reset remain responsible for recovery from the fall.

`native_bot1.cuh` adds opt-in entry points and separate recovery state. Existing
compact `State`, `Input`, `Decision`, `activate`, `update`, and `locomotion` retain
their behavior. This change does not bind Bot1 into a runtime or alter a model.

## Recovered state and command contract

Private recovered methods below are in
`C:\rekagent\work\controller-audit-isil\IsilDump\REKApp\REKApp`.
These are derived semantics, not a reproduction of the recovered source.

`AIOpponentController.txt:192` (`UpdateStateMachine`, RVA `0x2368570`),
`:1645` (`UpdateFaultEStopCycle`, `0x23680e0`), and `:1837`
(`DriveRecovery`, `0x2367430`) establish this pseudocode:

```text
On an Update while the brain is active:
  Decrease the tactical state timer by this Update's delta time.
  If the robot is fallen or its input controller reports runner recovery:
    Write zero velocity without changing tactical phase or minimum-footwork timer.
    If motor shutdown hold is absent, clear the fault timer and fault latch.
    Otherwise accumulate the fault timer. Request one EStop toggle at the
      configured delay when disengaged, or at 0.5 s when engaged; flip the
      fault latch and reset the timer to zero. Discard timer overshoot.
    If no longer fallen, clear the straighten-success latch.
    Else if not dampened, request Dampen.
    Else if Straighten has not succeeded, request Straighten and latch its result.
    Else if recovery is armed, request prone for orientation 0, supine otherwise.
    Return without tactical action selection or random draws.
  Otherwise clear the fault timer/latch and run the ordinary tactical state.
```

The straighten latch is not cleared by the ordinary tactical branch or a
Dampen request. A failed Straighten is retried on every active Update, with no
retry delay. An already completed attack enters tactical Recovering without
CancelPunch; an expired still-playing attack requests cancellation. The opt-in
wrapper preserves this distinction while leaving the compact wrapper unchanged.

| Local current-G1 command | Grounded result |
| --- | --- |
| Dampen | Reject if remote-driven, already dampened, or under motor shutdown hold. Otherwise enter dampening and return true. |
| Straighten | Remote-driven, undampened, or motor-held requests fail. Otherwise dispatch `IPolicyRunner.Straighten`, which returns false for Sonic G1. |
| Get-up | Unreachable from a fresh current-G1 recovery latch. The new local adapter reports an unexpected get-up request as unsupported. It does not synthesize a trajectory or substitute neutral input. |
| ToggleEStop | The AI requests the toggle on the fault schedule. The recovered local input controller only applies it through an Engine runner; there is no Sonic runner EStop branch. A request is not evidence of a motor-state change. |

Evidence: `RobotInputController.txt:7462` (`ExecuteSpecial`, `0x226d360`),
`:8323` (`ToggleEStop`, `0x226f4d0`); `Robot.txt:55898`
(`RequestDampen`, `0x23d74c0`), `:55964` (`RequestStraighten`, `0x23d7600`),
and `:56100` (`RequestGetUp`, `0x23d7500`). `REKBehaviour.txt:23` identifies
the remote-driven flag. RequestStraighten invokes interface slot 8.
`IPolicyRunner.txt:65` contains an actual native return-false implementation.
The complete Sonic method inventory has no Straighten override, and its base
`REKBehaviour` has none. Sonic implements `IPolicyRunner` directly.

The existing private v7 probe
`C:\rekagent\evidence\motion-parity-D21-20260903T071000Z\artifacts\subagent-g1-spawn-mujoco_asset_probe_v7.json`
identifies SonicPolicyRunner on `g1_29dof_Prefab_SONIC`, sharedassets0 path 3188,
serialized SHA256
`40f11944265946fb0a00747f52b172d094ade069979bf4fe26c75367676a63f5`.
Both get-up references at lines 161326 and 161330 have file/path ID zero.
`SonicPolicyRunner.txt:13319` requires initialization, an unpaused runner,
a composer, and both clips for CanGetUp. Therefore CanGetUp is false here.
The recovery-watchdog settings of 12 s and 35 degrees do not create missing clips.
The v8 probe at
`C:\Users\Daniel\codex-wr64-puffer-scenario-7cc6ce1\ocean\rek\evidence\evidence_out\mujoco_asset_probe_v8.json`
independently agrees and records AI fault delay 3 s at line 140128. The new API
requires the delay as input; it does not invent a default for an unmeasured hold.

## Lifecycle and cadence required by a physical adapter

- `activate_g1(State&, input_present)` transitions an inactive possessed brain
  to Settling with the native initial delay. `deactivate_g1(State&, input_present)`
  enters Inactive and reports a zero-velocity write when a sink is present.
  Both preserve RNG, counters, minimum timer, reposition vector, and separate
  recovery state. Use the existing seed initializer only for a fresh environment.
- Counted body reset preserves AI phase/timers. `FightCoordinator.txt:74657`
  (`ResetBothToSpawn`) only invokes both robots' `ResetToSpawn`.
  `RobotInputController.txt:13678` (`HandleResetComplete`) clears locomotion,
  braking, and momentum then plays idle, without activating/deactivating the AI.
  AIOpponentController has no reset event handler. In contrast, round boundaries
  release/repossess brains: `FightCoordinator.txt:2866`, `:4471`, `:4753`,
  and `AIOpponentController.txt:2330`, `:3329`.
- InputController.IsActive is the independent bool at offset `0x6a`
  (`RobotInputController.txt:5088`); only Activate/Deactivate assign it in the
  current input-controller hierarchy. Fall, dampening, and counted body reset
  do not deactivate it. Do not substitute `!dampened`, `!resetting`, or the
  Sonic motor/history controller-enable flag for input activity.
- `update_g1` takes separate `fallen`, `runner_recovering`, `dampened`,
  `motor_shutdown_hold`, `recovery_armed`, orientation, and fault-delay inputs.
  Its outputs include special request, EStop request, zero-velocity write, and
  the existing tactical decision. Apply the synchronous special result using
  `recovery_special_result`; only Straighten updates the success latch.
- `fixed_locomotion_g1` is a separate explicit event. It reports no write when
  the brain or input sink is inactive. Native FixedUpdate does not test own
  fall/recovery before writing tactical-phase locomotion
  (`AIOpponentController.txt:140`, `:6682`). A recovery Update can write zero and
  a subsequent FixedUpdate can overwrite it while the sink remains active.
  Caller scheduling must be explicit. These APIs do not establish server cadence.
- Sonic IsRecovering means its recovery coroutine exists
  (`SonicPolicyRunner.txt:13298`), not that the robot is fallen/dampened/resetting.
  Only the get-up handler starts it; reset completion stops it. With the current
  null clips and false Straighten, normal fall/count/reset leaves it false.
- Preserve existing physical fall measurement, gain dampening, countdown, and
  reset execution. An unmeasured motor shutdown hold or changed recovery assets
  requires explicit support rather than assuming a successful special command.

Geometry is also source-defined: `DistanceToOpponent` (`AIOpponentController.txt:5268`,
RVA `0x2367250`) measures planar root distance, ignoring Unity Y. In MuJoCo this
is XY distance. `AngleToOpponent` (`:4476`, `0x2366600`) horizontally projects
Robot.Forward and root displacement, returns zero when either squared projected
length is below `0.001f`, and computes their normalized signed angle about
Unity +Y. With MuJoCo vectors `f` and `d`, its sign is the sign of
`fy*dx - fx*dy`, using positive sign at zero, and its magnitude is the clamped
dot-product angle. Exactly behind therefore gives +180 degrees. Raw negative
atan2 is not identical at the signed-zero boundary. The earlier
[G1 heading proof](observable-balance-heading-20260921.md) establishes horizontal
root +X. Inputs should use both fighters' same pre-physics root snapshot.
`OpponentIsDown` (`AIOpponentController.txt:5692`) tests only the opponent's
confirmed `isFallen` flag, not a falling qualifier or a motor-suspension flag.

## Provenance and CPU verification

Current installed binary/assets and v7 provenance are pinned in the linked
heading proof. Additional SHA256 values checked for this recovery work:

| Private source or artifact | SHA256 |
| --- | --- |
| `AIOpponentController.txt` | `330956e4c257623fb78b0d619783c94b9a1f476bae4cf4e7c554491eb69c73e2` |
| `IPolicyRunner.txt` | `cddccbd0dac12bff62b5d69ec8f5a2191077ce7ddfb10e15e8981335e571777a` |
| `REKBehaviour.txt` | `23cd835247d11ae8a5263fab691084af9b0f5279ac86a4f497209bc9fc9718f7` |
| `RobotInputController.txt` | `f248df08449e3ff0706ce15ea07e4d58517f2fc9ed3f3143473fa48c4323bc21` |
| `SonicPolicyRunner.txt` | `5c7668aa79591cd84dfd120856ecdf96554309c85a2d5a425e8f42636381ab58` |
| `FightCoordinator.txt` | `9e847da90b34d96db852cf06c3dcd04760d92f4b4f84f525a2de140bbe6d42f5` |
| `C:\rekagent\work\rek-current-interop-b20ca0d\inventory-dummy-rek.json` | `d2daf98b8d545692262f0f68509fd7f9a5ffc31ec030559b156fce4e0e1721f2` |
| Private v8 probe named above | `b132eb19cb7b223a87ee3885c16e521e82d7e99006c09ed63e8cc899ad057686` |

CPU-only WSL Ubuntu-22.04 checks: GCC C++17 `-O2 -Wall -Wextra -Werror` passed
the unchanged 20,042-check Bot1 suite and the new 198-check recovery suite.
Clang C++17 `-O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined` passed
the same 198 recovery checks with no sanitizer diagnostic. Tests exercise all
tactical phases during falls, retries, result feedback, fault delays/overshoot,
guard combinations, lifecycle preservation, cancellation, and independent event
ordering. Artifacts are in WSL `/tmp/rek-bot1-g1-recovery-y0E0Fn`.

This is a client-source/static-configuration contract with CPU fixture evidence.
It does not measure authoritative live function dispatch, runtime field values,
Update/FixedUpdate cadence, Unity RNG, or physical response. No GPU test, game
launch, UI input, bridge mutation, or proprietary-source publication was used.
