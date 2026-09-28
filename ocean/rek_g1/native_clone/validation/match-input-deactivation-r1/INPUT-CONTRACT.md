# Match input deactivation

The original recovered client executable deactivates controls when a round ends and while the next round is prepared. It reactivates them in `ActivateRound`. The native clone previously accepted external direct commands throughout those phases.

Source directory: `C:/rekagent/work/controller-audit-isil/IsilDump/REKApp/REKApp`.

- `FightCoordinator.txt:69941`: `currentRound.IsActive=0`; line69943 calls `DeactivateControls` in `EndRound`.
- `FightCoordinator.txt:68637`: `StartRound` calls `DeactivateControls`; line68924 `ActivateRound` calls `ActivateControls`.
- `FightCoordinator.txt:65907` and66031: deactivation reaches the local input controller and both command sinks.
- `RobotInputController.txt:11134`: `Deactivate` zeroes the robot's velocity command. Line11179 cancels the composer's action. Line11238 plays idle for the nonvisual robot. Lines11241–11243 clear locomotion-active, stop-braking and momentum; line11256 clears isActive.
- `RobotInputController.txt:3086` and3492: native incoming velocity/move handlers check input-controller isActive before execution. Update/FixedUpdate also guard it at lines10308/10452.

This supports stopping dispatch between rounds. It does not establish every authoritative-server callback cadence. The global listener order remains separately unresolved.

Runtime integration tracks the active phase in one private byte per fighter. Immediately after each combat substep, an active-to-inactive edge calls the scheduler deactivation operation once. Subsequent inactive ticks do not repeatedly restart the idle clip. Round preparation still owns physical reset; control deactivation does not reset physics, motor history, ownership or the fight state.

The additive match pre-dispatch path ignores incoming movement/move/cancel requests while inactive, continues building the existing idle reference when the motor is unsuspended, and retains the direct/categorical mode. `INPUT_INACTIVE` is explicit clone-side rejection telemetry, not an invented original wire enum. The original request is preserved for this feedback; no move is queued for later activation. Tick-local categorical semantic output is cleared at the deactivation edge to avoid retaining a discrete-action assertion after the original CancelAction.

`runtime.cu` before this change is preserved in `original/runtime.cu`, SHA256 `2176c91043b70c468b1743c04f515d2ceec2b7e66c3d86412901d8332efbd91f`. `test_runtime_edges.py` extracts the exact new CUDA kernel body for a CPU launch-index test. Its26 assertions pass across two independent arenas, repeated inactive substeps, countdown, activation, fight-over and idle. It does not execute GPU code.

The scheduler implementation and tests are a separately reviewed collaboration contribution. Final source hashes and integrated GPU validation must be read from the final integration receipt; this document alone is not a runtime-success claim.
