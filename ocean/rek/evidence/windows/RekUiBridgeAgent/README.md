# REK UI State Bridge Agent prototype

This directory contains an uninstalled BepInEx IL2CPP plugin prototype for the current Windows REK build. It is a local state-observation bridge for a separate UI mirror. It does not reproduce the REK visuals by itself.

## Build binding

The plugin compiles against the current generated interop at:

`C:\rekagent\work\rek-current-interop-b20ca0d\interop-v1.5.3`

It fails closed unless the installed build has both of these measured hashes:

- `GameAssembly.dll`: `6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412`
- `global-metadata.dat`: `e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd`

## Transport and protocol

The bridge uses only the Windows named pipe `rek-ui-bridge-v1`. `PipeOptions.CurrentUserOnly` is mandatory. Each accepted connection must also pass a fail-closed `GetNamedPipeClientComputerNameW` check against the REK host name, or return Win32 `ERROR_PIPE_LOCAL` (229), which directly identifies a local pipe. Wine 11 exposes that computer-name API only as a stub, so an `EntryPointNotFoundException` activates a narrow fallback: `GetNamedPipeClientProcessId` must return a nonzero process that resolves as live in the same local process namespace. All other failures reject the connection. The hello record names the verification method in `local_client_verification`. There is one client and at most 32 pending requests.

Requests are newline-delimited UTF-8 JSON, at most 4096 bytes. Responses are
newline-delimited UTF-8 JSON and may be larger because active-fighter state
contains the measured ordered bone names. The pinned client rejects any response
larger than 1 MiB. Request IDs are 1 to 64 ASCII letters, digits, `.`, `_`, `:`,
or `-`.

State request:

```json
{"type":"get_state","request_id":"mirror-1"}
```

The server also pushes a `state` record when its stable observed state changes. State is sampled from Unity `LateUpdate` and includes:

- active scene and `LobbyShellController.CurrentScreen`
- game-menu open state and pane
- UI Toolkit focused element name, type, path, display text, bounds, and enabled/visible state
- Unity EventSystem selected GameObject path and selectable state
- known Home-screen card states and whether each card is focused
- displayed Home user label, without reading `GameContext.AccessToken`
- measured private-AI and post-fight prompt predicates for a future Space gate
- Unity frame/time plus exact game, metadata, and plugin hashes

Text from input fields or elements whose names indicate passwords, secrets, tokens, cookies, credentials, recovery, or MFA is redacted.

## Input status

The parser recognizes only these exact atomic key names:

`Left`, `Right`, `Up`, `Down`, `Enter`, `Escape`, `Space`

All atomic input requests are currently rejected from Unity's main thread with `applied:false` and reason `verified_process_targeted_unity_input_delivery_not_implemented`. Global `SendInput`, window messages, and process-wide keyboard injection are intentionally absent because they cannot provide verified REK-only delivery. The isolated continuous controller described below uses recovered `RobotInputController` and game-menu methods only.

The published `space_gate_would_allow` field requires current-build evidence of a connected client-only solo session, an AI opponent slot with no client or human markers, an inactive round, and a visible enabled winning PostFight continue prompt. No path generates Space. During an explicitly started continuous-controller run, the same predicate may authorize one direct `GameMenuController.HandlePostFightContinue()` request for that inactive transition. The controller then requires a new, non-reused active round identity before it resumes.

Current-build native code at `GameMenuController.HandlePostFightContinue` RVA
`0x23AAE90` branches on `postFightIsWinner`. A win requests
`FightCoordinator.SendPostFightIntent(stay:true)` and closes the menu. A loss
calls `ExitToLobby`. The controller therefore stops with an explicit limitation
instead of attempting an automatic same-session restart after a loss.

`EnterSolo` invokes the current build's `KothScreenController.OnSoloClicked()`
only from the observed Free Play screen, with no connected session and the
visible enabled `soloButton`. Current-build localization identifies this route
as `PRACTICE ALONE`, `Claim an arena to yourself`, and `Private Practice`.
Issuing the request is not proof that the server accepted it. Controlled
acquisition still requires the active Unity multiplayer session's measured
`ISession.IsPrivate` property to be true. Session IDs and room codes are not read
or published.

Semantic mutation is disabled on native Windows regardless of foreground
state. The isolated Spark X session is the only enabled execution surface, and
it is accepted only when the Wine-only
`ntdll!wine_get_version` export returns exact version `11.13` and all three
inherited process environment values exactly match
`REK_EVIDENCE_ISOLATED_SESSION=spark-x98`, `DISPLAY=:98`, and
`WINEPREFIX=/opt/codexrook/wineprefix`. A native Windows process cannot satisfy
the Wine export proof by setting environment strings. The boolean result and
proof are published as `foreground.isolated_session_verified` and
`foreground.isolated_session_proof`.

## Rendered command-edge markers

An accepted measured schedule on the isolated Spark session displays 24 fixed
8 by 8 pixel marker cells at the top left of the REK render. Each cell begins
exact black and changes once to exact magenta after its corresponding semantic
velocity or move command has been applied on the Unity fixed-update thread.
The change is visible in the first subsequent `OnGUI` render and persists after
the schedule completes. Native Windows never displays the strip because it
cannot pass the isolated-session mutation proof.

Each transition is also emitted as `rendered_command_marker_edge` with schema
`rek.rendered_command_marker.v1`, its unique selector, command identity,
schedule run ID, region, colors, Unity frame, and fixed time. The producer
asserts
`first_post_marker_frame_is_first_rendered_frame_after_command_edge`. The
lossless frame capture and `video_clock_anchor.py` must still machine-detect the
exact persistent transition and bind the resulting video and trace hashes.
Neither the marker event nor its appearance proves server acceptance.

## Single-motion trials

`StartSingleMotionTrial` is available only through an exclusive local pipe
lease in the verified Spark X `:98` Wine session. It accepts the 12 selectors
defined by `rek.single_motion_trial.v1`. A trial requires a new active round
that was requested by the same connection, unchanged private Bot 1 session
identity, exact T800 versus T800 pairing, an exact neutral velocity, no pending
move, special, or emergency-stop request, and a complete finite initial-state
record. Round identities are consumed before motion begins and cannot be
reused. The bridge stops the trial if its lease, session identity, round,
pairing, controller, owned command value, or pending-command state changes.

Trial records carry UTC, Stopwatch, Unity frame and time fields compatible with
the evidence recorder's clocks. A `single_motion_trial_client_request` record
means only that the recovered client send method returned. Every record states
that server acceptance and authoritative execution are unknown.

## Continuous private Bot 1 controller

`StartContinuousBotController` is mutually exclusive with measured schedules
and single-motion trials. It runs only with an exclusive pipe lease in the
verified Spark X `:98` session and a proven private, unranked Sparring Bot 1
session. The local fighter must have exact semantic and measured runtime T800
identity. The opponent must have the exact measured T800 ordered-bone runtime
signature. Its potentially stale semantic robot ID is not trusted for
acceptance, and any semantic/runtime disagreement is retained in telemetry.

At 50 Hz, the controller measures both fighter root transforms, emits distance,
bearing, and heading telemetry, turns to face the opponent, and round-robins the
six pinned T800 attack methods only after local motion completion and readiness
are observed. The deterministic round robin is explicitly labeled as an audit
controller divergence from the shipped opponent's randomized attack policy.
Per-move range and angle values are explicitly non-calibrated. Facing mirrors
the current-build `AIOpponentController.ComputeFacingYaw` at RVA `0x2366E20`:
the signed `AngleToOpponent` bearing uses a `17.5` degree deadband, magnitude
ramps as `abs(angle) / 45`, saturates at the serialized yaw command magnitude
`1.5`, and negates the bearing sign. Thus an opponent to local right produces
negative yaw and an opponent to local left produces positive yaw.
Pinned impact times, lead times, and release times are serialized asset metadata,
not observed runtime timing. Clean-hit and fall counters are emitted separately
without causal attribution to a request.

Fall recovery uses only recovered semantic calls and mirrors the current-build
`AIOpponentController.DriveRecovery` at RVA `0x2367430`. While fallen and not
dampened it requests `Dampen` (`SpecialCommand` 4). Once dampened it requests
`Straighten` (`SpecialCommand` 1) once. After recovery is armed it requests the
measured suggested prone or supine get-up (`SpecialCommand` 2 or 3). Emergency
stop is not part of ordinary fall recovery. It is reserved for the separate
build-pinned `motorShutdownHold` fault cycle at
`AIOpponentController.UpdateFaultEStopCycle` RVA `0x23680E0`, after the
serialized 3-second delay and with a 0.5-second engaged hold. No physical Space
mapping is asserted. Every lifecycle record is limited to client request edges
and local observations. It never establishes server acceptance or
authoritative execution.

## Current limitations

- The plugin has been compiled, its protocol/local-pipe code has standalone
  tests, and its hash-bound runtime field access has produced controlled traces
  on the measured Windows build. Runtime deployment must use the exact reviewed
  plugin hash.
- Focus is polled once per `LateUpdate`. A transient that begins and ends inside one frame can be missed.
- Pointer-hover styling that does not change UI Toolkit focus or EventSystem selection is not represented.
- Home cards have explicit mirror records. Other screens expose the focused element only, not a complete visual tree.
- The state record carries timing and semantic state, not pixels, animations, textures, fonts, or audio.
- Named-pipe delivery adds scheduling latency. The Unity frame/time is the observation timestamp; receipt time is not identical to render time.
- An `applied` input acknowledgement does not exist because no input path has been accepted as target-exclusive.

No network client, credential reader, screenshot capture, proprietary binary copy, game-file write, REK launch, or REK restart is implemented.
