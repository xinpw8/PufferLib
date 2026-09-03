# REK UI State Bridge Agent prototype

This directory contains an uninstalled BepInEx IL2CPP plugin prototype for the current Windows REK build. It is a local state-observation bridge for a separate UI mirror. It does not reproduce the REK visuals by itself.

## Build binding

The plugin compiles against the current generated interop at:

`C:\rekagent\work\rek-current-interop-b20ca0d\interop-v1.5.3`

It fails closed unless the installed build has both of these measured hashes:

- `GameAssembly.dll`: `6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412`
- `global-metadata.dat`: `e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd`

## Transport and protocol

The bridge uses only the Windows named pipe `rek-ui-bridge-v1`. `PipeOptions.CurrentUserOnly` is mandatory. Each accepted connection must also pass a fail-closed `GetNamedPipeClientComputerNameW` check against the REK host name, or return Win32 `ERROR_PIPE_LOCAL` (229), which directly identifies a local pipe. There is one client and at most 32 pending requests.

Messages are newline-delimited UTF-8 JSON, at most 4096 bytes. Request IDs are 1 to 64 ASCII letters, digits, `.`, `_`, `:`, or `-`.

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

All input requests are currently rejected from Unity's main thread with `applied:false` and reason `verified_process_targeted_unity_input_delivery_not_implemented`. There is no autonomous input loop. Global `SendInput`, window messages, and process-wide keyboard injection are intentionally absent because they cannot provide verified REK-only delivery.

The published `space_gate_would_allow` field is observation only. It requires current-build evidence of a connected client-only solo session, an AI opponent slot with no client or human markers, an inactive round, and a visible enabled PostFight continue prompt. It never generates Space.

`EnterSolo` is deliberately rejected. The recovered `KothScreenController`
route calls `OnSoloClicked()`, which enters the public solo arena. `IsSolo`,
unranked state, an arena ID, and an empty opponent client slot do not distinguish
that case from a private room. Controlled acquisition additionally requires the
active Unity multiplayer session's measured `ISession.IsPrivate` property to be
true. Session IDs and room codes are not read or published.

## Current limitations

- The plugin has been compiled, its protocol/local-pipe code has standalone
  tests, and its hash-bound runtime field access has produced controlled traces
  on the measured Windows build. The repository contains source only.
- Focus is polled once per `LateUpdate`. A transient that begins and ends inside one frame can be missed.
- Pointer-hover styling that does not change UI Toolkit focus or EventSystem selection is not represented.
- Home cards have explicit mirror records. Other screens expose the focused element only, not a complete visual tree.
- The state record carries timing and semantic state, not pixels, animations, textures, fonts, or audio.
- Named-pipe delivery adds scheduling latency. The Unity frame/time is the observation timestamp; receipt time is not identical to render time.
- An `applied` input acknowledgement does not exist because no input path has been accepted as target-exclusive.

No network client, credential reader, screenshot capture, proprietary binary copy, game-file write, REK launch, or REK restart is implemented.
