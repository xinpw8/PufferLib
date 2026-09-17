# REK UI pipe client

`RekUiPipeClient.exe policy-relay <expected-bridge-sha256>` provides a persistent
stdin/stdout JSON-line connection for the G1 transfer experiment. It initially
requests state only, verifies the live REK process, build, bridge hash and
isolated Spark session, then accepts explicit requests. It never acquires a
lease or starts a match by itself. See
[the source/action contract](../RekUiBridgeAgent/G1_POLICY_STREAM.md).

This build-pinned client is the only supported caller for
`RekUiBridgeAgent` on the isolated Spark REK runtime. It opens the local
current-user pipe, proves that the server process is the live `REK.exe`, and
validates the REK binaries, bridge binary, isolated Wine session, command
schedule, private-session state, Bot 1 identity, and exact T800 pairing before
allowing a measured schedule.

```powershell
RekUiPipeClient.exe state [--bridge-sha256=HASH] [output.jsonl] [timeout_seconds]
RekUiPipeClient.exe enter-private output.jsonl [timeout_seconds]
RekUiPipeClient.exe schedule output.jsonl [timeout_seconds]
RekUiPipeClient.exe trial selector output.jsonl [timeout_seconds]
RekUiPipeClient.exe controller output.jsonl [run_seconds|until-ended]
RekUiPipeClient.exe g1-held output.jsonl [timeout_seconds]
```

The optional `--bridge-sha256=` argument applies only to `state` and requires
exactly 64 lowercase hexadecimal characters. It replaces only that invocation's
expected bridge SHA256; all existing hello, REK process, game build, bridge version,
isolated Spark host and absent-lease checks remain required. Other modes reject
the option before connecting. Omitting it preserves the legacy compiled hash.
`state` sends one `get_state` request and never acquires a lease, starts a stream,
or sends Stop/Release cleanup commands. With no output path it writes no transcript.
The persistent `policy-relay` mode sends Stop/Release on normal stdin closure and
must not be used for a strictly read-only state query.

`enter-private` uses only the recovered Unity main-thread methods for Login,
Free Play, and `KothScreenController.OnSoloClicked()`. The request acknowledgement
does not count as server acceptance. The command succeeds only after state proves
the active multiplayer session is private and the opponent is exactly Sparring
Bot 1 with no human client in its slot.

`schedule` requires a proven private Bot 1 session. If the round is idle, it
issues the recovered `StartRound` semantic command and waits for active gameplay.
It then requires exact T800 versus T800 identity, validates every command
boundary and the terminal move-send and neutral-send counts, confirms lease
release, and confirms the released state. Thus a successful `enter-private`
followed by `schedule` covers the cold-start path without keyboard input.

`trial` accepts exactly `forward`, `backward`, `strafe-left`, `strafe-right`,
`yaw-left`, `yaw-right`, `move-2`, `move-3`, `move-4`, `move-5`, `move-9`, or
`move-10`. It requires an inactive private Bot 1 session, starts and binds a
new unique round, verifies exact T800 versus T800 identity and a complete
pending-free initial-state record, and consumes that round identity before the
single action can start. The transcript validates the local command edge and
the return of the recovered client send method. It does not claim that the
server accepted the request or that authoritative execution occurred.

`controller` starts the continuous private Bot 1 controller. An integer duration
from 1 to 600 seconds retains the bounded mode and defaults to 120 seconds.
`until-ended` holds the exclusive lease and streams until the bridge emits
`continuous_controller_end`; Ctrl+C requests a semantic controller stop before
the lease is released. It requires exact local semantic and measured runtime
T800 identity plus the opponent's exact measured T800 runtime bone signature.
The opponent semantic robot ID may be stale and is recorded but is not trusted.
The transcript validates controller, round, request, local-motion, recovery,
and telemetry records, but can establish only that client methods returned and
that local state transitions were observed. It cannot establish server
acceptance or authoritative execution.

`g1-held` is the machine command for a fresh-round G1 held-input measurement.
It requires an inactive proven solo Sparring Bot 1 session, sends only the
existing semantic `StartRound` request, waits for a new active round, and then
requires exact G1 versus G1 runtime signatures before starting the schedule.
The default timeout is 150 seconds. The client verifies every 50 Hz schedule
tick, every 500 Hz kick lifecycle observation, each translation release and
direction-specific `TransitionSettled` result, every local return and request
lifecycle record, the immutable round identity, the exact schedule hash, and
terminal coverage. Translation remains held at its kick edge, releases 100 ms
later, and stays neutral for the rest of the four-second observation window.
It publishes a validated partial transcript and returns nonzero if the round
ends or lifecycle coverage is incomplete. It does not emit keyboard, mouse, or
gamepad input.

Mutation modes require a new transcript path. Records are flushed as they arrive
to a uniquely named `.partial-*` file. A complete run is flushed to disk and
published to the requested path with a no-overwrite rename only after terminal
validation and confirmed lease release. Failed partials remain available for
diagnosis and cannot be mistaken for successful artifacts.
