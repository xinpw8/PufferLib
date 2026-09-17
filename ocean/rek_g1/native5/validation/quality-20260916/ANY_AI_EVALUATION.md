# Any private REK AI evaluation

User authorization on 2026-09-17 UTC expands policy evaluation to any REK bot.
The native game chooses the opponent. No difficulty setter was introduced.

## Scope change

The live driver uses `StartG1PolicyStreamAnyAi`, `StartG1PolicyRound`, and
`ExitLostG1PolicySession`. It requires the new `policy_proven` and
`policy_active_gameplay_proven` fields. Legacy `proven`,
`active_gameplay_proven`, and `StartG1PolicyStream` retain exact Bot1 semantics.
Reading an ineligible legacy Bot1 proof no longer erases a valid higher-bot
private route. Human, occupancy, endpoint, and runtime changes still reject it.

Each active stream pins the measured `client_ai_difficulty` and
`sparring_bot_number`. Both native action boundaries and the driver reject an
identity change. The source stream and final summary record the opponent.
The existing isolated Spark marker, client-only private solo route, no-human
occupancy, exact G1 fighter pairing, control lease, round identity, action
mask, and stale-action restrictions remain required. This change does not
make the 223-feature G1 checkpoint compatible with another robot model.

Known byte difficulties 0 through 255 and consistent bot numbers are covered
offline. This is an input-contract range, not a claim that REK offers 256
distinct bot configurations. Unknown identity is rejected. A higher bot is
no longer exited simply to seek Bot1.

[24 Node tests](any-ai/node-tests.txt), [6,433 relay assertions](any-ai/relay-tests.json),
and [290 protocol cases](any-ai/protocol-tests.json) passed. The bridge build
had zero warnings/errors. [Build/source hashes](any-ai/validation-manifest.json)
and [deployment logs](any-ai/any-ai-deployment/stdout.txt) are preserved.
Only the owned isolated Spark game process was restarted. The previous DLL
was backed up outside the plugin directory. No Windows input was emitted.

## Executed evaluation

All policy trials used the unchanged native CUDA BF16 checkpoint
`f87dae69a777e4ac28782bdee89b30d7434208d773be75bf97f56fba4a52b07e`,
worker seed 73, rendered-pose observation projection, and 50 Hz target cadence.
No training, reward, simulator-physics, or policy-weight change was made here.
Execution was the installed authentic REK client under Wine/Box64 on
`spark-4ae3`, isolated display `:98`; it was not a Windows ground-truth run.

| Trial | Actual opponent | Recorded outcome | Locally accepted attacks | Process exit |
| --- | --- | --- | ---: | ---: |
| r1 | Bot1, difficulty 0 | Round 1 won 18:10 by points | 95/95 | 0 |
| r2 | Bot1, difficulty 0 | Round 2 incomplete; last observed 2:2 with 80.41536 s left | 31/31 | 2 |
| r3 | Unobserved | Startup failed before bridge availability or input | 0 | 2 |
| r4 | Bot1, difficulty 0 | Fresh round 1 lost 5:17 by points; transport fix deployed | 67/67 | 0 |

r1 ran from 01:00:25 to 01:02:24 UTC. It entered through the native private
solo route and exercised `ReadyPrivateAiSession` successfully, including
waiting for the spawned active fighter pair before policy actions. There were
5,640 predictions and 5,639 applied actions. One nonattack action was rejected
at round shutdown. The terminal native winner index was 0, the controlled slot.
Owned velocity was neutralized and the control lease released. Sources:
[summary](any-ai/live-any-ai-r1/trial/summary.json),
[command timeline](any-ai/live-any-ai-r1/stdout.jsonl), and
[passive audit](any-ai/live-any-ai-r1.audit.json).

The game automatically advanced to round 2. r2 attached with 115.69108 s left,
so it was not a fully controlled round from its first tick. Telemetry stopped
after 1,702 successful actions. The driver detected `source_stream_missing`;
stop and release requests timed out. It correctly returned exit 2 despite
earlier successful actions. The last in-progress score is not a draw or a
completed result. Sources: [summary](any-ai/live-any-ai-r2/trial/summary.json),
[command timeline](any-ai/live-any-ai-r2/stdout.jsonl), and
[passive audit](any-ai/live-any-ai-r2.audit.json).

The game remained responsive to a later fresh pipe connection. At 01:07:20 UTC,
[passive state](any-ai/any-ai-recovery-state.json) showed no control lease, no
running policy stream, and an inactive private AI arena. The game bridge log
recorded `Pipe server connection failed: NullReferenceException` between
connections 2 and 3. The exception's originating operation was not established.
Source inspection identified a liveness defect: the server awaited the reader
alone and could fail to observe an already-faulted writer until the reader
closed. The per-frame send itself queues data; a synchronous Unity-thread pipe
write deadlock was not demonstrated.

## Transport repair and verification

A real local-pipe regression injected a writer serialization exception while
the client kept its side open. The original server left the connection identity
active and the test timed out. After the fix, either loop ending retires the
connection, clears its identity before cancellation, closes the pipe, and
drains both tasks with a bounded wait. The next client can connect and exchange
messages. Fixed diagnostics identify the failing loop and exception type
without recording exception messages.

The [before-fix failure](any-ai/transport/writer-failure-before.json),
[302 passing protocol cases](any-ai/transport/protocol-tests.json), and
[new build manifest](any-ai/transport/validation-manifest.json) are preserved.
Tests cover writer failure, reader failure, reconnect, and stuck-task cleanup.
The new DLL is
`6c88b9e3718f5549c146865a4a336515385e5b019fd8321a6f5e1fec4253a5e4`;
the relay is unchanged. It was [deployed on Spark](any-ai/any-ai-transport-deployment/stdout.txt)
after verifying and stopping only owned process 4111, with a verified backup.

r3 did not reach a policy stream. Its relay timed out connecting, the game
process subsequently was absent, and startup logs ended before bridge
availability. No input was sent and this is not an evaluation result. See
[r3 summary](any-ai/live-any-ai-r3/trial/summary.json) and
[relay error](any-ai/live-any-ai-r3/trial/relay.stderr.txt). The existing isolated
launcher was retried without changing any source, policy, or authentication.

The retry, r4, ran a fresh round from 01:14:49 to 01:16:48 UTC and completed
with a 5:17 points loss. There were 5,727 predictions, 5,726 applied actions,
and 67/67 locally accepted attacks. The one rejected nonattack action crossed
the terminal boundary. There was no source-stream timeout; stop and release
were acknowledged, owned velocity was neutralized, and the control lease was
released. Sources: [summary](any-ai/live-any-ai-r4/trial/summary.json),
[timeline](any-ai/live-any-ai-r4/stdout.jsonl), and
[passive audit](any-ai/live-any-ai-r4.audit.json). This verifies one normal
full-round session after deployment. Fault-reconnection behavior was exercised
by the offline local-pipe regression, not by inducing a fault in the live game.

## Interpretation and retained evidence

The two completed rounds here produced one win and one loss. This demonstrates
one authentic round win, not a completed match win, a
reliable win rate, a superhuman policy, or simulator parity. The same checkpoint
previously lost 4:14 in the [prior trial](PRIVATE_READY_AND_LIVE_R3.md).
Bot1 was assigned in every observed round here; higher-bot acceptance has offline tests
but no completed higher-bot live trial yet. Removing the opponent restriction
cannot be credited with improved policy strength from these observations.

Raw requests, stdout, stderr, game observations, worker actions, configs, and
hashes remain under
`/home/spark-advantage/rek-training/policy-quality-20260916-r1/`, in
`live-any-ai-r1/` through `live-any-ai-r4/`. Published files are aggregate reports and text copies;
raw hashes refer to the private originals. No game binaries, model weights,
credentials, or account tokens are included.
