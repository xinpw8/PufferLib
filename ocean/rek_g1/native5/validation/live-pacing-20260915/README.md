# Live action acknowledgment pacing

Trial r2 produced 423 inferences and 423 game acknowledgments. Inspection found
observations 2 through 15 arrived before the first acknowledgment. Actions from
observations 2, 3, and 4 were subsequently applied in the same Unity frame 3537,
at identical Unity time 61.67360656179138. The orchestrator had released its
in-flight slot immediately after writing each action to the relay.

The corrected orchestrator retains that slot until the game acknowledges the
exact request ID, observation sequence, round hash, and action. While occupied,
incoming observations are discarded. Following acknowledgment, the next source
must have measured Unity QPC ticks strictly greater than the acknowledgment.
Pre-acknowledgment sources are discarded even if they arrive later. Nothing is
queued or replayed. Rejected actions advance the same measured boundary.

Missing clocks fail explicitly. Non-increasing clocks cannot substitute host
time. An unacknowledged action stops the run after the bounded watchdog timeout.
Existing shutdown stops streaming and releases bridge-owned controls.

Setup also handles an observed lost private session only when solo Bot 1 scope
is proven, `round_active` is false, `round_inactive` is true, the post-fight prompt
is visible, `post_fight_is_winner` is false, and private-entry recovery is enabled.
It cannot issue a midround exit and does not add a policy quit action.

Eight Node tests passed on both Windows and Spark. They include one-in-flight
retention, mismatched/duplicate acknowledgments, stale sources, rejected actions,
missing/regressing clocks, and every required lost-session guard. No GPU code or
physics was changed. These unit tests do not claim a successful subsequent live
trial.

[Spark output](test.stdout.txt), [status](test.status.json), [command](test.command.json),
and [source hashes](hashes.json).
