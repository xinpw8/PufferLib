# G1 source telemetry and policy requests

Schema `rek.g1_policy_source.v1` reports client-local observations. It does not
claim authoritative server state, command acceptance, simulator parity, or a
complete 223-value policy observation. Missing quantities are JSON null.

`RekUiPipeClient policy-relay <expected-bridge-sha256>` keeps one verified local
named-pipe connection. It starts without a control lease or automatic input.
Requests and responses are JSON lines on stdin/stdout. EOF closes the lease.

Read-only requests are `get_state` and `get_policy_state`, each with a unique
`request_id`. Control requires explicit `AcquireExclusiveControl`, then
`StartG1PolicyStream`. Existing `StartRound` can be requested separately. No
opponent actions are exposed. `StopG1PolicyStream` and `ReleaseExclusiveControl`
neutralize owned commands. Native Windows never passes the isolation guard.

An action request has exactly these keys:

```json
{"type":"policy_action","request_id":"a1","round_identity_sha256":"64 lowercase hex characters","observation_sequence":1,"action":17}
```

The observation must have been emitted during the active stream, have the same
immutable round identity, be newer than the last consumed action observation,
and be at most 250 ms old according to producer QPC. Requests are revalidated on
the Unity main thread. Local action rejection does not queue or retry a move.
After the first valid action, 250 ms without another valid action stops and
neutralizes the stream. Startup has a bounded 1 s grace period while commands
remain neutral. The per-action 250 ms source-age limit applies during startup
as well.

Version0.4.9 publishes continuous telemetry from the actual Unity LateUpdate
callback, once per rendered frame at most, with a producer-QPC minimum interval
of 20 ms. Fixed-step catch-up never republishes the same rendered pose. Native
held velocity and scope checks remain in the 500 Hz fixed-step path. The
watchdog is checked in Update after queued commands have been validated and
dispatched. During an active policy stream, ordinary automatic UI-state
capture is suspended to avoid duplicating the expensive per-frame snapshot;
explicit read-only requests remain supported. A final source observation can
still be emitted on scope loss before the stream terminates.

`g1_policy_state` records have these fields:

- `schema`, `observation_sequence`, `round_identity_sha256`, `local_slot`,
  `phase`, `stream_active`, `global_input_emitted:false`,
  `authority_scope:"client_replicated_and_local_observations"`,
  `server_acceptance:"unknown"`.
- `clock`: `utc`, `unity_frame`, `unity_time`, `unity_fixed_time`,
  `qpc_ticks`, `qpc_frequency_hz`.
- `fighters[2]`, in network-slot order: `root_position_xyz`,
  `root_rotation_xyzw`, `root_linear_velocity_xyz`,
  `root_angular_velocity_xyz`, `visual_only`, `player_controlled`, `falling`,
  `fallen`, `dampened`, `resetting`, `motor_shutdown`, `tilt_angle`,
  `floor_contact_count`, `bone_names[30]`, `bone_local_rotations_xyzw[30][4]`,
  `bone_world_positions_xyz[30][3]`,
  `base_linear_velocity_local_xyz`, `base_angular_velocity_local_xyz`,
  `joint_positions:null`, `joint_velocities:null`, `last_hit:null`,
  `runner:{available,is_done,is_recovering,current_motion_name,motion_frame_index,current_move_index,motion_identity_status}`.
  `current_motion_name` and `current_move_index` are null: the recovered
  MotionSequence is not a named clip and has no proven clip-identity mapping.
  Root velocity getters are local Robot fields. Their remote-authority validity
  is unknown for visual-only robots. Base velocities are null when the native
  TryGetBaseVelocityLocal call reports unavailable.
- `input`, for the local fighter only: `active`, `punching`, `recovering`,
  `velocity_command_xyz`, `pending_move`, `pending_move_index`,
  `pending_special`, `pending_estop`, `desired_action`, `requested_move_index`,
  `requested_move_qpc_ticks`, `move_request_pending_transport`,
  `move_send_method_returned`, `action_busy`, `action_busy_source`, `allow_move_interrupt`,
  `native_transition_settled:{forward,backward,strafe_left,strafe_right}`.
  `desired_action` is this bridge's retained held category. `requested_move_index`
  and its QPC timestamp describe the latest local request, not a server-accepted
  or currently playing move. Both are null outside an active stream. For
  visual-only fighters, `action_busy` is null and `action_busy_source` is
  `unavailable_visual_only_client`; the raw punching/recovering flags remain
  local diagnostic getter values. Version0.4.8 treats the request lifecycle as
  complete for transport only after the matching SendMoveEvent returns and the
  native pending field clears. It does not wait for an inapplicable local
  playback flag, estimate a busy interval, or claim playback completion.
- `round`: `number`, `duration`, `time_remaining`, `active`, `redo`,
  `clean_hits[2]`, `falls[2]`, `result`, `result_value`, `winner_index`, `knockout`.
- `fight`: `current_round`, `rounds_won[2]`, `result`, `result_value`,
  `winner_index`. `referee:null`: no referee packet subscription is implemented
  by this stream. Round clean hits/falls are measured replicated counters.
- `action_mask[33]`: native client transport and bridge-owned command gates,
  not authoritative server readiness or distance/facing restrictions.
  `action_mask_source` makes that distinction explicit.

`phase` is the authentic enum:0 Idle,1 RoundActive,2 RoundEnd,3 BetweenRounds,
4 FightOver,5 Setup,6 Sandbox. The compact candidate uses2 for active, so an
observation adapter must explicitly translate semantics. RoundResult is0
InProgress,1 WonByPoints,2 WonByKO,3 Tie,4 Redo. FightResult is0 InProgress,1
WonByRounds,2 WonByTKO. These enum constants and the RoundState getter list were
read directly from the measured interop assembly metadata. RoundState exposes
CleanHits and Falls, with no separate Score or Points getter; no synthetic score
formula is applied by this source stream.

Actions 0/1 mean hold/release. Actions 2..15 are W,S,A,D,Q,E,WQ,WE,SQ,SE,AQ,AE,DQ,DE.
The recovered command axes are forward, strafe, yaw; A and Q use positive
strafe/yaw, D and E use negative. Keyboard yaw ramp uses the measured native
configuration. Categories16..32 map to move indices
`6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16`.
Translation must be released and its native TransitionSettled predicate true
before an attack is requested. The attack mask checks retained held translation
as well as instantaneous native velocity: a transient zero in the latter does
not release a retained W/S/A/D command. Attack requests preempt outgoing yaw while their
transport is pending. On visual-only clients, desired yaw may resume when the
send returns and pending clears. The authentic server retains its own action
execution gates. The bridge never queues or retries a rejected request.

`g1_policy_action` acknowledges the action, source observation, request ID,
round, local disposition, optional ExecuteMoveByIndex boolean and current
pending move. A local acknowledgement does not prove network dispatch or
server acceptance. `g1_policy_end` gives the stop reason and whether exact
owned local velocity was neutralized and a neutral send method returned.

`g1_policy_dispatch` records the actual matching SendMoveEvent return, its
original request/observation identifiers and move index. This is transport
evidence only. Execution, completion and server acceptance remain unknown.
