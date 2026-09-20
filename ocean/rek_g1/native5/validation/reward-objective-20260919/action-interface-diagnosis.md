# Authentic action-interface diagnosis

Read-only review, 2026-09-19. Four complete private Sparring Bot 1 rounds under
`C:\rekagent\work\reward-objective-20260919-r1`. No runtime interaction,
deployment or training was performed for this diagnosis.

| Trial | Final awarded points | Five-point awards | Other awards | Native attack dispatch returns |
| --- | --- | --- | --- | --- |
| point_difference_v1-r1 | 10:6 | 5:0 | 5:6 | 101 |
| round_outcome_v1-r2 | 12:9 | 5:0 | 7:9 | 95 |
| point_difference_v1-r2 | 15:19 | 10:10 | 5:9 | 101 |
| round_outcome_v1-r4 | 20:15 | 10:5 | 10:10 | 114 |

These are packet-observed awarded points, not strike counts. The last win's
entire margin is the five-point award difference. The loss has equal five-point
totals and a four-point deficit in other awards. Each trial's sole action
rejection is the terminal ownership race. There is no observed lost attack
dispatch, and no dispatch proves server acceptance, execution or contact.

## Verified interface discrepancy

`live_transfer/encode_live.cpp:190` uses fixed candidate move durations because
visual clients do not expose authoritative action busy state. It marks the four
rounds busy for approximately 104.3, 99.9, 103.7 and 103.1 seconds, respectively.
At line 217 it zeros desired-command features 176..178 during those windows.

`fast_runtime.cu:250` allows pending yaw changes during an attack, but lines
269..275 return before root/yaw integration. Its feature publication also zeros
the desired commands during attacks. In contrast,
`Plugin.G1PolicyStream.cs:160` clears the visual client's transport-in-flight
state after the move send returns. Lines 296..310 then continue the held yaw
ramp through native request transport.

Correlating encoder QPC windows with the independent recorder's exact
`REK_Input` prefix projections gives:

| Trial | Input requests during projected busy | With nonzero yaw | With nonzero translation |
| --- | --- | --- | --- |
| point_difference_v1-r1 | 4,743 | 3,089 | 0 |
| round_outcome_v1-r2 | 3,973 | 2,028 | 0 |
| point_difference_v1-r2 | 5,205 | 3,539 | 0 |
| round_outcome_v1-r4 | 5,062 | 2,822 | 0 |

The median absolute nonzero busy-period yaw request is 0.069 in the loss and
0.068 in the last win. Those are requested command values, not measured angular
velocities. Most source snapshots report zero `velocity_command_xyz` even while
the independently captured native requests are nonzero. The source publication
boundary therefore cannot be treated as a record of the last transmitted input.
The encoder's explicit owned `desired_action` avoids that sampled-field problem.

Smallest independently trainable correction: a versioned owned-command
observation contract that retains pending desired yaw during candidate busy,
using the same rule in the runtime and live encoder. Client yaw-ramp state can
also be represented if captured explicitly. Neither change establishes the
server's response to those commands. Do not replace the native request ramp by
the compact model's root angular velocity, or shorten the busy durations on this
evidence alone. The current frozen checkpoint schema must remain unchanged.

## Tactical opportunities and limits

- The loss requests 17/101 attacks with opponent bearing beyond 90 degrees and
  19/101 at root distance above 1 m. The last win requests 19/114 and 49/114,
  respectively. These receipt-context measurements do not identify misses.
- Non-five-point local award contexts in the loss span 0.533..0.779 m and
  absolute projected bearing 0.2..52.4 degrees. In the last win they span
  0.527..0.827 m and 0.6..56.8 degrees. Request and award geometry are promising
  policy-learning inputs, but these few events do not justify hard range gates
  or causal labels.
- The loss makes seven attack requests while the local received count bit is
  active; the last win makes four. All sampled visual `fallen` and round `falls`
  values remain false/zero. Received referee data exposes this blind spot, but
  adding a live-only count gate would create a training mismatch in the
  constant-upright runtime.
- The current bearing is projected pelvis-local positive X. Recovered
  `Robot.get_Forward` at RVA `0x23e0100` uses negative root-right plus serialized
  `forwardYawOffset`; the live stream does not record that offset. This review
  does not prove an axis error and does not justify a 90/180-degree correction.
- Native dispatch age from the source observation has p10/median/p90 of
  19/22/35 ms in the loss and 20/23/32 ms in the last win. There is no evidence
  that a transport-latency fix explains the outcome difference.

## Reproduction evidence

For each `live-<trial>` directory use `trial/summary.json`,
`trial/relay.stdout.jsonl`, `trial/encoder.stdout.jsonl`, and
`contact-analysis/{summary.json,score-events.jsonl}`. Contact summaries bind the
native recorder capture by concurrent frame, QPC and both root poses, and list
the input hashes. Aggregate calculations above use each encoder observation's
busy state at the latest preceding QPC and do not interpolate packet effects.

The loss relay SHA256 is
`d84bb5d75cda966010d6efbd602e9569a39089710f94df122dc7c7f9ece4e94e`.
Its recorder capture is
`C:\rekagent\evidence\runtime\rek-private-ai-protocol-v7\rek-private-ai-root-motion-20260919T232016.8516086Z-pid190584-b8eda200563c4b0b8e70406aa0a22f41.jsonl`,
SHA256 `15b60588f0bc1773a2d2d9397c9a33845b1f73a29f128931933483417d8ef1fc`.
Private recovered call evidence is
`C:\rekagent\work\controller-audit-isil\IsilDump\REKApp\REKApp\Robot.txt:38200`
and `RobotInputController.txt:250`. No raw packet, player-state record or
proprietary code dump is included in this note.
