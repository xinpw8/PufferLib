# Observable balance projection

`observable_balance.h` defines the opt-in 223-float schema
`rek.native5.observable_balance.v1`. **It is not connected to either
`runtime.cu` or `live_transfer/encode_live.cpp`.** Existing checkpoints and the
ongoing physical baseline still use their existing observations. Integration
requires fresh training and the same new projection at evaluation.

The physical producer currently exposes support contacts, physical fall phases,
controller route/busy state, instantaneous body/joint velocities and contact
attribution. The live producer has rendered poses, received points/count state,
client requests and pose differences. The shared projection removes unsupported
internal state, computes velocities from preceding observed poses in both
producers, and represents missing data explicitly.

## Exact feature mapping

Per-fighter columns use `b=0` for the actor and `b=86` for the opponent. Both
snapshot inputs use absolute fighter slots; `actor_slot` determines ordering.

| Columns | Meaning |
| --- | --- |
| `b+0..2` | Common-frame root position. Opponent X/Y are replaced at 86/87 below. |
| `b+3..6` | Normalized, sign-canonical common-frame root quaternion WXYZ. |
| `b+7..8` | Root-origin position difference / observed time, rotated into the current horizontal local +X heading. Requires history and heading availability. |
| `b+9` | World-vertical root-origin position difference / observed time. Requires history. |
| `b+12` | Wrapped difference of horizontal root heading / observed time. Availability is `b+76`. |
| `b+13..41` | Pose-projected hinge twist angles, wrapped to one revolution. Availability is `b+74`. |
| `b+42..70` | Wrapped pose-projected joint differences / observed time. Availability is `b+75`. |
| `b+71` | Current horizontal local +X heading is observable. A vertical forward axis has no horizontal heading. |
| `b+72` | Quaternion-derived root up-axis tilt divided by pi, in [0,1]. No fall threshold is applied. |
| `b+73` | Absolute root height, not standing-height ratio or inferred floor height. |
| `b+74` | Current projected joint pose is available. |
| `b+75` | Current and preceding projected joints and observation history are available. |
| `b+76` | Current and preceding headings and observation history are available. |
| 86, 87 | Horizontal root distance; actor-relative bearing divided by pi. Bearing is zero padding if heading is unavailable or distance is zero. |
| 172..175 | Current actor horizontal-heading quaternion. Availability follows column 71. |
| 184, 185 | Absolute actor slot; observation phase 4 at terminal, 2 when active, otherwise 0. |
| 188, 189 | Round duration and remaining time, each divided by 120 s. |
| 190, 191 | Observed actor/opponent awarded-point counters, including referee awards. |
| 202 | Received/native referee count state is available. |
| 203 | Same-round, same-perspective preceding observation is available with a positive interval no greater than 250 ms. |
| 204, 205 | Actor/opponent count-active bits. Zero padding when column 202 is zero. |
| 217, 218 | Awarded-point counter changes over the observation interval. Availability follows column 203. These are not hit counts or causal contact labels. |

Every unlisted column is structurally unavailable and always zero padding. This
includes roll/pitch angular velocity, support contacts, inferred/physical down
flags, fall-classification state, recovery timers, last-hit proxies, controller
route/busy state, owned-yaw intent, count duration and contact-event counts.
`feature_mask()` produces the same 166-retained-column binary mask for both
producers. Per-sample availability remains separate from this structural mask.

## Adapter and history obligations

- Live roots transform Unity XYZ into common XZY, and Unity XYZW into common
  WXYZ `(-w,x,z,y)`, followed by normalization and canonical quaternion sign.
  The physical producer supplies its root origin and root quaternion in the
  same common frame. Root origin means the exported free-root position, not
  the body center of mass. Geometry remains numerical world coordinates;
  Unity-to-metre calibration is unverified.
- `project_joint()` is shared swing/twist projection using a model rest
  orientation, local measured bone quaternion and corresponding unit hinge
  axis. The physical adapter must supply equivalent local body orientations
  and model conventions. Raw physical hinge coordinates must not be declared
  equal to client-projected joints without establishing that mapping. Unknown
  joint correspondence uses `joint_pose_available=0`. Off-axis residual is
  diagnostic and is never a fall/contact label.
- Both adapters provide current observation time in seconds from one clock
  domain and a round key that changes at genuine round boundaries. The live
  adapter derives elapsed time from QPC ticks/frequency and binds round identity
  and referee receipt to the current process/lifecycle. The physical adapter
  provides observation-step time and its round identity. These clock/key
  bindings are adapter obligations; the header does not infer them.
- Preserve exactly the preceding observation snapshot. A first observation,
  new round, perspective change, nonpositive interval or gap over 250 ms masks
  derivatives and point deltas. Keep the current valid snapshot as the next
  history entry. Explicit observation-stream resets also clear history.
- Do not clear history on a privileged physical fall or same-round body reset,
  and do not substitute collision-sweep/route-entry snapshots. Those events
  produce observed pose changes on the live side. The shared projection must
  expose the same observable difference. This does not make physical teleports
  equal to the client's potentially interpolated display motion.
- Live `referee_available` requires the existing verified, fresh,
  lifecycle-bound received-state contract. Populate the count mask from its
  two received bits. The physical adapter uses native fight `count_active[]`.
  A count receipt describes client-observed state, so matching the encoder does
  not eliminate network/display delay. No server execution, reset completion,
  self-fall cause or opponent action is inferred.

## Verification

`observable_balance_test.cpp` exercises fixed Unity +Y yaw and tilted-root
fixtures, local +X bearing, 4,096 varied native/common versus Unity/common pose
pairs, both fighter perspectives, quaternion sign aliases, yaw/joint branch
crossings, shared hinge-twist projection, missing joints/referee state,
same-round body displacement, history boundaries and invalid inputs. It uses
no GPU, game process or policy.

The CPU regression passes 1,172,058 assertions; the maximum discrepancy between
equivalent Unity-converted and directly supplied common-frame observations is
1.49011612e-08. `observable_balance_cuda_test.cu` also runs 64 synthetic snapshot
pairs on Spark and compares all 14,272 output features against the CPU path.
That comparison passes with maximum absolute discrepancy zero. Neither test
executes physics, a policy or a game process.

Equivalent-input tests establish encoding and topology behavior. They do
not establish authentic REK pose calibration, physics parity, matched action
masks or a deployable policy. The CUDA test uses the same header compiled with
`nvcc -O2 -std=c++17 -arch=sm_121`.

Compile and run with C++17:

```text
g++ -O2 -std=c++17 -Wall -Wextra -Wpedantic observable_balance_test.cpp -o observable_balance_test
./observable_balance_test
```
