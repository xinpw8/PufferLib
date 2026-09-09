# G1 semantic action and Sonic policy adapter

## Scope

The current `rek_g1` binding is a native semantic two-robot environment. The
trainable policy does not command 29 actuators. It selects a held locomotion
state or one of the recovered G1 kick routes. A build-pinned Sonic adapter then
constructs policy observations, executes the supplied encoder and decoder, and
converts the output into joint targets for MuJoCo.

This is an executable public-family candidate. It is not yet a parity-accepted
REK clone. Current-service encoder and decoder byte identity remains unknown.

## High-level action ABI

The Puffer action is one categorical value. Multiple heads are not used because
independent direction, duration, and attack heads would generate combinations
whose legality depends on hidden scheduler state.

The 20 categories are:

| Category | Meaning |
| --- | --- |
| 0 | continue active segment |
| 1 | neutral |
| 2 to 15 | validated W, S, A, D, Q, and E held combinations |
| 16 to 19 | kick routes for runtime moves 6 to 9 |

The held combinations are neutral, W, S, A, D, Q, E, and Q/E paired with one
cardinal translation. Contradictory pairs such as W+S, A+D, and Q+E are
invalid. F is absent because no installed G1 binding for it has been recovered.

Ocean transports the selected category through a binary32 action buffer. The
adapter requires a finite, exact integer in range before it mutates scheduler
state. Invalid values fail the vector.

## Held-input and scheduling behavior

One Puffer transition is one 50 Hz controller action. Every action repeats the
complete desired held state. A bit remaining set means held; clearing it means
released.

W, S, A, and D affect translation immediately. Q and E use the recovered
keyboard yaw ramp. Each controller tick adds `elapsed / ramp_duration`, clamps
the magnitude to one, and applies the requested sign. A sign change clears the
old accumulation before advancing the new sign. Release clears the ramp.

Translation can be held for multiple actions and cannot be reduced to an input
edge. This is required for appreciable movement. Q or E can coexist with a
translation hold. If a kick is accepted, effective yaw becomes zero while the
desired yaw state and ramp continue. Yaw resumes after the kick.

A kick cannot start while translation is active, the locomotion transition is
unsettled, another action is busy, or fall/reset state suspends the runner. A
rejected attack edge is currently dropped. Whether the current REK service
queues such an edge is unknown. A controlled input and acknowledgement capture
must resolve it before parity acceptance.

Locomotion segment length is an explicit training-interface parameter. Kick
segment lengths are configured compositor traversal ticks. The current values
come from the reconstructed compositor state machine and are not measurements
of physical action completion, hit latency, or attack effectiveness.

## Low-level policy adapter

The native adapter owns one mutable runner state per robot and shares immutable
model and motion data. A controller action follows this order:

1. Resolve semantic input and route transitions.
2. Compose the required reference windows.
3. Construct encoder and decoder observations from physical state, reference
   state, prior actions, history, heading, and command filtering.
4. Execute the compatible batch encoder and decoder sessions.
5. Reorder and scale outputs into joint targets.
6. Step both fighters through ten shared 2 ms MuJoCo updates.
7. Update physical, fall, hit, score, referee, reward, terminal, and mask state.

Rows that are dampened or resetting do not advance policy, previous actions,
history, motion cursor, reference frame, heading forgiveness, or command
filter. MuJoCo still advances. A fallen row holds its last joint targets with
the live proportional gain, damping gain, and force limit multiplied by exact
binary32 0.1.

## Motion routes

`native_motion_routes.c` pins RobotConfig path 2722 and the exact static
MocapClipConfig, NPZ identity, direction, mirror flag, frame count, and runtime
move index for idle, forward, backward, left/right strafe, left/right turn, and
four kicks. `g1_semantic_assets.c` loads only the generated binary32 bundle
whose byte sizes and SHA-256 values match its compiled manifest.

Asset frame count is metadata. It does not by itself define action duration.
Runtime composition includes authored frame bounds, playback direction and
speed, looping, blend windows, mirroring, and entry matching. The loader also
applies the exact recovered clip-heading normalization before exposing a clip.

## Timing and reset

Recorded client evidence establishes 500 Unity fixed updates per second and ten
fixed updates for each 50 Hz command action. Contacts and falls are measured on
every 2 ms step.

`ResetBothToSpawn` is two phase:

1. clear fall/contact/reset state and apply both root spawn poses;
2. after one fixed update, restore motor state, zero every joint position and
   velocity, reapply root poses, reset composer and policy state, clear command
   and locomotion state, then publish reset completion. The completed reset
   remains at policy tick zero until the next policy action.

The fight reset grace is 2.0 s. The robot-local fallback reset grace is 0.5 s.
An arena terminal is published for one outer action. Only that arena resets
before its next action.

## Model identity

The binding requires explicit regular files for the encoder and decoder. The
candidate build generator validates their manifest, graph contract, byte size,
and SHA-256 and compiles the accepted identities into the extension. Runtime
rechecks the same identities before opening ONNX Runtime sessions. Symlinks,
missing hashes, shape-only compatibility, and unlisted graphs fail closed.

The installed Steam containers inspected so far do not expose the current
service's model payloads, and no local model cache with authoritative byte
identity was found. I don't know the current-service weights. To obtain that
identity, capture the live runner's resolved model assets or an authenticated
runtime manifest, hash the actual bytes, and verify them against controller
tensors from the same build. Until then, reports must call the supplied models
the validated public GEAR-SONIC family candidate.

## Observation, reward, and terminal contract

Schema 3 exposes 223 binary32 values per robot. It includes both complete robot
states, fall state, relative command and composer state, scores, attribution,
referee state, round state, and per-action events. Paired rows observe the same
arena with self and opponent exchanged.

The reward is the row's score delta minus the opponent score delta. This is a
zero-sum training design, not a recovered REK reward. Both rows terminate when
the recovered round-end event occurs. Terminal observation is the final state
of that episode.

## Acceptance sequence

Implementation correctness and REK parity are separate gates:

1. Strict C tests, sanitizers, static analysis, deterministic reset, row-local
   isolation, and complete ABI tests must pass on x86-64 and Spark aarch64.
2. Every held category and kick route must execute through the native binding.
3. Controlled Windows captures must include matched initial physical state,
   actual accepted inputs, opponent actions, contact/fall/score events, and
   multiple repeats.
4. Those repeats define REK's variance envelope for each trajectory and event.
5. Held-out action sequences must replay through the Spark environment.
6. Parity is accepted only when all specified errors are no greater than the
   corresponding repeated-run variance.

The current single-fight held-input trace lacks several items in step 3. It is
useful for diagnosing coordinate and heading errors, but it cannot accept or
reject parity.
