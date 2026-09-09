# Native Puffer vector boundary

`binding.c` registers the native semantic G1 duel candidate. It does not use
the former direct-actuator diagnostic. One process owns the immutable semantic
asset bundle, shared ONNX sessions, native motion state, paired MuJoCo arenas,
combat state, and the Puffer bridge.

## Shape and ownership

One Puffer row is one robot. The row count must be positive, even, and ordered
by arena as player then opponent. One arena owns one MuJoCo model instance with
both robots, which is required for physical contact. A single Puffer buffer is
currently required so both fighters are staged before the same physics step.

The binding exposes:

* observation schema 3, 223 binary32 values per row;
* one 20-category action head;
* one binary32 reward and terminal per row;
* a 20-byte action-validity mask per row.

`MY_VEC_STEP` invokes one complete batch transition. The scalar `c_step` and
`c_reset` entry points fail closed because scalar stepping cannot preserve the
shared-contact barrier. `MY_VEC_STEP_RANGE` accepts only the complete buffer.

## Action contract

Category 0 continues the active scheduled command. Categories 1 through 15
start neutral or a validated held W, S, A, D, Q, and E combination. Categories
16 through 19 start runtime moves 6 through 9. The four kick durations supplied
to the constructor are configured compositor traversal lengths. They must be
positive and must match the route assets. They are not measurements of physical
completion or attack effectiveness.

The scheduler retains desired held input across continuation ticks. While a
kick is active, category 0 retains desired yaw and the neutral, Q, and E
categories update it without restarting the kick. Effective yaw remains zero
until the kick completes; translation and kick-start categories stay masked.
Retaining and advancing the ramp is provisional candidate behavior, not a
recovered current REK parity fact. Adapter-table kick templates are normalized
to neutral and inherit current desired Q/E at dispatch; direct semantic
commands remain permitted to carry yaw.

The idle action mask rejects starts that violate translation-settle, busy, or
kick-preemption rules. Invalid categorical floats, partial callbacks, invalid
next facts, and mask-generation failures poison the vector. Batch callbacks
declare both the runtime-facts ABI version and `sizeof(RekG1RuntimeFacts)`; a
stale callback is rejected before invocation. Callback rewards must be finite
and terminals must be exactly 0 or 1 before terminal-driven scheduler resets
are accepted. Only a complete batch reset can recover a poisoned vector.

The Python `VecEnv` binding exposes the host mask through the read-only
`action_mask_ptr` and `action_mask_size` properties. The CUDA binding also
exposes `gpu_action_mask_ptr`. `action_mask_size` is the per-row byte stride;
the host allocation contains `total_agents * action_mask_size` bytes and stays
valid only until `VecEnv.close()`. These are views of the mask already owned and
updated by `StaticVec`; the binding does not allocate a second policy mask.

## Native transition

Every outer transition performs the following sequence:

1. Decode all Puffer actions and stage both fighters' semantic commands.
2. Compose route references and run the fixed-batch policy graph. Suspended rows
   are included in the graph invocation because the verified ONNX artifacts have
   exact static batch dimensions, but their outputs are discarded and their
   policy history is restored byte-for-byte.
3. Advance ten 2 ms MuJoCo fixed updates.
4. After every shared-contact step, measure contacts and fall state, filter hit
   candidates, update attribution and scoring, and advance the referee.
5. Publish paired observations, zero-sum score-delta training rewards, terminal
   events, and the next action masks.

Contact attribution for a fixed update uses the fall phase that existed before
new fall events from that same update are committed. Attribution-only contacts
can update last-struck state outside an active round, but cannot award points.
Only the affected arena is reset after a terminal event.

## Fall and reset execution

The selected serialized G1 prefab has no prone or supine recovery clips, but
runtime clip injection has not been excluded. Current REK `CanGetUp` therefore
remains unknown. The candidate provisionally selects `CanGetUp=false` and the
3.0 s no-recovery referee branch. Under that candidate selection, a committed
fall suspends policy and compositor progress while holding the last joint
targets under exact binary32 0.1 retained gain, damping, and effort limits.
MuJoCo physics continues, while the suspended native 50 Hz policy-evaluation
counter remains frozen. The diagnostic counter is not the installed runner's
2 ms `stepCounter`.

The recovered `ResetBothToSpawn` boundary is two phase. The roots are restored
at the reset request. After one 2 ms fixed update, joint positions and velocities
are reset, row-local policy history and prior actions are cleared, composers and
command filters return to idle, and fall suppression continues for the remaining
2.0 s grace period. A reset completing inside an outer transition leaves its
policy and motion counters at zero; only the next genuine policy transition
advances them. Whole-vector initialization uses a deterministic immediate reset
before any transition is observed.

## Required process inputs

The binding requires these environment paths and has no working-directory
fallback:

* `REK_G1_SEMANTIC_ASSETS_DIR`
* `REK_G1_ENCODER_ONNX`
* `REK_G1_DECODER_ONNX`

The semantic bundle loader checks exact sizes and SHA-256 identities compiled
from its generated manifest. The encoder and decoder must also match the
compiled candidate-family identity. Regular-file, no-symlink, byte-length, and
digest checks run before ONNX Runtime opens either graph. Model payloads are not
committed to the repository.

## Authority and remaining gates

The exact current REK service policy weights remain unknown. A graph that has
compatible tensor shapes is not enough to identify it. This runtime is a
public-family candidate until current-build capture proves a stronger identity.

The remaining acceptance work is empirical:

1. collect repeated controlled REK trajectories for every held movement and
   kick, including matched initial state and opponent timeline;
2. preserve action acceptance, contact, fall, score, and reset event timing;
3. construct the repeated-run variance envelope;
4. replay held-out sequences through this binding;
5. accept parity only if every specified trajectory and event error is within
   that envelope.

The present smoke and held-trace replay verify native execution, deterministic
reset, ABI integrity, paired views, action coverage, and diagnostic throughput.
They do not establish REK parity.
