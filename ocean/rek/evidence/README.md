# REK evidence package

Tooling to reverse-engineer one exact REK build into a behavioural clone, and to
prove — against REK, not against our own expectations — whether a candidate
clone actually reproduces it.

## Acceptance criterion

A clone is accepted when held-out action sequences replay with trajectory and
event errors no greater than REK's own repeated-run variance. Nothing weaker
counts, and nothing here can substitute for it: a test suite written against
self-authored rules only confirms that the code does what its author expected,
which is exactly how a plausible but wrong model passes everything.

Throughput work begins after that test passes, not before.

## Status: Windows prerequisites collected; replay gate incomplete

Evidence has now been collected from the pinned Windows installation. The
package contains a build inventory, an inventory verification result, a static
survey, IL2CPP probe and recovery artifacts, network reconnaissance, and a
controlled private-AI authority experiment whose measured verdict is
`remote_authority`. A passive client-fixed recorder has also produced runtime
command and client-visible state captures from private AI rounds.

The pinned client now also has a bounded native controller-path artifact. It
records exact GameAssembly method extents, audited client-side controller
formulas, serialized T800 move pointers, and the native force-limit construction.
It explicitly leaves private-AI runtime activation and server equivalence
unknown. The private-AI client is visual-only, so those static client methods
cannot be promoted to the authoritative transition function without further
evidence.

Eight full `Sparring Bot 1` captures have been imported with the v2 command
identity. They contain 7,135 server-snapshot callback events, 7,127 unique
snapshot ticks, and 7,119 one-step transitions across two endpoint/session
groups. `snapshot_transition_baseline.py` evaluates those groups with whole
groups held out. This is a diagnostic dataset, not a canonical replay artifact.

A separate replay-only staging component has derived one bounded schedule from
concrete transport invocations in a finalized `Sparring Bot 1` capture. The
schedule contains 400 observed velocity boundaries and six observed move calls.
All 11,119 source sample records agree on local slot 0, opponent slot 1,
`Sparring Bot 1`, and AI difficulty 0; those measured conditions are bound into
the manifest. The component is not installed or armed, and its offline build
and tests do not count as a controlled trace.

That is prerequisite evidence, not clone validation. The existing canonical
`.trace` files and `envelope.json` predate
`rek.client_fixed.command_schedule.v2`; they do not bind repeats to the same
measured command schedule and are therefore rejected. No canonical parity
report exists. New command-identified repeats and held-out action sequences are
still required.

The tools must be run against the real Windows installation. A package of
scripts is not an evidence package, and that remains checkable rather than a
matter of opinion:

```
python check_artifacts.py --dir evidence_out/
```

It reports which artifacts exist, whether each is well formed, and whether they
all describe the same build — traces from one client build and a survey from
another cannot be reasoned about together, and nothing else in the pipeline
notices. It exits non-zero until the package is complete. On the current
artifacts it prints `stage: statics surveyed` because the legacy traces and
envelope fail the command-identity gate.

## Rules of evidence

1. Every implemented transition rule cites direct static evidence, runtime
   evidence, or a controlled REK experiment.
2. Field names, animation durations, UI readouts and manufacturer datasheets are
   reconnaissance. They never establish a mechanism.
3. Unknowns stay unknown. No tool here emits a default, a fallback or an
   inferred value in place of a measurement.
4. Every trace and every recovered constant names the build fingerprint it came
   from.

## Sequence

| Step | State | Tool |
|---|---|---|
| 1. Pin and inventory one exact build | **collected** | `inventory.py` |
| 2. Determine where private-AI physics executes | **collected: `remote_authority`** | `authority_test.py` (`net_observe.py` is recon only) |
| 3. Recover the input → controller / network path | **partial: exact client request projections and native client semantics recovered; server acceptance and authoritative activation unknown** | passive recorder, `client_fixed_import.py`, `controller_path.py` |
| 4. Inventory physics, bodies, models, native code | **collected** (static half) | `static_survey.py` |
| 4b. IL2CPP names and inference runtime | **collected** | `il2cpp_probe.py` |
| 4c. IL2CPP type and method recovery | artifact present; exact native controller extents pinned; runtime validation incomplete | Il2CppDumper output, `controller_path.py` |
| 5. Tick-level recorder | **partial: client timing plus raw replicated pose/fight packets; no server tick** | passive recorder, `raw_bone_validate.py`, `client_fixed_import.py`, format in `trace.py` |
| 6. Controlled traces and differential replay | action-rich traces and an offline replay schedule are present; runtime repeats and held-out clone replay are missing | `differ.py`, `calibrate_envelope.py`, `measured_parity.py` |
| Completeness gate over all of the above | **tooled** | `check_artifacts.py` |

Steps 3 and 5 are only partial. The passive recorder projects exact packed
`REK_Input` and `REK_Move` request bodies from the send-method arguments and
copies exact `REK_Bones`, `REK_FightState`, `REK_Score`, and `REK_Hit` bodies
before the client consumes them. It does not control the keyboard or mouse.
The request prefixes prove method invocation and projected bytes, not send
completion, delivery, acceptance, or execution. The receive packets expose no
server tick, command acknowledgement, active move/profile identity, joint
velocity, torque, policy observation/output, or hidden controller state. Those
unknowns remain absent rather than being filled with inferred values.

### Running the collection steps

Steps 1, 4 and 4b are non-interactive and are driven together, so they cannot be
run out of order or against different builds:

```
python collect.py --out evidence_out
```

It pins the build, probes the IL2CPP and native binaries, surveys the Unity
assets, then reports the package state and what is still needed. Each step
records whether it succeeded: a run without UnityPy still pins the build and
probes the binaries, and says which step could not run and why, rather than
producing a directory that looks like a build with nothing in it.

The authority test is not in there and cannot be — it needs someone playing the
game and marking what they see while the network is cut.

### 1. Pin the build

```
python inventory.py --out inventory.json
python inventory.py --verify inventory.json     # non-zero if the build moved
```

Hashes every file and derives three Merkle roots: `manifest` over everything,
`immutable` over every shipped file — this is the identity traces cite — and
`behavioural` over the subset most likely to matter, for triage only. The
identity deliberately does not depend on a hand-picked category list, so a
change cannot hide in a bucket nobody thought to enumerate: an Addressables
bundle, a controller weight file, a Burst library, a physics plugin. Volatile
files are recorded but excluded from the identity. Run `--verify` before
trusting any earlier trace.

This pins the **client**. If the simulation turns out to be remote, the server
version is not pinned by anything here, and traces must carry endpoint, session,
protocol and any server-reported version — `trace.py` refuses a
server-authoritative trace without them.

### 2. Where does practice physics run?

The fork that determines everything after it. Reflex Arc states that live
matches use an authoritative dedicated server; that says nothing about practice.

**Socket enumeration cannot answer this.** Practice could hold authentication,
telemetry, leaderboard or presence sockets while simulating locally; it could be
server-driven over a mostly idle socket; it could run local prediction corrected
remotely; or it could contact the server only at reset and result submission.
Socket presence distinguishes none of those, so `net_observe.py` is
reconnaissance only.

The decisive experiment is intervention:

```
python authority_test.py --name REK --out authority_practice.json
```

Load into practice, type `block`, keep issuing inputs for at least a minute,
attempt a fall, a recovery, a score and an arena reset, and mark what you see
from a fixed vocabulary. The tool never touches your firewall — it prompts, or
runs commands you supply — and it timestamps everything.

| Observed while blocked | Verdict |
|---|---|
| state keeps evolving, reset and scoring complete | `local_authority` |
| state keeps evolving, no interaction confirmed | `local_authority_weak` |
| world freezes, or inputs stop taking effect | `remote_authority` |
| state continues, then visibly corrects | `local_prediction_remote_correction` |
| the game kept talking to the network | `inconclusive` — the block failed |

That last row matters most: the verdict is withheld unless the tool can show, on
the game's own sockets, that the block actually applied. A failed intervention
read as evidence is worse than no evidence. Repeat with added latency and packet
loss where the first run is ambiguous.

- **Locally simulated** → instrument the local simulation directly.
- **Server-owned** → the recorder sits above the transport, capturing commands
  sent and state replicated back, and every field below that the client never
  receives is simply unavailable.

### 4. What is in the build

```
python static_survey.py --inventory inventory.json --out static_survey.json
```

Reports TimeManager (the actual tick rate), PhysicsManager, every
`ArticulationBody` / `Rigidbody` / joint component with its drives and limits,
collision geometry, shipped inference models, native plugins and Burst
libraries, the rig the characters are skinned to, the animation clips, and
whether the build is IL2CPP. Anything not found is listed under `absent`.

Every record carries a **role**: `authoritative`, `candidate_lead`,
`client_render_only`, `unknown_role`, `absent`. Nothing static is ever marked
authoritative — presence in the build is not participation in the transition
function, and only a runtime trace or controlled experiment can promote a
record. Serialized components are `unknown_role`; name matches and clips are
`candidate_lead`.

**On animation clips.** They are catalogued, with durations and events, because
a physics-based controller can still be driven by reference motions, phase
signals or skill latents drawn from a motion library — the two claims are not in
conflict. What must not recur is a clip duration becoming a startup, active or
recovery window. That inference produced the discarded model. How clips are
consumed is established by tracing the code that reads them, not by reading the
clips.

The survey also emits `not_recoverable_statically`, naming what this step
cannot settle: the physics scene practice actually runs, the controller
observation vector, how outputs become joint targets, recurrent state and skill
phase, contact-to-score logic, input buffering, network schemas, execution order
within a tick, and every server-side parameter.

If it is IL2CPP — and a Unity Windows build almost certainly is — types,
methods and the network schema need Il2CppDumper against `GameAssembly.dll` and
`global-metadata.dat`. The survey prints both paths and hashes for that step.

Part of it does not need a decompiler:

```
python il2cpp_probe.py --inventory inventory.json --out il2cpp_probe.json
```

It verifies `global-metadata.dat` really is IL2CPP metadata and reports its
version, so the right dumper can be chosen — and reads nothing past the header,
because everything past it moves between versions. Then it extracts
identifier-shaped strings from the metadata and every native binary and sorts
them into buckets: controller, physics, netcode, match rules, input, animation,
inference runtime. This is `strings` with domain classification. A name in a
binary proves a name is in a binary; every one is a `candidate_lead` and a
target for instrumentation, which is exactly what step 3 needs in order to have
somewhere to attach.

Finding an inference runtime establishes only that the executable contains an
inference path. It does not establish that a model is present or active. In this
build, ONNX Runtime and the complete `EngineAIPolicyRunner` call path are present,
but all 45 T800 ONNX/config/trajectory pointers are null. Runtime logs show the
runner aborting initialization at the first missing profile before the client
switches both fighters to network-driven visual-only mode.

### 4c. Bounded native controller semantics

`controller_path.py` turns the completed native audit into one reproducible,
build-pinned JSON artifact. It requires every source explicitly and refuses any
hash, build, RVA extent, ISIL method block, serialized object, move pointer, or
force-limit input that differs from the audited Windows build:

```
python controller_path.py --inventory evidence_out/inventory.json --recovery evidence_out/il2cpp_recovery.json --probe evidence_out/mujoco_asset_probe_v8.json --game-assembly "C:/Program Files (x86)/Steam/steamapps/common/REK Alpha Test/GameAssembly.dll" --global-metadata "C:/Program Files (x86)/Steam/steamapps/common/REK Alpha Test/REK_Data/il2cpp_data/Metadata/global-metadata.dat" --isil-dir C:/rekagent/work/controller-audit-isil/IsilDump/REKApp/REKApp --out evidence_out/controller_path.json
```

Re-run the same inputs with `--verify evidence_out/controller_path.json` to
require byte-for-byte reproduction. The artifact cites exact native bytes and
normalized ISIL blocks for every formula. Cpp2IL dummy assemblies contribute
names, signatures and Unity type trees only. Their restored method bodies are
never consumed as semantic evidence.

The recovered formulas describe output clipping, default-pose and residual
joint targets, transition interpolation, per-joint and aggregate torque limits,
walk target clamps, MuJoCo implicit-PD parameter writes, and runtime force-limit
selection. The serialized T800 table preserves all twelve RobotConfig move
pointers, including the exact six null and six non-null path IDs. The v8 probe
also binds all six referenced `MocapClipConfig` objects and preserves their
object hashes, names, profile strings, impact windows, blending, reversal
settings, and null `npzFile` pointers. These configuration fields do not contain
the move trajectories. The table also preserves that all 45 runner profiles
have null serialized policy/config/trajectory pointers. No replacement values
are inferred.

This artifact remains static client evidence. It does not establish which
runner the dedicated server executes, the server build, active profile assets,
observation construction, model weights or hidden state, authoritative motor
cache values, server-tick command alignment, or move-profile execution. Each of
those is emitted as a hard unknown with the measurement needed to resolve it.
`controller_path.py` is therefore separate from `collect.py`, whose inputs do
not include the independently generated native analysis.

`t800_runtime_boundary.py` combines that static artifact with the allowlisted
Windows `Player.log` lifecycle and the remote-authority artifact. It verifies
that each accepted missing-profile message has `LoadProfile` and `Init` stack
frames, then counts the matching T800 visual-only and network-client
transitions. The generated `t800_client_runtime_boundary.json` establishes that
the observed client aborted local runner initialization and rendered network
poses. It leaves the authoritative controller payload location unknown.

```
python t800_runtime_boundary.py --player-log C:/Users/Daniel/AppData/LocalLow/REK/REK/Player.log --controller-path evidence_out/controller_path.json --authority evidence_out/authority_private_ai.json --out evidence_out/t800_client_runtime_boundary.json
```

### 5. Recorder

A passive Windows recorder and `client_fixed_import.py` now populate the
`trace.py` format with client-fixed state and transport-call evidence. The
importer uses `rek.client_fixed.command_schedule.v2`, which hashes measured
velocity commands and discrete move, special, and emergency-stop invocation
timing. Missing discrete action identity fails closed. Older imported traces do
not satisfy this schema and cannot establish a repeat envelope.

The staged v0.6.1 recorder emits `rek.private_ai.protocol.v6`. It fails closed
before opening a capture unless the pinned client is running at a measured
`0.002 s` fixed step in an active, unranked solo round; the current multiplayer
session reports `IsPrivate`; the opponent is exactly `Sparring Bot 1` at client
AI difficulty 0 with no human or client in that slot; and both visual-only
fighters are exactly T800. Local identity requires the case-sensitive
`fighterIdentities[localSlot].RobotID == "t800"` result and the exact ordered
26-bone `LINK_*` runtime signature. Opponent identity requires the same exact
runtime signature; a missing or stale opponent semantic ID is retained as an
explicit mismatch instead of replacing the measured runtime identity. A
non-T800 opponent runtime still fails closed. The recorder also rejects an
absent, inactive, or zero-extent `Camera.main`. No finalized v6 capture is
claimed here yet.

Protocol v6 adds one `root_pose_sample` for every captured client
`FixedUpdate`, declared and validated as a contiguous 500 Hz stream. Each
sample preserves both fighters' measured world root position and rotation,
their `Camera.WorldToScreenPoint` coordinates and visibility flags, plus the
selected camera's transform, view and projection matrices, viewport and pixel
geometry, render-target state, clip planes, display, and render scale. Matrix
order and Unity's bottom-left screen-coordinate convention are explicit. These
are root and camera measurements only, not inferred joint state, image-derived
tracking, contacts, velocities, or server state.

Capture bounds, compact samples, root samples, and outbound request edges carry
both UTC and `System.Diagnostics.Stopwatch.GetTimestamp()` values. On
high-resolution Windows systems the latter is QueryPerformanceCounter-backed;
the capture records its frequency. The v6 validator requires explicit UTC,
bounded monotonic Stopwatch values, a root sample for every fixed tick, the
measured `0.002 s` fixed-time cadence, and an unchanged camera instance and
render geometry. Outbound timestamps identify the client-observed `Send*`
method edge. They do not establish network send completion or a server-side
execution time.

The staged recorder retains the protocol-boundary evidence introduced by v0.5.1.
Prefixes copy `FastBufferReader` bytes without advancing the reader. For each
`REK_Bones` packet it preserves the two-byte header and every transmitted world
position and rotation; a postfix binds that body to the decoded snapshot ring
entry. The validator recognizes only the exact ordered `t800_26` and `g1_30`
layouts, whose bodies are 730 and 842 bytes respectively. The mapping is
measured from the scoped runtime objects:
`engineai_t800_FactoryPolicy(Clone)` carries the 26-name `LINK_*` sequence, and
`g1_29dof_Prefab_SONIC(Clone)` carries the 30-name `pelvis` through
`right_wrist_yaw_link` sequence. The v6 header correctly declares T800 as 26
bones and 730 bytes and pins the ordered-name signature. Historical v0.5.1
headers declared `t800_bone_count=30` and `t800_body_bytes=842`; those are known
format mislabels. Backward-compatible validation checks them only as pinned v5
literals and never uses them for fighter classification. The native sender sets
an intended interval of `0.02 s`, equivalent to 50 Hz, using unreliable
delivery.

Both protocol schemas preserve raw `REK_FightState` (33 bytes, reliable,
nominal 10 Hz), `REK_Score` (7 bytes, reliable), and `REK_Hit` (29 bytes,
unreliable) packets. A FightState postfix correlates each copied body with the
applied client state. Score and referee fields are authoritative event labels,
but the Hit packet is effects telemetry and has no fighter identity. Clearing a
referee count is not uniquely a successful get-up event.

On the outbound side, prefixes project the exact 13-byte `REK_Input` and
two-byte `REK_Move` bodies, including float32 bit patterns and byte truncation.
The actual boundary is the individual send methods called from `LateUpdate`;
the unused `ClientSendFrame` helper is not hooked or treated as a cadence.
Every outbound record is explicitly request-only with null server tick and
acceptance fields and no observed acknowledgement.

All protocol and pose timestamps remain client observation times. None of the
recovered packet bodies contains a server tick, server send timestamp, command
sequence, move identity, acceptance result, active policy state, joint velocity,
torque, controller observation or output, model weights, or hidden state.

Validate any finalized v6 or historical v5 JSONL capture before an importer or
model consumes it:

```
python raw_bone_validate.py --raw C:/rekagent/evidence/runtime/rek-private-ai-protocol-v6/<capture>.jsonl --out evidence_out/<capture>.protocol-validation.json
python raw_bone_validate.py --raw C:/rekagent/evidence/runtime/rek-private-ai-protocol-v5/<capture>.jsonl --out evidence_out/<capture>.protocol-validation.json
```

The validator pins each schema to its exact recorder version and DLL hash. It
requires each redundant JSON number to round-trip to the exact IEEE-754
binary32 bits in the base64 body, including preserving a `-0` token's
negative-zero sign. It checks body hashes and exact packet lengths, verifies all
per-channel sequences, requires one-to-one FightState and bone postfix
correlation, verifies request-only semantics, and rejects raw arena/session
identifiers. Only an irreversible SHA-256 session identity is retained for
repeat grouping. Three complete private Bot 1 captures from recorder v0.5.1
remain validated historical evidence, including a mixed `t800_26` versus
`g1_30` capture. They do not satisfy the stricter v6 private T800-vs-T800 and
500 Hz root-stream claims.

Neither recorder schema is complete enough for clone acceptance. Both observe
the client boundary of a server-authoritative mode, not server-only physics or
controller state, and neither can make an incomplete channel set equivalent to
the full transition state.

`snapshot_transition_baseline.py` is a measured-input system-identification
diagnostic over an exact allowlist of replicated root position, linear velocity,
and angular velocity channels. Its inputs are only observed transport-call
payloads and counts. It never treats controller status flags as actions. It
coalesces duplicate callbacks at the same client tick while preserving and
hashing their exact multiplicity, rejects decreasing ticks and callback sequence
gaps, and evaluates with complete endpoint/session groups held out.

On the current eight full traces, adding the nine measured input features did
not improve held-out one-step error. At lag zero, pooled RMSE changed from
`0.0213202742` for state-only autoregression to `0.0213991724` with inputs, a
`0.3701%` degradation. Inputs lost on all eight held-out traces at lags zero,
one, and two; the negative-lag placebo also degraded. Windows and DGX Spark
reports matched in structure and all nonnumeric values, and all 3,042 numeric
fields agreed within relative tolerance `1e-10` and absolute tolerance `1e-12`.
This negative result does not establish action alignment, a simulator, or
parity, and the generated report marks all three claims false.

The format does not hardcode a channel list: the recorder declares what it
actually captured, and a channel that was not captured is absent rather than
present and zero — which is what stops a reduced-order model from passing by
staying silent about state it never had. Suggested naming, so both sides agree:

```
cmd.<pilot>.<field>          raw and processed input, skill/mode selection,
                             guard, button edge vs held, sequence number
tick.client, tick.server     and command acknowledgement, if networked
root.<pilot>.pos.{x,y,z}     root pose
root.<pilot>.quat.{x,y,z,w}
root.<pilot>.vel.*           linear and angular velocity
root.<pilot>.angvel.*
joint.<pilot>.<name>.pos     joint state and actuator commands
joint.<pilot>.<name>.vel
joint.<pilot>.<name>.target
ctrl.<pilot>.obs[i]          controller observations, outputs, recurrent state
ctrl.<pilot>.act[i]
ctrl.<pilot>.hidden[i]
contact.<pilot>.<body>.*     contact points, normals, impulses, foot/body flags
round.timer, round.state     round and scoring state
score.<pilot>, downs.<pilot>
rng.state
```

Events carry the discrete occurrences: `hit`, `score`, `fall`, `getup`,
`round_start`, `round_end`, `ko`.

**Every REK channel must cite where it came from.** The writer refuses one that
does not, with a citation naming a `class`, `method`, `serialized_field`,
`transport_message`, `runtime_address` or `controlled_experiment`:

```python
TraceWriter(path, ['root.0.pos.x'], fingerprint, 'rek', provenance={
    'root.0.pos.x': {'kind': 'serialized_field',
                     'ref': 'ArticulationBody.m_AnchorPosition.x'}})
```

The recorder is written only after the static survey and the control-path trace
identify the real fields, and this is what stops a channel being invented in
between: once it is in the file, a guessed channel is indistinguishable from a
measured one. `check_artifacts.py` rejects a REK trace with uncited channels,
and `differ.py compare` prints the citation for any channel that fails, so a
disagreement points at the thing it was read from. Clone traces need no
citations — a clone's channels come from its own source.

Do not infer balance from a UI bar when pelvis orientation, support contacts,
joint state and controller outputs can be captured.

### 6. Differential validation

```
python differ.py baseline rek_a.trace rek_b.trace rek_c.trace --out envelope.json
python differ.py compare  rek_a.trace clone.trace --envelope envelope.json
python differ.py compare  rek_a.trace clone.trace --envelope envelope.json \
                          --mode short-horizon --window 30
python differ.py distributional --rek rek_*.trace --clone clone_*.trace
```

`baseline` runs first and measures how far REK differs from itself across
repeats of one experiment. That spread is the acceptance envelope. If REK is
deterministic for a given seed the envelope collapses to zero and the comparison
becomes exact equality — the stronger result, and one the tooling will not
soften.

Three properties that decide whether the gate is worth anything:

- **The envelope is a quantile, not a maximum.** Default acceptance is p99, with
  the max reported alongside; a channel whose max exceeds its p99 more than
  tenfold is called out, because one anomalous REK run would otherwise buy the
  clone that much slack. `--accept-at max` exists and is the permissive setting.
- **Events are scored by precision and recall** against a matching window, not
  by zipping two lists. Zipping turns one dropped hit into a timing error on
  every hit after it.
- **Open-loop comparison has a horizon.** Contact-rich humanoid dynamics amplify
  any difference, so past the first divergence agreement is luck and
  disagreement is the same error re-counted. The report gives
  `first_divergent_tick` and `valid_horizon_ticks`, and says so when most of a
  run is past that point. Validate beyond it with `--mode short-horizon`
  transition tests from injected states, or with repeated closed-loop
  experiments compared distributionally.

**Distributional mode** is the other way past the horizon, and the one that
survives chaos outright. Instead of asking whether two trajectories coincide, it
asks whether many independent episodes produce the same distribution of
outcomes: hit counts, event timings, episode lengths, terminal states, per-
channel summaries. Each statistic is compared with a two-sample KS test, and the
threshold is REK's own split-half disagreement on that same statistic — so once
again the oracle sets the bar rather than a number anyone picked.

It refuses to run on fewer than four REK episodes, and says plainly that below
about twenty a side a pass means "not yet contradicted" rather than parity. A
clone that never emits an event at all fails rather than passing by having no
distribution to disagree with.

Hidden controller state must be captured or reconstructed: identical visible
poses evolve differently when recurrent state or skill phase differs. `compare`
names any `ctrl.*.hidden` channel REK recorded that the clone did not.

Use at least three REK runs. With two there is one difference per tick and the
quantiles are indistinguishable from the max; `baseline` says so.

All repeats must also use the same measured fixed-substep sample phase relative
to the command schedule. A trace sampled before a 50 Hz command boundary is not
interchangeable with one sampled after it. The importer records
`command_sample_phase_substeps`, and both `check_artifacts.py` and `differ.py`
reject mixed phases. Cropping, relabeling, interpolation, and last-value holding
do not convert one phase into another because they would substitute unmeasured
state for a missing sample.

### Experiment matrix

Each experiment is run repeatedly from a repeatable reset, several times in REK
to establish the envelope and once in the clone to test it: neutral input; each
locomotion direction; each attack or skill command; guard initiation and
release; every command from stationary and moving states; every pairwise
simultaneous command; several distances and relative angles; arena-boundary
interactions; L100 vs L100, L100 vs H100, H100 vs H100; interrupted attacks;
contact during recovery; falls and get-up; and identical commands repeated
across many resets.

## Checks

```
python test_evidence.py
```

Covers fingerprint stability and sensitivity, trace round-tripping, and that the
differ actually fails a clone outside REK's own spread — including the case
where the clone simply does not record state that REK has. It says nothing about
whether any clone is faithful. Only REK can answer that.
