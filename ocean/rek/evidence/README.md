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

## Status: no evidence has been collected yet

This directory contains instrumentation, not results. None of the following
exist, and until they do every substantive question about REK is unanswered:

```
inventory.json                    static_survey.json
inventory verification result     model/controller inventory
practice-mode socket trace        live-mode socket trace
network-interruption result       runtime command trace
runtime state trace
```

The tools have to be run against the real Windows installation and their raw
outputs committed. A package of scripts is not an evidence package, and that is
checkable rather than a matter of opinion:

```
python check_artifacts.py --dir evidence_out/
```

It reports which artifacts exist, whether each is well formed, and whether they
all describe the same build — traces from one client build and a survey from
another cannot be reasoned about together, and nothing else in the pipeline
notices. It exits non-zero until the package is complete, and today it prints
`stage: no evidence`.

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
| 1. Pin and inventory one exact build | **tooled** | `inventory.py` |
| 2. Determine where practice physics executes | **tooled** | `authority_test.py` (`net_observe.py` is recon only) |
| 3. Recover the input → controller / network path | needs runtime instrumentation | — |
| 4. Inventory physics, bodies, models, native code | **tooled** (static half) | `static_survey.py` |
| 4b. IL2CPP names and inference runtime | **tooled** | `il2cpp_probe.py` |
| 4c. IL2CPP type and method recovery | needs Il2CppDumper | — |
| 5. Tick-level recorder | blocked on 2, 3, 4 | format in `trace.py` |
| 6. Controlled traces and differential replay | **tooled** | `differ.py` |
| Completeness gate over all of the above | **tooled** | `check_artifacts.py` |

Steps 3 and 5 are deliberately not written yet. Where the state lives, and what
the command interface actually is, are the outputs of steps 2 and 4 — writing a
recorder before those land would mean guessing at the very things this package
exists to measure.

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

The one near-decisive result it can produce is whether an inference runtime is
linked at all. `Unity.Sentis`, `Barracuda` or `onnxruntime` symbols mean a neural
controller runs in the client and its weights and tensor shapes are recoverable.
Their absence across every native binary is evidence against, though not proof —
it could be statically inlined, name-mangled, packed, or server-side only.

### 5. Recorder

Not written. `trace.py` defines the format it must write, so the recorder and
the clone emit the same thing and are directly comparable.

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
