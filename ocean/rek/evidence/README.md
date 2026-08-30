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
| 2. Determine where practice physics executes | **tooled** | `net_observe.py` |
| 3. Recover the input → controller / network path | needs runtime instrumentation | — |
| 4. Inventory physics, bodies, models, native code | **tooled** (static half) | `static_survey.py` |
| 4b. IL2CPP type and method recovery | needs Il2CppDumper | — |
| 5. Tick-level recorder | blocked on 2, 3, 4 | format in `trace.py` |
| 6. Controlled traces and differential replay | **tooled** | `differ.py` |

Steps 3 and 5 are deliberately not written yet. Where the state lives, and what
the command interface actually is, are the outputs of steps 2 and 4 — writing a
recorder before those land would mean guessing at the very things this package
exists to measure.

### 1. Pin the build

```
python inventory.py --out inventory.json
python inventory.py --verify inventory.json     # non-zero if the build moved
```

Hashes everything, reads the Steam `buildid`, and derives one fingerprint from
the decisive files. Logs and crash dumps are excluded so that ordinary noise
does not invalidate the identity, while a real update does. Run `--verify`
before trusting any earlier trace.

### 2. Where does practice physics run?

The fork that determines everything after it. Reflex Arc states that live
matches use an authoritative dedicated server; that says nothing about practice.

```
python net_observe.py --name REK --seconds 120 --note "practice, solo" --out net_practice.json
python net_observe.py --name REK --seconds 120 --note "live match"     --out net_match.json
```

The contrast between the two runs is the evidence, not either alone. Confirm
with the firewall: block the process and see whether practice still steps.

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
libraries, and whether the build is IL2CPP. Anything not found is listed under
`absent` rather than filled in. Name matches are recorded under `name_hits` and
are explicitly not findings.

If it is IL2CPP — and a Unity Windows build almost certainly is — types,
methods and the network schema need Il2CppDumper against `GameAssembly.dll` and
`global-metadata.dat`. The survey prints both paths and hashes for that step.

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

Do not infer balance from a UI bar when pelvis orientation, support contacts,
joint state and controller outputs can be captured.

### 6. Differential validation

```
python differ.py baseline rek_a.trace rek_b.trace rek_c.trace --out envelope.json
python differ.py compare  rek_a.trace clone.trace --envelope envelope.json
```

`baseline` runs first and measures how far REK differs from itself across
repeats of one experiment. That spread is the acceptance envelope. If REK is
deterministic for a given seed the envelope collapses to zero and the comparison
becomes exact equality — the stronger result, and one the tooling will not
soften. `compare` exits non-zero on any channel or event outside the envelope,
and reports the first divergent tick.

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
