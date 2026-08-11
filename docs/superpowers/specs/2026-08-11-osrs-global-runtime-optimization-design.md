# OSRS Global Runtime Optimization Design

Date: 2026-08-11

## Goal

Make every OSRS encounter in the current integration branch faster and simpler without changing config files, observation layouts, action layouts, combat rules, rewards, or terminal rules. Equal-cost routes may choose different legal tiles. Goldens may change only when an intentional route tie or RNG-order change explains the diff.

Finish the current Inferno route pass first. Optimize the shared SDK and the integrated Inferno, Colosseum, Zulrah, and NH PvP encounters next. Merge Muspah last and adapt both Muspah wrappers to the final shared APIs.

## Current evidence

Fresh one-million-step single-thread profiles on the M4 Pro give these baselines:

| Encounter | Steps per second | Dominant costs |
|---|---:|---|
| Inferno | 756,048 | Attack-route field 25%, observation write 15%, mask write 11% |
| Colosseum | 444,261 | NPC pathfinding 33%, observation write 26%, mask write 10% |
| Zulrah | 344,646 | Pathfinding 62% combined, observation write 19% |
| NH PvP | 531,411 | Pathfinding 45%, loadout resolution 21%, masks 20% |

The completed Inferno route work raised paired mean throughput from 447,050 to 610,390 SPS in the first pass, then from 677,657 to 752,251 SPS in the second pass. The combined estimate is about 68%. Shared route construction remains the largest proven source of gain.

## Scope

### In scope

- Shared collision, LOS, pathfinding, interaction, inventory, observation, and mask primitives under `ocean/osrs/`
- Integrated Inferno and Colosseum native wrappers
- Integrated Zulrah and NH PvP encounter paths
- Encounter-local work justified by a fresh profile after the shared cutover
- Muspah and Zulrah native wrapper integration after the shared substrate is stable
- Focused regression tests, mechanics goldens, builds, viewer smoke checks, and paired benchmarks

### Out of scope

- Config changes
- Reward tuning
- Curriculum tuning
- Policy hyperparameter tuning
- Puffer trainer internals
- A unified ECS or SoA rewrite
- Observation or action width changes
- Speculative caches without measured reuse

## Architecture

Encounter code keeps policy. The SDK owns execution mechanisms.

```text
Encounter policy
  waves, spawns, NPC decisions, rewards
        |
        v
Shared tick mechanisms
  player actions, interactions, combat, inventory
        |
        +--> immutable arena topology
        |      step masks, footprint masks, LOS bitsets
        |      thread-local route scratch
        |
        +--> canonical item metadata
               click semantics, gear slot, observation code
               direct byte-mask output
```

The cutover uses one shared mechanism for each concern. Old mechanisms are deleted after every caller migrates.

## Arena topology

Add one process-shared immutable arena topology per encounter. Build it during encounter setup before worker threads start. Each encounter context holds a `const` pointer to that topology. Do not hide it behind lazy global lookup.

The topology contains:

- Arena origin, width, and height
- Static blocked tiles
- Static footprint-blocked masks for supported NPC sizes
- One eight-bit legal-step mask per tile
- Tile-pair LOS bitsets
- A topology revision used by derived route caches

A maximum dimension of 64 covers NH PvP's 61 by 28 arena and every current encounter. Invalid dimensions abort during setup.

Colosseum's private static blocked, footprint, and LOS tables become the reference implementation, then move into the SDK. Inferno, Zulrah, and NH PvP use the same representation. The encounter supplies static geometry. The SDK builds and queries the tables.

Static topology uses one process-wide allocation per encounter and becomes immutable after setup. Per-env dynamic occupancy remains in encounter state.

## Route solver

Replace `pathfind_step` and `pathfind_step_arena` with one bounded route solver.

The solver uses one thread-local scratch workspace per worker:

- `uint16_t` generation stamps
- Packed `uint16_t` tile queue entries
- One-byte route directions
- Route depth required by fallback and shortest-path checks

The solver never allocates during a tick. It never clears a 104 by 104 stack matrix. Generation wrap clears the stamp array once and resumes at generation one.

The topology's legal-step mask replaces repeated collision-map traversal for static geometry. Dynamic blockers remain an explicit callback. Diagonal movement checks both cardinal edges once. The current duplicate diagonal collision calls are removed.

The result is a tagged outcome:

- Reached source
- Route found
- Fallback selected
- Unreachable

A successful result carries the selected destination, first step, and optional run step. This prevents contradictory `found`, destination, and delta combinations.

One route solve may provide both walking steps when the dynamic blocker revision stays unchanged. Otherwise the second step runs a new query. Equal-cost route tie changes are allowed. Every returned step must remain legal and shortest for the selected destination.

## Route cache lifetime

Route caches are derived state, not source state.

A cache key contains:

- Topology revision
- Dynamic blocker revision
- Source
- Target or target rectangle
- Actor footprint
- Route mode

A cache hit requires every key field to match. Encounter code increments the blocker revision when relevant occupancy changes. A reset or snapshot restore invalidates every derived route cache. Snapshots do not serialize route scratch or route caches.

If an encounter cannot expose a sound blocker revision, it does not use the cache. There is no heuristic reuse.

## Inventory representation

Replace repeated classification of parallel inventory fields with canonical content metadata.

Each cell stores a `uint8_t` canonical content code and a `uint16_t` raw OSRS ID. Generated metadata indexed by content code provides:

- Item index
- Consumable kind
- Dose
- Click action
- Gear slot
- Observation code
- Static affordance features

Constructors and mutation functions create valid cells. Callers do not write item, dose, or click fields independently. This removes illegal combinations and repeated `osrs_consumable_click_registry_index` switches.

The generated item observation table remains the source for static features. Dynamic values such as equipped state, current healing fraction, cooldown legality, and special energy remain dynamic.

Every inventory constructor, click mutation, drink transition, swap, snapshot, restore, renderer, observation writer, mask writer, and test migrates in the same cutover. Old fields and classifiers are deleted.

## Observation and mask output

Observation widths and indices do not change.

Observation writers use the canonical cell code and generated metadata directly. They do not rebuild a full affordance record to recover one code or one heal value.

Change `EncounterDef.write_mask` to write bytes. Native wrappers pass their action-mask buffer directly. The viewer and diagnostics consume byte masks. Delete temporary float masks and float-to-byte conversion loops.

Observation and mask output remain pure functions of encounter state and context. Output caching is allowed only for static table rows. Dynamic state is always written from the current tick.

## Encounter-local work

Reprofile after the shared cutover. Keep local work narrow.

### Inferno

- Reassess attack-route field construction
- Move static LOS to shared topology
- Remove redundant observation-slot refreshes only when state revisions prove reuse

### Colosseum

- Replace private topology tables with the shared topology
- Reassess occupancy-aware NPC routing
- Share route fields only when blocker revisions prove identical inputs
- Reassess Venator candidate selection

### Zulrah

- Remove full-grid path clearing through the shared solver
- Reassess cloud destination selection
- Keep phase and hazard policy encounter-owned

### NH PvP

- Resolve each selected loadout once per agent tick
- Pass the resolved value through masks, movement, switches, and combat
- Keep opponent policy and combat decisions encounter-owned

### Muspah

Merge after the APIs above stop moving. Adapt `osrs_muspah`, `osrs_muspah_native`, and the Zulrah native wrapper during the merge. Do not import duplicate pathfinding, collision, inventory, or mask stacks.

## Failure handling

Abort on defects:

- Invalid arena dimensions
- Topology build failure
- Queue overflow
- Invalid item content code
- Invalid generated metadata
- Unknown config keys
- Impossible tagged states

Return route failure as data. Missing routes and unreachable targets are expected outcomes.

Do not add silent caps, timeout fallbacks, lazy concurrent initialization, or stale-cache recovery.

## Verification

### Route properties

Run exhaustive static checks for each arena and randomized dynamic-blocker checks:

- Every step is adjacent and legal
- Every diagonal satisfies both cardinal edge constraints
- A found route reaches its selected destination
- The optimized solver and the test-only reference BFS agree on reachability and shortest distance when given the same blockers
- Fallback selection minimizes the defined distance and route-cost ordering
- Cached and uncached queries produce equally valid outcomes
- A run step equals two valid sequential walk steps
- Generation wrap preserves results

Keep a simple reference BFS in tests only.

### Data contracts

- Every inventory content code round-trips
- Generated metadata matches existing click, dose, gear-slot, and observation semantics
- Drink transitions select the correct next dose
- Observation widths, indices, and values stay unchanged
- Mask widths, indices, and values stay unchanged
- Reset, snapshot, restore, reward, terminal, and combat traces remain deterministic

### Encounter gates

Run the shared player, interaction, inventory, collision, and route suites. Run focused Inferno, Colosseum, Zulrah, NH PvP, and later Muspah suites. Build every affected native wrapper. Run each encounter through the viewer and profiler.

Reseed a semantic golden only when an intentional equal-cost route or RNG-order change explains every diff. Unrelated golden changes fail the gate.

### Performance gates

Use paired runs with identical binaries, inputs, machine state, and profile action streams.

- No encounter may regress beyond measured noise
- Each patch must improve its target median
- The aggregate geometric-mean SPS must rise
- Sampling must show the intended hotspot fell
- Final validation must include actual vectorized training throughput

Continue while a measured hotspot supports a simpler change with material gain. Reject caches whose lifecycle costs more complexity than their measured gain.

## Clean cutover order

1. Freeze reference binaries and benchmark inputs
2. Add route property tests and a test-only reference BFS
3. Add shared topology and route scratch
4. Migrate every current route caller and delete old pathfinders
5. Migrate inventory representation and generated metadata
6. Migrate byte mask output and delete adapters
7. Reprofile all current encounters
8. Apply measured encounter-local changes
9. Run full mechanics, build, viewer, and performance gates
10. Merge Muspah and adapt its wrappers to the final APIs
