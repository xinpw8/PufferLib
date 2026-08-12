# Task 5 report

## Result

Implementation complete. Main validation pending.

Base: `a506c5a462c4a035e9d79d842d35655ac7a4102c`

## Implementation

- Moved Colosseum arena bounds, static footprint occupancy, legal steps, and directed static LOS to the shared `EncounterArenaTopology`.
- Added the tile-blocked topology LOS build mode that preserves Colosseum's legacy directional endpoint rules.
- Deleted the private blocked-tile array, footprint table, static LOS table, and their initialization latch.
- Routed player, NPC, forecast, spawn, Sol, modifier, observation, mask, lab, and render geometry through explicit topology-owning contexts.
- Kept NPC occupancy, player occupancy, modifiers, hazards, and active state dynamic.
- Deleted the process-global legacy context and contextless reset, step, combat, lab, and attack wrappers.
- Made supported callbacks reject NULL context. Standalone probes and tests now initialize and finalize explicit contexts.
- Added exhaustive static tile, footprint, directed LOS, endpoint-law, legal-step, and forecast parity checks to existing Colosseum tests.
- Added lifecycle checks for NULL and unfinalized contexts.
- Kept the checked shared footprint query unchanged and added a trusted finalized, size-in-range query for validated Colosseum path contexts. The hot route blocker now pays lifecycle and size checks once when it creates the path context instead of once per explored tile.
- Preserved the observation and mask layouts. No item tables changed.

## Observed evidence

The interrupted worker's compile attempts exposed stale private-geometry and old-signature callers in `test_colosseum_modifiers.c`. The first main gate then reproduced a stale `col_player_walkable` callback-data crash in the modifier test. Source inspection found the same callback misuse in the skyfall probe and movement trace, plus unfinalized contexts in five standalone probes. Recovery migrated those callers to explicit geometry contexts and finalized every audited context before reset, step, or topology query.

After that fix, exact, golden, and determinism gates passed. Twelve paired 5M-step profiles regressed on every seed, with a final/base geometric mean of `0.978398718` and median SPS of `482699` versus `493402`. Equivalent samples isolated repeated finalized-state and footprint-size checks under `col_npc_path_blocked`. The follow-up moves those checks to `col_npc_path_ctx_begin` and retains raw bounds handling on every tile query. No compile, test, profile, or sanitizer result was observed after the performance edit.

## Pending main validation

Run the Task 5 gates from the brief:

```bash
clang -std=c11 -O2 -I. ocean/osrs/tests/test_colosseum_modifiers.c -lm -o /tmp/test_colosseum_modifiers && /tmp/test_colosseum_modifiers
clang -std=c11 -O2 -I. ocean/osrs/tests/test_colosseum_forecast_exact.c -lm -o /tmp/test_colosseum_forecast_exact && /tmp/test_colosseum_forecast_exact
clang -std=c11 -O2 -I. ocean/osrs/tests/test_colosseum_golden.c -lm -o /tmp/test_colosseum_golden && /tmp/test_colosseum_golden
clang -std=c11 -O2 -I. ocean/osrs/tests/probe_colo_sim_invariant.c -lm -o /tmp/probe_colo_sim_invariant && /tmp/probe_colo_sim_invariant
```

Also run the 10k modifier battery, Sol spear geometry checks, one-worker versus eight-worker determinism, C11 and C++17 include gates, sanitizer gates, twelve paired profiles, and `sample`. Compare all semantic hashes with the Task 4 baseline. Do not reseed.
