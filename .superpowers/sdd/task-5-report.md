# Task 5 report

## Result

Implementation and main validation complete.

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

After that fix, exact, golden, and determinism gates passed. Twelve paired 5M-step profiles regressed on every seed, with a final/base geometric mean of `0.978398718` and median SPS of `482699` versus `493402`. Equivalent samples isolated repeated finalized-state and footprint-size checks under `col_npc_path_blocked`. The follow-up moves those checks to `col_npc_path_ctx_begin` and retains raw bounds handling on every tile query.

## Main validation

- Modifier battery: `13549/13550`. Every Colosseum topology and Sol spear check passed. The sole failure is the inherited generic item-table code `101` mismatch that reproduces at the Task 5 base.
- Exact forecast: topology LOS `1335180` directed pairs, footprint `5709` NPC-size checks, landing `25` actions, route equivalence `22511700` queries, attack routes `1656` queries, and the Task 4 golden fixture all passed.
- Golden: `12/12` configs matched.
- Sim invariant: all 12 printed hashes matched the detached Task 5 base byte for byte.
- Determinism: one worker versus eight workers and repeated eight-worker runs were identical across 256 envs in both late-start modes.
- Portability: the C11 topology setup and C++17 Colosseum header gates passed.
- Sanitizers: the topology setup passed UBSan. The host ASan runtime deadlocked before `main` in `AsanInitInternal`, so no ASan result exists.
- Performance: all 12 paired 5M-step runs improved. Final/base ratios were `1.001197`, `1.013857`, `1.005567`, `1.013254`, `1.008387`, `1.005730`, `1.008313`, `1.003657`, `1.006833`, `1.000941`, `1.013913`, and `1.003901`. The geometric mean was `1.007119621`. Median SPS rose from `487283` to `491122`.
- The post-fix sample no longer attributes route-solver samples to the checked Colosseum footprint wrapper.
