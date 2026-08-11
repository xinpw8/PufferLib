# Task 3 report

Status: DONE

## Result

Inferno, Colosseum, Zulrah, and NH PvP now use `EncounterArenaTopology` plus `EncounterRouteBlockers` through one shared route engine. Production callers no longer use `EncounterArenaAttackRouteField`, `encounter_build_arena_attack_route_field`, `encounter_arena_attack_route_landing`, `encounter_pathfind_arena_attack_approach`, or the old arena BFS helpers.

The engine returns `EncounterRouteResult` with `ROUTE_REACHED_TARGET`, `ROUTE_REACHED_FALLBACK`, `ROUTE_UNREACHABLE`, or `ROUTE_INVALID_INPUT`. Expected route failure is data. Invalid internal state, unsupported topology bounds, queue overflow, corrupt predecessor state, stale topology revision, and ownership violations abort.

`EncounterRouteInput` carries the immutable topology and dynamic blocker callback, context, and revision. Query scratch and reusable source/reverse fields are thread-local fixed storage. Queries allocate nothing and take no locks. One finalized topology is process-shared per encounter and each encounter context stores a const pointer. Dynamic occupancy stays in encounter state.

Player chase state moved out of serialized `OsrsInteraction` state into actor-local `OsrsActorRouteCache`. Cache validation covers topology revision, blocker revision, source/current actor position, actor size, target position and size, attack range, movement mode, and route-cost policy. Restore/reset paths clear the context cache. `OsrsInteraction` retains its fixed 244-byte serialized route reserve, but raw base artifacts intentionally differ where the removed serialized cache had data. Forecast comparison validates each file's raw record hash before canonicalizing route storage.

Equal-cost behavior remains explicit in `EncounterRouteCostPolicy`. Attack-chase OSRS order uses the packed source field. Colosseum NPC OSRS-aggro paths use the reverse field with preserved south-first edge ordering. `OSRS_PLAYER_MOVE_ACTION` uses the direct one-action policy. `OSRS_PLAYER_MOVE_DESTINATION` uses OSRS routing in Inferno, Colosseum, Zulrah, and NH PvP.

## TDD evidence

### RED

The first route-result tests were added to `test_osrs_player_step.c` before the production contract. This command failed to compile because the tagged route types and solver did not exist:

```text
clang -std=c11 -O2 -I. ocean/osrs/tests/test_osrs_player_step.c -lm -o /tmp/test_osrs_player_step_task3
```

The first migrated forecast comparison exposed route-order drift:

```text
/tmp/test_colosseum_forecast_exact_task3 --compare /tmp/task3-colosseum-forecast
colosseum exact mismatch at byte 1947076: expected 117 got 39
```

Changing all explicit movement to direct routing proved incorrect for Zulrah:

```text
clang -std=c11 -O2 -I. ocean/osrs/tests/test_zulrah_golden.c -lm -o /tmp/test_zulrah_golden_task3 && /tmp/test_zulrah_golden_task3
```

All six golden hashes changed. Restoring Zulrah explicit movement to `ENCOUNTER_ROUTE_COST_OSRS` restored 6/6 exact hashes.

The initial exhaustive reference comparison caught an invalid overlap case in the test driver:

```text
clang -std=c11 -O2 -I. ocean/osrs/tests/test_colosseum_forecast_exact.c -lm -o /tmp/test_colosseum_forecast_exact_task3 && /tmp/test_colosseum_forecast_exact_task3 --attack-route-selftest
route equivalence mismatch scenario=open-arena source=(0,13) target=(0,13,2) expected=(0,2,13,1,0,2) actual=(0,1,12,1,0,2)
```

The exhaustive driver was corrected to exclude actor/target overlap. The production interaction layer handles overlap through its explicit escape contract rather than normal attack-chase routing.

### GREEN

Tagged outcomes, cache invalidation, direct movement, tie order, topology bounds, dynamic blockers, and actor-local reuse:

```text
clang -std=c11 -O2 -I. ocean/osrs/tests/test_osrs_player_step.c -lm -o /tmp/test_osrs_player_step_task3 && /tmp/test_osrs_player_step_task3
```

Result: 73/73 passed.

Shared NPC movement, footprint masks, diagonal edge clearance, overlap rewrite, melee policy, and current overlap:

```text
clang -std=c11 -O2 -I. ocean/osrs/tests/test_osrs_npc_movement.c -lm -o /tmp/test_osrs_npc_movement_task3 && /tmp/test_osrs_npc_movement_task3
```

Result: 29/29 passed.

Exhaustive optimized-versus-reference BFS:

```text
/tmp/test_colosseum_forecast_exact_task3 --attack-route-selftest
/tmp/test_inferno_forecast_exact_task3 --attack-route-selftest
clang -std=c11 -O2 -I. -DTEST_ROUTE_TOPOLOGY_NH_PVP ocean/osrs/tests/test_encounter_route_topology_setup.c -lm -o /tmp/test_route_topology_nh_pvp && /tmp/test_route_topology_nh_pvp
```

Results:

```text
colosseum exhaustive route equivalence PASS: 22511700 source-target-range queries across 2 blocker fields
colosseum attack route property selftest PASS: 1656 target queries across 3 fields
inferno exhaustive route equivalence PASS: 19048431 source-target-range queries across 2 blocker fields
nh_pvp target-directed BFS equivalence PASS: 5418624 source-target queries across 2 blocker fields
```

The test-only reference performs an independent legacy FIFO BFS and landing scan. The checks cover every walkable source, every statically valid target anchor, target sizes through each encounter maximum, every encounter player attack range, and representative dynamic blocker revisions. Each comparison pins outcome, destination, shortest distance, first step, run step, and waypoint sequence.

Topology finalization, immutable content identity, construction-order independence, process reuse, and bounds:

```text
clang -std=c11 -O2 -I. -DTEST_ROUTE_TOPOLOGY_INFERNO ocean/osrs/tests/test_encounter_route_topology_setup.c -lm -o /tmp/test_route_topology_inferno && /tmp/test_route_topology_inferno
clang -std=c11 -O2 -I. -DTEST_ROUTE_TOPOLOGY_COLOSSEUM ocean/osrs/tests/test_encounter_route_topology_setup.c -lm -o /tmp/test_route_topology_colosseum && /tmp/test_route_topology_colosseum
clang -std=c11 -O2 -I. -DTEST_ROUTE_TOPOLOGY_ZULRAH ocean/osrs/tests/test_encounter_route_topology_setup.c -lm -o /tmp/test_route_topology_zulrah && /tmp/test_route_topology_zulrah
clang -std=c11 -O2 -I. -DTEST_ROUTE_TOPOLOGY_NH_PVP ocean/osrs/tests/test_encounter_route_topology_setup.c -lm -o /tmp/test_route_topology_nh_pvp && /tmp/test_route_topology_nh_pvp
```

Results: Inferno 870 open and 0 blocked, Colosseum 865 open and 291 blocked, Zulrah 69 open and 715 blocked, and NH PvP 1648 open and 60 blocked. Every setup test passed order independence and identical-map process reuse.

Forecast fixtures came from base commit `8fd9feb0fc5c19428751a16cc6b2b82aa2f9e731`. Each expected and current raw state first validates against the state hash stored in its own record. The comparator then clears the removed serialized route storage in both states, clears Colosseum's route-profiling-only blocker counters, recomputes the canonical state hashes, and compares canonical record headers, forecasts, observations, masks, and state bytes. Raw base bytes are not claimed identical because the deleted route cache occupied the serialized reserve.

```text
clang -std=c11 -O2 -I. ocean/osrs/tests/test_colosseum_forecast_exact.c -lm -o /tmp/test_colosseum_forecast_exact_task3
clang -std=c11 -O2 -I. ocean/osrs/tests/test_inferno_forecast_exact.c -lm -o /tmp/test_inferno_forecast_exact_task3
/tmp/test_colosseum_forecast_exact_task3 --compare /tmp/task3-colosseum-forecast
/tmp/test_inferno_forecast_exact_task3 --compare /tmp/task3-inferno-forecast
```

Results: both raw record-integrity checks and canonical semantic comparisons passed. Colosseum also passed 1,336,336 LoS pairs, 12,005 footprint checks, 25 landing actions, 22,511,700 exhaustive route queries, and 1,656 attack-route property queries.

Focused encounter mechanics:

```text
clang -std=c11 -O2 -I. ocean/osrs/tests/test_inferno_attack_styles.c -lm -o /tmp/test_inferno_attack_styles_task3 && /tmp/test_inferno_attack_styles_task3
clang -std=c11 -O2 -I. ocean/osrs/tests/test_zulrah_hit_delay.c -lm -o /tmp/test_zulrah_hit_delay_task3 && /tmp/test_zulrah_hit_delay_task3
clang -std=c11 -O2 -I. ocean/osrs/tests/test_zulrah_golden.c -lm -o /tmp/test_zulrah_golden_task3 && /tmp/test_zulrah_golden_task3
clang -std=c11 -O2 -I. ocean/osrs/tests/test_osrs_pvp_pending_hits.c -lm -o /tmp/test_osrs_pvp_pending_hits_task3 && /tmp/test_osrs_pvp_pending_hits_task3
```

Results:

- Inferno: 2041/2041 passed
- Zulrah hit delay and roll order: 313/313 passed
- Zulrah golden master: 6/6 hashes match baseline
- NH PvP pending hits: 9/9 passed

An extra full `test_colosseum_modifiers.c` run produced 13545/13546 with only `generic item table code 101 feature 13 matches base-99 semantics` failing. The untouched base worktree produced the identical 13545/13546 failure. Task 3 did not change item encoding. Every route, movement, LoS, forecast, snapshot, combat, and modifier check in that binary passed.

## Standalone API consumers

No clangd indexer or harness LSP-reference operation was available. Before deleting exported route symbols, textual reference discovery covered production, tests, wrappers, and standalone benchmarks. Consumers importing the deleted API were migrated to the test-only reference header where comparison with the old implementation is intentional:

- `ocean/osrs/tests/test_osrs_player_step.c`
- `ocean/osrs/tests/test_colosseum_forecast_exact.c`
- `ocean/osrs/tests/test_colosseum_modifiers.c`
- `ocean/osrs/tests/test_inferno_attack_styles.c`
- `ocean/osrs/tests/bench_colo_step.c`
- `ocean/osrs/tests/bench_colosseum_forecast_profile.c`
- `ocean/osrs_colosseum/osrs_colosseum.h`
- `ocean/osrs_inferno/osrs_inferno.h`

Final production search for `EncounterArenaAttackRouteField`, `encounter_build_arena_attack_route_field`, `encounter_arena_attack_route_landing`, `encounter_pathfind_arena_attack_approach`, `pathfind_step_arena`, `BFS_VISIT`, `bfs_via`, and `route_field` returned matches only from `ocean/osrs/tests/osrs_route_reference.h` and tests calling that reference. No production match remains.

## Build and portability gates

Available native viewer wrappers:

```text
OUTPUT_NAME=/tmp/osrs_task3_inferno_wrapper ./build.sh osrs_inferno --fast
OUTPUT_NAME=/tmp/osrs_task3_colosseum_wrapper ./build.sh osrs_colosseum --fast
```

Results:

```text
Built: .//tmp/osrs_task3_inferno_wrapper
Built: .//tmp/osrs_task3_colosseum_wrapper
```

The repository has no separate Zulrah or NH PvP wrapper directories. Both compile into the shared Inferno viewer/profile wrapper and were exercised through `--encounter zulrah` and `--encounter nh_pvp`.

The macOS ARM64 `--cpu` wrapper mode is unavailable because PufferLib's CPU backend includes x86-only intrinsics. It failed before encounter compilation. CPU translation portability was checked directly for every encounter header:

```text
clang -std=c11 -O2 -I. /tmp/route_portability_inferno.c -lm -o /tmp/route_portability_inferno_c
clang -std=c11 -O2 -I. /tmp/route_portability_colosseum.c -lm -o /tmp/route_portability_colosseum_c
clang -std=c11 -O2 -I. /tmp/route_portability_zulrah.c -lm -o /tmp/route_portability_zulrah_c
clang -std=c11 -O2 -I. /tmp/route_portability_nh_pvp.c -lm -o /tmp/route_portability_nh_pvp_c
g++-16 -std=c++17 -O2 -I. -x c++ /tmp/route_portability_inferno.c -lm -o /tmp/route_portability_inferno_cpp
g++-16 -std=c++17 -O2 -I. -x c++ /tmp/route_portability_colosseum.c -lm -o /tmp/route_portability_colosseum_cpp
g++-16 -std=c++17 -O2 -I. -x c++ /tmp/route_portability_zulrah.c -lm -o /tmp/route_portability_zulrah_cpp
g++-16 -std=c++17 -O2 -I. -x c++ /tmp/route_portability_nh_pvp.c -lm -o /tmp/route_portability_nh_pvp_cpp
```

Result: all eight C/C++17 translation and link gates passed.

## Paired performance

Method: base binary from clean `8fd9feb0fc5c19428751a16cc6b2b82aa2f9e731`, final current binary, 100,000 steps, paired policy seeds 1 through 12, identical encounter and seed per pair. The clean base profiling copy received one benchmark-only seeding patch matching the final driver: parse `--policy-seed` and call `srand(profile_seed)` once before the profile loop. It changed no encounter, route, action, observation, reward, or terminal code. Both binaries therefore consumed the same deterministic policy stream for each pair. Command shape:

```text
BINARY --profile --encounter ENCOUNTER --profile-steps 100000 --policy-seed SEED
```

| Encounter | Base SPS by seed 1..12 | Current SPS by seed 1..12 | Base GM | Current GM | Ratio GM |
|---|---|---|---:|---:|---:|
| Inferno | 722423, 774821, 776633, 793531, 785521, 799968, 761905, 773317, 781378, 791615, 784757, 772809 | 829187, 891234, 868523, 876939, 869664, 899159, 890076, 883556, 877093, 888976, 881073, 877332 | 776313 | 877566 | 1.1304x |
| Colosseum | 412613, 411399, 418468, 408986, 412565, 419336, 419925, 410521, 412223, 413810, 413414, 411748 | 498925, 497033, 496966, 491809, 495818, 499266, 500726, 492795, 498962, 489615, 505953, 498507 | 413737 | 497180 | 1.2017x |
| Zulrah | 345111, 352209, 344043, 345543, 346371, 344942, 341495, 348594, 360659, 345458, 345342, 351240 | 703438, 690322, 693577, 699364, 701095, 702134, 697024, 700535, 706869, 699658, 689422, 685594 | 347550 | 697392 | 2.0066x |
| NH PvP | 557336, 553547, 514634, 508024, 524115, 511470, 526840, 516726, 519591, 513313, 519651, 527209 | 683289, 663724, 655768, 688848, 675489, 674832, 669456, 679768, 674132, 672278, 678178, 672839 | 524161 | 673999 | 1.2859x |

Aggregate geometric mean of the four encounter ratios: 1.3683x. No encounter regressed.

## Changed files

Full `8fd9feb0fc5c19428751a16cc6b2b82aa2f9e731..HEAD` task delta:

- `.superpowers/sdd/task-3-report.md`
- `ocean/osrs/osrs_collision.h`
- `ocean/osrs/osrs_encounter.h`
- `ocean/osrs/osrs_encounter_player.h`
- `ocean/osrs/osrs_interaction.h`
- `ocean/osrs/osrs_pathfinding.h`
- `ocean/osrs/osrs_pvp_actions.h`
- `ocean/osrs/osrs_pvp_api.h`
- `ocean/osrs/osrs_pvp_movement.h`
- `ocean/osrs/osrs_render.h`
- `ocean/osrs/osrs_visual.c`
- `ocean/osrs/encounters/encounter_nh_pvp.h`
- `ocean/osrs/encounters/encounter_zulrah.h`
- `ocean/osrs/encounters/colosseum/encounter_colosseum_boss.inc`
- `ocean/osrs/encounters/colosseum/encounter_colosseum_combat.inc`
- `ocean/osrs/encounters/colosseum/encounter_colosseum_helpers.inc`
- `ocean/osrs/encounters/colosseum/encounter_colosseum_model.inc`
- `ocean/osrs/encounters/colosseum/encounter_colosseum_movement.inc`
- `ocean/osrs/encounters/colosseum/encounter_colosseum_player_actions.inc`
- `ocean/osrs/encounters/colosseum/encounter_colosseum_render_snapshot.inc`
- `ocean/osrs/encounters/colosseum/encounter_colosseum_reset_spawn.inc`
- `ocean/osrs/encounters/colosseum/encounter_colosseum_reward_step.inc`
- `ocean/osrs/encounters/inferno/encounter_inferno_helpers.inc`
- `ocean/osrs/encounters/inferno/encounter_inferno_model.inc`
- `ocean/osrs/encounters/inferno/encounter_inferno_player_actions.inc`
- `ocean/osrs/encounters/inferno/encounter_inferno_render_snapshot.inc`
- `ocean/osrs/encounters/inferno/encounter_inferno_reset_spawn.inc`
- `ocean/osrs/encounters/inferno/encounter_inferno_reward_step.inc`
- `ocean/osrs/tests/osrs_route_reference.h`
- `ocean/osrs/tests/test_encounter_route_topology_setup.c`
- `ocean/osrs/tests/test_osrs_player_step.c`
- `ocean/osrs/tests/test_colosseum_forecast_exact.c`
- `ocean/osrs/tests/test_colosseum_modifiers.c`
- `ocean/osrs/tests/test_inferno_attack_styles.c`
- `ocean/osrs/tests/test_inferno_forecast_exact.c`
- `ocean/osrs/tests/test_zulrah_golden.c`
- `ocean/osrs/tests/test_zulrah_hit_delay.c`
- `ocean/osrs/tests/bench_colo_step.c`
- `ocean/osrs/tests/bench_colosseum_forecast_profile.c`
- `ocean/osrs_colosseum/osrs_colosseum.h`
- `ocean/osrs_inferno/osrs_inferno.h`

## Self-review

- Topology stays immutable after finalization. No dynamic actor, pillar, clamp, cloud, or opponent occupancy entered it.
- Contexts hold const process-shared topology pointers.
- Route scratch is thread-local, fixed-size, and generation-guarded or reset before reads.
- Query hot paths allocate nothing, take no locks, and have no fallback to the deleted production implementation.
- Actor route caches validate every required key. Changed blocker revisions and target geometry reroute from the actor's current tile.
- Source/reverse fields validate topology revision, blocker identity/revision, source or target geometry, actor size, movement mode, and cost policy as applicable.
- Dynamic blocker memoization distinguishes unknown, open, and blocked footprints and never mutates topology.
- Fallback distance, FIFO/depth tie-breaking, first step, run step, and waypoint compression match the independent references across 46,978,755 exhaustive queries.
- Inferno and Colosseum canonical forecasts are byte-exact after clearing removed serialized route storage and route-profiling counters. Raw records first pass their own stored-hash integrity checks. Zulrah's six hashes remain exact. Snapshot state size and serialized interaction layout remain unchanged.
- Production search finds no old route field or arena BFS symbol. The only legacy implementation is isolated in `ocean/osrs/tests/osrs_route_reference.h`.
- Final `git diff --check` passed.
