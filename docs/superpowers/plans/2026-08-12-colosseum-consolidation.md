# Colosseum Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the latest Colosseum behavior onto the optimized 5c branch and make the run38 `933`-observation, `451`-mask checkpoint contract canonical.

**Architecture:** Treat the current 5c branch as the structural source of truth. Port the net behavior from `0fc4b50c..8281518ea` and the Colosseum-only behavior from `3812fd454`, adapting every call to the current shared topology, route, context, and item-metadata APIs.

**Tech Stack:** C11 single-header encounter implementation, shared OSRS SDK, clang, local Metal viewer, deterministic golden and semantic-hash probes.

## Global Constraints

- Keep `EncounterArenaTopology`, the shared route engine, canonical inventory content codes, and shared item metadata.
- Canonical model contract: `COLO_NUM_OBS == 933`, `COLO_ACTION_MASK_SIZE == 451`, spell head `{none, Death Charge}`.
- Canonical snapshot version: `25`.
- Preserve `laser_obs_mode` and require `1` for run38 evaluation.
- Port the verified `osrs-assets-v22` manifest SHA. Do not port the Puffer source-layout migration, old item-table files, stale baselines, or the altered FNV seed.
- Do not modify Inferno, Zulrah, NH PvP, Puffer internals, checkpoints, generated assets, or local untracked artifacts.
- Every semantic change gets a failing contract test before implementation.

---

### Task 1: Isolate and establish the red baseline

**Files:**
- Modify: `ocean/osrs/asset_manifest.json`
- Modify later: `ocean/osrs/encounters/colosseum/*.inc`
- Test: `ocean/osrs/tests/test_colosseum_modifiers.c`

**Interfaces:**
- Consumes: `valtteri/colo-5c-port` at the plan commit.
- Produces: `/private/tmp/puffer-colo-consolidation` on `valtteri/colo-consolidation`.

- [ ] **Step 1: Create the temporary branch and worktree**

```bash
pwd
git worktree add /private/tmp/puffer-colo-consolidation -b valtteri/colo-consolidation valtteri/colo-5c-port
```

Expected: worktree starts at the plan commit with no tracked modifications.

- [ ] **Step 2: Pin and install the published asset archive**

Set the `osrs-assets-v22` SHA to `2f908da5b5ddf148c0cbef48ad6334c5253c04c6cd83782340bc2f1c5dffc7ad`, then run:

```bash
bash ocean/osrs/scripts/setup-data.sh
```

Expected: the downloaded release archive matches the manifest and installs the ignored local data.

- [ ] **Step 3: Compile and run the current focused baseline**

```bash
clang -std=c11 -O2 -I. ocean/osrs/tests/test_colosseum_modifiers.c -lm -o /tmp/colo_consolidation_baseline
/tmp/colo_consolidation_baseline
```

Expected: `13550/13550 tests passed`.

- [ ] **Step 4: Confirm current interface values before adding red tests**

Use the existing `test_primary_head_resolution` and fuzz assertions.

Expected: `COLO_NUM_OBS == 922`, `COLO_ACTION_MASK_SIZE == 452`, `COLO_SPELL_DIM == 3`.

---

### Task 2: Restore honest Sol laser observations

**Files:**
- Modify: `config/osrs_colosseum.ini`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_boss.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_helpers.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_model.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_obs_mask.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_render_snapshot.inc`
- Test: `ocean/osrs/tests/test_colosseum_modifiers.c`

**Interfaces:**
- Consumes: current `ColoSolState`, `ColosseumConfig`, `ColosseumLog`, and `col_write_boss_obs`.
- Produces: `laser_obs_mode`, twelve added boss-observation floats, and episode laser diagnostics.

- [ ] **Step 1: Add failing laser-pack contract tests**

Add assertions that the enabled pack reports one cooldown plus four records of three floats and that disabled mode zeroes exactly those thirteen laser-pack channels without changing width. Assert the intermediate interface:

```c
CHECK("laser pack expands obs 922 to 934", COLO_NUM_OBS == 934);
```

Run the modifiers suite. Expected: compile or assertion failure because the pack and config field do not exist.

- [ ] **Step 2: Add final laser state and configuration fields**

Add to `ColoSolState`:

```c
int laser_volley_active;
int laser_show_seen;
int laser_aligned_show;
int laser_n_aligned_show;
int laser_n_active_at_fire;
```

Add `int laser_obs_mode` to `ColosseumConfig` with default `1`, binary config parsing, and `laser_obs_mode = 1` in `config/osrs_colosseum.ini`. Add the nine `laser_*` float counters from `8281518ea` to `ColosseumLog` and preserve their existing metric names.

- [ ] **Step 3: Port the final multi-crystal observation writer**

Define:

```c
#define COLO_SOL_LASER_CRYSTAL_FEATS 3
#define COLO_SOL_LASER_OBS_SIZE \
    (1 + COLO_SOL_MAX_CRYSTALS * COLO_SOL_LASER_CRYSTAL_FEATS)
#define COLO_BOSS_OBS_PREMOVE_TAIL_SIZE (2 + COLO_SOL_LASER_OBS_SIZE)
```

Change `col_write_boss_obs` to accept `int laser_obs_mode`. Write the normalized volley cooldown, then each crystal's active flag, signed line delta, and normalized freeze timer. Advance the same indices with zeros when disabled.

- [ ] **Step 4: Port final laser event accounting**

Reset volley diagnostics at Sol reset. At show, prefire, and fire transitions, count active and aligned crystals and update the existing episode log fields. Do not port the canceled next-style or pending-style experiments.

- [ ] **Step 5: Run and commit**

```bash
clang -std=c11 -O2 -I. ocean/osrs/tests/test_colosseum_modifiers.c -lm -o /tmp/colo_laser_pack
/tmp/colo_laser_pack
git add config/osrs_colosseum.ini ocean/osrs/encounters/colosseum ocean/osrs/tests/test_colosseum_modifiers.c
git commit -m "Restore honest Sol laser observations"
```

Expected: modifiers suite passes with `COLO_NUM_OBS == 934` and mask `452`.

---

### Task 3: Port the Manticore activation state machine

**Files:**
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_model.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_combat.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_obs_mask.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_render_snapshot.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_reset_spawn.inc`
- Test: `ocean/osrs/tests/test_colosseum_modifiers.c`

**Interfaces:**
- Consumes: `col_npc_has_los_to_player(s, ctx, npc)` and current peer-stagger logic.
- Produces: `ColoManticorePhase` and explicit activation, target-wait, initial-charge, and repeating states.

- [ ] **Step 1: Add the failing WIP Manticore contract**

Port `test_manticore_initial_charge_timing_and_target_gate` from `3812fd454`, but route line-of-sight through `ColosseumContext`. Change misleading test text from “visible tick” to “spawn tick”.

Expected red conditions: the current Manticore arms before the seventh spawn tick, exposes orbs too early, and lacks the phase enum.

- [ ] **Step 2: Add the explicit phase type**

```c
typedef enum {
    COLO_MANTICORE_PHASE_ACTIVATING = 0,
    COLO_MANTICORE_PHASE_WAITING_FOR_TARGET,
    COLO_MANTICORE_PHASE_CHARGING,
    COLO_MANTICORE_PHASE_REPEATING,
} ColoManticorePhase;

typedef struct {
    ColoManticorePhase phase;
    int cycle_step;
    AttackStyle orb_style[3];
    AttackStyle fixed_orb_style[3];
} ColoManticoreState;

#define COLO_MANTICORE_ACTIVATION_TICKS 7
#define COLO_MANTICORE_CHARGE_TICKS 10
#define COLO_MANTICORE_ARM_ANIMATION_TICKS 3
```

Initialize Manticores in `ACTIVATING` with attack timer `7`.

- [ ] **Step 3: Implement target gating and phase transitions**

Use:

```c
static int col_manticore_has_target(
    const ColosseumState* s,
    const ColosseumContext* ctx,
    const ColoNPC* npc,
    const ColoNpcStats* stats
) {
    int dist = col_npc_dist_to_player(s, npc);
    return dist >= 1 && dist <= stats->attack_range &&
        col_npc_has_los_to_player(s, ctx, npc);
}
```

Count activation down independently of line of sight. In `WAITING_FOR_TARGET`, arm only after a valid target appears. In `CHARGING`, consume ten full ticks, then require a valid target before the first shot. Switch to `REPEATING` when firing starts. Adapt peer staggering so only equivalent ready states delay each other.

- [ ] **Step 4: Hide premature telegraphs and oracle data**

Make Manticore orbs and next-prayer data invisible during activation, target wait, and the first three charge ticks. Preserve existing one-tick barrage tells after the first shot.

- [ ] **Step 5: Run and commit**

```bash
clang -std=c11 -O2 -I. ocean/osrs/tests/test_colosseum_modifiers.c -lm -o /tmp/colo_manticore
/tmp/colo_manticore
git add ocean/osrs/encounters/colosseum ocean/osrs/tests/test_colosseum_modifiers.c
git commit -m "Model Manticore activation and charge"
```

Expected: all Manticore timing, telegraph, shared-cycle, and stagger tests pass.

---

### Task 4: Make the thrall lifecycle automatic

**Files:**
- Modify: `config/osrs_colosseum.ini`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_helpers.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_lab_json.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_mask_render.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_model.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_obs_mask.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_player_actions.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_reset_spawn.inc`
- Test: `ocean/osrs/tests/test_colosseum_modifiers.c`

**Interfaces:**
- Consumes: `tick_scratch.player_attacked`, `player_attack_npc_idx`, current Magic, loadout profile, and pending-hit queues.
- Produces: automatic thrall state and final `933`/`451` model contract.

- [ ] **Step 1: Add the failing automatic-thrall test**

Port the final `test_thrall_regression` from `3812fd454`. Assert reset lifetime, attack-triggered target acquisition, twelve-tick aggro, four-tick attacks, expiry resummon, profile multiplier, and Sol immunity.

Expected: compile failure on `thrall_aggro_ticks_left` and `thrall_lifetime_total`.

- [ ] **Step 2: Replace representational state**

Remove `thrall_active` and `thrall_recast_cd`. Add:

```c
int thrall_target_slot;
int thrall_aggro_ticks_left;
int thrall_lifetime_left;
int thrall_lifetime_total;
int thrall_attack_timer;
```

Define `COLO_THRALL_AGGRO_TICKS 12`. Compute lifetime from current Magic, doubled only for `COLO_LOADOUT_PROFILE_SPEEDRUN`.

- [ ] **Step 3: Implement automatic reset, targeting, and expiry**

`col_resummon_thrall` sets no target, zero aggro, full derived lifetime, and a four-tick attack timer. Call it at episode reset and expiry. Acquire a target only when `tick_scratch.player_attacked` names a live non-Sol NPC. Decrement aggro only on ticks without a new player attack. Clear invalid or expired targets.

- [ ] **Step 4: Remove the summon action and rewrite observations**

Set `COLO_SPELL_DIM` to `2`, keep only `COLO_SPELL_DEATH_CHARGE = 1`, delete the summon mask branch, and reduce `COLO_THRALL_DC_OBS_SIZE` from `6` to `5`. Observe normalized lifetime, normalized aggro, Death Charge active, window, and cooldown. Render the implicit thrall unconditionally.

Set `action_mask_size = 451`. Assert `COLO_NUM_OBS == 933` and `COLO_ACTION_MASK_SIZE == 451`.

- [ ] **Step 5: Run and commit**

```bash
clang -std=c11 -O2 -I. ocean/osrs/tests/test_colosseum_modifiers.c -lm -o /tmp/colo_thrall
/tmp/colo_thrall
git add config/osrs_colosseum.ini ocean/osrs/encounters/colosseum ocean/osrs/tests/test_colosseum_modifiers.c
git commit -m "Make Colosseum thralls automatic"
```

Expected: thrall and Death Charge contracts pass at `933`/`451`.

---

### Task 5: Move Volatility to corpse removal

**Files:**
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_combat.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_model.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_modifiers.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_render_snapshot.inc`
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_reward_step.inc`
- Test: `ocean/osrs/tests/test_colosseum_modifiers.c`

**Interfaces:**
- Consumes: `ColosseumContext.route_topology` and NPC death linger.
- Produces: context-owned Volatility render events and corpse-removal damage.

- [ ] **Step 1: Add the failing corpse-removal contract**

Port `test_volatility_explodes_on_corpse_removal` from `3812fd454`. Assert no lethal-hit explosion, no explosion during linger, damage on removal, and one family-specific spot animation.

Expected: current code damages on the lethal hit and emits no removal event.

- [ ] **Step 2: Add context-owned event storage**

```c
int volatility_explosion_count;
int volatility_explosion_x[COLO_MAX_NPCS];
int volatility_explosion_y[COLO_MAX_NPCS];
ColoNpcType volatility_explosion_type[COLO_MAX_NPCS];
```

Clear the count in `col_context_clear_render_events`.

- [ ] **Step 3: Move the explosion trigger**

Remove `col_mod_volatility_on_death` from `col_apply_npc_death`. Change death-linger ticking to accept `ColosseumContext*`. Immediately before deactivating a corpse, call `col_mod_volatility_on_corpse_removed` with the shared finalized topology and queue its render event.

- [ ] **Step 4: Emit family-specific spot animations**

Map human, Manticore, Colossus, Minotaur, and Sol families to GFX IDs `2713`, `2721`, `2722`, `2723`, and `2724`. Emit a stationary overlay projectile at the corpse origin during `col_render_post_tick_ctx`.

- [ ] **Step 5: Run and commit**

```bash
clang -std=c11 -O2 -I. ocean/osrs/tests/test_colosseum_modifiers.c -lm -o /tmp/colo_volatility
/tmp/colo_volatility
git add ocean/osrs/encounters/colosseum ocean/osrs/tests/test_colosseum_modifiers.c
git commit -m "Explode Volatility on corpse removal"
```

Expected: modifiers suite passes and the new render event appears exactly once.

---

### Task 6: Finalize snapshot and deterministic contracts

**Files:**
- Modify: `ocean/osrs/encounters/colosseum/encounter_colosseum_render_snapshot.inc`
- Modify: `ocean/osrs/tests/test_colosseum_golden.c`
- Modify: `ocean/osrs/tests/probe_colo_sim_invariant.c`
- Modify only if required: focused Colosseum forecast and topology tests

**Interfaces:**
- Consumes: final state layout and `933`/`451` interface.
- Produces: snapshot version `25` and explained deterministic baselines.

- [ ] **Step 1: Bump and test snapshot version**

Set:

```c
#define COLO_SNAPSHOT_VERSION 25u
```

Keep stale-version rejection and round-trip tests. Expected: version 24 abort fixture rejects and version 25 round trip passes.

- [ ] **Step 2: Run deterministic probes before changing baselines**

```bash
clang -std=c11 -O2 -I. ocean/osrs/tests/test_colosseum_golden.c -lm -o /tmp/colo_golden_new
/tmp/colo_golden_new
clang -std=c11 -O2 -I. ocean/osrs/tests/probe_colo_sim_invariant.c -lm -o /tmp/colo_sim_new
/tmp/colo_sim_new
```

Expected: digest mismatches caused by laser observation width, Manticore timing, thrall state, and Volatility timing.

- [ ] **Step 3: Update only observed final digests**

Keep `FNV_OFFSET` unchanged at `1469598103934665603ULL`. Replace the 12 golden and 12 sim-invariant expected values with the values printed by the final candidate. Add a short reason beside the baseline arrays only where an existing baseline explanation already exists.

- [ ] **Step 4: Re-run exact deterministic suites**

Expected: `12/12 configs match baseline` in both binaries.

- [ ] **Step 5: Commit**

```bash
git add ocean/osrs/encounters/colosseum/encounter_colosseum_render_snapshot.inc ocean/osrs/tests/test_colosseum_golden.c ocean/osrs/tests/probe_colo_sim_invariant.c
git commit -m "Finalize consolidated Colosseum contract"
```

---

### Task 7: Verify run38 and integrate

**Files:**
- No source changes unless a real verification failure identifies a defect.

**Interfaces:**
- Consumes: final consolidation branch.
- Produces: verified canonical `valtteri/colo-5c-port` and no temporary worktree.

- [ ] **Step 1: Run focused shared and encounter suites**

Compile and run:

```text
ocean/osrs/tests/test_colosseum_modifiers.c
ocean/osrs/tests/test_colosseum_golden.c
ocean/osrs/tests/probe_colo_sim_invariant.c
ocean/osrs/tests/test_colosseum_forecast_exact.c
ocean/osrs/tests/test_encounter_route_topology_setup.c
ocean/osrs/tests/test_osrs_inventory_actions.c
ocean/osrs/tests/test_osrs_inventory_clicks.c
ocean/osrs/tests/test_osrs_item_effect_masks.c
ocean/osrs/tests/test_inferno_attack_styles.c
ocean/osrs/tests/test_inferno_lab.c
```

Expected: every suite passes with no unexplained changed digest.

- [ ] **Step 2: Run worker-count determinism**

```bash
clang -std=c11 -O2 -I. -Xpreprocessor -fopenmp \
  -I/opt/homebrew/opt/libomp/include \
  ocean/osrs/tests/probe_env_thread_determinism.c \
  -L/opt/homebrew/opt/libomp/lib -lomp -lm \
  -o /tmp/colo_worker_determinism
/tmp/colo_worker_determinism
```

Expected: every one-worker versus multi-worker comparison reports `IDENTICAL`.

- [ ] **Step 3: Build and smoke the local viewer**

```bash
./build.sh osrs_colosseum --local
./osrs_colosseum \
  --model checkpoints/colo-w11-protein-best/run38_378535936.bin \
  --start-wave 11 \
  --hidden-size 512 \
  --num-layers 2
```

Expected: model loads, viewer reports observations `933` and mask `451`, and a wave-11 episode advances without aborting.

- [ ] **Step 4: Commit any verification-driven fix separately**

Only if a reproduced failure required a source correction. Re-run the exact failed check after the fix.

- [ ] **Step 5: Fast-forward canonical and remove isolation**

From the canonical checkout:

```bash
pwd
git merge --ff-only valtteri/colo-consolidation
git worktree remove /private/tmp/puffer-colo-consolidation
git branch -d valtteri/colo-consolidation
```

Expected: `valtteri/colo-5c-port` points at the verified consolidation tip, the temporary path is gone, and existing untracked local artifacts remain untouched.
