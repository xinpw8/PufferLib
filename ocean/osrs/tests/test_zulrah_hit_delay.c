#include <stdio.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_zulrah.h"
#include "ocean/osrs/tests/osrs_test_check.h"

/** osrs-engine-quirks section 8, transcribed from the spec table rather than
    from the sim. is_player adds the NPC-processes-first tick. */
static int spec8_ranged_delay(int distance, int is_player) {
    return 1 + (3 + distance) / 6 + (is_player ? 1 : 0);
}

static int spec8_thrown_delay(int distance, int is_player) {
    return 1 + distance / 6 + (is_player ? 1 : 0);
}

static int spec8_magic_delay(int distance, int is_player) {
    return 1 + (1 + distance) / 3 + (is_player ? 1 : 0);
}

static ZulrahContext g_ctx;
static ZulrahState g_state;
static CollisionMap* g_collision_map;


static ZulrahState* fresh_state(uint32_t seed) {
    EncounterState* state = (EncounterState*)&g_state;
    EncounterContext* context = (EncounterContext*)&g_ctx;
    ENCOUNTER_ZULRAH.init_context(context);
    ENCOUNTER_ZULRAH.init_state(state, context);
    if (!g_collision_map)
        g_collision_map = collision_map_load("ocean/osrs/data/zulrah.cmap");
    if (!g_collision_map) abort();
    ENCOUNTER_ZULRAH.put_ptr(state, context, "collision_map", g_collision_map);
    ENCOUNTER_ZULRAH.put_int(state, context, "world_offset_x", 2256);
    ENCOUNTER_ZULRAH.put_int(state, context, "world_offset_y", 3061);

    ENCOUNTER_ZULRAH.put_int(state, context, "gear_tier", 0);
    ENCOUNTER_ZULRAH.put_int(state, context, "gear_tier_mode", ZUL_GEAR_TIER_FIXED);
    ENCOUNTER_ZULRAH.put_int(state, context, "episode_mode", ZUL_EPISODE_SINGLE_KILL);
    ENCOUNTER_ZULRAH.finalize_context(state, context);
    ENCOUNTER_ZULRAH.reset(state, context, seed);
    return &g_state;
}

/** Places the player `distance` tiles (edge to edge) due south of Zulrah. */
static void place_at_distance(ZulrahState* s, int distance) {
    s->zulrah.x = 10;
    s->zulrah.y = 20;
    s->player.x = 10;
    s->player.y = s->zulrah.y - distance;
    int actual = encounter_projectile_distance(
        s->zulrah.x, s->zulrah.y, ZUL_NPC_SIZE,
        s->player.x, s->player.y, 1,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    if (actual != distance) {
        printf("  SETUP ERROR: wanted distance %d, got %d\n", distance, actual);
        tests_failed++;
    }
}

/** Fires one Zulrah attack and returns the resolve pass on which the player's
    hitpoints actually drop. 0 means it never landed within the budget. */
static int ticks_until_player_hit(ZulrahState* s, AttackStyle style) {
    s->player.prayer = PRAYER_NONE;
    s->player.current_hitpoints = s->player.base_hitpoints;

    if (style == ATTACK_STYLE_RANGED) zul_attack_ranged(s);
    else zul_attack_magic(s);

    if (encounter_pending_hit_queue_damage_sum(&s->player_pending_hits) <= 0)
        return -1;

    for (int t = 1; t <= 16; t++) {
        int before = s->player.current_hitpoints;
        encounter_resolve_player_pending_hits_observed(
            &s->player_pending_hits, &s->player, s->player.prayer,
            &s->damage_received_this_tick, NULL, NULL,
            zul_player_hit_landed, s);
        if (s->player.current_hitpoints < before) return t;
    }
    return 0;
}

static int ticks_until_zulrah_hit(ZulrahState* s, AttackStyle style, int damage) {
    s->zulrah.current_hitpoints = 500;
    zul_queue_zulrah_hit(s, damage, style, 0);

    for (int t = 1; t <= 16; t++) {
        int before = s->zulrah.current_hitpoints;
        zul_resolve_zulrah_pending_hits(s);
        if (s->zulrah.current_hitpoints < before) return t;
    }
    return 0;
}

static void test_npc_hit_delay_by_distance(void) {
    printf("--- Zulrah -> player hit delay matches section 8 ---\n");

    static const int DISTANCES[] = { 1, 2, 3, 5, 6, 8, 9, 10 };
    const int n = (int)(sizeof(DISTANCES) / sizeof(DISTANCES[0]));

    for (int i = 0; i < n; i++) {
        int d = DISTANCES[i];
        int expected = spec8_ranged_delay(d, 0);

        int landed = 0;
        for (int attempt = 0; attempt < 64 && !landed; attempt++) {
            ZulrahState* s = fresh_state(0x5EED0000u + (uint32_t)(d * 64 + attempt));
            place_at_distance(s, d);
            int got = ticks_until_player_hit(s, ATTACK_STYLE_RANGED);
            if (got < 0) continue;
            landed = 1;
            char label[96];
            snprintf(label, sizeof(label), "ranged delay at distance %d", d);
            ASSERT_INT_EQ(label, got, expected);
        }
        CHECK("ranged attack eventually connects for the delay probe", landed);
    }

    for (int i = 0; i < n; i++) {
        int d = DISTANCES[i];
        int expected = spec8_magic_delay(d, 0);

        int landed = 0;
        for (int attempt = 0; attempt < 64 && !landed; attempt++) {
            ZulrahState* s = fresh_state(0x71D30000u + (uint32_t)(d * 64 + attempt));
            place_at_distance(s, d);
            int got = ticks_until_player_hit(s, ATTACK_STYLE_MAGIC);
            if (got < 0) continue;
            landed = 1;
            char label[96];
            snprintf(label, sizeof(label), "magic delay at distance %d", d);
            ASSERT_INT_EQ(label, got, expected);
        }
        CHECK("magic attack eventually connects for the delay probe", landed);
    }
}

static void test_player_hit_delay_by_distance(void) {
    printf("\n--- player -> Zulrah hit delay carries the +1 ---\n");

    static const int DISTANCES[] = { 1, 2, 3, 5, 6, 8, 9, 10 };
    const int n = (int)(sizeof(DISTANCES) / sizeof(DISTANCES[0]));

    for (int i = 0; i < n; i++) {
        int d = DISTANCES[i];
        ZulrahState* s = fresh_state(0x11110000u + (uint32_t)d);
        place_at_distance(s, d);

        uint8_t weapon = s->player.equipped[GEAR_SLOT_WEAPON];
        int is_thrown = (weapon == ITEM_TOXIC_BLOWPIPE);

        int expected_ranged = is_thrown
            ? spec8_thrown_delay(d, 1)
            : spec8_ranged_delay(d, 1);
        char label[96];
        snprintf(label, sizeof(label), "player ranged delay at distance %d", d);
        ASSERT_INT_EQ(label,
            ticks_until_zulrah_hit(s, ATTACK_STYLE_RANGED, 7), expected_ranged);
    }

    for (int i = 0; i < n; i++) {
        int d = DISTANCES[i];
        ZulrahState* s = fresh_state(0x22220000u + (uint32_t)d);
        place_at_distance(s, d);
        s->player.equipped[GEAR_SLOT_WEAPON] = ITEM_SANGUINESTI_STAFF;

        char label[96];
        snprintf(label, sizeof(label), "player magic delay at distance %d", d);
        ASSERT_INT_EQ(label,
            ticks_until_zulrah_hit(s, ATTACK_STYLE_MAGIC, 7),
            spec8_magic_delay(d, 1));
    }
}

static void test_damage_drawn_before_accuracy(void) {
    printf("\n--- damage roll precedes the accuracy roll ---\n");

    int hits_checked = 0;
    for (uint32_t seed = 0; seed < 400 && hits_checked < 40; seed++) {
        ZulrahState* s = fresh_state(0x3C3C0000u + seed);
        place_at_distance(s, 5);
        s->player.prayer = PRAYER_NONE;
        encounter_pending_hit_queue_clear(&s->player_pending_hits);

        uint32_t probe = s->rng_state;
        int first_draw = encounter_rand_int(
            &probe, MONSTER_DATABASE[MON_ZULRAH_GREEN].max_hit + 1);

        zul_attack_ranged(s);

        int queued = encounter_pending_hit_queue_damage_sum(&s->player_pending_hits);
        if (queued <= 0) continue;
        hits_checked++;
        ASSERT_INT_EQ("ranged queued damage equals the first RNG draw",
            queued, first_draw);
    }
    CHECK("collected ranged hits for the draw-order probe", hits_checked > 0);

    int magic_checked = 0;
    for (uint32_t seed = 0; seed < 400 && magic_checked < 40; seed++) {
        ZulrahState* s = fresh_state(0x4D4D0000u + seed);
        place_at_distance(s, 5);
        s->player.prayer = PRAYER_NONE;
        encounter_pending_hit_queue_clear(&s->player_pending_hits);

        uint32_t probe = s->rng_state;
        int first_draw = encounter_rand_int(
            &probe, MONSTER_DATABASE[MON_ZULRAH_BLUE].max_hit + 1);

        zul_attack_magic(s);

        int queued = encounter_pending_hit_queue_damage_sum(&s->player_pending_hits);
        if (queued <= 0) continue;
        magic_checked++;
        ASSERT_INT_EQ("magic queued damage equals the first RNG draw",
            queued, first_draw);
    }
    CHECK("collected magic hits for the draw-order probe", magic_checked > 0);
}

static void test_prayer_does_not_perturb_the_rng_stream(void) {
    printf("\n--- prayer state leaves the RNG stream untouched ---\n");

    for (uint32_t seed = 0; seed < 32; seed++) {
        ZulrahState* a = fresh_state(0x5A5A0000u + seed);
        place_at_distance(a, 5);
        a->player.prayer = PRAYER_NONE;
        uint32_t rng_before = a->rng_state;
        zul_attack_ranged(a);
        uint32_t unprayed_rng = a->rng_state;

        ZulrahState* b = fresh_state(0x5A5A0000u + seed);
        place_at_distance(b, 5);
        b->player.prayer = PRAYER_PROTECT_RANGED;
        b->rng_state = rng_before;
        zul_attack_ranged(b);

        ASSERT_INT_EQ("ranged: prayer does not change RNG advancement",
            (int)b->rng_state, (int)unprayed_rng);
        ASSERT_INT_EQ("ranged: correct prayer freezes damage to zero",
            encounter_pending_hit_queue_damage_sum(&b->player_pending_hits), 0);
    }

    for (uint32_t seed = 0; seed < 32; seed++) {
        ZulrahState* a = fresh_state(0x6B6B0000u + seed);
        place_at_distance(a, 5);
        a->player.prayer = PRAYER_NONE;
        uint32_t rng_before = a->rng_state;
        zul_attack_magic(a);
        uint32_t unprayed_rng = a->rng_state;

        ZulrahState* b = fresh_state(0x6B6B0000u + seed);
        place_at_distance(b, 5);
        b->player.prayer = PRAYER_PROTECT_MAGIC;
        b->rng_state = rng_before;
        zul_attack_magic(b);

        ASSERT_INT_EQ("magic: prayer does not change RNG advancement",
            (int)b->rng_state, (int)unprayed_rng);
        ASSERT_INT_EQ("magic: correct prayer freezes damage to zero",
            encounter_pending_hit_queue_damage_sum(&b->player_pending_hits), 0);
    }
}

/** Section 11: the protect check belongs to the tick the attack is calculated.
    Flicking on after the throw must not save the player, and flicking off after
    the throw must not doom one already blocked. */
static void test_prayer_resolves_at_throw_not_landing(void) {
    printf("\n--- protect prayer resolves at the throw tick ---\n");

    int late_on_checked = 0;
    for (uint32_t seed = 0; seed < 200 && late_on_checked < 20; seed++) {
        ZulrahState* s = fresh_state(0x7E7E0000u + seed);
        place_at_distance(s, 5);
        s->player.prayer = PRAYER_NONE;
        s->player.current_hitpoints = s->player.base_hitpoints;
        encounter_pending_hit_queue_clear(&s->player_pending_hits);

        zul_attack_ranged(s);
        if (encounter_pending_hit_queue_damage_sum(&s->player_pending_hits) <= 0)
            continue;
        late_on_checked++;

        s->player.prayer = PRAYER_PROTECT_RANGED;
        for (int t = 0; t < 8; t++)
            encounter_resolve_player_pending_hits_observed(
                &s->player_pending_hits, &s->player, s->player.prayer,
                &s->damage_received_this_tick, NULL, NULL,
                zul_player_hit_landed, s);

        CHECK("praying after the throw does not cancel the hit",
            s->player.current_hitpoints < s->player.base_hitpoints);
    }
    CHECK("collected throws for the late-prayer probe", late_on_checked > 0);

    for (uint32_t seed = 0; seed < 32; seed++) {
        ZulrahState* s = fresh_state(0x8F8F0000u + seed);
        place_at_distance(s, 5);
        s->player.prayer = PRAYER_PROTECT_RANGED;
        s->player.current_hitpoints = s->player.base_hitpoints;
        encounter_pending_hit_queue_clear(&s->player_pending_hits);

        zul_attack_ranged(s);
        s->player.prayer = PRAYER_NONE;
        for (int t = 0; t < 8; t++)
            encounter_resolve_player_pending_hits_observed(
                &s->player_pending_hits, &s->player, s->player.prayer,
                &s->damage_received_this_tick, NULL, NULL,
                zul_player_hit_landed, s);

        ASSERT_INT_EQ("dropping prayer after the throw does not revive the hit",
            s->player.current_hitpoints, s->player.base_hitpoints);
    }
}

static void test_melee_stare_reads_prayer_at_calculation(void) {
    printf("\n--- melee stare reads prayer at the calculation tick ---\n");

    {
        ZulrahState* s = fresh_state(0x9A9A0001u);
        s->zulrah.x = 10; s->zulrah.y = 20;
        s->player.x = 10; s->player.y = 18;
        s->player.current_hitpoints = s->player.base_hitpoints;
        s->player.prayer = PRAYER_NONE;

        zul_fire_action(s, &g_ctx, ZA_MELEE);
        s->player.prayer = PRAYER_PROTECT_MELEE;
        zul_melee_hit(s);

        CHECK("praying melee after the stare began still takes damage",
            s->player.current_hitpoints < s->player.base_hitpoints);
    }

    {
        ZulrahState* s = fresh_state(0x9A9A0002u);
        s->zulrah.x = 10; s->zulrah.y = 20;
        s->player.x = 10; s->player.y = 18;
        s->player.current_hitpoints = s->player.base_hitpoints;
        s->player.prayer = PRAYER_PROTECT_MELEE;

        zul_fire_action(s, &g_ctx, ZA_MELEE);
        s->player.prayer = PRAYER_NONE;
        zul_melee_hit(s);

        ASSERT_INT_EQ("dropping melee prayer after the stare began stays blocked",
            s->player.current_hitpoints, s->player.base_hitpoints);
    }
}

static void test_topology_geometry_parity(void) {
    printf("\n--- Zulrah static topology matches the arena collision map ---\n");
    ZulrahState* s = fresh_state(0x7a110001u);
    (void)s;
    const EncounterArenaTopology* topology = g_ctx.route_topology;
    CHECK("Zulrah topology covers the 28 by 28 local arena",
        topology->origin_x == 0 && topology->origin_y == 0 &&
        topology->width == ZUL_ARENA_SIZE &&
        topology->height == ZUL_ARENA_SIZE);
    CHECK("Zulrah pillar lines preserve the encounter's open LOS rule",
        topology->static_los_mode == ENCOUNTER_ARENA_TOPOLOGY_LOS_OPEN);

    int open_tiles = 0;
    int step_checks = 0;
    for (int x = 0; x < ZUL_ARENA_SIZE; x++) {
        for (int y = 0; y < ZUL_ARENA_SIZE; y++) {
            int expected_walkable = collision_tile_walkable(
                g_collision_map, 0, x + 2256, y + 3061);
            CHECK("Zulrah player tile parity",
                !encounter_arena_topology_tile_blocked(topology, x, y) ==
                    expected_walkable);
            if (!expected_walkable) continue;
            open_tiles++;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int expected_step = collision_traversable_step(
                        g_collision_map, 0,
                        x + 2256, y + 3061, dx, dy);
                    CHECK("Zulrah player step parity",
                        encounter_arena_topology_step_allowed(
                            topology, x, y, 1, dx, dy) == expected_step);
                    step_checks++;
                }
            }
        }
    }
    ASSERT_INT_EQ("Zulrah collision map has 69 allowed player tiles",
        open_tiles, 69);
    ASSERT_INT_EQ("Zulrah checks every direction from every allowed tile",
        step_checks, 69 * 8);

    int phase_targets = 0;
    for (int rotation = 0; rotation < ZUL_NUM_ROTATIONS; rotation++) {
        for (int phase_index = 0;
                phase_index < ZUL_ROT_LENGTHS[rotation];
                phase_index++) {
            const ZulRotationPhase* phase =
                &ZUL_ROTATIONS[rotation][phase_index];
            int target_x = ZUL_POSITIONS[phase->position][0];
            int target_y = ZUL_POSITIONS[phase->position][1];
            CHECK("every Zulrah phase target footprint stays in topology",
                encounter_arena_topology_contains(
                    topology, target_x, target_y) &&
                encounter_arena_topology_contains(
                    topology,
                    target_x + ZUL_NPC_SIZE - 1,
                    target_y + ZUL_NPC_SIZE - 1));
            CHECK("every Zulrah phase stand tile is allowed",
                !encounter_arena_topology_tile_blocked(
                    topology,
                    ZUL_STAND_COORDS[phase->stand][0],
                    ZUL_STAND_COORDS[phase->stand][1]));
            if (phase->stall != ZUL_STAND_NONE) {
                CHECK("every Zulrah phase stall tile is allowed",
                    !encounter_arena_topology_tile_blocked(
                        topology,
                        ZUL_STAND_COORDS[phase->stall][0],
                        ZUL_STAND_COORDS[phase->stall][1]));
            }
            phase_targets++;
        }
    }
    ASSERT_INT_EQ("all 47 Zulrah phase targets are pinned",
        phase_targets, 47);
}

static void test_unified_player_contract_dimensions(void) {
    printf("\n--- unified player contract dimensions ---\n");
    CHECK("Zulrah uses shared action heads",
        ZUL_NUM_ACTION_HEADS == OSRS_BASE_NUM_ACTION_HEADS);
    CHECK("Zulrah primary head uses shared target layout",
        ZUL_ACTION_HEAD_DIMS[OSRS_HEAD_PRIMARY] ==
            OSRS_PRIMARY_DIM(ZUL_OBS_NPC_SLOTS));
    CHECK("Zulrah overhead head uses shared dimension",
        ZUL_ACTION_HEAD_DIMS[OSRS_HEAD_OVERHEAD] == OSRS_OVERHEAD_DIM);
    CHECK("Zulrah spell head uses shared dimension",
        ZUL_ACTION_HEAD_DIMS[OSRS_HEAD_SPELL] == OSRS_SPELL_DIM);
    CHECK("Zulrah observation uses shared prefix",
        ZUL_NUM_OBS == OSRS_SHARED_OBS_SIZE + 104);
    CHECK("Zulrah action mask uses shared dimensions",
        ZUL_ACTION_MASK_SIZE == OSRS_BASE_ACTION_MASK_SIZE(ZUL_OBS_NPC_SLOTS));
}
static void test_unified_player_contract_semantics(void) {
    printf("\n--- unified player contract semantics ---\n");
    ZulrahState* state = fresh_state(0xA11CE001u);
    state->player_stunned_ticks = 3;
    float observation[ZUL_NUM_OBS];
    float mask[ZUL_ACTION_MASK_SIZE];
    zul_write_obs(
        (EncounterState*)state, (EncounterContext*)&g_ctx, observation);
    zul_write_mask(
        (EncounterState*)state, (EncounterContext*)&g_ctx, mask);
    unsigned char byte_mask[ZUL_ACTION_MASK_SIZE];
    zul_write_mask_bytes(
        (EncounterState*)state, (EncounterContext*)&g_ctx, byte_mask);
    for (int i = 0; i < ZUL_ACTION_MASK_SIZE; i++)
        CHECK("Zulrah byte mask matches the float contract",
            (float)byte_mask[i] == mask[i]);

    ASSERT_FLOAT_NEAR("shared hitpoints lead Zulrah observation",
        observation[0],
        (float)state->player.current_hitpoints /
            (float)state->player.base_hitpoints,
        0.0f);
    ASSERT_FLOAT_NEAR("shared inventory code is in Zulrah observation",
        observation[OSRS_SHARED_OBS_INVENTORY_START],
        osrs_inventory_cell_obs_code_encode(
            state->player.inventory_cells[0].content_code),
        0.0f);
    ASSERT_FLOAT_NEAR("encounter observation exposes melee stun duration",
        observation[ZUL_OBS_AFTER_SHARED],
        3.0f / ZUL_MELEE_STUN_TICKS,
        0.0f);
    ASSERT_FLOAT_NEAR("Zulrah state follows melee stun",
        observation[ZUL_OBS_AFTER_SHARED + 1],
        (float)state->zulrah.current_hitpoints /
            (float)MONSTER_DATABASE[MON_ZULRAH_GREEN].hp,
        0.0f);

    int spell_offset =
        osrs_base_action_head_mask_offset(ZUL_OBS_NPC_SLOTS, OSRS_HEAD_SPELL);
    ASSERT_FLOAT_NEAR("Zulrah no-spell action stays legal",
        mask[spell_offset], 1.0f, 0.0f);
    for (int spell = 1; spell < OSRS_SPELL_DIM; spell++) {
        ASSERT_FLOAT_NEAR("unsupported Zulrah spells stay masked",
            mask[spell_offset + spell], 0.0f, 0.0f);
    }
}


static void test_weaponless_player_observation_stays_defined(void) {
    ZulrahState* state = fresh_state(0xDEADu);
    state->player.equipped[GEAR_SLOT_WEAPON] = ITEM_NONE;
    zul_mark_live_stats_dirty(state);

    ASSERT_INT_EQ("weaponless player keeps the ranged stats contract",
        zul_player_equipped_attack_style(state), ATTACK_STYLE_RANGED);
    float observation[ZUL_NUM_OBS];
    ENCOUNTER_ZULRAH.write_obs(
        (EncounterState*)state, (EncounterContext*)&g_ctx, observation);
    CHECK("weaponless player observation remains defined", observation[0] >= 0.0f);
}


int main(void) {
    printf("zulrah hit-delay and roll-order regressions\n\n");
    test_unified_player_contract_dimensions();
    test_unified_player_contract_semantics();
    test_weaponless_player_observation_stays_defined();

    test_npc_hit_delay_by_distance();
    test_player_hit_delay_by_distance();
    test_damage_drawn_before_accuracy();
    test_prayer_does_not_perturb_the_rng_stream();
    test_prayer_resolves_at_throw_not_landing();
    test_topology_geometry_parity();
    test_melee_stare_reads_prayer_at_calculation();

    return osrs_test_summary();
}
