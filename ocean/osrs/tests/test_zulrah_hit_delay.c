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

static ZulrahState* fresh_state(uint32_t seed) {
    EncounterState* state = (EncounterState*)&g_state;
    EncounterContext* context = (EncounterContext*)&g_ctx;
    ENCOUNTER_ZULRAH.init_context(context);
    ENCOUNTER_ZULRAH.init_state(state, context);
    ENCOUNTER_ZULRAH.put_int(state, context, "gear_tier", 0);
    ENCOUNTER_ZULRAH.put_int(state, context, "gear_tier_mode", ZUL_GEAR_TIER_FIXED);
    ENCOUNTER_ZULRAH.put_int(state, context, "episode_mode", ZUL_EPISODE_SINGLE_KILL);
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
        zul_resolve_player_pending_hits(s);
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

/** The canonical contract draws damage FIRST. Under the old inverted order the
    accuracy roll consumed draw 1, so the queued damage could not match. */
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

/** Prayer decides whether damage is frozen to zero, never how many draws the
    attack consumes. This is what the skipped-roll defect broke. */
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
        for (int t = 0; t < 8; t++) zul_resolve_player_pending_hits(s);

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
        for (int t = 0; t < 8; t++) zul_resolve_player_pending_hits(s);

        ASSERT_INT_EQ("dropping prayer after the throw does not revive the hit",
            s->player.current_hitpoints, s->player.base_hitpoints);
    }
}

/** Defect 3 mirror image: the melee stare froze its protect check at landing,
    three ticks after Zulrah committed to the attack. */
static void test_melee_stare_reads_prayer_at_calculation(void) {
    printf("\n--- melee stare reads prayer at the calculation tick ---\n");

    {
        ZulrahState* s = fresh_state(0x9A9A0001u);
        s->zulrah.x = 10; s->zulrah.y = 20;
        s->player.x = 10; s->player.y = 18;
        s->player.current_hitpoints = s->player.base_hitpoints;
        s->player.prayer = PRAYER_NONE;

        zul_melee_start(s);
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

        zul_melee_start(s);
        s->player.prayer = PRAYER_NONE;
        zul_melee_hit(s);

        ASSERT_INT_EQ("dropping melee prayer after the stare began stays blocked",
            s->player.current_hitpoints, s->player.base_hitpoints);
    }
}

int main(void) {
    printf("zulrah hit-delay and roll-order regressions\n\n");

    test_npc_hit_delay_by_distance();
    test_player_hit_delay_by_distance();
    test_damage_drawn_before_accuracy();
    test_prayer_does_not_perturb_the_rng_stream();
    test_prayer_resolves_at_throw_not_landing();
    test_melee_stare_reads_prayer_at_calculation();

    return osrs_test_summary();
}
