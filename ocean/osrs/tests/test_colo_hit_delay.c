#include <stdio.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"
#include "ocean/osrs/tests/osrs_test_check.h"

/** Section 8 of osrs-engine-quirks, transcribed independently of the sim.
    Player-on-NPC projectiles land one tick later than the base table. */
static int section8_player_hit_delay(AttackStyle style, int distance) {
    switch (style) {
        case ATTACK_STYLE_MELEE:
            return 0;
        case ATTACK_STYLE_RANGED:
            return 1 + (3 + distance) / 6 + 1;
        case ATTACK_STYLE_MAGIC:
            return 1 + (1 + distance) / 3 + 1;
        default:
            break;
    }
    abort();
}

static void clear_npcs(ColosseumState* s) {
    memset(s->npcs, 0, sizeof(s->npcs));
    memset(s->npc_collision_flags, 0, sizeof(s->npc_collision_flags));
    memset(s->totems, 0, sizeof(s->totems));
    memset(s->bees, 0, sizeof(s->bees));
    col_rebuild_player_collision_flags(s);
}

static void init_state(ColosseumState* s, ColosseumContext* ctx, int px, int py) {
    col_init_context_typed(ctx);
    ctx->config.late_start_state_mode = 0;
    col_finalize_route_topology(ctx);
    memset(s, 0, sizeof(*s));
    col_reset_ctx((EncounterState*)s, (EncounterContext*)ctx, 4141u);
    clear_npcs(s);
    s->modifiers.draft_pending = 0;
    s->player.x = px;
    s->player.y = py;
    col_rebuild_player_collision_flags(s);
}

static ColoWeaponSet weapon_set_for_style(AttackStyle style) {
    switch (style) {
        case ATTACK_STYLE_MELEE:  return COLO_GEAR_MELEE;
        case ATTACK_STYLE_RANGED: return COLO_GEAR_RANGED;
        case ATTACK_STYLE_MAGIC:  return COLO_GEAR_MAGIC;
        default: break;
    }
    abort();
}

typedef struct {
    int distance;
    int queued_ticks;
    int check_prayer;
    int hit_count;
} QueuedSwing;

/** Fires one real player swing at an NPC placed `offset` tiles east of the
    player and reports what the sim actually queued on that NPC. */
static QueuedSwing swing_at_offset(AttackStyle style, int offset) {
    ColosseumState s;
    ColosseumContext ctx;
    int px = 16;
    int py = 16;
    init_state(&s, &ctx, px, py);
    col_apply_weapon_set(&s, weapon_set_for_style(style));
    col_init_npc(&s, 0, COLO_FREMENNIK_ARCHER, px + offset, py);

    ColoNPC* npc = &s.npcs[0];
    int distance = encounter_projectile_distance(
        s.player.x, s.player.y, 1,
        npc->x, npc->y, col_npc_effective_size(npc),
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);

    col_player_attack_target_ctx(&s, &ctx, 0);

    QueuedSwing out = {
        .distance = distance,
        .queued_ticks = npc->pending_hits.count > 0
            ? npc->pending_hits.hits[0].ticks_remaining
            : -1,
        .check_prayer = npc->pending_hits.count > 0
            ? npc->pending_hits.hits[0].check_prayer
            : -1,
        .hit_count = npc->pending_hits.count,
    };
    return out;
}

static void test_timing_helper_matches_section8(void) {
    const AttackStyle styles[3] = {
        ATTACK_STYLE_MELEE, ATTACK_STYLE_RANGED, ATTACK_STYLE_MAGIC
    };
    const uint8_t weapons[3] = {
        ITEM_SCYTHE_OF_VITUR, ITEM_TWISTED_BOW, ITEM_TUMEKENS_SHADOW
    };
    const char* labels[3] = { "melee", "ranged", "magic" };

    for (int i = 0; i < 3; i++) {
        for (int d = 0; d <= 12; d++) {
            char label[96];
            snprintf(label, sizeof(label),
                "col_player_projectile_timing %s d=%d matches section 8",
                labels[i], d);
            EncounterProjectileTiming timing =
                col_player_projectile_timing(styles[i], weapons[i], d);
            ASSERT_INT_EQ(label, timing.damage_delay_ticks,
                section8_player_hit_delay(styles[i], d));
        }
    }
}

static void test_queued_delay_matches_section8(void) {
    const AttackStyle styles[3] = {
        ATTACK_STYLE_MELEE, ATTACK_STYLE_RANGED, ATTACK_STYLE_MAGIC
    };
    const char* labels[3] = { "melee", "ranged", "magic" };
    const int melee_offsets[1] = { 1 };
    const int ranged_offsets[6] = { 1, 2, 3, 6, 9, 10 };

    for (int i = 0; i < 3; i++) {
        const int* offsets = styles[i] == ATTACK_STYLE_MELEE
            ? melee_offsets : ranged_offsets;
        int n = styles[i] == ATTACK_STYLE_MELEE ? 1 : 6;
        for (int k = 0; k < n; k++) {
            QueuedSwing swing = swing_at_offset(styles[i], offsets[k]);
            char label[128];
            snprintf(label, sizeof(label),
                "queued %s hit at d=%d lands on the section 8 tick",
                labels[i], swing.distance);
            ASSERT_INT_EQ(label, swing.queued_ticks,
                section8_player_hit_delay(styles[i], swing.distance));

            snprintf(label, sizeof(label),
                "queued %s hit at d=%d resolves prayer at throw, not on landing",
                labels[i], swing.distance);
            ASSERT_INT_EQ(label, swing.check_prayer, 0);
        }
    }
}

static void test_magic_delay_grows_with_distance(void) {
    QueuedSwing near = swing_at_offset(ATTACK_STYLE_MAGIC, 1);
    QueuedSwing far = swing_at_offset(ATTACK_STYLE_MAGIC, 9);
    CHECK("magic hit delay grows with distance rather than staying flat",
        far.queued_ticks > near.queued_ticks);
    CHECK("magic swings still queue exactly one splat",
        near.hit_count == 1 && far.hit_count == 1);
}

int main(void) {
    printf("colosseum player hit delay\n");
    test_timing_helper_matches_section8();
    test_queued_delay_matches_section8();
    test_magic_delay_grows_with_distance();
    return osrs_test_summary();
}
