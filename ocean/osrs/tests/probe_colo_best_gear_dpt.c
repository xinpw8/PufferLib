#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"

#include "ocean/osrs/tests/osrs_test_check.h"

static void loadout_reset(ColosseumState* s, ColosseumContext* ctx, int mode,
                          float frac, uint32_t seed) {
    col_init_context_typed(ctx);
    ctx->config.loadout_profile_mode = mode;
    ctx->config.beginner_loadout_fraction = frac;
    col_finalize_route_topology(ctx);
    memset(s, 0, sizeof(*s));
    col_reset_ctx((EncounterState*)s, (EncounterContext*)ctx, seed);
}

static int setup_contains(const uint8_t setup[NUM_GEAR_SLOTS], uint8_t item) {
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
        if (setup[slot] == item) return 1;
    return 0;
}

static void test_argmax_setup_and_style(void) {
    printf("test_argmax_setup_and_style (T1, T3)\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 771);
    col_build_npc_stats();
    s.player.current_attack = 118;
    s.player.current_strength = 118;
    s.player.current_ranged = 112;
    s.player.current_magic = 99;
    col_mark_live_loadout_dirty(&s);

    ColoBestGear best[COLO_NUM_WEAPON_SETS][COLO_NUM_NPC_TYPES];
    col_build_best_gear_table(&s, best);

    const ColoBestGear* bm = &best[COLO_GEAR_MAGIC][COLO_FREMENNIK_BERSERKER];
    CHECK("T1 magic best weapon is Tumeken's shadow",
        bm->setup[GEAR_SLOT_WEAPON] == ITEM_TUMEKENS_SHADOW);
    CHECK("T1 magic best setup includes Occult necklace",
        setup_contains(bm->setup, ITEM_OCCULT_NECKLACE));
    CHECK("T1 magic best setup includes Confliction gauntlets",
        setup_contains(bm->setup, ITEM_CONFLICTION_GAUNTLETS));
    CHECK("T1 magic best setup includes Avernic treads",
        setup_contains(bm->setup, ITEM_AVERNIC_TREADS));

    const ColoBestGear* jm = &best[COLO_GEAR_MAGIC][COLO_JAGUAR_WARRIOR];
    ColoNPC jnpc = (ColoNPC){
        .type = COLO_JAGUAR_WARRIOR,
        .hp = COLO_NPC_STATS[COLO_JAGUAR_WARRIOR].hp,
        .max_hp = COLO_NPC_STATS[COLO_JAGUAR_WARRIOR].hp,
        .size = COLO_NPC_STATS[COLO_JAGUAR_WARRIOR].size,
        .active = 1, .death_ticks = 0,
    };
    float ref = col_expected_dpt_for_equipment_vs_npc(&s, jm->setup, &jnpc, 1);
    CHECK("T1 jaguar magic argmax DPT reproducible through the leaf (accuracy-weighted)",
        fabsf(ref - jm->dpt) < 1e-3f);
    CHECK("T1 jaguar magic argmax setup is the shadow kit",
        jm->setup[GEAR_SLOT_WEAPON] == ITEM_TUMEKENS_SHADOW &&
        setup_contains(jm->setup, ITEM_OCCULT_NECKLACE) &&
        setup_contains(jm->setup, ITEM_CONFLICTION_GAUNTLETS));

    struct { ColoNpcType type; int want_style; const char* name; } spec[] = {
        { COLO_FREMENNIK_BERSERKER, COLO_GEAR_MAGIC,  "berserker -> magic" },
        { COLO_FREMENNIK_ARCHER,    COLO_GEAR_MELEE,  "archer -> melee" },
        { COLO_FREMENNIK_SEER,      COLO_GEAR_RANGED, "seer -> ranged" },
        { COLO_SERPENT_SHAMAN,      COLO_GEAR_RANGED, "serpent -> ranged" },
    };
    for (int k = 0; k < (int)(sizeof(spec) / sizeof(spec[0])); k++) {
        int argmax = -1;
        float best_dpt = -1.0f;
        for (int style = 0; style < COLO_NUM_WEAPON_SETS; style++)
            if (best[style][spec[k].type].dpt > best_dpt) {
                best_dpt = best[style][spec[k].type].dpt;
                argmax = style;
            }
        char label[96];
        snprintf(label, sizeof(label), "T3 argmax style: %s", spec[k].name);
        CHECK(label, argmax == spec[k].want_style);
    }
}

static void test_beats_worn_single_swap(void) {
    printf("test_beats_worn_single_swap (T2)\n");
    int modes[] = { COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY,
                    COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY };
    for (int m = 0; m < 2; m++) {
        ColosseumContext ctx;
        ColosseumState s;
        loadout_reset(&s, &ctx, modes[m], 0.0f, 401 + m);
        col_build_npc_stats();
        s.player.current_attack = 118;
        s.player.current_strength = 118;
        s.player.current_ranged = 112;
        s.player.current_magic = 99;
        col_mark_live_loadout_dirty(&s);

        ColoBestGear best[COLO_NUM_WEAPON_SETS][COLO_NUM_NPC_TYPES];
        col_build_best_gear_table(&s, best);

        uint8_t weapons[64];
        int nweap = 0;
        uint8_t add_seen[256] = {0};
        for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
            uint8_t w = s.player.equipped[slot];
            if (w != ITEM_NONE && item_is_weapon(w) && !add_seen[w]) {
                add_seen[w] = 1; weapons[nweap++] = w;
            }
        }
        for (int cell = 0; cell < COLO_INVENTORY_DISPLAY_SLOTS; cell++) {
            uint8_t w =
                osrs_inventory_cell_item_index(&s.player.inventory_cells[cell]);
            if (w != ITEM_NONE && item_is_weapon(w) && !add_seen[w]) {
                add_seen[w] = 1; weapons[nweap++] = w;
            }
        }

        int violations = 0;
        for (int wi = 0; wi < nweap; wi++) {
            uint8_t w = weapons[wi];
            int wstyle = get_item_attack_style(w);
            if (wstyle < 1 || wstyle > 3) continue;
            ColoWeaponSet style_set = (ColoWeaponSet)(wstyle - 1);
            uint8_t worn[NUM_GEAR_SLOTS];
            memcpy(worn, s.player.equipped, NUM_GEAR_SLOTS);
            worn[GEAR_SLOT_WEAPON] = w;
            if (item_is_two_handed(w)) worn[GEAR_SLOT_SHIELD] = ITEM_NONE;
            for (int type = 0; type < COLO_NUM_NPC_TYPES; type++) {
                if (col_type_is_hazard_entity((ColoNpcType)type)) continue;
                ColoNPC npc = (ColoNPC){
                    .type = (ColoNpcType)type,
                    .hp = COLO_NPC_STATS[type].hp,
                    .max_hp = COLO_NPC_STATS[type].hp,
                    .size = COLO_NPC_STATS[type].size,
                    .active = 1, .death_ticks = 0,
                };
                float worn_dpt = col_expected_dpt_for_equipment_vs_npc(&s, worn, &npc, 1);
                if (best[style_set][type].dpt < worn_dpt - 1e-3f) violations++;
            }
        }
        char label[96];
        snprintf(label, sizeof(label), "T2 best-gear >= worn single-swap (profile %d)", m);
        CHECK(label, violations == 0);
    }
}

static void test_equip_and_augury_same_tick(void) {
    printf("test_equip_and_augury_same_tick (T4)\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 99);
    col_build_npc_stats();

    int shadow_cell = -1;
    for (int cell = 0; cell < COLO_INVENTORY_DISPLAY_SLOTS; cell++)
        if (osrs_inventory_cell_item_index(
                &s.player.inventory_cells[cell]) == ITEM_TUMEKENS_SHADOW) {
            shadow_cell = cell;
            break;
        }
    CHECK("T4 shadow is in the bag", shadow_cell >= 0);
    if (shadow_cell < 0) return;

    CHECK("T4 starts on a non-magic weapon",
        col_equipped_weapon_attack_style(&s.player) != ATTACK_STYLE_MAGIC);
    s.player.current_prayer = 99;

    static float mask[COLO_ACTION_MASK_SIZE];
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    int off_base = -1;
    {
        int offset = 0;
        for (int h = 0; h < COLO_HEAD_OFFENSIVE; h++) offset += COLO_ACTION_DIMS[h];
        off_base = offset;
    }
    int augury_bit = off_base + 4;
    CHECK("T4 mask allows Augury on a non-magic weapon (points-only)",
        mask[augury_bit] == 1.0f);

    int act[COLO_NUM_ACTION_HEADS] = {0};
    act[COLO_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)] = shadow_cell + 1;
    act[COLO_HEAD_OFFENSIVE] = 4;
    col_step_ctx((EncounterState*)&s, (EncounterContext*)&ctx, act);

    CHECK("T4 weapon equipped to shadow on the swap tick",
        s.player.equipped[GEAR_SLOT_WEAPON] == ITEM_TUMEKENS_SHADOW);
    CHECK("T4 offensive prayer is Augury after the swap tick",
        s.player.offensive_prayer == OFFENSIVE_PRAYER_AUGURY);

    const EncounterLoadoutStats* live = col_live_loadout_stats(&s);
    EncounterLoadoutStats no_aug;
    int cur_magic = s.player.current_magic;
    encounter_compute_loadout_stats(
        s.player.equipped, ATTACK_STYLE_MAGIC, OFFENSIVE_PRAYER_NONE,
        cur_magic, FIGHT_STYLE_ACCURATE,
        col_weapon_set_spell_base_damage(&s, COLO_GEAR_MAGIC), &no_aug);
    encounter_update_loadout_level(&no_aug, OFFENSIVE_PRAYER_NONE, cur_magic, cur_magic);
    CHECK("T4 live stats are the magic style (shadow equipped)",
        live->style == ATTACK_STYLE_MAGIC);
    CHECK("T4 Augury raises effective level over no-prayer",
        live->eff_level > no_aug.eff_level);
}

static void test_per_cell_marginal_bit(void) {
    printf("test_per_cell_marginal_bit (T5)\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 55);
    col_build_npc_stats();
    s.player.current_attack = 118;
    s.player.current_strength = 118;
    s.player.current_ranged = 112;
    s.player.current_magic = 99;
    col_mark_live_loadout_dirty(&s);

    memset(s.npcs, 0, sizeof(s.npcs));
    s.npcs[0] = (ColoNPC){
        .type = COLO_FREMENNIK_BERSERKER,
        .hp = COLO_NPC_STATS[COLO_FREMENNIK_BERSERKER].hp,
        .max_hp = COLO_NPC_STATS[COLO_FREMENNIK_BERSERKER].hp,
        .size = COLO_NPC_STATS[COLO_FREMENNIK_BERSERKER].size,
        .x = 18, .y = 18, .active = 1, .death_ticks = 0,
    };
    s.player.x = 18; s.player.y = 20;
    osrs_interaction_set(&s.interaction, 0);

    ColoBestGear best[COLO_NUM_WEAPON_SETS][COLO_NUM_NPC_TYPES];
    col_build_best_gear_table(&s, best);

    int magic_was_argmax = -1;
    {
        float bd = -1.0f;
        for (int style = 0; style < COLO_NUM_WEAPON_SETS; style++)
            if (best[style][COLO_FREMENNIK_BERSERKER].dpt > bd) {
                bd = best[style][COLO_FREMENNIK_BERSERKER].dpt;
                magic_was_argmax = style;
            }
    }
    CHECK("T5 magic is the argmax with shadow present", magic_was_argmax == COLO_GEAR_MAGIC);
    float magic_dpt_with = best[COLO_GEAR_MAGIC][COLO_FREMENNIK_BERSERKER].dpt;

    for (int cell = 0; cell < COLO_INVENTORY_DISPLAY_SLOTS; cell++)
        if (osrs_inventory_cell_item_index(
                &s.player.inventory_cells[cell]) == ITEM_TUMEKENS_SHADOW)
            s.player.inventory_cells[cell] = osrs_inventory_cell_empty();
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
        if (s.player.equipped[slot] == ITEM_TUMEKENS_SHADOW)
            s.player.equipped[slot] = ITEM_NONE;

    ColoBestGear best2[COLO_NUM_WEAPON_SETS][COLO_NUM_NPC_TYPES];
    col_build_best_gear_table(&s, best2);
    float magic_dpt_without = best2[COLO_GEAR_MAGIC][COLO_FREMENNIK_BERSERKER].dpt;
    CHECK("T5 removing shadow drops the magic best DPT",
        magic_dpt_without < magic_dpt_with - 1e-3f);
    int argmax_without = -1;
    {
        float bd = -1.0f;
        for (int style = 0; style < COLO_NUM_WEAPON_SETS; style++)
            if (best2[style][COLO_FREMENNIK_BERSERKER].dpt > bd) {
                bd = best2[style][COLO_FREMENNIK_BERSERKER].dpt;
                argmax_without = style;
            }
    }
    CHECK("T5 removing shadow flips the berserker argmax away from magic",
        argmax_without != COLO_GEAR_MAGIC);
}

static void test_memo_result_preserving(void) {
    printf("test_memo_result_preserving (T6)\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 313);
    col_build_npc_stats();
    s.player.current_attack = 118;
    s.player.current_strength = 118;
    s.player.current_ranged = 112;
    s.player.current_magic = 99;
    col_mark_live_loadout_dirty(&s);

    ColoBestGear best[COLO_NUM_WEAPON_SETS][COLO_NUM_NPC_TYPES];
    col_build_best_gear_table(&s, best);

    int mismatches = 0;
    for (int style = 0; style < COLO_NUM_WEAPON_SETS; style++) {
        for (int type = 0; type < COLO_NUM_NPC_TYPES; type++) {
            if (col_type_is_hazard_entity((ColoNpcType)type)) continue;
            const ColoBestGear* bg = &best[style][type];
            if (bg->dpt < 0.0f) continue;
            ColoNPC npc = (ColoNPC){
                .type = (ColoNpcType)type,
                .hp = COLO_NPC_STATS[type].hp,
                .max_hp = COLO_NPC_STATS[type].hp,
                .size = COLO_NPC_STATS[type].size,
                .active = 1, .death_ticks = 0,
            };
            float direct = col_expected_dpt_for_equipment_vs_npc(&s, bg->setup, &npc, 1);
            if (direct != bg->dpt) mismatches++;
        }
    }
    CHECK("T6 memoized oracle DPT bit-identical to un-memoized leaf", mismatches == 0);
}

static void test_confliction_reference(void) {
    printf("test_confliction_reference (T7)\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 717);
    col_build_npc_stats();
    s.player.current_magic = 99;
    col_mark_live_loadout_dirty(&s);

    ColoNPC npc = (ColoNPC){
        .type = COLO_JAGUAR_WARRIOR,
        .hp = COLO_NPC_STATS[COLO_JAGUAR_WARRIOR].hp,
        .max_hp = COLO_NPC_STATS[COLO_JAGUAR_WARRIOR].hp,
        .size = COLO_NPC_STATS[COLO_JAGUAR_WARRIOR].size,
        .active = 1, .death_ticks = 0,
    };

    uint8_t setup_1h[NUM_GEAR_SLOTS];
    memset(setup_1h, ITEM_NONE, NUM_GEAR_SLOTS);
    setup_1h[GEAR_SLOT_WEAPON] = ITEM_TRIDENT_OF_SWAMP;
    setup_1h[GEAR_SLOT_HANDS] = ITEM_CONFLICTION_GAUNTLETS;
    setup_1h[GEAR_SLOT_NECK] = ITEM_OCCULT_NECKLACE;

    uint8_t setup_1h_nocon[NUM_GEAR_SLOTS];
    memcpy(setup_1h_nocon, setup_1h, NUM_GEAR_SLOTS);
    setup_1h_nocon[GEAR_SLOT_HANDS] = ITEM_NONE;

    float dpt_con = col_expected_dpt_for_equipment_vs_npc(&s, setup_1h, &npc, 1);
    float dpt_nocon = col_expected_dpt_for_equipment_vs_npc(&s, setup_1h_nocon, &npc, 1);

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        setup_1h, ATTACK_STYLE_MAGIC, OFFENSIVE_PRAYER_AUGURY, 99, FIGHT_STYLE_ACCURATE,
        col_weapon_set_spell_base_damage(&s, COLO_GEAR_MAGIC), &stats);
    encounter_update_loadout_level(&stats, OFFENSIVE_PRAYER_AUGURY, 99, 99);
    OsrsEquipmentEffectProfile effects;
    encounter_derive_loadout_effect_profile(setup_1h, &effects);
    CHECK("T7 confliction applies for a 1h magic weapon",
        osrs_confliction_can_apply(&effects, ATTACK_STYLE_MAGIC, ITEM_TRIDENT_OF_SWAMP, 1));

    int att_roll = osrs_player_att_roll(stats.eff_level, stats.attack_bonus);
    const ColoNpcStats* ns = &COLO_NPC_STATS[COLO_JAGUAR_WARRIOR];
    int def_roll = col_npc_target_def_roll(
        &npc, ns, ATTACK_STYLE_MAGIC, MELEE_STYLE_STAB);
    float single = osrs_hit_chance(att_roll, def_roll);
    float dbl = osrs_hit_chance_double(att_roll, def_roll);
    float ref_hit = dbl / (1.0f + dbl - single);
    float ref_dpt = ref_hit * (0.0f + (float)stats.max_hit) * 0.5f / (float)stats.attack_speed;
    CHECK("T7 confliction DPT matches the reference steady-state formula",
        fabsf(dpt_con - ref_dpt) < 1e-2f);
    CHECK("T7 confliction (charged double-acc) beats no-confliction single roll",
        dpt_con > dpt_nocon - 1e-6f && ref_hit > single);

    uint8_t setup_2h[NUM_GEAR_SLOTS];
    memset(setup_2h, ITEM_NONE, NUM_GEAR_SLOTS);
    setup_2h[GEAR_SLOT_WEAPON] = ITEM_TUMEKENS_SHADOW;
    setup_2h[GEAR_SLOT_HANDS] = ITEM_CONFLICTION_GAUNTLETS;
    OsrsEquipmentEffectProfile effects_2h;
    encounter_derive_loadout_effect_profile(setup_2h, &effects_2h);
    CHECK("T7 confliction DISABLED with a 2h weapon (shadow)",
        !osrs_confliction_can_apply(&effects_2h, ATTACK_STYLE_MAGIC, ITEM_TUMEKENS_SHADOW, 1));
}

static void test_best_is_locally_optimal(void) {
    printf("test_best_is_locally_optimal (T8)\n");
    int modes[] = { COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY,
                    COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY };
    int violations = 0;
    for (int m = 0; m < 2; m++) {
        ColosseumContext ctx;
        ColosseumState s;
        loadout_reset(&s, &ctx, modes[m], 0.0f, 808 + m);
        col_build_npc_stats();
        s.player.current_attack = 118;
        s.player.current_strength = 118;
        s.player.current_ranged = 112;
        s.player.current_magic = 99;
        col_mark_live_loadout_dirty(&s);
        ColoBestGear best[COLO_NUM_WEAPON_SETS][COLO_NUM_NPC_TYPES];
        col_build_best_gear_table(&s, best);

        for (int style = 0; style < COLO_NUM_WEAPON_SETS; style++) {
            for (int type = 0; type < COLO_NUM_NPC_TYPES; type++) {
                if (col_type_is_hazard_entity((ColoNpcType)type)) continue;
                const ColoBestGear* bg = &best[style][type];
                if (bg->dpt < 0.0f) continue;
                ColoNPC npc = col_matchup_representative_npc((ColoNpcType)type);
                for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
                    if (slot == GEAR_SLOT_WEAPON) continue;
                    if (bg->setup[slot] == ITEM_NONE) continue;
                    uint8_t variant[NUM_GEAR_SLOTS];
                    memcpy(variant, bg->setup, NUM_GEAR_SLOTS);
                    variant[slot] = ITEM_NONE;
                    float v = col_expected_dpt_for_equipment_vs_npc(&s, variant, &npc, 1);
                    if (v > bg->dpt + 1e-4f) violations++;
                }
            }
        }
    }
    CHECK("T8 argmax setup is locally optimal under single-slot empties", violations == 0);
}

static void calibrate_norm(void) {
    printf("NORM calibration sweep (max best-gear DPT over profile x weapon x type)\n");
    int modes[] = { COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY,
                    COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY };
    const char* names[] = { "speedrun", "beginner" };
    float global_max = 0.0f;
    for (int m = 0; m < 2; m++) {
        ColosseumContext ctx;
        ColosseumState s;
        loadout_reset(&s, &ctx, modes[m], 0.0f, 1000 + m);
        col_build_npc_stats();
        s.player.current_attack = 118;
        s.player.current_strength = 118;
        s.player.current_ranged = 112;
        s.player.current_magic = 99;
        col_mark_live_loadout_dirty(&s);
        ColoBestGear best[COLO_NUM_WEAPON_SETS][COLO_NUM_NPC_TYPES];
        col_build_best_gear_table(&s, best);
        float pmax = 0.0f;
        int pmax_style = -1, pmax_type = -1;
        for (int style = 0; style < COLO_NUM_WEAPON_SETS; style++)
            for (int type = 0; type < COLO_NUM_NPC_TYPES; type++)
                if (best[style][type].dpt > pmax) {
                    pmax = best[style][type].dpt;
                    pmax_style = style; pmax_type = type;
                }
        printf("  %-9s max best-gear DPT = %.3f (style %d, type %s)\n",
            names[m], pmax, pmax_style, colo_npc_type_name(pmax_type));
        if (pmax > global_max) global_max = pmax;
    }
    printf("  GLOBAL max best-gear DPT = %.3f\n", global_max);
    printf("  NORM for top setup at ~0.9 -> %.1f ; current COLO_EXPECTED_DPT_NORM = %.1f\n",
        global_max / 0.9f, (double)COLO_EXPECTED_DPT_NORM);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--calibrate") == 0) {
        calibrate_norm();
        return 0;
    }
    printf("colosseum best-gear DPT oracle probe\n\n");
    test_argmax_setup_and_style();
    test_beats_worn_single_swap();
    test_equip_and_augury_same_tick();
    test_per_cell_marginal_bit();
    test_memo_result_preserving();
    test_confliction_reference();
    test_best_is_locally_optimal();
    return osrs_test_summary();
}
