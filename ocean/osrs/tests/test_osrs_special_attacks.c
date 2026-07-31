#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ocean/osrs/osrs_encounter.h"
#include "ocean/osrs/osrs_special_attacks.h"

#include "ocean/osrs/tests/osrs_test_check.h"

static Player make_maxed_player(void) {
    Player p;
    memset(&p, 0, sizeof(p));
    p.base_hitpoints = 99; p.current_hitpoints = 99;
    p.base_prayer = 99;    p.current_prayer = 99;
    p.base_attack = 99;    p.current_attack = 99;
    p.base_strength = 99;  p.current_strength = 99;
    p.base_defence = 99;   p.current_defence = 99;
    p.base_ranged = 99;    p.current_ranged = 99;
    p.base_magic = 99;     p.current_magic = 99;
    return p;
}

static int loadout_stats_equal(const EncounterLoadoutStats* a, const EncounterLoadoutStats* b) {
    return a->attack_bonus == b->attack_bonus &&
           a->strength_bonus == b->strength_bonus &&
           a->eff_level == b->eff_level &&
           a->max_hit == b->max_hit &&
           a->attack_speed == b->attack_speed &&
           a->attack_range == b->attack_range &&
           a->style == b->style &&
           a->fight_style == b->fight_style &&
           a->def_stab == b->def_stab &&
           a->def_slash == b->def_slash &&
           a->def_crush == b->def_crush &&
           a->def_magic == b->def_magic &&
           a->def_ranged == b->def_ranged &&
           a->att_prayer_mult == b->att_prayer_mult &&
           a->str_prayer_mult == b->str_prayer_mult &&
           a->spell_base_damage == b->spell_base_damage;
}

static void test_consumable_amounts_and_laws(void) {
    printf("test_consumable_amounts_and_laws\n");

    CHECK("super restore amount at 99 is 32", osrs_super_restore_amount(99) == 32);
    CHECK("sanfew amount at 99 is 33", osrs_sanfew_restore_amount(99) == 33);
    CHECK("sanfew out-restores super restore above level 80",
        osrs_sanfew_restore_amount(99) > osrs_super_restore_amount(99) &&
        osrs_sanfew_restore_amount(60) < osrs_super_restore_amount(60));
    CHECK("super combat boost at 99 is 19", osrs_super_combat_boost_amount(99) == 19);
    CHECK("ranging boost at 99 is 13", osrs_ranging_boost_amount(99) == 13);
    CHECK("brew heal at 99 is 16", osrs_brew_heal_amount(99) == 16);
    CHECK("brew defence boost at 99 is 21", osrs_brew_defence_boost_amount(99) == 21);

    int pvp_float_caps_match_integer_helpers = 1;
    for (int level = 1; level <= 99; level++) {
        int old_combat = (int)floorf(level * 0.15f) + 5;
        int old_ranged = (int)floorf(level * 0.10f) + 4;
        int old_brew_defence = (int)floorf(2.0f + (0.20f * level));
        if (old_combat != osrs_super_combat_boost_amount(level) ||
                old_ranged != osrs_ranging_boost_amount(level) ||
                old_brew_defence != osrs_brew_defence_boost_amount(level)) {
            pvp_float_caps_match_integer_helpers = 0;
        }
    }
    CHECK("PvP float cap expressions match integer helpers for levels 1..99",
        pvp_float_caps_match_integer_helpers);

    CHECK("osrs_drink_potion super restore matches the amount helper",
        osrs_drink_potion(POTION_SUPER_RESTORE, 0, 99, 0).prayer_restored ==
        osrs_super_restore_amount(99));
    CHECK("osrs_drink_potion sanfew matches and cures venom",
        osrs_drink_potion(POTION_SANFEW, 0, 99, 0).prayer_restored ==
            osrs_sanfew_restore_amount(99) &&
        osrs_drink_potion(POTION_SANFEW, 0, 99, 0).venom_cured == 1);
    CHECK("osrs_drink_potion super combat matches the amount helper",
        osrs_drink_potion(POTION_SUPER_COMBAT, 0, 99, 0).level_boost ==
        osrs_super_combat_boost_amount(99));
    CHECK("osrs_brew_effect heal matches the amount helper",
        osrs_brew_effect(99, 99, 99, 99, 99, 99).hp_healed == osrs_brew_heal_amount(99));

    BrewResult fresh = osrs_brew_effect(99, 99, 99, 99, 99, 99);
    BrewResult drained = osrs_brew_effect(99, 99, 50, 50, 50, 50);
    CHECK("brew def boost computes from base defence (21 at 99)",
        fresh.def_boost == 21 && drained.def_boost == 21);
    CHECK("brew drains diminish with the current level",
        drained.att_drain < fresh.att_drain && drained.att_drain == 7);

    Player p = make_maxed_player();
    p.current_attack = 1; p.current_strength = 40; p.current_ranged = 98;
    for (int i = 0; i < 20; i++) encounter_restore_stats(&p);
    CHECK("restore converges to base and never overshoots",
        p.current_attack == 99 && p.current_strength == 99 &&
        p.current_ranged == 99 && p.current_magic == 99);

    p = make_maxed_player();
    encounter_super_combat_boost(&p);
    encounter_super_combat_boost(&p);
    CHECK("super combat caps at base + boost (118)",
        p.current_attack == 118 && p.current_strength == 118 && p.current_defence == 118);
    encounter_ranging_boost(&p);
    encounter_ranging_boost(&p);
    CHECK("ranging caps at base + boost (112)", p.current_ranged == 112);

    p = make_maxed_player();
    for (int i = 0; i < 4; i++) encounter_brew_drain_stats(&p);
    CHECK("brews drain offensive stats", p.current_attack < 99 && p.current_magic < 99);
    CHECK("repeated brews cap defence at base + base-level boost (120)",
        p.current_defence == 99 + 21);
    Player half_def = make_maxed_player();
    half_def.current_defence = 50;
    encounter_brew_drain_stats(&half_def);
    CHECK("brew def boost is base-derived even when defence is drained",
        half_def.current_defence == 50 + 21);
    for (int i = 0; i < 20; i++) encounter_restore_stats(&p);
    CHECK("restores recover every brewed-down stat to base",
        p.current_attack == 99 && p.current_strength == 99 &&
        p.current_ranged == 99 && p.current_magic == 99);

    Player a = make_maxed_player(); a.current_attack = 1;
    Player b = make_maxed_player(); b.current_attack = 1;
    encounter_restore_stats(&a);
    encounter_sanfew_restore_stats(&b);
    CHECK("per-dose stat recovery follows the formulas (33 vs 32 at 99)",
        a.current_attack == 33 && b.current_attack == 34);
}

static void drift_ticks(Player* p, int* timer, EncounterStatDriftPins pins, int ticks) {
    for (int i = 0; i < ticks; i++) encounter_tick_stat_drift(p, timer, pins);
}

static int player_combat_and_hp_at_base(const Player* p) {
    return p->current_attack == p->base_attack &&
           p->current_strength == p->base_strength &&
           p->current_defence == p->base_defence &&
           p->current_ranged == p->base_ranged &&
           p->current_magic == p->base_magic &&
           p->current_hitpoints == p->base_hitpoints;
}

static void test_stat_drift_laws(void) {
    printf("test_stat_drift_laws\n");

    Player p = make_maxed_player();
    p.current_attack = 118;
    p.current_strength = 80;
    p.current_defence = 120;
    p.current_ranged = 112;
    p.current_magic = 70;
    p.current_hitpoints = 50;
    int timer = 0;
    drift_ticks(&p, &timer, encounter_stat_drift_no_pins(), 99);
    CHECK("stat drift waits for a complete 100-tick cycle",
        p.current_attack == 118 && p.current_strength == 80 &&
        p.current_hitpoints == 50 && timer == 99);
    drift_ticks(&p, &timer, encounter_stat_drift_no_pins(), 1);
    CHECK("stat drift moves every combat stat and HP one toward base",
        p.current_attack == 117 && p.current_strength == 81 &&
        p.current_defence == 119 && p.current_ranged == 111 &&
        p.current_magic == 71 && p.current_hitpoints == 51 && timer == 0);

    p = make_maxed_player();
    p.current_hitpoints = 115;
    timer = 0;
    drift_ticks(&p, &timer, encounter_stat_drift_no_pins(), 100);
    CHECK("overhealed hitpoints decay one per 100-tick cycle",
        p.current_hitpoints == 114);

    p = make_maxed_player();
    p.current_hitpoints = 0;
    timer = ENCOUNTER_STAT_DRIFT_TICKS - 1;
    drift_ticks(&p, &timer, encounter_stat_drift_no_pins(), 1);
    CHECK("natural HP regeneration does not revive a dead player",
        p.current_hitpoints == 0);

    p = make_maxed_player();
    p.current_attack = 118;
    p.current_strength = 80;
    p.current_defence = 120;
    p.current_ranged = 112;
    p.current_magic = 70;
    p.current_hitpoints = 115;
    timer = 0;
    for (int cycle = 0; cycle < 50; cycle++)
        drift_ticks(&p, &timer, encounter_stat_drift_no_pins(), ENCOUNTER_STAT_DRIFT_TICKS);
    CHECK("repeated stat drift converges every combat stat and HP to base",
        player_combat_and_hp_at_base(&p));

    p = make_maxed_player();
    encounter_super_combat_boost(&p);
    p.current_strength = 122;
    p.current_defence = 50;
    timer = 0;
    EncounterStatDriftPins pins = encounter_divine_super_combat_pins(&p);
    drift_ticks(&p, &timer, pins, ENCOUNTER_STAT_DRIFT_TICKS);
    CHECK("divine super combat pins boosted stats through drift",
        p.current_attack == 118 && p.current_strength == 121 &&
        p.current_defence == 118);
    drift_ticks(&p, &timer, pins, ENCOUNTER_STAT_DRIFT_TICKS * 4);
    CHECK("divine super combat floor holds for later drift cycles",
        p.current_attack == 118 && p.current_strength == 118 &&
        p.current_defence == 118);

    p = make_maxed_player();
    encounter_ranging_boost(&p);
    timer = 0;
    pins = encounter_divine_ranging_pins(&p);
    drift_ticks(&p, &timer, pins, ENCOUNTER_STAT_DRIFT_TICKS * 3);
    CHECK("divine ranging pins ranged through drift", p.current_ranged == 112);
}

static void test_spec_costs_and_sgs(void) {
    printf("test_spec_costs_and_sgs\n");

    CHECK("cost table: claws/SGS/elder maul 50, DDS 25, statius 35, non-spec 0",
        osrs_spec_cost(ITEM_DRAGON_CLAWS) == 50 &&
        osrs_spec_cost(ITEM_SGS) == 50 &&
        osrs_spec_cost(ITEM_ELDER_MAUL) == 50 &&
        osrs_spec_cost(ITEM_DRAGON_DAGGER) == 25 &&
        osrs_spec_cost(ITEM_STATIUS_WARHAMMER) == 35 &&
        osrs_spec_cost(ITEM_SCYTHE_OF_VITUR) == 0);

    uint32_t rng = 4242;
    int landed = 0, missed = 0;
    for (int i = 0; i < 4000; i++) {
        SpecResult r = osrs_resolve_spec(ITEM_SGS, 20000, 50, 12000, 99, &rng);
        if (r.total_damage > 0) {
            landed++;
            int expected_heal = r.total_damage / 2 > 10 ? r.total_damage / 2 : 10;
            int expected_pray = r.total_damage / 4 > 5 ? r.total_damage / 4 : 5;
            if (r.heal != expected_heal || r.prayer_restore != expected_pray) {
                CHECK("SGS heal/prayer follow max(d/2,10) / max(d/4,5)", 0);
                return;
            }
        } else {
            missed++;
            if (r.heal != 0 || r.prayer_restore != 0) {
                CHECK("a missed SGS spec restores nothing", 0);
                return;
            }
        }
    }
    CHECK("SGS sample hit both outcomes", landed > 0 && missed > 0);
    CHECK("SGS heal/prayer follow max(d/2,10) / max(d/4,5)", 1);
}

static void test_two_handed_loadout_shield_suppression(void) {
    printf("test_two_handed_loadout_shield_suppression\n");

    uint8_t two_handed_with_shield[NUM_GEAR_SLOTS];
    uint8_t two_handed_without_shield[NUM_GEAR_SLOTS];
    uint8_t one_handed_with_shield[NUM_GEAR_SLOTS];
    uint8_t one_handed_without_shield[NUM_GEAR_SLOTS];
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        two_handed_with_shield[slot] = ITEM_NONE;
        two_handed_without_shield[slot] = ITEM_NONE;
        one_handed_with_shield[slot] = ITEM_NONE;
        one_handed_without_shield[slot] = ITEM_NONE;
    }

    two_handed_with_shield[GEAR_SLOT_WEAPON] = ITEM_SGS;
    two_handed_with_shield[GEAR_SLOT_SHIELD] = ITEM_DRAGON_DEFENDER;
    two_handed_without_shield[GEAR_SLOT_WEAPON] = ITEM_SGS;
    one_handed_with_shield[GEAR_SLOT_WEAPON] = ITEM_DRAGON_CLAWS;
    one_handed_with_shield[GEAR_SLOT_SHIELD] = ITEM_DRAGON_DEFENDER;
    one_handed_without_shield[GEAR_SLOT_WEAPON] = ITEM_DRAGON_CLAWS;

    EncounterLoadoutStats sgs_with_shield;
    EncounterLoadoutStats sgs_without_shield;
    EncounterLoadoutStats claws_with_shield;
    EncounterLoadoutStats claws_without_shield;
    encounter_compute_loadout_stats(two_handed_with_shield, ATTACK_STYLE_MELEE,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AGGRESSIVE, 0, &sgs_with_shield);
    encounter_compute_loadout_stats(two_handed_without_shield, ATTACK_STYLE_MELEE,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AGGRESSIVE, 0, &sgs_without_shield);
    encounter_compute_loadout_stats(one_handed_with_shield, ATTACK_STYLE_MELEE,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AGGRESSIVE, 0, &claws_with_shield);
    encounter_compute_loadout_stats(one_handed_without_shield, ATTACK_STYLE_MELEE,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AGGRESSIVE, 0, &claws_without_shield);

    CHECK("SGS is marked two-handed and dragon claws are marked one-handed",
        item_is_two_handed(ITEM_SGS) == 1 &&
        item_is_two_handed(ITEM_DRAGON_CLAWS) == 0);
    CHECK("2h loadout stats ignore an occupied shield slot",
        loadout_stats_equal(&sgs_with_shield, &sgs_without_shield));
    CHECK("1h loadout stats keep dragon defender bonuses",
        claws_with_shield.attack_bonus > claws_without_shield.attack_bonus &&
        claws_with_shield.strength_bonus ==
            claws_without_shield.strength_bonus +
            ITEM_DATABASE[ITEM_DRAGON_DEFENDER].melee_strength);

    uint8_t two_handed_with_effect_shield[NUM_GEAR_SLOTS];
    uint8_t one_handed_with_effect_shield[NUM_GEAR_SLOTS];
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        two_handed_with_effect_shield[slot] = ITEM_NONE;
        one_handed_with_effect_shield[slot] = ITEM_NONE;
    }
    two_handed_with_effect_shield[GEAR_SLOT_WEAPON] = ITEM_SGS;
    two_handed_with_effect_shield[GEAR_SLOT_SHIELD] = ITEM_ELYSIAN_SPIRIT_SHIELD;
    one_handed_with_effect_shield[GEAR_SLOT_WEAPON] = ITEM_DRAGON_CLAWS;
    one_handed_with_effect_shield[GEAR_SLOT_SHIELD] = ITEM_ELYSIAN_SPIRIT_SHIELD;

    OsrsEquipmentEffectProfile two_handed_profile;
    OsrsEquipmentEffectProfile one_handed_profile;
    encounter_derive_loadout_effect_profile(
        two_handed_with_effect_shield, &two_handed_profile);
    encounter_derive_loadout_effect_profile(
        one_handed_with_effect_shield, &one_handed_profile);
    CHECK("2h effect profile ignores shield effect bits",
        two_handed_profile.shield_item == ITEM_NONE &&
        !osrs_effect_profile_has(&two_handed_profile, OSRS_ITEM_EFFECT_ELYSIAN));
    CHECK("1h effect profile keeps shield effect bits",
        one_handed_profile.shield_item == ITEM_ELYSIAN_SPIRIT_SHIELD &&
        osrs_effect_profile_has(&one_handed_profile, OSRS_ITEM_EFFECT_ELYSIAN));
}

static void test_magic_effective_attack_level_law(void) {
    printf("test_magic_effective_attack_level_law\n");

    uint8_t powered_staff_loadout[NUM_GEAR_SLOTS];
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        powered_staff_loadout[slot] = ITEM_NONE;
    }
    powered_staff_loadout[GEAR_SLOT_WEAPON] = ITEM_TRIDENT_OF_SWAMP;

    EncounterLoadoutStats accurate;
    EncounterLoadoutStats longrange;
    encounter_compute_loadout_stats(powered_staff_loadout, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_AUGURY, 82, FIGHT_STYLE_ACCURATE, 30, &accurate);
    encounter_compute_loadout_stats(powered_staff_loadout, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_AUGURY, 82, FIGHT_STYLE_LONGRANGE, 30, &longrange);

    CHECK("E12 accurate magic eff is floor(82*1.25)+2+9 = 113",
        accurate.eff_level == 113);
    CHECK("E12 longrange magic eff is floor(82*1.25)+9 = 111",
        longrange.eff_level == 111);

    encounter_update_loadout_level(&accurate, OFFENSIVE_PRAYER_AUGURY, 82, 82);
    encounter_update_loadout_level(&longrange, OFFENSIVE_PRAYER_AUGURY, 82, 82);
    CHECK("E12 dynamic magic recompute preserves accurate and longrange laws",
        accurate.eff_level == 113 && longrange.eff_level == 111);
}

static void test_spec_force_max_laws(void) {
    printf("test_spec_force_max_laws\n");

    typedef struct {
        int item;
        int total;
        int hits;
        int def_drain;
        int magic_def_drain;
        int freeze_ticks;
        int attack_speed_override;
        int heal;
        int prayer_restore;
    } ForceMaxCase;

    const ForceMaxCase cases[] = {
        { ITEM_AGS, 55, 1, 0, 0, 0, 0, 0, 0 },
        { ITEM_DRAGON_CLAWS, 77, 4, 0, 0, 0, 0, 0, 0 },
        { ITEM_STATIUS_WARHAMMER, 50, 1, 60, 0, 0, 0, 0, 0 },
        { ITEM_BGS, 48, 1, 48, 0, 0, 0, 0, 0 },
        { ITEM_ZGS, 44, 1, 0, 0, 32, 0, 0, 0 },
        { ITEM_SGS, 44, 1, 0, 0, 0, 0, 22, 11 },
        { ITEM_ANCIENT_GS, 44, 1, 0, 0, 0, 0, 0, 0 },
        { ITEM_VESTAS, 48, 1, 0, 0, 0, 0, 0, 0 },
        { ITEM_VOIDWAKER, 60, 1, 0, 0, 0, 0, 0, 0 },
        { ITEM_GRANITE_MAUL, 40, 1, 0, 0, 0, 1, 0, 0 },
        { ITEM_DRAGON_DAGGER, 92, 2, 0, 0, 0, 0, 0, 0 },
        { ITEM_ELDER_MAUL, 40, 1, 70, 0, 0, 0, 0, 0 },
        { ITEM_TOXIC_BLOWPIPE, 60, 1, 0, 0, 0, 0, 30, 0 },
        { ITEM_MAGIC_SHORTBOW_I, 80, 2, 0, 0, 0, 0, 0, 0 },
        { ITEM_DARK_BOW, 96, 2, 0, 0, 0, 0, 0, 0 },
        { ITEM_HEAVY_BALLISTA, 50, 1, 0, 0, 0, 0, 0, 0 },
        { ITEM_ZARYTE_CROSSBOW, 40, 1, 0, 0, 0, 0, 0, 0 },
        { ITEM_MORRIGANS_JAVELIN, 48, 1, 0, 0, 0, 0, 0, 0 },
        { ITEM_ARMADYL_CROSSBOW, 40, 1, 0, 0, 0, 0, 0, 0 },
        { ITEM_VOLATILE_STAFF, 58, 1, 0, 0, 0, 0, 0, 0 },
        { ITEM_EYE_OF_AYAK, 52, 1, 0, 52, 0, 5, 0, 0 },
    };

    int all_cases_match = 1;
    for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++) {
        SpecResult r = {0};
        osrs_spec_result_force_max(&r, cases[i].item, 40, 200);
        if (r.total_damage != cases[i].total || r.num_hits != cases[i].hits ||
            r.def_drain != cases[i].def_drain ||
            r.magic_def_drain != cases[i].magic_def_drain ||
            r.freeze_ticks != cases[i].freeze_ticks ||
            r.attack_speed_override != cases[i].attack_speed_override ||
            r.heal != cases[i].heal ||
            r.prayer_restore != cases[i].prayer_restore ||
            r.spec_cost != osrs_spec_cost(cases[i].item)) {
            all_cases_match = 0;
        }
    }
    CHECK("force-max covers every resolver weapon's best outcome", all_cases_match);

    SpecResult claws = {0};
    osrs_spec_result_force_max(&claws, ITEM_DRAGON_CLAWS, 40, 200);
    CHECK("force-max claws uses first-success t=79 floor split with observable total 77",
        claws.damage[0] == 39 && claws.damage[1] == 19 &&
        claws.damage[2] == 9 && claws.damage[3] == 10 &&
        claws.total_damage == 77);

    SpecResult sgs = {0};
    osrs_spec_result_force_max(&sgs, ITEM_SGS, 8, 200);
    CHECK("force-max SGS recomputes heal/prayer minimums from the forced total",
        sgs.total_damage == 8 && sgs.heal == 10 && sgs.prayer_restore == 5);

    SpecResult elder = {0};
    osrs_spec_result_force_max(&elder, ITEM_ELDER_MAUL, 40, 200);
    CHECK("force-max elder maul re-applies 35% target defence drain",
        elder.total_damage == 40 && elder.def_drain == 70);
}

static void claws_expected_splats(int branch, int total, int out[4]) {
    out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
    switch (branch) {
        case 0:
            out[0] = total / 2; out[1] = total / 4;
            out[2] = total / 8; out[3] = total / 8 + 1;
            break;
        case 1:
            out[0] = total / 2; out[1] = total / 4; out[2] = total / 4 + 1;
            break;
        case 2:
            out[0] = total / 2; out[1] = total / 2 + 1;
            break;
        case 3:
            out[0] = total + 1;
            break;
    }
}

static int claws_classify(const SpecResult* r, int max_hit, int* out_branch) {
    if (r->damage[0] == 0 && r->damage[1] == 0 &&
        r->damage[2] == 0 && r->damage[3] == 0) {
        *out_branch = 5;
        return 1;
    }
    if (r->damage[0] == 1 && r->damage[1] == 1 &&
        r->damage[2] == 0 && r->damage[3] == 0) {
        *out_branch = 4;
        return 1;
    }
    for (int k = 0; k < 4; k++) {
        int low = max_hit * (4 - k) / 4;
        int high = max_hit + low - 1;
        for (int t = low; t <= high; t++) {
            int e[4];
            claws_expected_splats(k, t, e);
            if (r->damage[0] == e[0] && r->damage[1] == e[1] &&
                r->damage[2] == e[2] && r->damage[3] == e[3]) {
                *out_branch = k;
                return 1;
            }
        }
    }
    return 0;
}

static void test_claws_and_def_drains(void) {
    printf("test_claws_and_def_drains\n");

    uint32_t rng = 1337;
    int seen_total[40] = {0};
    int classified_ok = 1, branch0 = 0, min_total = 1 << 30, max_total = 0;
    for (int i = 0; i < 20000; i++) {
        SpecResult r = osrs_resolve_spec(ITEM_DRAGON_CLAWS, 2000000, 40, 1, 99, &rng);
        int branch;
        if (!claws_classify(&r, 40, &branch)) { classified_ok = 0; break; }
        if (branch != 0) continue;
        branch0++;
        int total = r.damage[0] + r.damage[1] + r.damage[2] + r.damage[3];
        if (total != r.total_damage) { classified_ok = 0; break; }

        for (int t = 40; t <= 79; t++) {
            int e[4];
            claws_expected_splats(0, t, e);
            if (r.damage[0] == e[0] && r.damage[1] == e[1] &&
                r.damage[2] == e[2] && r.damage[3] == e[3]) {
                seen_total[t - 40] = 1;
            }
        }
        if (total < min_total) min_total = total;
        if (total > max_total) max_total = total;
    }
    int covered = 1;
    for (int i = 0; i < 40; i++) covered &= seen_total[i];
    CHECK("claws splats always match a dps-calc branch table", classified_ok);
    CHECK("claws first-success branch dominates at p~=1", branch0 >= 19990);
    CHECK("claws first-success draws cover every total in [40, 79]", covered);
    CHECK("claws observable extremes match the floor splits (41 / 77 at max 40)",
        min_total == 20 + 10 + 5 + 6 && max_total == 39 + 19 + 9 + 10);

    rng = 777;
    int seen_class[6] = {0};
    int contested_ok = 1;
    for (int i = 0; i < 50000; i++) {
        SpecResult r = osrs_resolve_spec(ITEM_DRAGON_CLAWS, 10000, 40, 10000, 99, &rng);
        int branch;
        if (r.num_hits != 4 || !claws_classify(&r, 40, &branch)) {
            contested_ok = 0;
            break;
        }
        seen_class[branch] = 1;
    }
    CHECK("claws contested-accuracy tuples all match dps-calc tables", contested_ok);
    CHECK("claws all four branches and both all-miss outcomes observed",
        seen_class[0] && seen_class[1] && seen_class[2] &&
        seen_class[3] && seen_class[4] && seen_class[5]);

    rng = 99;
    int maul_drained = 0, statius_drained = 0;
    for (int i = 0; i < 2000 && (!maul_drained || !statius_drained); i++) {
        SpecResult m = osrs_resolve_spec(ITEM_ELDER_MAUL, 20000, 40, 8000, 200, &rng);
        if (m.damage[0] > 0 && !maul_drained) {
            CHECK("elder maul drains 35% of target def", m.def_drain == 200 * 35 / 100);
            maul_drained = 1;
        }
        SpecResult st = osrs_resolve_spec(ITEM_STATIUS_WARHAMMER, 20000, 40, 8000, 200, &rng);
        if (st.damage[0] > 0 && !statius_drained) {
            CHECK("statius drains 30% of target def", st.def_drain == 200 * 30 / 100);
            statius_drained = 1;
        }
    }
    CHECK("both drain weapons landed in the sample", maul_drained && statius_drained);
}

static void test_item_effect_laws(void) {
    printf("test_item_effect_laws\n");

    OsrsItemEffectState state;
    osrs_item_effect_state_init(&state);

    OsrsEquipmentEffectProfile empty;
    memset(&empty, 0, sizeof(empty));
    OsrsPreparedAttackEffects id = osrs_prepare_attack_effects(
        &empty, &state, ITEM_SCYTHE_OF_VITUR, ATTACK_STYLE_MELEE,
        OSRS_MAGIC_ATTACK_NONE, osrs_target_ref_none(), 1, 12345, 50,
        osrs_target_effect_context_magic(300, 80), 99, 99);
    CHECK("empty profile is the identity on rolls",
        id.attack_roll == 12345 && id.max_hit == 50 && id.use_double_accuracy == 0);

    OsrsEquipmentEffectProfile tbow_profile;
    memset(&tbow_profile, 0, sizeof(tbow_profile));
    tbow_profile.effect_mask = OSRS_ITEM_EFFECT_TWISTED_BOW;
    int prev_hit = -1, prev_roll = -1, monotone = 1;
    for (int magic = 1; magic <= 350; magic += 7) {
        OsrsPreparedAttackEffects e = osrs_prepare_attack_effects(
            &tbow_profile, &state, ITEM_TWISTED_BOW, ATTACK_STYLE_RANGED,
            OSRS_MAGIC_ATTACK_NONE, osrs_target_ref_none(), 1, 20000, 80,
            osrs_target_effect_context_magic(magic, 0), 99, 99);
        if (e.max_hit < prev_hit || e.attack_roll < prev_roll) monotone = 0;
        prev_hit = e.max_hit;
        prev_roll = e.attack_roll;
    }
    CHECK("tbow scaling is monotone in target magic", monotone);

    CHECK("crystal points are helm 1 / legs 2 / body 3",
        osrs_crystal_armour_points(ITEM_CRYSTAL_HELM) == 1 &&
        osrs_crystal_armour_points(ITEM_CRYSTAL_LEGS) == 2 &&
        osrs_crystal_armour_points(ITEM_CRYSTAL_BODY) == 3);
    OsrsEquipmentEffectProfile crystal;
    memset(&crystal, 0, sizeof(crystal));
    crystal.effect_mask = OSRS_ITEM_EFFECT_CRYSTAL_ARMOUR;
    crystal.crystal_armour_points = 6;
    OsrsPreparedAttackEffects bowfa = osrs_prepare_attack_effects(
        &crystal, &state, ITEM_BOW_OF_FAERDHINEN, ATTACK_STYLE_RANGED,
        OSRS_MAGIC_ATTACK_NONE, osrs_target_ref_none(), 1, 20000, 40,
        osrs_target_effect_context_magic(100, 0), 99, 99);
    CHECK("full crystal scales bowfa by 26/20 accuracy and 46/40 damage",
        bowfa.attack_roll == 20000 * 26 / 20 && bowfa.max_hit == 40 * 46 / 40);

    OsrsEquipmentEffectProfile fang;
    memset(&fang, 0, sizeof(fang));
    fang.effect_mask = OSRS_ITEM_EFFECT_FANG;
    CHECK("fang generated item carries the shared fang effect",
        ITEM_DATABASE[ITEM_OSMUMTENS_FANG].effect_mask == OSRS_ITEM_EFFECT_FANG);

    int fang_bounds_ok = 1;
    for (int max_hit = 1; max_hit <= 99; max_hit++) {
        OsrsPreparedAttackEffects prepared = osrs_prepare_attack_effects_for_melee_style(
            &fang, &state, ITEM_OSMUMTENS_FANG, ATTACK_STYLE_MELEE,
            MELEE_STYLE_STAB, OSRS_MAGIC_ATTACK_NONE, osrs_target_ref_none(), 1,
            12345, max_hit, osrs_target_effect_context_none(), 99, 99);
        int shrink = osrs_fang_hit_bound_shrink(max_hit);
        if (prepared.min_hit != shrink || prepared.max_hit != max_hit - shrink) {
            fang_bounds_ok = 0;
        }
    }
    CHECK("fang min/max bounds follow floor(max*3/20) shrink", fang_bounds_ok);

    OsrsPreparedAttackEffects fang_stab = osrs_prepare_attack_effects_for_melee_style(
        &fang, &state, ITEM_OSMUMTENS_FANG, ATTACK_STYLE_MELEE,
        MELEE_STYLE_STAB, OSRS_MAGIC_ATTACK_NONE, osrs_target_ref_none(), 1,
        20000, 50, osrs_target_effect_context_none(), 99, 99);
    OsrsPreparedAttackEffects fang_slash = osrs_prepare_attack_effects_for_melee_style(
        &fang, &state, ITEM_OSMUMTENS_FANG, ATTACK_STYLE_MELEE,
        MELEE_STYLE_SLASH, OSRS_MAGIC_ATTACK_NONE, osrs_target_ref_none(), 1,
        20000, 50, osrs_target_effect_context_none(), 99, 99);
    CHECK("fang accuracy reroll applies only on stab",
        fang_stab.use_fang_accuracy == 1 && fang_slash.use_fang_accuracy == 0);

    int double_accuracy_monotone = 1;
    for (int attack_roll = 1; attack_roll <= 20000; attack_roll += 137) {
        for (int def_roll = 1; def_roll <= 20000; def_roll += 251) {
            if (osrs_hit_chance_double(attack_roll, def_roll) + 0.000001f <
                osrs_hit_chance(attack_roll, def_roll)) {
                double_accuracy_monotone = 0;
            }
        }
    }
    CHECK("fang double-roll accuracy never lowers hit chance", double_accuracy_monotone);

    CHECK("scythe splats: 1 vs 1x1, 2 vs 2x2, 3 vs 3x3+",
        osrs_scythe_splats_for_target_size(1) == 1 &&
        osrs_scythe_splats_for_target_size(2) == 2 &&
        osrs_scythe_splats_for_target_size(3) == 3 &&
        osrs_scythe_splats_for_target_size(5) == 3);

    OsrsEquipmentEffectProfile fury;
    memset(&fury, 0, sizeof(fury));
    fury.effect_mask = OSRS_ITEM_EFFECT_BLOOD_FURY;
    uint32_t rng = 31337;
    int procs = 0, bad_heal = 0;
    for (int i = 0; i < 2000; i++) {
        int heal = osrs_blood_fury_heal_amount(
            &fury, ATTACK_STYLE_MELEE, 30, &rng);
        if (heal > 0) {
            procs++;
            if (heal != 9) bad_heal = 1;
        }
    }
    CHECK("blood fury heals exactly 30% on every proc", !bad_heal);
    CHECK("blood fury procs near the 20% rate", procs > 280 && procs < 520);
    int ranged_heal = osrs_blood_fury_heal_amount(
        &fury, ATTACK_STYLE_RANGED, 30, &rng);
    CHECK("blood fury never procs on ranged damage", ranged_heal == 0);
}

int main(void) {
    test_consumable_amounts_and_laws();
    test_stat_drift_laws();
    test_spec_costs_and_sgs();
    test_two_handed_loadout_shield_suppression();
    test_magic_effective_attack_level_law();
    test_spec_force_max_laws();
    test_claws_and_def_drains();
    test_item_effect_laws();

    return osrs_test_summary();
}
