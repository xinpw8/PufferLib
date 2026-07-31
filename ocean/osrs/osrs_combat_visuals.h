#ifndef OSRS_COMBAT_VISUALS_H
#define OSRS_COMBAT_VISUALS_H

#include "osrs_items.h"
#include "osrs_types.h"
#include "osrs_gfx_ids.h"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    OSRS_COMBAT_VISUAL_STYLE_ANY = -1,
    OSRS_COMBAT_VISUAL_STANCE_ANY = -1,
    OSRS_COMBAT_VISUAL_NO_ANIMATION = -1,
    OSRS_PLAYER_UNARMED_ATTACK_ANIM = 422,
    OSRS_PLAYER_POWERED_STAFF_ATTACK_ANIM = 1167,
    OSRS_COMBAT_PROJECTILE_MISSING = -1,
};

typedef enum {
    OSRS_ITEM_ID_RUNE_ARROW = 892,
    OSRS_ITEM_ID_ABYSSAL_WHIP = 4151,
    OSRS_ITEM_ID_GRANITE_MAUL = 4153,
    OSRS_ITEM_ID_AHRIMS_STAFF = 4710,
    OSRS_ITEM_ID_DRAGON_DAGGER = 5698,
    OSRS_ITEM_ID_RUNE_CROSSBOW = 9185,
    OSRS_ITEM_ID_DIAMOND_BOLTS_E = 9243,
    OSRS_ITEM_ID_DARK_BOW = 11235,
    OSRS_ITEM_ID_DRAGON_ARROWS = 11212,
    OSRS_ITEM_ID_DRAGON_DART = 11230,
    OSRS_ITEM_ID_STAFF_OF_THE_DEAD = 11791,
    OSRS_ITEM_ID_ARMADYL_CROSSBOW = 11785,
    OSRS_ITEM_ID_ARMADYL_GODSWORD = 11802,
    OSRS_ITEM_ID_MAGIC_SHORTBOW_I = 12788,
    OSRS_ITEM_ID_TRIDENT_OF_THE_SWAMP = 12899,
    OSRS_ITEM_ID_TOXIC_BLOWPIPE = 12926,
    OSRS_ITEM_ID_DRAGON_CLAWS = 13652,
    OSRS_ITEM_ID_ZURIELS_STAFF = 13867,
    OSRS_ITEM_ID_HEAVY_BALLISTA = 19481,
    OSRS_ITEM_ID_TWISTED_BOW = 20997,
    OSRS_ITEM_ID_ELDER_MAUL = 21003,
    OSRS_ITEM_ID_KODAI_WAND = 21006,
    OSRS_ITEM_ID_GHRAZI_RAPIER = 22324,
    OSRS_ITEM_ID_SANGUINESTI_STAFF = 22481,
    OSRS_ITEM_ID_VESTAS_LONGSWORD = 22613,
    OSRS_ITEM_ID_STATIUSS_WARHAMMER = 22622,
    OSRS_ITEM_ID_MORRIGANS_JAVELIN = 22636,
    OSRS_ITEM_ID_INQUISITORS_MACE = 24417,
    OSRS_ITEM_ID_VOLATILE_NIGHTMARE_STAFF = 24424,
    OSRS_ITEM_ID_BOW_OF_FAERDHINEN = 25865,
    OSRS_ITEM_ID_ANCIENT_GODSWORD = 26233,
    OSRS_ITEM_ID_ZARYTE_CROSSBOW = 26374,
    OSRS_ITEM_ID_VOIDWAKER = 27690,
    OSRS_ITEM_ID_DRAGON_HUNTER_WAND = 30070,
    OSRS_ITEM_ID_EYE_OF_AYAK = 31113,
    OSRS_ITEM_ID_TUMEKENS_SHADOW = 27275,
} OsrsCombatVisualItemId;

typedef enum {
    OSRS_COMBAT_VISUAL_KIND_ITEM = 1,
    OSRS_COMBAT_VISUAL_KIND_SPELL = 2,
    OSRS_COMBAT_VISUAL_KIND_NPC = 3,
    OSRS_COMBAT_VISUAL_KIND_SPECIAL = 4,
} OsrsCombatVisualKind;

typedef enum {
    OSRS_COMBAT_PROJECTILE_NONE = 0,
    OSRS_COMBAT_PROJECTILE_BOLT,
    OSRS_COMBAT_PROJECTILE_RUNE_ARROW,
    OSRS_COMBAT_PROJECTILE_DRAGON_ARROW,
    OSRS_COMBAT_PROJECTILE_DRAGON_DART,
    OSRS_COMBAT_PROJECTILE_TRIDENT,
} OsrsCombatProjectileVisual;

typedef enum {
    OSRS_COMBAT_VISUAL_SPELL_NONE = 0,
    OSRS_COMBAT_VISUAL_SPELL_ICE_BARRAGE = 1,
    OSRS_COMBAT_VISUAL_SPELL_BLOOD_BARRAGE = 2,
} OsrsCombatVisualSpell;

typedef struct {
    int32_t launch_spotanim_id;
    int32_t travel_spotanim_id;
    int32_t impact_spotanim_id;
    int32_t projectile_model_id;
    int32_t projectile_anim_id;
    int16_t hit_delay;
    int16_t client_delay;
    int16_t projectile_start_height;
    int16_t projectile_end_height;
    int16_t projectile_delay;
    int16_t projectile_angle;
    int16_t projectile_length_adjustment;
    int16_t projectile_progress;
    int16_t projectile_step_multiplier;
    int16_t projectile_count;
} OsrsCombatProjectileProfile;

typedef struct {
    int16_t projectile_start_height;
    int16_t projectile_end_height;
    int16_t projectile_delay;
    int16_t projectile_angle;
    int16_t projectile_length_adjustment;
    int16_t projectile_progress;
    int16_t projectile_step_multiplier;
} OsrsCombatAltProjectileProfile;

typedef struct {
    uint8_t kind;
    int32_t key_id;
    const char* key_name;
    int8_t style;
    int8_t stance_idx;
    int16_t attack_anim_id;
    OsrsCombatProjectileProfile projectile;
    OsrsCombatAltProjectileProfile alt_projectile;
    int32_t aux_travel_spotanim_id;
    int32_t aux_impact_spotanim_id;
    int32_t aux_projectile_model_id;
    int32_t aux_projectile_anim_id;
    int8_t impact_on_last_only;
    int32_t double_launch_spotanim_id;
} OsrsCombatVisualRow;

enum {
    OSRS_PROJECTILE_MODEL_ARROW = 3136,
    OSRS_PROJECTILE_MODEL_VENATOR_BOLT = 46993,
    OSRS_PROJECTILE_MODEL_TRIDENT = 20825,
    OSRS_PROJECTILE_MODEL_DRAGON_ARROW = 26377,
    OSRS_PROJECTILE_MODEL_DRAGON_DART = 26379,
    OSRS_PROJECTILE_ANIM_TRIDENT = 5462,
    OSRS_PROJECTILE_ANIM_DRAGON_ARROW = 6622,
    OSRS_PROJECTILE_ANIM_DRAGON_DART = 6622,
    OSRS_COMBAT_PROJECTILE_SEQUENCE_MAX = 8,
};

#include "osrs_combat_visuals_generated.h"

#define OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING OSRS_COMBAT_PROJECTILE_MISSING
#define OSRS_COMBAT_VISUAL_COLOSSEUM_ALT_NONE \
    {OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, \
     OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, \
     OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, \
     OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING}
#define OSRS_COMBAT_VISUAL_COLOSSEUM_PROJECTILE(launch, travel, impact, model, anim, hit, client, start, end, delay, angle, len, progress, step, count) \
    {launch, travel, impact, model, anim, hit, client, start, end, delay, angle, \
     len, progress, step, count}
#define OSRS_COMBAT_VISUAL_COLOSSEUM_NPC(npc_id, attack_style, attack_anim, projectile_profile) \
    {(uint8_t)OSRS_COMBAT_VISUAL_KIND_NPC, npc_id, "", (int8_t)attack_style, \
     (int8_t)OSRS_COMBAT_VISUAL_STANCE_ANY, (int16_t)attack_anim, \
     projectile_profile, OSRS_COMBAT_VISUAL_COLOSSEUM_ALT_NONE, \
     OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, \
     OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, \
     (int8_t)0, OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING}

static const OsrsCombatVisualRow OSRS_COMBAT_VISUAL_COLOSSEUM_ROWS[] = {
    OSRS_COMBAT_VISUAL_COLOSSEUM_NPC(
        12814, ATTACK_STYLE_RANGED, 10850,
        OSRS_COMBAT_VISUAL_COLOSSEUM_PROJECTILE(
            24, 15,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            3136, OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 16,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 1)),
    OSRS_COMBAT_VISUAL_COLOSSEUM_NPC(
        12815, ATTACK_STYLE_MAGIC, 10853,
        OSRS_COMBAT_VISUAL_COLOSSEUM_PROJECTILE(
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 328, 329,
            5091, 1577,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 16,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 1)),
    OSRS_COMBAT_VISUAL_COLOSSEUM_NPC(
        12811, ATTACK_STYLE_MAGIC, 10859,
        OSRS_COMBAT_VISUAL_COLOSSEUM_PROJECTILE(
            1458, 1459, 1460,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 16,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 1)),
    OSRS_COMBAT_VISUAL_COLOSSEUM_NPC(
        12817, ATTACK_STYLE_RANGED, 10892,
        OSRS_COMBAT_VISUAL_COLOSSEUM_PROJECTILE(

            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 2673, 2676,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 16,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 1)),
    OSRS_COMBAT_VISUAL_COLOSSEUM_NPC(
        12818, ATTACK_STYLE_MELEE, 10869,
        OSRS_COMBAT_VISUAL_COLOSSEUM_PROJECTILE(
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 2685, 2686,
            51213, 10328,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 16,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 1)),
    OSRS_COMBAT_VISUAL_COLOSSEUM_NPC(
        12818, ATTACK_STYLE_RANGED, 10869,
        OSRS_COMBAT_VISUAL_COLOSSEUM_PROJECTILE(
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 2683, 2684,
            51221, 10327,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 16,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 1)),
    OSRS_COMBAT_VISUAL_COLOSSEUM_NPC(
        12818, ATTACK_STYLE_MAGIC, 10869,
        OSRS_COMBAT_VISUAL_COLOSSEUM_PROJECTILE(
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 2681, 2682,
            51215, 10329,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 16,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 1)),
    OSRS_COMBAT_VISUAL_COLOSSEUM_NPC(
        12819, ATTACK_STYLE_MAGIC, 10903,
        OSRS_COMBAT_VISUAL_COLOSSEUM_PROJECTILE(
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 2679,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            51210,
            10903,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 16,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING,
            OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING, 1)),
};

static const size_t OSRS_COMBAT_VISUAL_COLOSSEUM_ROW_COUNT =
    sizeof(OSRS_COMBAT_VISUAL_COLOSSEUM_ROWS) /
    sizeof(OSRS_COMBAT_VISUAL_COLOSSEUM_ROWS[0]);

#undef OSRS_COMBAT_VISUAL_COLOSSEUM_NPC
#undef OSRS_COMBAT_VISUAL_COLOSSEUM_PROJECTILE
#undef OSRS_COMBAT_VISUAL_COLOSSEUM_ALT_NONE
#undef OSRS_COMBAT_VISUAL_COLOSSEUM_MISSING

typedef struct {
    uint16_t item_id;
    int16_t special_attack_anim_id;
} OsrsCombatSpecialFallback;

typedef struct {
    uint16_t weapon_item_id;
    uint16_t projectile_item_id;
    OsrsCombatProjectileVisual visual;
} OsrsCombatWeaponProjectileDefault;

typedef struct {
    OsrsCombatProjectileProfile projectile;
    int16_t sequence_index;
    int16_t sequence_count;
} OsrsCombatProjectileSequencePart;

static const OsrsCombatProjectileProfile OSRS_POWERED_STAFF_PROJECTILE_PROFILE = {
    GFX_TRIDENT_CAST, GFX_TRIDENT_PROJ, GFX_TRIDENT_IMPACT,
    OSRS_PROJECTILE_MODEL_TRIDENT, OSRS_PROJECTILE_ANIM_TRIDENT,
    3, 3, 160, 120, OSRS_COMBAT_PROJECTILE_MISSING, 16,
    OSRS_COMBAT_PROJECTILE_MISSING, OSRS_COMBAT_PROJECTILE_MISSING,
    OSRS_COMBAT_PROJECTILE_MISSING, 1
};

static const OsrsCombatProjectileProfile OSRS_TUMEKENS_SHADOW_PROJECTILE_PROFILE = {
    GFX_TUMEKENS_SHADOW_CAST, GFX_TUMEKENS_SHADOW_PROJ, GFX_TUMEKENS_SHADOW_IMPACT,
    OSRS_PROJECTILE_MODEL_TRIDENT, OSRS_PROJECTILE_ANIM_TRIDENT,
    3, 3, 160, 120, OSRS_COMBAT_PROJECTILE_MISSING, 16,
    OSRS_COMBAT_PROJECTILE_MISSING, OSRS_COMBAT_PROJECTILE_MISSING,
    OSRS_COMBAT_PROJECTILE_MISSING, 1
};

static const OsrsCombatSpecialFallback OSRS_COMBAT_SPECIAL_FALLBACKS[] = {
    {OSRS_ITEM_ID_GRANITE_MAUL, 1667},
    {OSRS_ITEM_ID_DRAGON_DAGGER, 1062},
    {OSRS_ITEM_ID_MAGIC_SHORTBOW_I, 1074},
    {OSRS_ITEM_ID_DRAGON_CLAWS, 7514},
    {OSRS_ITEM_ID_VESTAS_LONGSWORD, 7515},
    {OSRS_ITEM_ID_STATIUSS_WARHAMMER, 1378},
    {OSRS_ITEM_ID_INQUISITORS_MACE, 1060},
    {OSRS_ITEM_ID_VOLATILE_NIGHTMARE_STAFF, 8532},
    {OSRS_ITEM_ID_ANCIENT_GODSWORD, 7644},
    {OSRS_ITEM_ID_VOIDWAKER, 1378},
};

static const OsrsCombatWeaponProjectileDefault OSRS_RANGED_PROJECTILE_DEFAULTS[] = {
    {OSRS_ITEM_ID_RUNE_CROSSBOW, OSRS_ITEM_ID_DIAMOND_BOLTS_E, OSRS_COMBAT_PROJECTILE_BOLT},
    {OSRS_ITEM_ID_ARMADYL_CROSSBOW, OSRS_ITEM_ID_DIAMOND_BOLTS_E, OSRS_COMBAT_PROJECTILE_BOLT},
    {OSRS_ITEM_ID_HEAVY_BALLISTA, OSRS_ITEM_ID_DIAMOND_BOLTS_E, OSRS_COMBAT_PROJECTILE_BOLT},
    {OSRS_ITEM_ID_ZARYTE_CROSSBOW, OSRS_ITEM_ID_DIAMOND_BOLTS_E, OSRS_COMBAT_PROJECTILE_BOLT},
    {OSRS_ITEM_ID_DARK_BOW, OSRS_ITEM_ID_RUNE_ARROW, OSRS_COMBAT_PROJECTILE_RUNE_ARROW},
    {OSRS_ITEM_ID_MAGIC_SHORTBOW_I, OSRS_ITEM_ID_RUNE_ARROW, OSRS_COMBAT_PROJECTILE_RUNE_ARROW},
    {OSRS_ITEM_ID_BOW_OF_FAERDHINEN, OSRS_ITEM_ID_RUNE_ARROW, OSRS_COMBAT_PROJECTILE_RUNE_ARROW},
    {OSRS_ITEM_ID_TWISTED_BOW, OSRS_ITEM_ID_DRAGON_ARROWS, OSRS_COMBAT_PROJECTILE_DRAGON_ARROW},
    {OSRS_ITEM_ID_TOXIC_BLOWPIPE, OSRS_ITEM_ID_DRAGON_DART, OSRS_COMBAT_PROJECTILE_DRAGON_DART},
};

static const uint16_t OSRS_POWERED_STAFF_ITEMS[] = {
    OSRS_ITEM_ID_TRIDENT_OF_THE_SWAMP,
    OSRS_ITEM_ID_SANGUINESTI_STAFF,
    OSRS_ITEM_ID_EYE_OF_AYAK,
    OSRS_ITEM_ID_TUMEKENS_SHADOW,
};

static inline int osrs_combat_projectile_value_or(int value, int fallback) {
    return value == OSRS_COMBAT_PROJECTILE_MISSING ? fallback : value;
}

static inline int osrs_combat_visual_row_has_projectile(
    const OsrsCombatVisualRow* row
) {
    return row &&
        (row->projectile.launch_spotanim_id != OSRS_COMBAT_PROJECTILE_MISSING ||
         row->projectile.travel_spotanim_id != OSRS_COMBAT_PROJECTILE_MISSING ||
         row->projectile.impact_spotanim_id != OSRS_COMBAT_PROJECTILE_MISSING ||
         row->projectile.projectile_model_id != OSRS_COMBAT_PROJECTILE_MISSING ||
         row->aux_travel_spotanim_id != OSRS_COMBAT_PROJECTILE_MISSING ||
         row->aux_projectile_model_id != OSRS_COMBAT_PROJECTILE_MISSING);
}

static inline int osrs_combat_visual_projectile_count(
    const OsrsCombatVisualRow* row
) {
    if (!row || row->projectile.projectile_count < 1) return 1;
    if (row->projectile.projectile_count > 4) return 4;
    return row->projectile.projectile_count;
}

static inline int osrs_combat_visual_show_impact_for_projectile(
    const OsrsCombatVisualRow* effect,
    int sequence_index,
    int sequence_count
) {
    if (!effect || !effect->impact_on_last_only) return 1;
    return sequence_index == sequence_count - 1;
}

static inline int osrs_combat_projectile_profile_has_timing(
    const OsrsCombatProjectileProfile* profile
) {
    return profile &&
        profile->projectile_start_height != OSRS_COMBAT_PROJECTILE_MISSING &&
        profile->projectile_end_height != OSRS_COMBAT_PROJECTILE_MISSING &&
        profile->projectile_delay != OSRS_COMBAT_PROJECTILE_MISSING &&
        profile->projectile_angle != OSRS_COMBAT_PROJECTILE_MISSING &&
        profile->projectile_progress != OSRS_COMBAT_PROJECTILE_MISSING &&
        profile->projectile_step_multiplier != OSRS_COMBAT_PROJECTILE_MISSING;
}

static inline int osrs_combat_alt_projectile_profile_has_timing(
    const OsrsCombatAltProjectileProfile* profile
) {
    return profile &&
        profile->projectile_start_height != OSRS_COMBAT_PROJECTILE_MISSING &&
        profile->projectile_end_height != OSRS_COMBAT_PROJECTILE_MISSING &&
        profile->projectile_delay != OSRS_COMBAT_PROJECTILE_MISSING &&
        profile->projectile_angle != OSRS_COMBAT_PROJECTILE_MISSING &&
        profile->projectile_progress != OSRS_COMBAT_PROJECTILE_MISSING &&
        profile->projectile_step_multiplier != OSRS_COMBAT_PROJECTILE_MISSING;
}

static inline void osrs_combat_projectile_apply_timing(
    OsrsCombatProjectileProfile* out,
    const OsrsCombatProjectileProfile* timing
) {
    if (!out || !timing) return;
    out->hit_delay = timing->hit_delay;
    out->client_delay = timing->client_delay;
    out->projectile_start_height = timing->projectile_start_height;
    out->projectile_end_height = timing->projectile_end_height;
    out->projectile_delay = timing->projectile_delay;
    out->projectile_angle = timing->projectile_angle;
    out->projectile_length_adjustment = timing->projectile_length_adjustment;
    out->projectile_progress = timing->projectile_progress;
    out->projectile_step_multiplier = timing->projectile_step_multiplier;
}

static inline void osrs_combat_projectile_apply_alt_timing(
    OsrsCombatProjectileProfile* out,
    const OsrsCombatAltProjectileProfile* timing
) {
    if (!out || !timing) return;
    out->projectile_start_height = timing->projectile_start_height;
    out->projectile_end_height = timing->projectile_end_height;
    out->projectile_delay = timing->projectile_delay;
    out->projectile_angle = timing->projectile_angle;
    out->projectile_length_adjustment = timing->projectile_length_adjustment;
    out->projectile_progress = timing->projectile_progress;
    out->projectile_step_multiplier = timing->projectile_step_multiplier;
}

static inline int osrs_combat_visual_build_projectile_sequence(
    const OsrsCombatProjectileProfile* base,
    const OsrsCombatVisualRow* effect,
    OsrsCombatProjectileSequencePart* out,
    int capacity
) {
    if (!out || capacity < 0) abort();
    if (!base && !effect) return 0;

    int sequence_count = osrs_combat_visual_projectile_count(effect);
    int out_count = 0;
    const OsrsCombatProjectileProfile* timing =
        effect && osrs_combat_projectile_profile_has_timing(&effect->projectile)
            ? &effect->projectile
            : base;

    for (int i = 0; i < sequence_count; i++) {
        int show_impact =
            osrs_combat_visual_show_impact_for_projectile(effect, i, sequence_count);
        int use_alt = i > 0 && effect &&
            osrs_combat_alt_projectile_profile_has_timing(&effect->alt_projectile);

        if (effect && (
                effect->aux_travel_spotanim_id != OSRS_COMBAT_PROJECTILE_MISSING ||
                effect->aux_projectile_model_id != OSRS_COMBAT_PROJECTILE_MISSING)) {
            if (out_count >= capacity) return -1;
            OsrsCombatProjectileProfile aux = {
                .launch_spotanim_id = i == 0
                    ? effect->projectile.launch_spotanim_id
                    : OSRS_COMBAT_PROJECTILE_MISSING,
                .travel_spotanim_id = effect->aux_travel_spotanim_id,
                .impact_spotanim_id = show_impact
                    ? effect->aux_impact_spotanim_id
                    : OSRS_COMBAT_PROJECTILE_MISSING,
                .projectile_model_id = effect->aux_projectile_model_id,
                .projectile_anim_id = effect->aux_projectile_anim_id,
                .hit_delay = OSRS_COMBAT_PROJECTILE_MISSING,
                .client_delay = OSRS_COMBAT_PROJECTILE_MISSING,
                .projectile_start_height = OSRS_COMBAT_PROJECTILE_MISSING,
                .projectile_end_height = OSRS_COMBAT_PROJECTILE_MISSING,
                .projectile_delay = OSRS_COMBAT_PROJECTILE_MISSING,
                .projectile_angle = OSRS_COMBAT_PROJECTILE_MISSING,
                .projectile_length_adjustment = OSRS_COMBAT_PROJECTILE_MISSING,
                .projectile_progress = OSRS_COMBAT_PROJECTILE_MISSING,
                .projectile_step_multiplier = OSRS_COMBAT_PROJECTILE_MISSING,
                .projectile_count = (int16_t)sequence_count,
            };
            osrs_combat_projectile_apply_timing(&aux, timing);
            if (use_alt)
                osrs_combat_projectile_apply_alt_timing(&aux, &effect->alt_projectile);
            out[out_count++] = (OsrsCombatProjectileSequencePart){
                .projectile = aux,
                .sequence_index = (int16_t)i,
                .sequence_count = (int16_t)sequence_count,
            };
        }

        if (!base) continue;
        OsrsCombatProjectileProfile primary = *base;
        if (timing)
            osrs_combat_projectile_apply_timing(&primary, timing);
        if (use_alt)
            osrs_combat_projectile_apply_alt_timing(&primary, &effect->alt_projectile);

        if (effect && i == 0 &&
                effect->projectile.launch_spotanim_id != OSRS_COMBAT_PROJECTILE_MISSING &&
                effect->aux_travel_spotanim_id == OSRS_COMBAT_PROJECTILE_MISSING) {
            primary.launch_spotanim_id = effect->projectile.launch_spotanim_id;
        } else if (effect && i == 0 &&
                effect->double_launch_spotanim_id != OSRS_COMBAT_PROJECTILE_MISSING) {
            primary.launch_spotanim_id = effect->double_launch_spotanim_id;
        } else if (i > 0) {
            primary.launch_spotanim_id = OSRS_COMBAT_PROJECTILE_MISSING;
        }
        if (!show_impact)
            primary.impact_spotanim_id = OSRS_COMBAT_PROJECTILE_MISSING;
        primary.projectile_count = sequence_count;

        if (out_count >= capacity) return -1;
        out[out_count++] = (OsrsCombatProjectileSequencePart){
            .projectile = primary,
            .sequence_index = (int16_t)i,
            .sequence_count = (int16_t)sequence_count,
        };
    }
    return out_count;
}

static inline int osrs_combat_visual_style_matches(
    const OsrsCombatVisualRow* row, AttackStyle style
) {
    return row->style == OSRS_COMBAT_VISUAL_STYLE_ANY || row->style == (int8_t)style;
}

static inline int osrs_combat_visual_stance_matches(
    const OsrsCombatVisualRow* row, int stance_idx
) {
    return row->stance_idx < 0 || row->stance_idx == stance_idx;
}

static inline int osrs_combat_visual_fight_style_stance_idx(
    FightStyle fight_style
) {
    switch (fight_style) {
        case FIGHT_STYLE_ACCURATE:
        case FIGHT_STYLE_AUTOCAST:
            return 0;
        case FIGHT_STYLE_AGGRESSIVE:
        case FIGHT_STYLE_RAPID:
            return 1;
        case FIGHT_STYLE_CONTROLLED:
            return 2;
        case FIGHT_STYLE_DEFENSIVE:
        case FIGHT_STYLE_LONGRANGE:
        case FIGHT_STYLE_DEFENSIVE_AUTOCAST:
            return 3;
        default:
            fprintf(stderr, "unknown fight style: %d\n", fight_style);
            abort();
    }
}

static inline int osrs_combat_visual_key_matches(
    const OsrsCombatVisualRow* row,
    int kind,
    int32_t key_id,
    const char* key_name
) {
    if (row->kind != kind) return 0;
    if (key_id != OSRS_COMBAT_PROJECTILE_MISSING && row->key_id == key_id) return 1;
    return key_name && key_name[0] && row->key_name &&
        strcmp(row->key_name, key_name) == 0;
}

static inline const OsrsCombatVisualRow* osrs_combat_visual_find_row_in_table(
    const OsrsCombatVisualRow* rows,
    size_t row_count,
    int kind,
    int32_t key_id,
    const char* key_name,
    AttackStyle style,
    int stance_idx,
    int require_attack_anim,
    int require_projectile
) {
    const OsrsCombatVisualRow* fallback = NULL;
    const OsrsCombatVisualRow* style_fallback = NULL;
    const OsrsCombatVisualRow* stance_fallback = NULL;
    for (size_t i = 0; i < row_count; i++) {
        const OsrsCombatVisualRow* row = &rows[i];
        if (!osrs_combat_visual_key_matches(row, kind, key_id, key_name)) continue;
        if (!osrs_combat_visual_style_matches(row, style)) continue;
        if (!osrs_combat_visual_stance_matches(row, stance_idx)) continue;
        if (require_attack_anim &&
                row->attack_anim_id == OSRS_COMBAT_VISUAL_NO_ANIMATION) {
            continue;
        }
        if (require_projectile && !osrs_combat_visual_row_has_projectile(row)) {
            continue;
        }
        int exact_style = row->style == (int8_t)style;
        int any_style = row->style == OSRS_COMBAT_VISUAL_STYLE_ANY;
        int exact_stance = stance_idx >= 0 && row->stance_idx == stance_idx;
        int any_stance = row->stance_idx < 0;
        if (exact_style && exact_stance) return row;
        if (!style_fallback && exact_style && any_stance)
            style_fallback = row;
        if (!stance_fallback && any_style && exact_stance)
            stance_fallback = row;
        if (!fallback && any_stance)
            fallback = row;
    }
    return style_fallback ? style_fallback
        : (stance_fallback ? stance_fallback : fallback);
}

static inline const OsrsCombatVisualRow* osrs_combat_visual_find_row(
    int kind,
    int32_t key_id,
    const char* key_name,
    AttackStyle style,
    int stance_idx,
    int require_attack_anim,
    int require_projectile
) {
    const OsrsCombatVisualRow* colosseum_row =
        osrs_combat_visual_find_row_in_table(
            OSRS_COMBAT_VISUAL_COLOSSEUM_ROWS,
            OSRS_COMBAT_VISUAL_COLOSSEUM_ROW_COUNT,
            kind, key_id, key_name, style, stance_idx,
            require_attack_anim, require_projectile);
    if (colosseum_row) return colosseum_row;
    return osrs_combat_visual_find_row_in_table(
        OSRS_COMBAT_VISUAL_ROWS,
        OSRS_COMBAT_VISUAL_ROW_COUNT,
        kind, key_id, key_name, style, stance_idx,
        require_attack_anim, require_projectile);
}

static inline const OsrsCombatVisualRow* osrs_combat_visual_find_item_id_stance(
    uint16_t item_id, AttackStyle style, int stance_idx
) {
    return osrs_combat_visual_find_row(
        OSRS_COMBAT_VISUAL_KIND_ITEM, item_id, NULL, style, stance_idx, 1, 0);
}

static inline const OsrsCombatVisualRow* osrs_combat_visual_find_item_projectile_id(
    uint16_t item_id, AttackStyle style
) {
    return osrs_combat_visual_find_row(
        OSRS_COMBAT_VISUAL_KIND_ITEM, item_id, NULL, style,
        OSRS_COMBAT_VISUAL_STANCE_ANY, 0, 1);
}

static inline const OsrsCombatVisualRow* osrs_combat_visual_find_special_item_id(
    uint16_t item_id, AttackStyle style
) {
    return osrs_combat_visual_find_row(
        OSRS_COMBAT_VISUAL_KIND_SPECIAL, item_id, NULL, style,
        OSRS_COMBAT_VISUAL_STANCE_ANY, 1, 0);
}

static inline const OsrsCombatVisualRow* osrs_combat_visual_find_special_projectile_item_id(
    uint16_t item_id, AttackStyle style
) {
    return osrs_combat_visual_find_row(
        OSRS_COMBAT_VISUAL_KIND_SPECIAL, item_id, NULL, style,
        OSRS_COMBAT_VISUAL_STANCE_ANY, 0, 1);
}

static inline const OsrsCombatVisualRow* osrs_combat_visual_find_npc_id(
    uint16_t npc_id, AttackStyle style
) {
    return osrs_combat_visual_find_row(
        OSRS_COMBAT_VISUAL_KIND_NPC, npc_id, NULL, style,
        OSRS_COMBAT_VISUAL_STANCE_ANY, 0, 0);
}

static inline const char* osrs_combat_visual_spell_name(int spell_type) {
    switch (spell_type) {
        case OSRS_COMBAT_VISUAL_SPELL_ICE_BARRAGE: return "Ice Barrage";
        case OSRS_COMBAT_VISUAL_SPELL_BLOOD_BARRAGE: return "Blood Barrage";
        default: return NULL;
    }
}

static inline const OsrsCombatVisualRow* osrs_combat_visual_find_spell(
    int spell_type
) {
    return osrs_combat_visual_find_row(
        OSRS_COMBAT_VISUAL_KIND_SPELL,
        spell_type,
        osrs_combat_visual_spell_name(spell_type),
        ATTACK_STYLE_MAGIC,
        OSRS_COMBAT_VISUAL_STANCE_ANY,
        0,
        1);
}

static inline const OsrsCombatProjectileProfile* osrs_combat_visual_spell_projectile(
    int spell_type
) {
    const OsrsCombatVisualRow* row = osrs_combat_visual_find_spell(spell_type);
    return row ? &row->projectile : NULL;
}

static inline int osrs_combat_visual_special_fallback_anim(uint16_t item_id) {
    for (size_t i = 0;
            i < sizeof(OSRS_COMBAT_SPECIAL_FALLBACKS) /
                sizeof(OSRS_COMBAT_SPECIAL_FALLBACKS[0]);
            i++) {
        if (OSRS_COMBAT_SPECIAL_FALLBACKS[i].item_id == item_id) {
            return OSRS_COMBAT_SPECIAL_FALLBACKS[i].special_attack_anim_id;
        }
    }
    return OSRS_COMBAT_VISUAL_NO_ANIMATION;
}

static inline int osrs_combat_visual_weapon_attack_anim_for_stance(
    uint8_t item_db_idx,
    AttackStyle style,
    int stance_idx,
    int is_special,
    int fallback_anim_id
) {
    if (item_db_idx >= NUM_ITEMS) return fallback_anim_id;
    uint16_t item_id = ITEM_DATABASE[item_db_idx].item_id;
    if (is_special) {
        const OsrsCombatVisualRow* special =
            osrs_combat_visual_find_special_item_id(item_id, style);
        if (special) return special->attack_anim_id;
        int fallback_special = osrs_combat_visual_special_fallback_anim(item_id);
        if (fallback_special != OSRS_COMBAT_VISUAL_NO_ANIMATION)
            return fallback_special;
    }
    const OsrsCombatVisualRow* row =
        osrs_combat_visual_find_item_id_stance(item_id, style, stance_idx);
    if (!row) return fallback_anim_id;
    return row->attack_anim_id;
}

static inline int osrs_combat_visual_weapon_attack_anim_for_fight_style(
    uint8_t item_db_idx,
    AttackStyle style,
    FightStyle fight_style,
    int is_special,
    int fallback_anim_id
) {
    return osrs_combat_visual_weapon_attack_anim_for_stance(
        item_db_idx, style, osrs_combat_visual_fight_style_stance_idx(fight_style),
        is_special, fallback_anim_id);
}

static inline const OsrsCombatWeaponProjectileDefault*
osrs_combat_visual_default_ranged_projectile(uint16_t weapon_item_id) {
    for (size_t i = 0;
            i < sizeof(OSRS_RANGED_PROJECTILE_DEFAULTS) /
                sizeof(OSRS_RANGED_PROJECTILE_DEFAULTS[0]);
            i++) {
        if (OSRS_RANGED_PROJECTILE_DEFAULTS[i].weapon_item_id == weapon_item_id) {
            return &OSRS_RANGED_PROJECTILE_DEFAULTS[i];
        }
    }
    return NULL;
}

static inline const OsrsCombatProjectileProfile* osrs_combat_projectile_profile(
    OsrsCombatProjectileVisual visual
) {
    const OsrsCombatVisualRow* row = NULL;
    switch (visual) {
        case OSRS_COMBAT_PROJECTILE_NONE:
            return NULL;
        case OSRS_COMBAT_PROJECTILE_BOLT:
            row = osrs_combat_visual_find_item_projectile_id(
                OSRS_ITEM_ID_DIAMOND_BOLTS_E, ATTACK_STYLE_RANGED);
            break;
        case OSRS_COMBAT_PROJECTILE_RUNE_ARROW:
            row = osrs_combat_visual_find_item_projectile_id(
                OSRS_ITEM_ID_RUNE_ARROW, ATTACK_STYLE_RANGED);
            break;
        case OSRS_COMBAT_PROJECTILE_DRAGON_ARROW:
            row = osrs_combat_visual_find_item_projectile_id(
                OSRS_ITEM_ID_DRAGON_ARROWS, ATTACK_STYLE_RANGED);
            break;
        case OSRS_COMBAT_PROJECTILE_DRAGON_DART:
            row = osrs_combat_visual_find_item_projectile_id(
                OSRS_ITEM_ID_DRAGON_DART, ATTACK_STYLE_RANGED);
            break;
        case OSRS_COMBAT_PROJECTILE_TRIDENT:
            return &OSRS_POWERED_STAFF_PROJECTILE_PROFILE;
        default:
            fprintf(stderr, "unknown combat projectile visual: %d\n", visual);
            abort();
    }
    if (!row) {
        fprintf(stderr, "missing generated combat projectile visual: %d\n", visual);
        abort();
    }
    return &row->projectile;
}

static inline const OsrsCombatProjectileProfile* osrs_combat_visual_ranged_projectile_profile(
    uint8_t item_db_idx, OsrsCombatProjectileVisual fallback
) {
    if (item_db_idx >= NUM_ITEMS) return osrs_combat_projectile_profile(fallback);
    uint16_t item_id = ITEM_DATABASE[item_db_idx].item_id;
    const OsrsCombatVisualRow* item_projectile =
        osrs_combat_visual_find_item_projectile_id(item_id, ATTACK_STYLE_RANGED);
    if (item_projectile) return &item_projectile->projectile;
    const OsrsCombatWeaponProjectileDefault* default_projectile =
        osrs_combat_visual_default_ranged_projectile(item_id);
    if (default_projectile) {
        const OsrsCombatVisualRow* projectile =
            osrs_combat_visual_find_item_projectile_id(
                default_projectile->projectile_item_id, ATTACK_STYLE_RANGED);
        if (projectile) return &projectile->projectile;
    }
    return osrs_combat_projectile_profile(fallback);
}

static inline const OsrsCombatProjectileProfile* osrs_combat_visual_ranged_special_projectile_profile(
    uint8_t item_db_idx
) {
    if (item_db_idx >= NUM_ITEMS) return NULL;
    const OsrsCombatVisualRow* special =
        osrs_combat_visual_find_special_projectile_item_id(
            ITEM_DATABASE[item_db_idx].item_id, ATTACK_STYLE_RANGED);
    return special ? &special->projectile : NULL;
}

static inline int osrs_combat_visual_item_is_powered_staff(uint16_t item_id) {
    for (size_t i = 0;
            i < sizeof(OSRS_POWERED_STAFF_ITEMS) / sizeof(OSRS_POWERED_STAFF_ITEMS[0]);
            i++) {
        if (OSRS_POWERED_STAFF_ITEMS[i] == item_id) return 1;
    }
    return 0;
}

static inline OsrsCombatProjectileVisual osrs_combat_visual_magic_projectile(
    uint8_t item_db_idx
) {
    if (item_db_idx >= NUM_ITEMS) return OSRS_COMBAT_PROJECTILE_NONE;
    return osrs_combat_visual_item_is_powered_staff(
        ITEM_DATABASE[item_db_idx].item_id)
        ? OSRS_COMBAT_PROJECTILE_TRIDENT
        : OSRS_COMBAT_PROJECTILE_NONE;
}

static inline const OsrsCombatProjectileProfile* osrs_combat_visual_magic_projectile_profile(
    uint8_t item_db_idx
) {
    if (item_db_idx >= NUM_ITEMS) {
        return osrs_combat_projectile_profile(OSRS_COMBAT_PROJECTILE_NONE);
    }
    uint16_t item_id = ITEM_DATABASE[item_db_idx].item_id;
    const OsrsCombatVisualRow* item_projectile =
        osrs_combat_visual_find_item_projectile_id(item_id, ATTACK_STYLE_MAGIC);
    if (item_projectile) return &item_projectile->projectile;
    if (item_id == OSRS_ITEM_ID_TUMEKENS_SHADOW)
        return &OSRS_TUMEKENS_SHADOW_PROJECTILE_PROFILE;
    return osrs_combat_projectile_profile(
        osrs_combat_visual_magic_projectile(item_db_idx));
}

static inline int osrs_combat_visual_magic_attack_anim_for_fight_style(
    uint8_t item_db_idx, FightStyle fight_style, int is_special, int fallback_anim_id
) {
    if (item_db_idx >= NUM_ITEMS) return fallback_anim_id;
    if (!osrs_combat_visual_item_is_powered_staff(
            ITEM_DATABASE[item_db_idx].item_id)) {
        return fallback_anim_id;
    }
    int anim = osrs_combat_visual_weapon_attack_anim_for_fight_style(
        item_db_idx, ATTACK_STYLE_MAGIC, fight_style, is_special,
        OSRS_COMBAT_VISUAL_NO_ANIMATION);
    if (anim != OSRS_COMBAT_VISUAL_NO_ANIMATION) return anim;
    if (fight_style != FIGHT_STYLE_AUTOCAST) {
        anim = osrs_combat_visual_weapon_attack_anim_for_fight_style(
            item_db_idx, ATTACK_STYLE_MAGIC, FIGHT_STYLE_AUTOCAST, is_special,
            OSRS_COMBAT_VISUAL_NO_ANIMATION);
        if (anim != OSRS_COMBAT_VISUAL_NO_ANIMATION) return anim;
    }
    return OSRS_PLAYER_POWERED_STAFF_ATTACK_ANIM;
}

#endif
