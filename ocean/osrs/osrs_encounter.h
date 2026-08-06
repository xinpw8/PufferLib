#ifndef OSRS_ENCOUNTER_H
#define OSRS_ENCOUNTER_H

#include <ctype.h>
#include <errno.h>
#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "osrs_types.h"
#include "osrs_items.h"
#include "osrs_pathfinding.h"
#include "osrs_combat.h"
#include "osrs_consumables.h"
#include "osrs_item_effects.h"
#include "osrs_human_input_types.h"
#include "osrs_render_motion.h"
#include "osrs_lab.h"

typedef struct EncounterState EncounterState;
typedef struct EncounterContext EncounterContext;

#define ENCOUNTER_RENDER_HITS_MAX 32

static inline void encounter_abort_unknown_config(
    const char* encounter_name, const char* config_type, const char* key
) {
    fprintf(stderr, "%s unknown %s config key: %s\n",
        encounter_name, config_type, key);
    abort();
}

static inline int encounter_require_binary_config(
    const char* encounter_name, const char* key, int value
) {
    if (value != 0 && value != 1) {
        fprintf(stderr, "%s config %s must be 0 or 1, got %d\n",
            encounter_name, key, value);
        abort();
    }
    return value;
}

static inline int encounter_require_int_range_config(
    const char* encounter_name, const char* key, int value, int min_value, int max_value
) {
    if (value < min_value || value > max_value) {
        fprintf(stderr, "%s config %s must be in [%d, %d], got %d\n",
            encounter_name, key, min_value, max_value, value);
        abort();
    }
    return value;
}

#define ENCOUNTER_MAX_PENDING_HITS 32

typedef struct {
    int8_t active;
    int8_t ticks_remaining;
    int8_t attack_style;
    int8_t check_prayer;
    int8_t prayer_check_delay;

    int8_t spell_type;
    int8_t source_npc_type;
    int8_t source_npc_slot;
    int8_t hit_success;
    int8_t elysian_reduced;
    int16_t damage;
} EncounterPendingHit;

/* Replicated per NPC per env and streamed every step, so the width is load-bearing for
 * throughput, not tidiness. damage trails the int8 run so the record packs to 12 with no
 * padding; C++ also requires designated initializers in this order, so keep it last. */
static_assert(sizeof(EncounterPendingHit) == 12, "pending hit record must stay 12 bytes");

typedef struct {
    EncounterPendingHit hits[ENCOUNTER_MAX_PENDING_HITS];
    int count;
} EncounterPendingHitQueue;

static inline EncounterPendingHit encounter_pending_hit_resolved_at_throw(
    int raw_damage, int ticks_remaining, int attack_style, int overhead_prayer,
    int source_npc_type, int source_npc_slot, int accuracy_hit, int* out_prayed
) {
    EncounterProtectResolve pr =
        encounter_resolve_protect_at_throw(raw_damage, overhead_prayer, attack_style);
    if (out_prayed) *out_prayed = pr.prayed;
    return (EncounterPendingHit){
        .active = 1,
        .ticks_remaining = (int8_t)ticks_remaining,
        .attack_style = (int8_t)attack_style,
        .check_prayer = 0,
        .prayer_check_delay = 0,
        .spell_type = ENCOUNTER_SPELL_NONE,
        .source_npc_type = (int8_t)source_npc_type,
        .source_npc_slot = (int8_t)source_npc_slot,
        .hit_success = (int8_t)(accuracy_hit && !pr.prayed),
        .elysian_reduced = 0,
        .damage = (int16_t)pr.frozen_damage,
    };
}

static inline void encounter_pending_hit_queue_clear(EncounterPendingHitQueue* q) {
    memset(q, 0, sizeof(*q));
}

static inline EncounterPendingHit* encounter_pending_hit_queue_push(
    EncounterPendingHitQueue* q,
    EncounterPendingHit hit,
    const char* owner_label,
    int tick,
    int slot,
    int type
) {
    if (q->count < 0 || q->count > ENCOUNTER_MAX_PENDING_HITS) {
        fprintf(stderr,
            "%s pending-hit queue corrupt tick=%d slot=%d type=%d count=%d\n",
            owner_label, tick, slot, type, q->count);
        abort();
    }
    if (q->count >= ENCOUNTER_MAX_PENDING_HITS) {
        fprintf(stderr,
            "%s pending-hit queue overflow tick=%d slot=%d type=%d count=%d "
            "delay=%d style=%d spell=%d damage=%d\n",
            owner_label, tick, slot, type, q->count,
            hit.ticks_remaining, hit.attack_style, hit.spell_type, hit.damage);
        abort();
    }

    hit.active = 1;
    q->hits[q->count] = hit;
    return &q->hits[q->count++];
}

static inline void encounter_pending_hit_queue_remove(
    EncounterPendingHitQueue* q,
    int idx,
    const char* owner_label
) {
    if (q->count < 0 || q->count > ENCOUNTER_MAX_PENDING_HITS) {
        fprintf(stderr, "%s pending-hit queue corrupt before remove count=%d\n",
            owner_label, q->count);
        abort();
    }
    if (idx < 0 || idx >= q->count) {
        fprintf(stderr, "%s pending-hit queue invalid remove idx=%d count=%d\n",
            owner_label, idx, q->count);
        abort();
    }
    for (int i = idx + 1; i < q->count; i++)
        q->hits[i - 1] = q->hits[i];
    q->count--;
    memset(&q->hits[q->count], 0, sizeof(q->hits[q->count]));
}

static inline const EncounterPendingHit* encounter_pending_hit_queue_earliest(
    const EncounterPendingHitQueue* q
) {
    const EncounterPendingHit* best = NULL;
    for (int i = 0; i < q->count; i++) {
        const EncounterPendingHit* hit = &q->hits[i];
        if (!hit->active) continue;
        if (!best || hit->ticks_remaining < best->ticks_remaining)
            best = hit;
    }
    return best;
}

static inline int encounter_pending_hit_queue_damage_sum(
    const EncounterPendingHitQueue* q
) {
    int total = 0;
    for (int i = 0; i < q->count; i++) {
        if (q->hits[i].active)
            total += q->hits[i].damage;
    }
    return total;
}

#define ENCOUNTER_MAX_OVERLAY_TILES 16
#define ENCOUNTER_MAX_OVERLAY_ADDS 4
#define ENCOUNTER_MAX_OVERLAY_TILE_SHADOWS 48
#define ENCOUNTER_MAX_OVERLAY_FLOATING_MODELS 16
#define ENCOUNTER_MAX_ACTIVE_MODIFIERS 16
#define ENCOUNTER_OVERLAY_STATUS_TEXT_LEN 64

#define ENCOUNTER_MAX_OVERLAY_PROJECTILES 48

typedef enum {
    ENCOUNTER_PROJECTILE_MOTION_OSRS_FLIGHT = 0,
    ENCOUNTER_PROJECTILE_MOTION_TARGET_ANCHORED = 1,
} EncounterProjectileMotionMode;

typedef enum {
    ENCOUNTER_PROJECTILE_TARGET_FIXED = 0,
    ENCOUNTER_PROJECTILE_TARGET_PLAYER = 1,
    ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT = 2,
} EncounterProjectileTargetKind;

typedef struct {
    int active;
    int x, y;
    float scale;
} EncounterTileShadow;

typedef struct {
    int active;
    int anchor_kind;
    int npc_slot;
    int x, y;
    uint32_t model_id;
    int anim_id;
    float height_offset;
    float lateral_offset;
    float scale;
} EncounterFloatingModel;

typedef struct {
    int active;
    int modifier;
    int tier;
} EncounterActiveModifier;

typedef struct {
    struct { int x, y, active; } hazards[ENCOUNTER_MAX_OVERLAY_TILES];
    int hazard_count;

    int boss_x, boss_y, boss_visible;
    int boss_form;
    int boss_size;

    struct { int x, y, active, variant; } adds[ENCOUNTER_MAX_OVERLAY_ADDS];
    int add_count;

    EncounterTileShadow tile_shadows[ENCOUNTER_MAX_OVERLAY_TILE_SHADOWS];
    int tile_shadow_count;

    EncounterFloatingModel floating_models[ENCOUNTER_MAX_OVERLAY_FLOATING_MODELS];
    int floating_model_count;

    EncounterActiveModifier active_modifiers[ENCOUNTER_MAX_ACTIVE_MODIFIERS];
    int active_modifier_count;

    struct {
        int active;
        int src_x, src_y;
        int dst_x, dst_y;
        int style;
        int damage;
        int duration_ticks;
        int start_h;
        int end_h;
        int curve;
        float arc_height;
        int tracks_target;
        int source_kind;
        int source_npc_slot;
        int target_kind;
        int target_npc_slot;
        int start_delay;
        int motion_mode;
        float offset_x, offset_y, offset_z;
        int src_size;
        int dst_size;
        uint32_t model_id;
        int anim_id;
        int travel_gfx_id;
        int launch_gfx_id;
        int impact_gfx_id;
    } projectiles[ENCOUNTER_MAX_OVERLAY_PROJECTILES];
    int projectile_count;

    int melee_target_active;
    int melee_target_x, melee_target_y;

    int status_text_active;
    char status_text[ENCOUNTER_OVERLAY_STATUS_TEXT_LEN];
} EncounterOverlay;

static inline int encounter_attack_style_to_proj_style(int attack_style) {
    switch (attack_style) {
        case ATTACK_STYLE_RANGED: return 0;
        case ATTACK_STYLE_MAGIC:  return 1;
        case ATTACK_STYLE_MELEE:  return 2;
        default: return 0;
    }
}

static inline EncounterProjectileDelayKind encounter_projectile_delay_kind_for_style(
    AttackStyle style
) {
    switch (style) {
        case ATTACK_STYLE_MELEE:
            return ENCOUNTER_PROJECTILE_DELAY_MELEE;
        case ATTACK_STYLE_MAGIC:
            return ENCOUNTER_PROJECTILE_DELAY_MAGIC;
        case ATTACK_STYLE_RANGED:
            return ENCOUNTER_PROJECTILE_DELAY_RANGED;
        case ATTACK_STYLE_NONE:
            return ENCOUNTER_PROJECTILE_DELAY_MELEE;
    }
    abort();
}

static inline OffensivePrayer encounter_offensive_prayer_for_style(AttackStyle style) {
    switch (style) {
        case ATTACK_STYLE_MELEE:
            return OFFENSIVE_PRAYER_PIETY;
        case ATTACK_STYLE_RANGED:
            return OFFENSIVE_PRAYER_RIGOUR;
        case ATTACK_STYLE_MAGIC:
            return OFFENSIVE_PRAYER_AUGURY;
        case ATTACK_STYLE_NONE:
            return OFFENSIVE_PRAYER_NONE;
    }
    abort();
}

static inline int encounter_emit_projectile(
    EncounterOverlay* ov,
    int src_x, int src_y, int dst_x, int dst_y,
    int style, int damage,
    int duration_ticks, int start_h, int end_h, int curve,
    float arc_height, int tracks_target, int src_size, int dst_size,
    uint32_t model_id, int impact_gfx_id
) {
    if (ov->projectile_count >= ENCOUNTER_MAX_OVERLAY_PROJECTILES) {
        fprintf(stderr, "encounter overlay projectile capacity exceeded: %d\n",
            ENCOUNTER_MAX_OVERLAY_PROJECTILES);
        abort();
    }
    int i = ov->projectile_count++;
    ov->projectiles[i].active = 1;
    ov->projectiles[i].src_x = src_x;
    ov->projectiles[i].src_y = src_y;
    ov->projectiles[i].dst_x = dst_x;
    ov->projectiles[i].dst_y = dst_y;
    ov->projectiles[i].style = style;
    ov->projectiles[i].damage = damage;
    ov->projectiles[i].duration_ticks = duration_ticks;
    ov->projectiles[i].start_h = start_h;
    ov->projectiles[i].end_h = end_h;
    ov->projectiles[i].curve = curve;
    ov->projectiles[i].arc_height = arc_height;
    ov->projectiles[i].start_delay = 0;
    ov->projectiles[i].motion_mode = ENCOUNTER_PROJECTILE_MOTION_OSRS_FLIGHT;
    ov->projectiles[i].offset_x = 0.0f;
    ov->projectiles[i].offset_y = 0.0f;
    ov->projectiles[i].offset_z = 0.0f;
    ov->projectiles[i].tracks_target = tracks_target;
    ov->projectiles[i].source_kind = ENCOUNTER_PROJECTILE_TARGET_FIXED;
    ov->projectiles[i].source_npc_slot = -1;
    ov->projectiles[i].target_kind = tracks_target
        ? ENCOUNTER_PROJECTILE_TARGET_PLAYER
        : ENCOUNTER_PROJECTILE_TARGET_FIXED;
    ov->projectiles[i].target_npc_slot = -1;
    ov->projectiles[i].src_size = src_size;
    ov->projectiles[i].dst_size = dst_size;
    ov->projectiles[i].model_id = model_id;
    ov->projectiles[i].anim_id = -1;
    ov->projectiles[i].travel_gfx_id = 0;
    ov->projectiles[i].launch_gfx_id = 0;
    ov->projectiles[i].impact_gfx_id = impact_gfx_id;
    return i;
}

static inline void encounter_emit_tile_shadow(
    EncounterOverlay* ov, int x, int y, float scale
) {
    if (scale <= 0.0f) {
        fprintf(stderr, "encounter tile shadow scale must be positive: %f\n", scale);
        abort();
    }
    if (ov->tile_shadow_count >= ENCOUNTER_MAX_OVERLAY_TILE_SHADOWS) {
        fprintf(stderr, "encounter overlay tile shadow capacity exceeded: %d\n",
            ENCOUNTER_MAX_OVERLAY_TILE_SHADOWS);
        abort();
    }
    int i = ov->tile_shadow_count++;
    ov->tile_shadows[i] = (EncounterTileShadow){
        .active = 1,
        .x = x,
        .y = y,
        .scale = scale,
    };
}

static inline void encounter_emit_floating_model(
    EncounterOverlay* ov,
    int anchor_kind,
    int npc_slot,
    int x,
    int y,
    uint32_t model_id,
    int anim_id,
    float height_offset,
    float lateral_offset,
    float scale
) {
    if (model_id == 0 || scale <= 0.0f) {
        fprintf(stderr, "encounter floating model invalid model=%u scale=%f\n",
            model_id, scale);
        abort();
    }
    if (ov->floating_model_count >= ENCOUNTER_MAX_OVERLAY_FLOATING_MODELS) {
        fprintf(stderr, "encounter floating model capacity exceeded: %d\n",
            ENCOUNTER_MAX_OVERLAY_FLOATING_MODELS);
        abort();
    }
    int i = ov->floating_model_count++;
    ov->floating_models[i] = (EncounterFloatingModel){
        .active = 1,
        .anchor_kind = anchor_kind,
        .npc_slot = npc_slot,
        .x = x,
        .y = y,
        .model_id = model_id,
        .anim_id = anim_id,
        .height_offset = height_offset,
        .lateral_offset = lateral_offset,
        .scale = scale,
    };
}

static inline void encounter_emit_active_modifier(
    EncounterOverlay* ov,
    int modifier,
    int tier
) {
    if (modifier < 0 || tier <= 0) {
        fprintf(stderr, "encounter active modifier invalid modifier=%d tier=%d\n",
            modifier, tier);
        abort();
    }
    if (ov->active_modifier_count >= ENCOUNTER_MAX_ACTIVE_MODIFIERS) {
        fprintf(stderr, "encounter active modifier capacity exceeded: %d\n",
            ENCOUNTER_MAX_ACTIVE_MODIFIERS);
        abort();
    }
    int i = ov->active_modifier_count++;
    ov->active_modifiers[i] = (EncounterActiveModifier){
        .active = 1,
        .modifier = modifier,
        .tier = tier,
    };
}

static inline void encounter_require_projectile_slots(const EncounterOverlay* ov, int slots) {
    if (slots < 0 || ov->projectile_count + slots > ENCOUNTER_MAX_OVERLAY_PROJECTILES) {
        fprintf(stderr, "encounter overlay projectile capacity exceeded: need %d free from %d/%d\n",
            slots, ov->projectile_count, ENCOUNTER_MAX_OVERLAY_PROJECTILES);
        abort();
    }
}

static inline void encounter_require_projectile_index(const EncounterOverlay* ov, int projectile_idx) {
    if (projectile_idx < 0 || projectile_idx >= ov->projectile_count) {
        fprintf(stderr, "encounter projectile index out of range: %d/%d\n",
            projectile_idx, ov->projectile_count);
        abort();
    }
}

static inline void encounter_set_projectile_source_player(
    EncounterOverlay* ov, int projectile_idx
) {
    encounter_require_projectile_index(ov, projectile_idx);
    ov->projectiles[projectile_idx].source_kind = ENCOUNTER_PROJECTILE_TARGET_PLAYER;
    ov->projectiles[projectile_idx].source_npc_slot = -1;
}

static inline void encounter_set_projectile_source_npc_slot(
    EncounterOverlay* ov, int projectile_idx, int npc_slot
) {
    encounter_require_projectile_index(ov, projectile_idx);
    if (npc_slot < 0) {
        fprintf(stderr, "encounter projectile source npc slot is invalid: %d\n", npc_slot);
        abort();
    }
    ov->projectiles[projectile_idx].source_kind = ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT;
    ov->projectiles[projectile_idx].source_npc_slot = npc_slot;
}

static inline void encounter_set_projectile_target_npc_slot(
    EncounterOverlay* ov, int projectile_idx, int npc_slot
) {
    encounter_require_projectile_index(ov, projectile_idx);
    if (npc_slot < 0) {
        fprintf(stderr, "encounter projectile target npc slot is invalid: %d\n", npc_slot);
        abort();
    }
    ov->projectiles[projectile_idx].tracks_target = 1;
    ov->projectiles[projectile_idx].target_kind = ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT;
    ov->projectiles[projectile_idx].target_npc_slot = npc_slot;
}

static inline void encounter_set_projectile_motion_mode(
    EncounterOverlay* ov, int projectile_idx, int motion_mode
) {
    encounter_require_projectile_index(ov, projectile_idx);
    ov->projectiles[projectile_idx].motion_mode = motion_mode;
}

static inline void encounter_set_projectile_animation(
    EncounterOverlay* ov, int projectile_idx, int anim_id
) {
    encounter_require_projectile_index(ov, projectile_idx);
    ov->projectiles[projectile_idx].anim_id = anim_id;
}

static inline void encounter_set_projectile_launch_gfx(
    EncounterOverlay* ov, int projectile_idx, int launch_gfx_id
) {
    encounter_require_projectile_index(ov, projectile_idx);
    ov->projectiles[projectile_idx].launch_gfx_id = launch_gfx_id;
}

static inline void encounter_set_projectile_travel_gfx(
    EncounterOverlay* ov, int projectile_idx, int travel_gfx_id
) {
    encounter_require_projectile_index(ov, projectile_idx);
    ov->projectiles[projectile_idx].travel_gfx_id = travel_gfx_id;
}

static inline void encounter_set_projectile_offset(
    EncounterOverlay* ov, int projectile_idx,
    float offset_x, float offset_y, float offset_z
) {
    encounter_require_projectile_index(ov, projectile_idx);
    ov->projectiles[projectile_idx].offset_x = offset_x;
    ov->projectiles[projectile_idx].offset_y = offset_y;
    ov->projectiles[projectile_idx].offset_z = offset_z;
}

typedef struct {
    EntityType entity_type;
    int npc_def_id;
    int npc_visible;
    int npc_size;
    int npc_anim_id;
    int x, y;
    int dest_x, dest_y;
    int current_hitpoints, base_hitpoints;
    int special_energy;
    OverheadPrayer prayer;
    GearSet visible_gear;
    int frozen_ticks;
    int veng_active;
    int is_running;
    FightStyle fight_style;
    AttackStyle attack_style_this_tick;
    int magic_type_this_tick;
    int hit_landed_this_tick;
    int hit_damage;
    int render_hit_count;
    int render_hit_damage[ENCOUNTER_RENDER_HITS_MAX];
    int hit_was_successful;
    int hit_spell_type;
    int elysian_proc_this_tick;
    int cast_veng_this_tick;
    int ate_food_this_tick;
    int ate_karambwan_this_tick;
    int used_special_this_tick;
    uint8_t equipped[NUM_GEAR_SLOTS];
    int npc_slot;
    uint32_t npc_instance_id;
    RenderMovementKind render_movement_kind;
    int attack_target_entity_idx;
    const char* debug_npc_type_name;
    int debug_attack_timer;
    int debug_attack_style;
    int debug_manticore_state_active;
    int debug_manticore_cycle_step;
    int debug_manticore_orb_style[3];
} RenderEntity;

typedef enum {
    RENDER_ENTITY_FACE_MOVEMENT = 0,
    RENDER_ENTITY_FACE_ATTACK_TARGET = 1,
    RENDER_ENTITY_FACE_DEST_TILE = 2,
} RenderEntityFacingMode;

static inline int render_entity_find_previous_identity_index(
    const RenderEntity* previous,
    int previous_count,
    const int* previous_used,
    const RenderEntity* entity
) {
    if (entity->entity_type == ENTITY_PLAYER) {
        for (int j = 0; j < previous_count; j++) {
            if (!previous_used[j] && previous[j].entity_type == ENTITY_PLAYER) {
                return j;
            }
        }
        return -1;
    }

    if (entity->entity_type != ENTITY_NPC || entity->npc_slot < 0)
        return -1;

    for (int i = 0; i < previous_count; i++) {
        if (previous_used[i]) continue;
        if (previous[i].entity_type == ENTITY_NPC &&
                previous[i].npc_slot == entity->npc_slot &&
                previous[i].npc_def_id == entity->npc_def_id) {
            if ((previous[i].npc_instance_id || entity->npc_instance_id) &&
                    previous[i].npc_instance_id != entity->npc_instance_id) {
                continue;
            }
            return i;
        }
    }
    return -1;
}

static inline RenderEntityFacingMode render_entity_select_facing_mode(
    const RenderEntity* entity, int moved
) {
    if (entity->attack_target_entity_idx >= 0 || entity->current_hitpoints <= 0)
        return RENDER_ENTITY_FACE_ATTACK_TARGET;
    if (entity->attack_style_this_tick != ATTACK_STYLE_NONE)
        return RENDER_ENTITY_FACE_DEST_TILE;
    if (moved)
        return RENDER_ENTITY_FACE_MOVEMENT;
    return RENDER_ENTITY_FACE_DEST_TILE;
}

static inline void render_entity_from_player(const Player* p, RenderEntity* out) {
    memset(out, 0, sizeof(RenderEntity));
    out->entity_type = p->entity_type;
    out->npc_def_id = p->npc_def_id;
    out->npc_visible = p->npc_visible;
    out->npc_size = p->npc_size;
    out->npc_anim_id = p->npc_anim_id;
    out->x = p->x;
    out->y = p->y;
    out->dest_x = p->dest_x;
    out->dest_y = p->dest_y;
    out->current_hitpoints = p->current_hitpoints;
    out->base_hitpoints = p->base_hitpoints;
    out->special_energy = p->special_energy;
    out->prayer = p->prayer_display != PRAYER_NONE ? p->prayer_display : p->prayer;
    out->visible_gear = p->visible_gear;
    out->frozen_ticks = p->frozen_ticks;
    out->veng_active = p->veng_active;
    out->is_running = p->is_running;
    out->fight_style = p->fight_style;
    out->attack_style_this_tick = p->attack_style_this_tick;
    out->magic_type_this_tick = p->magic_type_this_tick;
    out->hit_landed_this_tick = p->hit_landed_this_tick;
    out->hit_damage = p->hit_damage;
    if (p->hit_landed_this_tick) {
        out->render_hit_count = 1;
        out->render_hit_damage[0] = p->hit_damage;
    }
    out->hit_was_successful = p->hit_was_successful;
    out->hit_spell_type = 0;
    out->elysian_proc_this_tick = p->elysian_proc_this_tick;
    out->cast_veng_this_tick = p->cast_veng_this_tick;
    out->ate_food_this_tick = p->ate_food_this_tick;
    out->ate_karambwan_this_tick = p->ate_karambwan_this_tick;
    out->used_special_this_tick = p->used_special_this_tick;
    memcpy(out->equipped, p->equipped, NUM_GEAR_SLOTS);
    out->npc_slot = -1;
    out->npc_instance_id = 0;
    out->render_movement_kind = RENDER_MOVEMENT_NORMAL;
    out->attack_target_entity_idx = -1;
}

static inline void encounter_resolve_attack_target(
    RenderEntity* entities, int count, int target_npc_slot
) {
    entities[0].attack_target_entity_idx = -1;
    if (target_npc_slot < 0) return;
    for (int i = 1; i < count; i++) {
        if (entities[i].npc_slot == target_npc_slot) {
            entities[0].attack_target_entity_idx = i;
            return;
        }
    }
}

#define ENCOUNTER_OVERHEAD_NO_CHANGE                    0
#define ENCOUNTER_OVERHEAD_OFF                          1
#define ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE            2
#define ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED           3
#define ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC            4
#define ENCOUNTER_OVERHEAD_SET_REFRESH_SMITE            5
#define ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION       6
#define ENCOUNTER_OVERHEAD_DIM_PVE                      5
#define ENCOUNTER_OVERHEAD_DIM_PVE_REDEMPTION           6

#define ENCOUNTER_OFFENSIVE_NO_CHANGE                   0
#define ENCOUNTER_OFFENSIVE_OFF                         1
#define ENCOUNTER_OFFENSIVE_SET_REFRESH_PIETY           2
#define ENCOUNTER_OFFENSIVE_SET_REFRESH_RIGOUR          3
#define ENCOUNTER_OFFENSIVE_SET_REFRESH_AUGURY          4
#define ENCOUNTER_OFFENSIVE_DIM                         5

static inline int encounter_apply_overhead_action(OverheadPrayer* overhead, int action) {
    OverheadPrayer target;
    switch (action) {
        case ENCOUNTER_OVERHEAD_NO_CHANGE:
            return 0;
        case ENCOUNTER_OVERHEAD_OFF:
            *overhead = PRAYER_NONE;
            return 0;
        case ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE:       target = PRAYER_PROTECT_MELEE;  break;
        case ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED:      target = PRAYER_PROTECT_RANGED; break;
        case ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC:       target = PRAYER_PROTECT_MAGIC;  break;
        case ENCOUNTER_OVERHEAD_SET_REFRESH_SMITE:       target = PRAYER_SMITE;          break;
        case ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION:  target = PRAYER_REDEMPTION;     break;
        default: return 0;
    }
    *overhead = target;
    return 1;
}

static inline int encounter_apply_offensive_action(OffensivePrayer* offensive, int action) {
    OffensivePrayer target;
    switch (action) {
        case ENCOUNTER_OFFENSIVE_NO_CHANGE:
            return 0;
        case ENCOUNTER_OFFENSIVE_OFF:
            *offensive = OFFENSIVE_PRAYER_NONE;
            return 0;
        case ENCOUNTER_OFFENSIVE_SET_REFRESH_PIETY:    target = OFFENSIVE_PRAYER_PIETY;  break;
        case ENCOUNTER_OFFENSIVE_SET_REFRESH_RIGOUR:   target = OFFENSIVE_PRAYER_RIGOUR; break;
        case ENCOUNTER_OFFENSIVE_SET_REFRESH_AUGURY:   target = OFFENSIVE_PRAYER_AUGURY; break;
        default: return 0;
    }
    *offensive = target;
    return 1;
}

#define ENCOUNTER_MOVE_ACTIONS 25

static const int ENCOUNTER_MOVE_TARGET_DX[25] = {
    0,
    -1, -1, -1, 0, 0, 1, 1, 1,
    -2, -2, -2, -2, -2,
    -1, -1,
    0, 0,
    1, 1,
    2, 2, 2, 2, 2
};
static const int ENCOUNTER_MOVE_TARGET_DY[25] = {
    0,
    -1, 0, 1, -1, 1, -1, 0, 1,
    -2, -1, 0, 1, 2,
    -2, 2,
    -2, 2,
    -2, 2,
    -2, -1, 0, 1, 2
};

typedef int (*encounter_walkable_fn)(void* ctx, int x, int y);

static inline int encounter_move_to_target(
    Player* p, int target_dx, int target_dy,
    encounter_walkable_fn is_walkable, void* ctx
) {
    int tx = p->x + target_dx;
    int ty = p->y + target_dy;
    int dist = abs(target_dx) > abs(target_dy) ? abs(target_dx) : abs(target_dy);
    int max_steps = dist;
    int steps = 0;

    for (int step = 0; step < max_steps; step++) {
        if (p->x == tx && p->y == ty) break;
        int dx = 0, dy = 0;
        if (tx > p->x) dx = 1; else if (tx < p->x) dx = -1;
        if (ty > p->y) dy = 1; else if (ty < p->y) dy = -1;

        int moved = 0;
        if (dx != 0 && dy != 0 &&
            is_walkable(ctx, p->x + dx, p->y + dy) &&
            is_walkable(ctx, p->x + dx, p->y) &&
            is_walkable(ctx, p->x, p->y + dy)) {
            p->x += dx; p->y += dy; moved = 1;
        } else if (dx != 0 && is_walkable(ctx, p->x + dx, p->y)) {
            p->x += dx; moved = 1;
        } else if (dy != 0 && is_walkable(ctx, p->x, p->y + dy)) {
            p->y += dy; moved = 1;
        }
        if (!moved) break;
        steps++;
    }

    p->is_running = (steps == 2);
    p->dest_x = p->x;
    p->dest_y = p->y;
    return steps;
}

static inline PathResult encounter_pathfind(
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    int src_x, int src_y, int dst_x, int dst_y,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx
) {
    return pathfind_step(cmap, 0,
        src_x + world_offset_x, src_y + world_offset_y,
        dst_x + world_offset_x, dst_y + world_offset_y,
        extra_blocked, blocked_ctx);
}

static inline PathResult encounter_pathfind_arena(
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    int src_x, int src_y, int dst_x, int dst_y,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    return pathfind_step_arena(cmap, 0,
        src_x + world_offset_x, src_y + world_offset_y,
        dst_x + world_offset_x, dst_y + world_offset_y,
        extra_blocked, blocked_ctx,
        arena_base_x + world_offset_x, arena_base_y + world_offset_y,
        arena_w, arena_h);
}

static inline int encounter_walk_toward(
    Player* p, int tx, int ty,
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    encounter_walkable_fn is_walkable, void* ctx,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    int steps = 0;
    for (int step = 0; step < 2; step++) {
        if (p->x == tx && p->y == ty) break;
        PathResult pr = (arena_w > 0)
            ? encounter_pathfind_arena(cmap, world_offset_x, world_offset_y,
                                       p->x, p->y, tx, ty,
                                       extra_blocked, blocked_ctx,
                                       arena_base_x, arena_base_y, arena_w, arena_h)
            : encounter_pathfind(cmap, world_offset_x, world_offset_y,
                                  p->x, p->y, tx, ty,
                                  extra_blocked, blocked_ctx);
        if (!pr.found || (pr.next_dx == 0 && pr.next_dy == 0)) break;
        int nx = p->x + pr.next_dx, ny = p->y + pr.next_dy;
        if (!is_walkable(ctx, nx, ny)) break;
        p->x = nx; p->y = ny;
        steps++;
    }
    p->is_running = (steps == 2);
    p->dest_x = p->x; p->dest_y = p->y;
    return steps;
}

static inline int encounter_move_toward_dest(
    Player* p, int* dest_x, int* dest_y,
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    encounter_walkable_fn is_walkable, void* ctx,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    if (*dest_x < 0 || *dest_y < 0) return 0;
    if (p->x == *dest_x && p->y == *dest_y) {
        *dest_x = -1; *dest_y = -1;
        return 0;
    }
    return encounter_walk_toward(p, *dest_x, *dest_y,
        cmap, world_offset_x, world_offset_y,
        is_walkable, ctx, extra_blocked, blocked_ctx,
        arena_base_x, arena_base_y, arena_w, arena_h);
}

static inline int encounter_entity_footprint_cardinal_adjacent(
    int ax, int ay, int a_size,
    int bx, int by, int b_size
) {
    int ax1 = ax + a_size - 1;
    int ay1 = ay + a_size - 1;
    int bx1 = bx + b_size - 1;
    int by1 = by + b_size - 1;

    int dx = 0;
    if (ax1 < bx) dx = bx - ax1;
    else if (bx1 < ax) dx = ax - bx1;

    int dy = 0;
    if (ay1 < by) dy = by - ay1;
    else if (by1 < ay) dy = ay - by1;

    return (dx + dy) == 1;
}

static inline int encounter_entity_footprints_overlap(
    int ax, int ay, int a_size,
    int bx, int by, int b_size
) {
    return !(ax + a_size <= bx || bx + b_size <= ax ||
             ay + a_size <= by || by + b_size <= ay);
}

typedef enum {
    OSRS_LOS_OPEN = 0,
    OSRS_LOS_BLOCKERS,
    OSRS_LOS_TILE,
} OsrsLosKind;

typedef struct {
    OsrsLosKind kind;
    const LOSBlocker* blockers;
    int blocker_count;
    int (*tile_blocked)(void* ctx, int x, int y);
    void* tile_ctx;
} OsrsLosQuery;

static inline OsrsLosQuery osrs_los_open(void) {
    OsrsLosQuery query;
    query.kind = OSRS_LOS_OPEN;
    query.blockers = NULL;
    query.blocker_count = 0;
    query.tile_blocked = NULL;
    query.tile_ctx = NULL;
    return query;
}

static inline OsrsLosQuery osrs_los_blockers(
    const LOSBlocker* blockers,
    int blocker_count
) {
    OsrsLosQuery query;
    query.kind = OSRS_LOS_BLOCKERS;
    query.blockers = blockers;
    query.blocker_count = blocker_count;
    query.tile_blocked = NULL;
    query.tile_ctx = NULL;
    return query;
}

static inline OsrsLosQuery osrs_los_tile(
    int (*tile_blocked)(void* ctx, int x, int y),
    void* tile_ctx
) {
    OsrsLosQuery query;
    query.kind = OSRS_LOS_TILE;
    query.blockers = NULL;
    query.blocker_count = 0;
    query.tile_blocked = tile_blocked;
    query.tile_ctx = tile_ctx;
    return query;
}

static inline const OsrsLosQuery* osrs_los_open_query(void) {
    static const OsrsLosQuery query = {
        OSRS_LOS_OPEN,
        NULL,
        0,
        NULL,
        NULL,
    };
    return &query;
}

static inline void osrs_los_require_query(
    const OsrsLosQuery* query,
    int attack_range
) {
    if (attack_range <= 1) return;
    if (!query) {
        fprintf(stderr, "missing OSRS LoS query for ranged attack\n");
        abort();
    }
    if (query->kind < OSRS_LOS_OPEN || query->kind > OSRS_LOS_TILE) {
        fprintf(stderr, "invalid OSRS LoS query kind: %d\n", (int)query->kind);
        abort();
    }
    if (query->kind == OSRS_LOS_BLOCKERS) {
        if (query->blocker_count < 0) {
            fprintf(stderr, "negative OSRS LoS blocker count: %d\n",
                query->blocker_count);
            abort();
        }
        if (query->blocker_count > 0 && !query->blockers) {
            fprintf(stderr, "OSRS LoS blocker query is missing blockers\n");
            abort();
        }
    }
    if (query->kind == OSRS_LOS_TILE && !query->tile_blocked) {
        fprintf(stderr, "OSRS tile LoS query is missing tile_blocked\n");
        abort();
    }
}

static inline int osrs_los_tile_ray_clear(
    const OsrsLosQuery* query,
    int x0, int y0,
    int x1, int y1
) {
    int dx = x1 - x0;
    int dy = y1 - y0;
    int adx = dx < 0 ? -dx : dx;
    int ady = dy < 0 ? -dy : dy;
    if (adx == 0 && ady == 0) return 1;
    if (query->tile_blocked(query->tile_ctx, x1, y1)) return 0;

    if (adx > ady) {
        int x = x0;
        int y_fp = y0 * LOS_FP_SCALE + LOS_FP_HALF;
        int slope = (dy * LOS_FP_SCALE) / adx;
        int x_inc = dx > 0 ? 1 : -1;
        if (dy < 0) y_fp -= 1;
        while (x != x1) {
            x += x_inc;
            int y = y_fp >> 16;
            if (query->tile_blocked(query->tile_ctx, x, y)) return 0;
            y_fp += slope;
            int new_y = y_fp >> 16;
            if (new_y != y &&
                    query->tile_blocked(query->tile_ctx, x, new_y))
                return 0;
        }
    } else {
        int y = y0;
        int x_fp = x0 * LOS_FP_SCALE + LOS_FP_HALF;
        int slope = (dx * LOS_FP_SCALE) / ady;
        int y_inc = dy > 0 ? 1 : -1;
        if (dx < 0) x_fp -= 1;
        while (y != y1) {
            y += y_inc;
            int x = x_fp >> 16;
            if (query->tile_blocked(query->tile_ctx, x, y)) return 0;
            x_fp += slope;
            int new_x = x_fp >> 16;
            if (new_x != x &&
                    query->tile_blocked(query->tile_ctx, new_x, y))
                return 0;
        }
    }
    return 1;
}

static inline int osrs_los_clear(
    const OsrsLosQuery* query,
    int px, int py, int psize,
    int tx, int ty, int tsize,
    int attack_range
) {
    osrs_los_require_query(query, attack_range);
    if (attack_range <= 1) return 1;

    switch (query->kind) {
        case OSRS_LOS_OPEN:
            return 1;

        case OSRS_LOS_BLOCKERS:
            return entity_has_line_of_sight(
                query->blockers,
                query->blocker_count,
                px,
                py,
                psize,
                tx,
                ty,
                tsize,
                attack_range);

        case OSRS_LOS_TILE: {
            int p_los_x = tx;
            if (p_los_x < px) p_los_x = px;
            if (p_los_x >= px + psize) p_los_x = px + psize - 1;
            int p_los_y = ty;
            if (p_los_y < py) p_los_y = py;
            if (p_los_y >= py + psize) p_los_y = py + psize - 1;

            int t_los_x = px;
            if (t_los_x < tx) t_los_x = tx;
            if (t_los_x >= tx + tsize) t_los_x = tx + tsize - 1;
            int t_los_y = py;
            if (t_los_y < ty) t_los_y = ty;
            if (t_los_y >= ty + tsize) t_los_y = ty + tsize - 1;

            return osrs_los_tile_ray_clear(
                query, t_los_x, t_los_y, p_los_x, p_los_y);
        }
    }

    fprintf(stderr, "unhandled OSRS LoS query kind: %d\n", (int)query->kind);
    abort();
}

static inline int encounter_player_can_attack(
    int player_x, int player_y,
    int target_x, int target_y, int target_size, int attack_range,
    const OsrsLosQuery* los_query
) {
    int dist = encounter_rect_distance(player_x, player_y, 1,
                                                   target_x, target_y, target_size);
    if (dist < 1 || dist > attack_range) return 0;
    if (attack_range == 1)
        return encounter_entity_footprint_cardinal_adjacent(
            player_x, player_y, 1, target_x, target_y, target_size);
    return osrs_los_clear(los_query,
        player_x, player_y, 1,
        target_x, target_y, target_size,
        attack_range);
}

#define ENCOUNTER_ATTACK_SEEK_MAX_TILES 128

typedef struct {
    int x;
    int y;
} EncounterAttackSeekTile;

static inline void encounter_attack_seek_add_tile(
    EncounterAttackSeekTile* tiles, int* count,
    int x, int y, int world_offset_x, int world_offset_y,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx
) {
    if (extra_blocked &&
            extra_blocked(blocked_ctx, x + world_offset_x, y + world_offset_y))
        return;
    if (*count >= ENCOUNTER_ATTACK_SEEK_MAX_TILES) {
        fprintf(stderr, "attack seek tile capacity exceeded: %d\n",
            ENCOUNTER_ATTACK_SEEK_MAX_TILES);
        abort();
    }
    tiles[*count] = (EncounterAttackSeekTile){x, y};
    (*count)++;
}

static inline int encounter_attack_seek_nearest_dsq(
    int x, int y, const EncounterAttackSeekTile* tiles, int count
) {
    int best = 0x3fffffff;
    for (int i = 0; i < count; i++) {
        int dx = x - tiles[i].x;
        int dy = y - tiles[i].y;
        int dsq = dx * dx + dy * dy;
        if (dsq < best) best = dsq;
    }
    return best;
}

static inline int encounter_attack_seek_has_exact_tile(
    int x, int y, const EncounterAttackSeekTile* tiles, int count
) {
    for (int i = 0; i < count; i++) {
        if (tiles[i].x == x && tiles[i].y == y) return 1;
    }
    return 0;
}

typedef struct {
    const CollisionMap* collision_map;
    encounter_walkable_fn is_walkable;
    void* walkable_ctx;
    pathfind_blocked_fn extra_blocked;
    void* blocked_ctx;
    int world_offset_x;
    int world_offset_y;
    int arena_base_x;
    int arena_base_y;
    int arena_w;
    int arena_h;
    int source_x;
    int source_y;
    int min_explored_x;
    int min_explored_y;
    int max_explored_x;
    int max_explored_y;
    uint16_t visit_order[PATHFIND_ARENA_MAX][PATHFIND_ARENA_MAX];
    uint16_t depth[PATHFIND_ARENA_MAX][PATHFIND_ARENA_MAX];
    int8_t via[PATHFIND_ARENA_MAX][PATHFIND_ARENA_MAX];
    int visited_count;
} EncounterArenaAttackRouteField;

typedef struct {
    int land_x;
    int land_y;
} EncounterAttackRouteLanding;

static inline int encounter_attack_route_field_extra_blocked(
    const EncounterArenaAttackRouteField* field,
    int x,
    int y
) {
    return field->extra_blocked &&
        field->extra_blocked(
            field->blocked_ctx,
            x + field->world_offset_x,
            y + field->world_offset_y);
}

static inline void encounter_build_arena_attack_route_field(
    EncounterArenaAttackRouteField* field,
    const CollisionMap* cmap,
    int world_offset_x,
    int world_offset_y,
    int source_x,
    int source_y,
    encounter_walkable_fn is_walkable,
    void* walkable_ctx,
    pathfind_blocked_fn extra_blocked,
    void* blocked_ctx,
    int arena_base_x,
    int arena_base_y,
    int arena_w,
    int arena_h
) {
    if (!field || !is_walkable) {
        fprintf(stderr, "attack route field is missing required input\n");
        abort();
    }
    if (arena_w <= 0 || arena_w > PATHFIND_ARENA_MAX ||
            arena_h <= 0 || arena_h > PATHFIND_ARENA_MAX) {
        fprintf(stderr, "attack route arena dimensions out of bounds: %dx%d\n",
            arena_w, arena_h);
        abort();
    }

    int local_source_x = source_x - arena_base_x;
    int local_source_y = source_y - arena_base_y;
    if (local_source_x < 0 || local_source_x >= arena_w ||
            local_source_y < 0 || local_source_y >= arena_h) {
        fprintf(stderr,
            "attack route source out of arena: source=(%d,%d) arena=(%d,%d,%d,%d)\n",
            source_x, source_y, arena_base_x, arena_base_y, arena_w, arena_h);
        abort();
    }

    memset(field, 0, sizeof(*field));
    field->collision_map = cmap;
    field->is_walkable = is_walkable;
    field->walkable_ctx = walkable_ctx;
    field->extra_blocked = extra_blocked;
    field->blocked_ctx = blocked_ctx;
    field->world_offset_x = world_offset_x;
    field->world_offset_y = world_offset_y;
    field->arena_base_x = arena_base_x;
    field->arena_base_y = arena_base_y;
    field->arena_w = arena_w;
    field->arena_h = arena_h;
    field->source_x = source_x;
    field->source_y = source_y;
    field->min_explored_x = local_source_x;
    field->min_explored_y = local_source_y;
    field->max_explored_x = local_source_x;
    field->max_explored_y = local_source_y;

    int queue_x[PATHFIND_MAX_QUEUE_ARENA];
    int queue_y[PATHFIND_MAX_QUEUE_ARENA];
    int head = 0;
    int tail = 0;
    field->visited_count = 1;
    field->visit_order[local_source_x][local_source_y] = 1;
    field->via[local_source_x][local_source_y] = VIA_START;
    pathfind_enqueue_or_abort(
        queue_x, queue_y, &tail, PATHFIND_MAX_QUEUE_ARENA,
        local_source_x, local_source_y);

    static const int route_dx[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    static const int route_dy[8] = {0, 0, -1, 1, -1, -1, 1, 1};
    static const int route_via[8] = {
        VIA_W, VIA_E, VIA_S, VIA_N, VIA_SW, VIA_SE, VIA_NW, VIA_NE
    };

    while (head < tail) {
        int cur_x = queue_x[head];
        int cur_y = queue_y[head];
        head++;

        int tile_x = arena_base_x + cur_x;
        int tile_y = arena_base_y + cur_y;
        int abs_x = tile_x + world_offset_x;
        int abs_y = tile_y + world_offset_y;
        uint16_t next_depth = field->depth[cur_x][cur_y] + 1;

        for (int i = 0; i < 8; i++) {
            int dx = route_dx[i];
            int dy = route_dy[i];
            int next_x = cur_x + dx;
            int next_y = cur_y + dy;
            if (next_x < 0 || next_x >= arena_w ||
                    next_y < 0 || next_y >= arena_h) {
                continue;
            }
            if (field->visit_order[next_x][next_y] != 0) continue;
            if (!collision_traversable_step(cmap, 0, abs_x, abs_y, dx, dy))
                continue;

            int next_tile_x = tile_x + dx;
            int next_tile_y = tile_y + dy;
            if (!is_walkable(walkable_ctx, next_tile_x, next_tile_y)) continue;
            if (extra_blocked &&
                    extra_blocked(
                        blocked_ctx,
                        next_tile_x + world_offset_x,
                        next_tile_y + world_offset_y)) {
                continue;
            }
            if (dx != 0 && dy != 0) {
                if (extra_blocked &&
                        (extra_blocked(
                            blocked_ctx,
                            tile_x + dx + world_offset_x,
                            tile_y + world_offset_y) ||
                         extra_blocked(
                            blocked_ctx,
                            tile_x + world_offset_x,
                            tile_y + dy + world_offset_y))) {
                    continue;
                }
                if (!is_walkable(walkable_ctx, tile_x + dx, tile_y) ||
                        !is_walkable(walkable_ctx, tile_x, tile_y + dy)) {
                    continue;
                }
            }

            pathfind_enqueue_or_abort(
                queue_x, queue_y, &tail, PATHFIND_MAX_QUEUE_ARENA,
                next_x, next_y);
            field->visited_count++;
            field->visit_order[next_x][next_y] =
                (uint16_t)field->visited_count;
            field->depth[next_x][next_y] = next_depth;
            field->via[next_x][next_y] = (int8_t)route_via[i];
            if (next_x < field->min_explored_x) field->min_explored_x = next_x;
            if (next_y < field->min_explored_y) field->min_explored_y = next_y;
            if (next_x > field->max_explored_x) field->max_explored_x = next_x;
            if (next_y > field->max_explored_y) field->max_explored_y = next_y;
        }
    }
}

static inline EncounterAttackRouteLanding
encounter_attack_route_overlap_landing(
    const EncounterArenaAttackRouteField* field,
    int target_x,
    int target_y,
    int target_size
) {
    Player player = {
        .x = field->source_x,
        .y = field->source_y,
    };
    int max_r = (target_size + 1) / 2 + 1;
    int best_dsq = 9999;
    int best_x = -1;
    int best_y = -1;
    for (int dy = -max_r; dy <= max_r; dy++) {
        for (int dx = -max_r; dx <= max_r; dx++) {
            if (dx == 0 && dy == 0) continue;
            int x = player.x + dx;
            int y = player.y + dy;
            if (!field->is_walkable(field->walkable_ctx, x, y)) continue;
            if (encounter_entity_footprints_overlap(
                    x, y, 1, target_x, target_y, target_size)) {
                continue;
            }
            int dsq = dx * dx + dy * dy;
            if (dsq < best_dsq) {
                best_dsq = dsq;
                best_x = x;
                best_y = y;
            }
        }
    }
    if (best_x >= 0) {
        encounter_walk_toward(
            &player, best_x, best_y,
            field->collision_map,
            field->world_offset_x,
            field->world_offset_y,
            field->is_walkable,
            field->walkable_ctx,
            field->extra_blocked,
            field->blocked_ctx,
            field->arena_base_x,
            field->arena_base_y,
            field->arena_w,
            field->arena_h);
    }
    return (EncounterAttackRouteLanding){
        .land_x = player.x,
        .land_y = player.y,
    };
}

static inline EncounterAttackRouteLanding encounter_arena_attack_route_landing(
    const EncounterArenaAttackRouteField* field,
    int target_x,
    int target_y,
    int target_size,
    int attack_range,
    const OsrsLosQuery* los_query
) {
    if (!field || !field->is_walkable ||
            field->arena_w <= 0 || field->arena_w > PATHFIND_ARENA_MAX ||
            field->arena_h <= 0 || field->arena_h > PATHFIND_ARENA_MAX ||
            field->visited_count <= 0 ||
            target_size <= 0 || attack_range <= 0) {
        fprintf(stderr, "invalid attack route landing query\n");
        abort();
    }

    EncounterAttackRouteLanding landing = {
        .land_x = field->source_x,
        .land_y = field->source_y,
    };
    int dist = encounter_rect_distance(
        field->source_x, field->source_y, 1,
        target_x, target_y, target_size);
    if (dist == 0) {
        return encounter_attack_route_overlap_landing(
            field, target_x, target_y, target_size);
    }
    if (encounter_player_can_attack(
            field->source_x, field->source_y,
            target_x, target_y, target_size, attack_range, los_query)) {
        return landing;
    }

    EncounterAttackSeekTile seek_tiles[ENCOUNTER_ATTACK_SEEK_MAX_TILES];
    int seek_count = 0;
    for (int xx = 0; xx < target_size; xx++) {
        int x = target_x + xx;
        encounter_attack_seek_add_tile(
            seek_tiles, &seek_count, x, target_y - 1,
            field->world_offset_x, field->world_offset_y,
            field->extra_blocked, field->blocked_ctx);
        encounter_attack_seek_add_tile(
            seek_tiles, &seek_count, x, target_y + target_size,
            field->world_offset_x, field->world_offset_y,
            field->extra_blocked, field->blocked_ctx);
    }
    for (int yy = 0; yy < target_size; yy++) {
        int y = target_y + yy;
        encounter_attack_seek_add_tile(
            seek_tiles, &seek_count, target_x - 1, y,
            field->world_offset_x, field->world_offset_y,
            field->extra_blocked, field->blocked_ctx);
        encounter_attack_seek_add_tile(
            seek_tiles, &seek_count, target_x + target_size, y,
            field->world_offset_x, field->world_offset_y,
            field->extra_blocked, field->blocked_ctx);
    }

    int selected_x = -1;
    int selected_y = -1;
    uint16_t selected_order = UINT16_MAX;
    for (int x = 0; x < field->arena_w; x++) {
        for (int y = 0; y < field->arena_h; y++) {
            uint16_t order = field->visit_order[x][y];
            if (order == 0 || order >= selected_order) continue;
            int tile_x = field->arena_base_x + x;
            int tile_y = field->arena_base_y + y;
            if (!field->is_walkable(field->walkable_ctx, tile_x, tile_y))
                continue;
            if (encounter_attack_route_field_extra_blocked(
                    field, tile_x, tile_y)) {
                continue;
            }
            int exact = seek_count > 0
                ? encounter_attack_seek_has_exact_tile(
                    tile_x, tile_y, seek_tiles, seek_count)
                : encounter_player_can_attack(
                    tile_x, tile_y,
                    target_x, target_y, target_size,
                    attack_range, los_query);
            if (!exact) continue;
            selected_x = x;
            selected_y = y;
            selected_order = order;
        }
    }

    if (selected_x < 0 && seek_count > 0) {
        int first_local_x = seek_tiles[0].x - field->arena_base_x;
        int first_local_y = seek_tiles[0].y - field->arena_base_y;
        int scan_min_x =
            field->min_explored_x > first_local_x - PATHFIND_MAX_FALLBACK_RADIUS
                ? field->min_explored_x
                : first_local_x - PATHFIND_MAX_FALLBACK_RADIUS;
        int scan_min_y =
            field->min_explored_y > first_local_y - PATHFIND_MAX_FALLBACK_RADIUS
                ? field->min_explored_y
                : first_local_y - PATHFIND_MAX_FALLBACK_RADIUS;
        int scan_max_x =
            field->max_explored_x > first_local_x + PATHFIND_MAX_FALLBACK_RADIUS
                ? field->max_explored_x
                : first_local_x + PATHFIND_MAX_FALLBACK_RADIUS;
        int scan_max_y =
            field->max_explored_y > first_local_y + PATHFIND_MAX_FALLBACK_RADIUS
                ? field->max_explored_y
                : first_local_y + PATHFIND_MAX_FALLBACK_RADIUS;
        if (scan_min_x < 0) scan_min_x = 0;
        if (scan_min_y < 0) scan_min_y = 0;
        if (scan_max_x >= field->arena_w) scan_max_x = field->arena_w - 1;
        if (scan_max_y >= field->arena_h) scan_max_y = field->arena_h - 1;

        int best_dsq = 0x3fffffff;
        int best_depth = 100;
        for (int x = scan_min_x; x <= scan_max_x; x++) {
            for (int y = scan_min_y; y <= scan_max_y; y++) {
                if (field->visit_order[x][y] == 0) continue;
                int tile_x = field->arena_base_x + x;
                int tile_y = field->arena_base_y + y;
                if (!field->is_walkable(field->walkable_ctx, tile_x, tile_y))
                    continue;
                int depth = field->depth[x][y];
                if (depth >= 100) continue;
                int dsq = encounter_attack_seek_nearest_dsq(
                    tile_x, tile_y, seek_tiles, seek_count);
                if (dsq < best_dsq ||
                        (dsq == best_dsq && depth < best_depth)) {
                    selected_x = x;
                    selected_y = y;
                    best_dsq = dsq;
                    best_depth = depth;
                }
            }
        }
    }
    if (selected_x < 0) return landing;

    int source_local_x = field->source_x - field->arena_base_x;
    int source_local_y = field->source_y - field->arena_base_y;
    int first_x = source_local_x;
    int first_y = source_local_y;
    int second_x = source_local_x;
    int second_y = source_local_y;
    int cur_x = selected_x;
    int cur_y = selected_y;
    uint16_t selected_depth = field->depth[selected_x][selected_y];
    while (field->depth[cur_x][cur_y] > 0) {
        uint16_t depth = field->depth[cur_x][cur_y];
        if (depth == 1) {
            first_x = cur_x;
            first_y = cur_y;
        } else if (depth == 2) {
            second_x = cur_x;
            second_y = cur_y;
        }

        int via = field->via[cur_x][cur_y];
        if (via == VIA_NONE || via == VIA_START) {
            fprintf(stderr,
                "broken attack route parent at local tile (%d,%d)\n",
                cur_x, cur_y);
            abort();
        }
        if (via & VIA_W) cur_x++;
        else if (via & VIA_E) cur_x--;
        if (via & VIA_S) cur_y++;
        else if (via & VIA_N) cur_y--;
        if (cur_x < 0 || cur_x >= field->arena_w ||
                cur_y < 0 || cur_y >= field->arena_h) {
            fprintf(stderr, "attack route parent left arena\n");
            abort();
        }
    }
    if (cur_x != source_local_x || cur_y != source_local_y) {
        fprintf(stderr, "attack route did not terminate at source\n");
        abort();
    }
    if (selected_depth == 0) return landing;

    landing.land_x = field->arena_base_x + first_x;
    landing.land_y = field->arena_base_y + first_y;
    if (encounter_player_can_attack(
            landing.land_x, landing.land_y,
            target_x, target_y, target_size, attack_range, los_query)) {
        return landing;
    }
    if (selected_depth >= 2) {
        landing.land_x = field->arena_base_x + second_x;
        landing.land_y = field->arena_base_y + second_y;
    }
    return landing;
}

static inline PathResult encounter_pathfind_arena_attack_approach(
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    int src_x, int src_y,
    int target_x, int target_y, int target_size, int attack_range,
    encounter_walkable_fn is_walkable, void* ctx,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    const OsrsLosQuery* los_query,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    PathResult result = {0, 0, 0, src_x, src_y};

    if (arena_w <= 0 || arena_w > PATHFIND_ARENA_MAX ||
        arena_h <= 0 || arena_h > PATHFIND_ARENA_MAX) {
        fprintf(stderr, "attack approach arena dimensions out of bounds: %dx%d\n",
            arena_w, arena_h);
        abort();
    }

    int local_src_x = src_x - arena_base_x;
    int local_src_y = src_y - arena_base_y;
    if (local_src_x < 0 || local_src_x >= arena_w ||
        local_src_y < 0 || local_src_y >= arena_h) {
        return result;
    }

    EncounterAttackSeekTile seek_tiles[ENCOUNTER_ATTACK_SEEK_MAX_TILES];
    int seek_count = 0;
    for (int xx = 0; xx < target_size; xx++) {
        int x = target_x + xx;
        encounter_attack_seek_add_tile(
            seek_tiles, &seek_count, x, target_y - 1,
            world_offset_x, world_offset_y, extra_blocked, blocked_ctx);
        encounter_attack_seek_add_tile(
            seek_tiles, &seek_count, x, target_y + target_size,
            world_offset_x, world_offset_y, extra_blocked, blocked_ctx);
    }
    for (int yy = 0; yy < target_size; yy++) {
        int y = target_y + yy;
        encounter_attack_seek_add_tile(
            seek_tiles, &seek_count, target_x - 1, y,
            world_offset_x, world_offset_y, extra_blocked, blocked_ctx);
        encounter_attack_seek_add_tile(
            seek_tiles, &seek_count, target_x + target_size, y,
            world_offset_x, world_offset_y, extra_blocked, blocked_ctx);
    }
    static OSRS_THREAD_LOCAL uint16_t approach_gen[PATHFIND_ARENA_MAX][PATHFIND_ARENA_MAX];
    static OSRS_THREAD_LOCAL int8_t approach_via[PATHFIND_ARENA_MAX][PATHFIND_ARENA_MAX];
    static OSRS_THREAD_LOCAL int16_t approach_cost[PATHFIND_ARENA_MAX][PATHFIND_ARENA_MAX];
    static OSRS_THREAD_LOCAL uint16_t approach_gen_counter = 0;
    approach_gen_counter++;
    if (approach_gen_counter == 0) {
        memset(approach_gen, 0, sizeof(approach_gen));
        approach_gen_counter = 1;
    }
    uint16_t gen = approach_gen_counter;

    #define APPROACH_VISITED(x, y) (approach_gen[(x)][(y)] == gen)
    #define APPROACH_VISIT(x, y, v, c) do { \
        approach_gen[(x)][(y)] = gen; \
        approach_via[(x)][(y)] = (v); \
        approach_cost[(x)][(y)] = (c); \
    } while(0)
    #define APPROACH_VIA(x, y) approach_via[(x)][(y)]
    #define APPROACH_COST(x, y) approach_cost[(x)][(y)]
    #define APPROACH_EB(x, y) \
        (extra_blocked && extra_blocked( \
            blocked_ctx, (x) + world_offset_x, (y) + world_offset_y))

    int queue_x[PATHFIND_MAX_QUEUE_ARENA];
    int queue_y[PATHFIND_MAX_QUEUE_ARENA];
    int head = 0;
    int tail = 0;
    APPROACH_VISIT(local_src_x, local_src_y, VIA_START, 0);
    pathfind_enqueue_or_abort(
        queue_x, queue_y, &tail, PATHFIND_MAX_QUEUE_ARENA,
        local_src_x, local_src_y);

    int selected_x = -1;
    int selected_y = -1;
    int min_explored_x = local_src_x;
    int min_explored_y = local_src_y;
    int max_explored_x = local_src_x;
    int max_explored_y = local_src_y;

    static const int dir_dx[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    static const int dir_dy[8] = {0, 0, -1, 1, -1, -1, 1, 1};
    static const int dir_via[8] = {
        VIA_W, VIA_E, VIA_S, VIA_N, VIA_SW, VIA_SE, VIA_NW, VIA_NE
    };

    while (head < tail) {
        int cur_x = queue_x[head];
        int cur_y = queue_y[head];
        head++;

        int tile_x = arena_base_x + cur_x;
        int tile_y = arena_base_y + cur_y;
        if (is_walkable(ctx, tile_x, tile_y) &&
                !APPROACH_EB(tile_x, tile_y) &&
                (seek_count > 0
                    ? encounter_attack_seek_has_exact_tile(
                        tile_x, tile_y, seek_tiles, seek_count)
                    : encounter_player_can_attack(
                        tile_x, tile_y, target_x, target_y, target_size,
                        attack_range, los_query))) {
            selected_x = cur_x;
            selected_y = cur_y;
            break;
        }

        int abs_x = tile_x + world_offset_x;
        int abs_y = tile_y + world_offset_y;
        int next_cost = APPROACH_COST(cur_x, cur_y) + 1;

        for (int i = 0; i < 8; i++) {
            int dx = dir_dx[i];
            int dy = dir_dy[i];
            int next_x = cur_x + dx;
            int next_y = cur_y + dy;
            if (next_x < 0 || next_x >= arena_w ||
                    next_y < 0 || next_y >= arena_h)
                continue;
            if (APPROACH_VISITED(next_x, next_y)) continue;
            if (!collision_traversable_step(cmap, 0, abs_x, abs_y, dx, dy))
                continue;

            int next_tile_x = tile_x + dx;
            int next_tile_y = tile_y + dy;
            if (!is_walkable(ctx, next_tile_x, next_tile_y)) continue;
            if (APPROACH_EB(next_tile_x, next_tile_y)) continue;
            if (dx != 0 && dy != 0) {
                if (APPROACH_EB(tile_x + dx, tile_y)) continue;
                if (APPROACH_EB(tile_x, tile_y + dy)) continue;
                if (!is_walkable(ctx, tile_x + dx, tile_y)) continue;
                if (!is_walkable(ctx, tile_x, tile_y + dy)) continue;
            }

            pathfind_enqueue_or_abort(
                queue_x, queue_y, &tail, PATHFIND_MAX_QUEUE_ARENA,
                next_x, next_y);
            APPROACH_VISIT(next_x, next_y, dir_via[i], next_cost);
            if (next_x < min_explored_x) min_explored_x = next_x;
            if (next_y < min_explored_y) min_explored_y = next_y;
            if (next_x > max_explored_x) max_explored_x = next_x;
            if (next_y > max_explored_y) max_explored_y = next_y;
        }
    }

    if (selected_x < 0 && seek_count > 0) {
        int first_local_x = seek_tiles[0].x - arena_base_x;
        int first_local_y = seek_tiles[0].y - arena_base_y;
        int scan_min_x = min_explored_x > first_local_x - PATHFIND_MAX_FALLBACK_RADIUS
            ? min_explored_x : first_local_x - PATHFIND_MAX_FALLBACK_RADIUS;
        int scan_min_y = min_explored_y > first_local_y - PATHFIND_MAX_FALLBACK_RADIUS
            ? min_explored_y : first_local_y - PATHFIND_MAX_FALLBACK_RADIUS;
        int scan_max_x = max_explored_x > first_local_x + PATHFIND_MAX_FALLBACK_RADIUS
            ? max_explored_x : first_local_x + PATHFIND_MAX_FALLBACK_RADIUS;
        int scan_max_y = max_explored_y > first_local_y + PATHFIND_MAX_FALLBACK_RADIUS
            ? max_explored_y : first_local_y + PATHFIND_MAX_FALLBACK_RADIUS;
        if (scan_min_x < 0) scan_min_x = 0;
        if (scan_min_y < 0) scan_min_y = 0;
        if (scan_max_x >= arena_w) scan_max_x = arena_w - 1;
        if (scan_max_y >= arena_h) scan_max_y = arena_h - 1;

        int best_dsq = 0x3fffffff;
        int best_cost = 100;
        for (int x = scan_min_x; x <= scan_max_x; x++) {
            for (int y = scan_min_y; y <= scan_max_y; y++) {
                if (!APPROACH_VISITED(x, y)) continue;
                int tile_x = arena_base_x + x;
                int tile_y = arena_base_y + y;
                if (!is_walkable(ctx, tile_x, tile_y)) continue;
                int cost = APPROACH_COST(x, y);
                if (cost >= 100) continue;
                int dsq = encounter_attack_seek_nearest_dsq(
                    tile_x, tile_y, seek_tiles, seek_count);
                if (dsq < best_dsq || (dsq == best_dsq && cost < best_cost)) {
                    selected_x = x;
                    selected_y = y;
                    best_dsq = dsq;
                    best_cost = cost;
                }
            }
        }
    }

    int cur_x = selected_x;
    int cur_y = selected_y;
    if (selected_x < 0) goto approach_done;

    result.found = 1;
    result.dest_x = arena_base_x + selected_x;
    result.dest_y = arena_base_y + selected_y;
    if (selected_x == local_src_x && selected_y == local_src_y)
        goto approach_done;

    while (1) {
        int v = APPROACH_VIA(cur_x, cur_y);
        int prev_x = cur_x;
        int prev_y = cur_y;
        if (v & VIA_W) prev_x++;
        else if (v & VIA_E) prev_x--;
        if (v & VIA_S) prev_y++;
        else if (v & VIA_N) prev_y--;

        if (prev_x == local_src_x && prev_y == local_src_y) {
            result.next_dx = cur_x - local_src_x;
            result.next_dy = cur_y - local_src_y;
            goto approach_done;
        }

        cur_x = prev_x;
        cur_y = prev_y;
        if (APPROACH_VIA(cur_x, cur_y) == VIA_NONE ||
                APPROACH_VIA(cur_x, cur_y) == VIA_START) {
            result.found = 0;
            result.next_dx = 0;
            result.next_dy = 0;
            goto approach_done;
        }
    }

approach_done:
    #undef APPROACH_VISITED
    #undef APPROACH_VISIT
    #undef APPROACH_VIA
    #undef APPROACH_COST
    #undef APPROACH_EB
    return result;
}

static inline int encounter_chase_attack_target(
    Player* p, int target_x, int target_y, int target_size, int attack_range,
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    encounter_walkable_fn is_walkable, void* ctx,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    const OsrsLosQuery* los_query,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    int dist = encounter_rect_distance(p->x, p->y, 1,
                                                   target_x, target_y, target_size);

    if (dist == 0) {
        int max_r = (target_size + 1) / 2 + 1;
        int best_dsq = 9999, bx = -1, by = -1;
        for (int dy = -max_r; dy <= max_r; dy++) {
            for (int dx = -max_r; dx <= max_r; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = p->x + dx, ny = p->y + dy;
                if (!is_walkable(ctx, nx, ny)) continue;
                if (encounter_entity_footprints_overlap(nx, ny, 1,
                                                        target_x, target_y, target_size))
                    continue;
                int d = dx * dx + dy * dy;
                if (d < best_dsq) { best_dsq = d; bx = nx; by = ny; }
            }
        }
        if (bx < 0) return 0;
        int steps = encounter_walk_toward(p, bx, by,
            cmap, world_offset_x, world_offset_y,
            is_walkable, ctx, extra_blocked, blocked_ctx,
            arena_base_x, arena_base_y, arena_w, arena_h);
        return steps > 0 ? 1 : 0;
    }

    if (encounter_player_can_attack(p->x, p->y, target_x, target_y,
                                     target_size, attack_range,
                                     los_query))
        return 0;

    int cx, cy;
    cx = -1;
    cy = -1;

    if (arena_w <= 0) {
        int scan_min_x = target_x - attack_range;
        int scan_max_x = target_x + target_size - 1 + attack_range;
        int scan_min_y = target_y - attack_range;
        int scan_max_y = target_y + target_size - 1 + attack_range;

        cx = -1;
        cy = -1;
        int best_player_dsq = 0x3fffffff;
        int best_target_dist = 0x3fffffff;
        if (scan_min_x <= scan_max_x && scan_min_y <= scan_max_y) {
            for (int yy = scan_min_y; yy <= scan_max_y; yy++) {
                for (int xx = scan_min_x; xx <= scan_max_x; xx++) {
                    if (!is_walkable(ctx, xx, yy)) continue;
                    if (!encounter_player_can_attack(xx, yy, target_x, target_y,
                            target_size, attack_range,
                            los_query))
                        continue;
                    int dx = xx - p->x;
                    int dy = yy - p->y;
                    int player_dsq = dx * dx + dy * dy;
                    int target_dist = encounter_rect_distance(
                        xx, yy, 1, target_x, target_y, target_size);
                    if (player_dsq < best_player_dsq ||
                            (player_dsq == best_player_dsq &&
                             target_dist < best_target_dist)) {
                        best_player_dsq = player_dsq;
                        best_target_dist = target_dist;
                        cx = xx;
                        cy = yy;
                    }
                }
            }
        }

        if (cx < 0) {
            cx = p->x < target_x ? target_x :
                 (p->x > target_x + target_size - 1 ? target_x + target_size - 1 : p->x);
            cy = p->y < target_y ? target_y :
                 (p->y > target_y + target_size - 1 ? target_y + target_size - 1 : p->y);
        }
    }

    int steps = 0;
    for (int step = 0; step < 2; step++) {
        if (encounter_player_can_attack(p->x, p->y, target_x, target_y,
                                         target_size, attack_range,
                                         los_query))
            break;
        PathResult pr = (arena_w > 0)
            ? encounter_pathfind_arena_attack_approach(
                cmap, world_offset_x, world_offset_y,
                p->x, p->y,
                target_x, target_y, target_size, attack_range,
                is_walkable, ctx,
                extra_blocked, blocked_ctx,
                los_query,
                arena_base_x, arena_base_y, arena_w, arena_h)
            : encounter_pathfind(cmap, world_offset_x, world_offset_y,
                p->x, p->y, cx, cy,
                extra_blocked, blocked_ctx);
        if (!pr.found || (pr.next_dx == 0 && pr.next_dy == 0)) break;
        int nx = p->x + pr.next_dx, ny = p->y + pr.next_dy;
        if (!is_walkable(ctx, nx, ny)) break;
        p->x = nx; p->y = ny;
        steps++;
    }
    p->is_running = (steps == 2);
    p->dest_x = p->x; p->dest_y = p->y;
    return steps > 0 ? 1 : 0;
}

typedef int (*encounter_npc_blocked_fn)(void* ctx, int x, int y, int size);
typedef int (*encounter_npc_overlap_hold_fn)(void* ctx);

typedef enum {
    ENCOUNTER_NPC_STEP_TRAVEL_TARGET = 0,
    ENCOUNTER_NPC_STEP_STOP_AT_MELEE = 1,
    ENCOUNTER_NPC_STEP_OSRS_AGGRO_TARGET = 2,
    ENCOUNTER_NPC_STEP_OSRS_AGGRO_STOP_AT_MELEE = 3,
} EncounterNpcStepPolicy;

#define ENCOUNTER_NPC_UNDER_PLAYER_NONE  0
#define ENCOUNTER_NPC_UNDER_PLAYER_MOVED 1
#define ENCOUNTER_NPC_UNDER_PLAYER_HELD  2
#define ENCOUNTER_NPC_UNDER_PLAYER_BLOCKED 3

static inline int encounter_npc_x_edge_clear(
    int x, int y, int size, int dx, int dy,
    encounter_npc_blocked_fn is_blocked, void* ctx
) {
    if (dx == 0) return 1;
    int ex = (dx == 1) ? x + size : x - 1;
    int y_start = (dy == -1) ? y - 1 : y;
    int y_end = (dy == 1) ? y + size : y + size - 1;
    for (int ey = y_start; ey <= y_end; ey++)
        if (is_blocked(ctx, ex, ey, 1)) return 0;
    return 1;
}

static inline int encounter_npc_y_edge_clear(
    int x, int y, int size, int dx, int dy,
    encounter_npc_blocked_fn is_blocked, void* ctx
) {
    if (dy == 0) return 1;
    int ey = (dy == 1) ? y + size : y - 1;
    int x_start = (dx == -1) ? x - 1 : x;
    int x_end = (dx == 1) ? x + size : x + size - 1;
    for (int ex = x_start; ex <= x_end; ex++)
        if (is_blocked(ctx, ex, ey, 1)) return 0;
    return 1;
}

static inline int encounter_npc_axis_gap(int a, int a_size, int b, int b_size) {
    int a_max = a + a_size - 1;
    int b_max = b + b_size - 1;
    if (a_max < b) return b - a_max;
    if (b_max < a) return a - b_max;
    return 0;
}

static inline int encounter_npc_try_step(
    int* x, int* y, int size, int dx, int dy,
    encounter_npc_blocked_fn is_blocked, void* ctx
) {
    if (dx == 0 && dy == 0) return 0;
    int x_clear = encounter_npc_x_edge_clear(*x, *y, size, dx, dy, is_blocked, ctx);
    int y_clear = encounter_npc_y_edge_clear(*x, *y, size, dx, dy, is_blocked, ctx);
    if (x_clear && y_clear) {
        *x += dx;
        *y += dy;
        return 1;
    }
    return 0;
}

static inline int encounter_npc_step_out_from_under(
    int* npc_x, int* npc_y, int npc_size,
    int player_x, int player_y,
    encounter_npc_blocked_fn is_blocked, void* ctx,
    encounter_npc_overlap_hold_fn hold_overlap,
    uint32_t* rng
) {
    if (!encounter_entity_footprints_overlap(
            *npc_x, *npc_y, npc_size, player_x, player_y, 1))
        return ENCOUNTER_NPC_UNDER_PLAYER_NONE;
    if (hold_overlap && hold_overlap(ctx)) return ENCOUNTER_NPC_UNDER_PLAYER_HELD;

    int axis = encounter_rand_int(rng, 2);
    int sign = encounter_rand_int(rng, 2) == 0 ? 1 : -1;
    int dx = axis == 0 ? sign : 0;
    int dy = axis == 1 ? sign : 0;

    if (encounter_npc_try_step(npc_x, npc_y, npc_size, dx, dy, is_blocked, ctx))
        return ENCOUNTER_NPC_UNDER_PLAYER_MOVED;
    return ENCOUNTER_NPC_UNDER_PLAYER_BLOCKED;
}

static inline int encounter_npc_step_toward_policy(
    int* x, int* y, int tx, int ty, int npc_size,
    int target_size, EncounterNpcStepPolicy policy,
    encounter_npc_blocked_fn is_blocked, void* ctx,
    encounter_npc_overlap_hold_fn hold_overlap,
    uint32_t* rng
) {
    int size = npc_size;
    int is_aggro_policy =
        policy == ENCOUNTER_NPC_STEP_OSRS_AGGRO_TARGET ||
        policy == ENCOUNTER_NPC_STEP_OSRS_AGGRO_STOP_AT_MELEE;
    int stops_at_melee =
        policy == ENCOUNTER_NPC_STEP_STOP_AT_MELEE ||
        policy == ENCOUNTER_NPC_STEP_OSRS_AGGRO_STOP_AT_MELEE;

    if (is_aggro_policy &&
            encounter_entity_footprints_overlap(
                *x, *y, size, tx, ty, target_size)) {
        assert(target_size == 1);
        assert(rng);
        int stepped = encounter_npc_step_out_from_under(
            x, y, size, tx, ty, is_blocked, ctx, hold_overlap, rng);
        return stepped == ENCOUNTER_NPC_UNDER_PLAYER_MOVED;
    }

    int x_gap = encounter_npc_axis_gap(*x, size, tx, target_size);
    int y_gap = encounter_npc_axis_gap(*y, size, ty, target_size);
    int raw_dx = tx - *x;
    int raw_dy = ty - *y;
    int dx = (raw_dx > 0) - (raw_dx < 0);
    int dy = (raw_dy > 0) - (raw_dy < 0);

    if (x_gap == 0 && y_gap == 0) return 0;
    if (stops_at_melee && x_gap + y_gap == 1) return 0;
    if (dx == 0 && dy == 0) return 0;

    if (stops_at_melee && x_gap == 1 && y_gap == 1) {
        return encounter_npc_try_step(x, y, size, dx, 0, is_blocked, ctx);
    }

    if (is_aggro_policy &&
            dx != 0 && dy != 0 &&
            encounter_entity_footprints_overlap(
                *x + dx, *y + dy, size, tx, ty, target_size)) {
        dy = 0;
    }

    if (dx != 0 && dy != 0 &&
        encounter_npc_try_step(x, y, size, dx, dy, is_blocked, ctx))
        return 1;
    if (dx != 0 && encounter_npc_try_step(x, y, size, dx, 0, is_blocked, ctx))
        return 1;
    if (dy != 0 && encounter_npc_try_step(x, y, size, 0, dy, is_blocked, ctx))
        return 1;
    return 0;
}

static inline void encounter_damage_player(
    Player* p, int damage, float* damage_tracker
) {
    if (damage > 0) {
        p->current_hitpoints -= damage;
        if (p->current_hitpoints < 0) p->current_hitpoints = 0;
        if (damage_tracker) *damage_tracker += (float)damage;
    }
    p->hit_landed_this_tick = 1;
    p->hit_damage = damage > 0 ? damage : 0;
}

static inline int encounter_damage_npc(
    int* hp, int* hit_landed, int* hit_damage, int damage
) {
    int applied = 0;
    if (damage > 0) {
        applied = damage > *hp ? *hp : damage;
        if (applied < 0) applied = 0;
        *hp -= applied;
        if (*hp < 0) *hp = 0;
    }
    *hit_landed = 1;
    *hit_damage = applied;
    return applied;
}

static inline int encounter_resolve_npc_pending_hit(
    EncounterPendingHit* ph,
    int* npc_hp, int* hit_landed, int* hit_damage,
    int* frozen_ticks, int* blood_heal_acc, float* damage_dealt_acc
) {
    (void)frozen_ticks;
    if (!ph->active) return 0;
    ph->ticks_remaining--;
    if (ph->ticks_remaining > 0) return 0;

    int dmg = encounter_damage_npc(npc_hp, hit_landed, hit_damage, ph->damage);
    if (damage_dealt_acc) *damage_dealt_acc += dmg;

    if (ph->spell_type == ENCOUNTER_SPELL_BLOOD && blood_heal_acc)
        *blood_heal_acc += dmg;

    ph->active = 0;
    return 1;
}

typedef void (*EncounterPendingHitObserver)(
    void* user, const EncounterPendingHit* hit, int damage_after_prayer,
    int damage_applied, int prayer_was_correct, int prayer_was_checked);

static inline void encounter_resolve_player_pending_hits_observed(
    EncounterPendingHitQueue* queue,
    Player* player, OverheadPrayer active_prayer,
    float* damage_received_acc, int* prayer_correct_count,
    int* off_prayer_hit_count,
    EncounterPendingHitObserver observer, void* observer_user
) {
    for (int i = 0; i < queue->count; i++) {
        EncounterPendingHit* hit = &queue->hits[i];

        if (hit->check_prayer && hit->prayer_check_delay > 0) {
            hit->prayer_check_delay--;
            if (hit->prayer_check_delay == 0) {
                if (encounter_prayer_correct_for_style(active_prayer, hit->attack_style)) {
                    hit->damage = 0;
                    if (prayer_correct_count) (*prayer_correct_count)++;
                } else if (hit->damage > 0 && hit->attack_style != ATTACK_STYLE_NONE) {
                    if (off_prayer_hit_count) (*off_prayer_hit_count)++;
                }
                hit->check_prayer = 0;
            }
        }
        hit->ticks_remaining--;
        if (hit->ticks_remaining <= 0) {
            int dmg = hit->damage;
            int checked = hit->check_prayer;
            int prayed = 0;
            if (hit->check_prayer) {
                prayed = encounter_prayer_correct_for_style(active_prayer, hit->attack_style);
                if (prayed) {
                    dmg = 0;
                    if (prayer_correct_count) (*prayer_correct_count)++;
                } else if (dmg > 0 && hit->attack_style != ATTACK_STYLE_NONE) {
                    if (off_prayer_hit_count) (*off_prayer_hit_count)++;
                }
            } else if (dmg > 0 && hit->attack_style != ATTACK_STYLE_NONE) {
                if (off_prayer_hit_count) (*off_prayer_hit_count)++;
            }

            int hitpoints_before = player->current_hitpoints;
            encounter_damage_player(player, dmg, NULL);
            int applied = hitpoints_before - player->current_hitpoints;
            if (damage_received_acc)
                *damage_received_acc += (float)applied;
            if (observer && hit->attack_style != ATTACK_STYLE_NONE)
                observer(observer_user, hit, dmg, applied, prayed, checked);
            encounter_pending_hit_queue_remove(queue, i, "player");
            i--;
        }
    }
}

static inline void encounter_clear_tick_flags(Player* p) {
    p->attack_style_this_tick = ATTACK_STYLE_NONE;
    p->magic_type_this_tick = 0;
    p->hit_landed_this_tick = 0;
    p->hit_damage = 0;
    p->hit_was_successful = 0;
    p->elysian_proc_this_tick = 0;
    p->cast_veng_this_tick = 0;
    p->ate_food_this_tick = 0;
    p->ate_karambwan_this_tick = 0;
    p->used_special_this_tick = 0;
}

static inline uint32_t encounter_resolve_seed(uint32_t saved_rng, uint32_t explicit_seed) {
    uint32_t rng = 12345;
    if (saved_rng != 0) rng = saved_rng;
    if (explicit_seed != 0) rng = explicit_seed;
    return rng;
}

static inline int encounter_overhead_drain_effect(OverheadPrayer prayer) {
    switch (prayer) {
        case PRAYER_PROTECT_MELEE:  return 12;
        case PRAYER_PROTECT_RANGED: return 12;
        case PRAYER_PROTECT_MAGIC:  return 12;
        case PRAYER_SMITE:          return 12;
        case PRAYER_REDEMPTION:     return 6;
        default: return 0;
    }
}

static inline int encounter_offensive_drain_effect(OffensivePrayer prayer) {
    switch (prayer) {
        case OFFENSIVE_PRAYER_PIETY:       return 24;
        case OFFENSIVE_PRAYER_RIGOUR:      return 24;
        case OFFENSIVE_PRAYER_AUGURY:      return 24;
        case OFFENSIVE_PRAYER_MELEE_LOW:   return 6;
        case OFFENSIVE_PRAYER_RANGED_LOW:  return 6;
        case OFFENSIVE_PRAYER_MAGIC_LOW:   return 6;
        default: return 0;
    }
}

static inline int encounter_player_prayer_bonus(const Player* p) {
    EquipmentBonuses bonuses;
    osrs_sum_equipment_bonuses(p->equipped, &bonuses);
    return bonuses.prayer;
}

static inline void encounter_drain_all_prayers(Player* p, int prayer_bonus) {
    int overhead_effect  = p->prayer_just_activated
        ? 0 : encounter_overhead_drain_effect(p->prayer);
    int offensive_effect = p->offensive_prayer_just_activated
        ? 0 : encounter_offensive_drain_effect(p->offensive_prayer);
    int total = overhead_effect + offensive_effect;

    p->prayer_just_activated = 0;
    p->offensive_prayer_just_activated = 0;

    if (p->current_prayer <= 0) {
        p->current_prayer = 0;
        p->prayer_drain_counter = 0;
        p->prayer = PRAYER_NONE;
        p->offensive_prayer = OFFENSIVE_PRAYER_NONE;
        return;
    }
    if (total <= 0) return;

    int drain_resistance = 60 + prayer_bonus * 2;
    p->prayer_drain_counter += total;
    while (p->prayer_drain_counter > drain_resistance) {
        p->current_prayer--;
        p->prayer_drain_counter -= drain_resistance;
        if (p->current_prayer <= 0) {
            p->current_prayer = 0;
            p->prayer_drain_counter = 0;
            p->prayer = PRAYER_NONE;
            p->offensive_prayer = OFFENSIVE_PRAYER_NONE;
            break;
        }
    }
}

typedef struct {
    int attack_bonus;
    int strength_bonus;
    int eff_level;
    int max_hit;
    int attack_speed;
    int attack_range;
    AttackStyle style;
    FightStyle fight_style;
    int def_stab, def_slash, def_crush, def_magic, def_ranged;
    float att_prayer_mult;
    float str_prayer_mult;
    int spell_base_damage;
} EncounterLoadoutStats;

static inline void encounter_effective_loadout_for_equipment(
    const uint8_t loadout[NUM_GEAR_SLOTS],
    uint8_t out[NUM_GEAR_SLOTS]
) {
    memcpy(out, loadout, NUM_GEAR_SLOTS);
    out[GEAR_SLOT_SHIELD] = osrs_suppress_shield_for_two_handed_weapon(
        out[GEAR_SLOT_WEAPON], out[GEAR_SLOT_SHIELD]);
}

static inline void encounter_derive_loadout_effect_profile(
    const uint8_t loadout[NUM_GEAR_SLOTS],
    OsrsEquipmentEffectProfile* out
) {
    uint8_t effective_loadout[NUM_GEAR_SLOTS];
    encounter_effective_loadout_for_equipment(loadout, effective_loadout);
    osrs_derive_equipment_effect_profile(effective_loadout, out);
}

static inline void encounter_offensive_prayer_mults(
    OffensivePrayer op, AttackStyle style, float* att_out, float* str_out
) {
    float att = 1.0f, str = 1.0f;
    switch (style) {
        case ATTACK_STYLE_MELEE:
            if (op == OFFENSIVE_PRAYER_PIETY) {
                att = 1.20f;
                str = 1.23f;
            } else if (op == OFFENSIVE_PRAYER_MELEE_LOW) {
                att = 1.15f;
                str = 1.15f;
            }
            break;
        case ATTACK_STYLE_RANGED:
            if (op == OFFENSIVE_PRAYER_RIGOUR) {
                att = 1.20f;
                str = 1.23f;
            } else if (op == OFFENSIVE_PRAYER_RANGED_LOW) {
                att = 1.15f;
                str = 1.15f;
            }
            break;
        case ATTACK_STYLE_MAGIC:
            if (op == OFFENSIVE_PRAYER_AUGURY) {
                att = 1.25f;
            } else if (op == OFFENSIVE_PRAYER_MAGIC_LOW) {
                att = 1.15f;
            }
            break;
        default: break;
    }
    *att_out = att;
    *str_out = str;
}

static inline void encounter_compute_loadout_stats(
    const uint8_t loadout[NUM_GEAR_SLOTS],
    AttackStyle style,
    OffensivePrayer offensive_prayer,
    int base_level,
    FightStyle fight_style,
    int spell_base_damage,
    EncounterLoadoutStats* out
) {
    memset(out, 0, sizeof(*out));
    out->style = style;
    out->fight_style = fight_style;

    uint8_t effective_loadout[NUM_GEAR_SLOTS];
    encounter_effective_loadout_for_equipment(loadout, effective_loadout);

    EquipmentBonuses eb;
    osrs_sum_equipment_bonuses(effective_loadout, &eb);

    const Item* weapon_item = get_item(loadout[GEAR_SLOT_WEAPON]);
    if (style == ATTACK_STYLE_MAGIC && weapon_item &&
            (weapon_item->effect_mask & OSRS_ITEM_EFFECT_TUMEKENS_SHADOW)) {
        eb.magic_damage *= 3;
        if (eb.magic_damage > 100) eb.magic_damage = 100;
    }

    out->def_stab = eb.defence_stab;
    out->def_slash = eb.defence_slash;
    out->def_crush = eb.defence_crush;
    out->def_magic = eb.defence_magic;
    out->def_ranged = eb.defence_ranged;

    out->attack_speed = eb.attack_speed + osrs_stance_speed_mod(fight_style);
    out->attack_range = eb.attack_range + osrs_stance_range_mod(fight_style);

    if (style == ATTACK_STYLE_MAGIC) {
        out->attack_bonus = eb.attack_magic;
    } else if (style == ATTACK_STYLE_RANGED) {
        out->attack_bonus = eb.attack_ranged;
    } else {
        out->attack_bonus = eb.attack_stab;
        if (eb.attack_slash > out->attack_bonus) out->attack_bonus = eb.attack_slash;
        if (eb.attack_crush > out->attack_bonus) out->attack_bonus = eb.attack_crush;
    }

    float att_prayer_mult, str_prayer_mult;
    encounter_offensive_prayer_mults(
        offensive_prayer, style, &att_prayer_mult, &str_prayer_mult);

    out->att_prayer_mult = att_prayer_mult;
    out->str_prayer_mult = str_prayer_mult;
    out->spell_base_damage = spell_base_damage;

    int att_stance_bonus = osrs_stance_att_bonus(fight_style, style);
    int str_stance_bonus = osrs_stance_str_bonus(fight_style);

    if (style == ATTACK_STYLE_MAGIC) {
        out->eff_level = osrs_magic_effective_attack_level(
            base_level, att_prayer_mult, fight_style);
    } else {
        out->eff_level = (int)(base_level * att_prayer_mult) + att_stance_bonus + 8;
    }

    int eff_str_level = (int)(base_level * str_prayer_mult) + str_stance_bonus + 8;

    float magic_dmg_prayer_mult = osrs_offensive_magic_dmg_mult(offensive_prayer);

    if (style == ATTACK_STYLE_RANGED) {
        out->strength_bonus = eb.ranged_strength;
        out->max_hit = (int)(0.5 + eff_str_level * (eb.ranged_strength + 64) / 640.0);
    } else if (style == ATTACK_STYLE_MAGIC) {
        out->strength_bonus = eb.magic_damage;
        out->max_hit = (int)(spell_base_damage * (1.0 + eb.magic_damage / 100.0) * magic_dmg_prayer_mult);
    } else {
        out->strength_bonus = eb.melee_strength;
        out->max_hit = (int)(0.5 + eff_str_level * (eb.melee_strength + 64) / 640.0);
    }
}

static inline void encounter_update_loadout_level(
    EncounterLoadoutStats* ls, OffensivePrayer offensive_prayer,
    int current_att_level, int current_str_level
) {
    float att_prayer_mult, str_prayer_mult;
    encounter_offensive_prayer_mults(
        offensive_prayer, ls->style, &att_prayer_mult, &str_prayer_mult);
    ls->att_prayer_mult = att_prayer_mult;
    ls->str_prayer_mult = str_prayer_mult;

    int att_stance_bonus = osrs_stance_att_bonus(ls->fight_style, ls->style);
    int str_stance_bonus = osrs_stance_str_bonus(ls->fight_style);
    if (ls->style == ATTACK_STYLE_MAGIC) {
        ls->eff_level = osrs_magic_effective_attack_level(
            current_att_level, att_prayer_mult, ls->fight_style);
        float magic_dmg_mult = osrs_offensive_magic_dmg_mult(offensive_prayer);
        ls->max_hit = (int)(ls->spell_base_damage * (1.0 + ls->strength_bonus / 100.0) * magic_dmg_mult);
    } else {
        ls->eff_level = (int)(current_att_level * att_prayer_mult) + att_stance_bonus + 8;
        int eff_str = (int)(current_str_level * str_prayer_mult) + str_stance_bonus + 8;
        ls->max_hit = (int)(0.5 + eff_str * (ls->strength_bonus + 64) / 640.0);
    }
}

static inline void encounter_compute_player_equipped_stats(
    Player* p,
    AttackStyle style,
    FightStyle fight_style,
    int spell_base_damage,
    EncounterLoadoutStats* out
) {
    int current_att = p->current_attack;
    int current_str = p->current_strength;
    if (style == ATTACK_STYLE_RANGED) {
        current_att = p->current_ranged;
        current_str = p->current_ranged;
    } else if (style == ATTACK_STYLE_MAGIC) {
        current_att = p->current_magic;
        current_str = p->current_magic;
    }
    encounter_compute_loadout_stats(
        p->equipped,
        style,
        p->offensive_prayer,
        current_att,
        fight_style,
        spell_base_damage,
        out);
    encounter_update_loadout_level(out, p->offensive_prayer, current_att, current_str);
}

static inline void encounter_init_maxed_player_combat_stats(
    Player* p,
    int prayer_level
) {
    p->base_attack = MAXED_BASE_ATTACK;
    p->base_strength = MAXED_BASE_STRENGTH;
    p->base_defence = MAXED_BASE_DEFENCE;
    p->base_ranged = MAXED_BASE_RANGED;
    p->base_magic = MAXED_BASE_MAGIC;
    p->base_prayer = prayer_level;
    p->base_hitpoints = MAXED_BASE_HITPOINTS;

    p->current_attack = p->base_attack;
    p->current_strength = p->base_strength;
    p->current_defence = p->base_defence;
    p->current_ranged = p->base_ranged;
    p->current_magic = p->base_magic;
    p->current_prayer = p->base_prayer;
    p->current_hitpoints = p->base_hitpoints;
}

static inline void encounter_apply_saturated_heart_boost(Player* p) {
    int boost = osrs_saturated_heart_magic_boost(p->base_magic);
    int cap = p->base_magic + boost;
    p->current_magic += boost;
    if (p->current_magic > cap) p->current_magic = cap;
    p->saturated_heart_active_ticks = 500;
}

static inline int encounter_tick_saturated_heart(Player* p) {
    if (p->saturated_heart_active_ticks <= 0) return 0;
    p->saturated_heart_active_ticks -= 1;
    if (p->saturated_heart_active_ticks > 0) return 0;
    if (p->current_magic <= p->base_magic) return 0;
    p->current_magic = p->base_magic;
    return 1;
}

static inline int encounter_saturated_heart_protects_magic(const Player* p) {
    if (p->saturated_heart_active_ticks <= 0) return 0;
    int cap = p->base_magic + osrs_saturated_heart_magic_boost(p->base_magic);
    return p->current_magic <= cap && p->current_magic > p->base_magic;
}

static inline int encounter_decay_stat_toward_base(int* current, int base) {
    if (*current > base) {
        *current -= 1;
        return 1;
    }
    if (*current < base) {
        *current += 1;
        return 1;
    }
    return 0;
}

static inline int encounter_decay_player_combat_stats_toward_base(Player* p) {
    int changed = 0;
    changed |= encounter_decay_stat_toward_base(&p->current_attack, p->base_attack);
    changed |= encounter_decay_stat_toward_base(&p->current_strength, p->base_strength);
    changed |= encounter_decay_stat_toward_base(&p->current_defence, p->base_defence);
    changed |= encounter_decay_stat_toward_base(&p->current_ranged, p->base_ranged);
    if (!encounter_saturated_heart_protects_magic(p))
        changed |= encounter_decay_stat_toward_base(&p->current_magic, p->base_magic);
    return changed;
}

#define ENCOUNTER_STAT_DRIFT_TICKS 100
#define ENCOUNTER_DIVINE_POTION_TICKS 500
#define ENCOUNTER_STAT_DRIFT_UNPINNED (-1)

typedef struct {
    int attack_floor;
    int strength_floor;
    int defence_floor;
    int ranged_floor;
    int magic_floor;
} EncounterStatDriftPins;

static inline EncounterStatDriftPins encounter_stat_drift_no_pins(void) {
    return (EncounterStatDriftPins){
        .attack_floor = ENCOUNTER_STAT_DRIFT_UNPINNED,
        .strength_floor = ENCOUNTER_STAT_DRIFT_UNPINNED,
        .defence_floor = ENCOUNTER_STAT_DRIFT_UNPINNED,
        .ranged_floor = ENCOUNTER_STAT_DRIFT_UNPINNED,
        .magic_floor = ENCOUNTER_STAT_DRIFT_UNPINNED,
    };
}

static inline int encounter_merge_stat_drift_floor(int a, int b) {
    if (a == ENCOUNTER_STAT_DRIFT_UNPINNED) return b;
    if (b == ENCOUNTER_STAT_DRIFT_UNPINNED) return a;
    return a > b ? a : b;
}

static inline EncounterStatDriftPins encounter_merge_stat_drift_pins(
    EncounterStatDriftPins a,
    EncounterStatDriftPins b
) {
    return (EncounterStatDriftPins){
        .attack_floor = encounter_merge_stat_drift_floor(a.attack_floor, b.attack_floor),
        .strength_floor = encounter_merge_stat_drift_floor(a.strength_floor, b.strength_floor),
        .defence_floor = encounter_merge_stat_drift_floor(a.defence_floor, b.defence_floor),
        .ranged_floor = encounter_merge_stat_drift_floor(a.ranged_floor, b.ranged_floor),
        .magic_floor = encounter_merge_stat_drift_floor(a.magic_floor, b.magic_floor),
    };
}

static inline EncounterStatDriftPins encounter_divine_super_combat_pins(const Player* p) {
    EncounterStatDriftPins pins = encounter_stat_drift_no_pins();
    pins.attack_floor = p->base_attack + osrs_super_combat_boost_amount(p->base_attack);
    pins.strength_floor =
        p->base_strength + osrs_super_combat_boost_amount(p->base_strength);
    pins.defence_floor = p->base_defence + osrs_super_combat_boost_amount(p->base_defence);
    return pins;
}

static inline EncounterStatDriftPins encounter_divine_ranging_pins(const Player* p) {
    EncounterStatDriftPins pins = encounter_stat_drift_no_pins();
    pins.ranged_floor = p->base_ranged + osrs_ranging_boost_amount(p->base_ranged);
    return pins;
}

static inline int encounter_enforce_stat_drift_floor(int* current, int base, int floor) {
    if (floor <= base) return 0;
    if (*current >= floor) return 0;
    *current = floor;
    return 1;
}

static inline int encounter_enforce_stat_drift_pins(Player* p, EncounterStatDriftPins pins) {
    int changed = 0;
    changed |= encounter_enforce_stat_drift_floor(
        &p->current_attack, p->base_attack, pins.attack_floor);
    changed |= encounter_enforce_stat_drift_floor(
        &p->current_strength, p->base_strength, pins.strength_floor);
    changed |= encounter_enforce_stat_drift_floor(
        &p->current_defence, p->base_defence, pins.defence_floor);
    changed |= encounter_enforce_stat_drift_floor(
        &p->current_ranged, p->base_ranged, pins.ranged_floor);
    changed |= encounter_enforce_stat_drift_floor(
        &p->current_magic, p->base_magic, pins.magic_floor);
    return changed;
}

static inline int encounter_drift_stat_toward_base_with_floor(
    int* current,
    int base,
    int floor
) {
    if (floor > base && *current <= floor) {
        if (*current == floor) return 0;
        *current = floor;
        return 1;
    }
    if (*current > base) {
        *current -= 1;
        if (floor > base && *current < floor) *current = floor;
        return 1;
    }
    if (*current < base) {
        *current += 1;
        return 1;
    }
    return 0;
}

static inline int encounter_tick_stat_drift(
    Player* p,
    int* stat_drift_timer,
    EncounterStatDriftPins pins
) {
    assert(p && stat_drift_timer);
    assert(*stat_drift_timer >= 0 && *stat_drift_timer < ENCOUNTER_STAT_DRIFT_TICKS);

    int changed = encounter_enforce_stat_drift_pins(p, pins);
    *stat_drift_timer += 1;
    if (*stat_drift_timer < ENCOUNTER_STAT_DRIFT_TICKS) return changed;
    *stat_drift_timer = 0;

    changed |= encounter_drift_stat_toward_base_with_floor(
        &p->current_attack, p->base_attack, pins.attack_floor);
    changed |= encounter_drift_stat_toward_base_with_floor(
        &p->current_strength, p->base_strength, pins.strength_floor);
    changed |= encounter_drift_stat_toward_base_with_floor(
        &p->current_defence, p->base_defence, pins.defence_floor);
    changed |= encounter_drift_stat_toward_base_with_floor(
        &p->current_ranged, p->base_ranged, pins.ranged_floor);
    changed |= encounter_drift_stat_toward_base_with_floor(
        &p->current_magic, p->base_magic, pins.magic_floor);
    if (p->current_hitpoints > 0)
        changed |= encounter_decay_stat_toward_base(&p->current_hitpoints, p->base_hitpoints);
    return changed;
}

typedef enum {
    ENCOUNTER_CONSUMABLE_STAT_EFFECT_NONE = 0,
    ENCOUNTER_CONSUMABLE_STAT_EFFECT_BREW_DRAIN,
    ENCOUNTER_CONSUMABLE_STAT_EFFECT_RESTORE,
    ENCOUNTER_CONSUMABLE_STAT_EFFECT_SANFEW,
} EncounterConsumableStatEffect;

static inline void encounter_apply_brew_heal_capped(
    Player* p, int brew_heal, int hitpoints_cap
) {
    p->current_hitpoints += brew_heal;
    if (p->current_hitpoints > hitpoints_cap) p->current_hitpoints = hitpoints_cap;
    p->ate_food_this_tick = 1;
}

static inline void encounter_apply_brew_heal(Player* p, int brew_heal) {
    encounter_apply_brew_heal_capped(p, brew_heal, p->base_hitpoints + brew_heal);
}

static inline void encounter_apply_brew_heal_and_timer(Player* p, int brew_heal) {
    encounter_apply_brew_heal(p, brew_heal);
    p->brew_doses--;
    p->potion_timer = 3;
}

static inline void encounter_add_prayer_restore(Player* p, int restore_amount) {
    p->current_prayer += restore_amount;
}

static inline void encounter_cap_prayer_restore(Player* p) {
    if (p->current_prayer > p->base_prayer)
        p->current_prayer = p->base_prayer;
}

static inline void encounter_finish_potion_dose(Player* p, int* doses) {
    (*doses)--;
    p->potion_timer = 3;
}

static inline void encounter_brew_drain_stats(Player* p) {
    BrewResult brew = osrs_brew_effect(p->base_hitpoints, p->base_defence,
                                       p->current_attack, p->current_strength,
                                       p->current_ranged, p->current_magic);

    p->current_attack -= brew.att_drain;
    if (p->current_attack < 0) p->current_attack = 0;
    p->current_strength -= brew.str_drain;
    if (p->current_strength < 0) p->current_strength = 0;
    p->current_ranged -= brew.range_drain;
    if (p->current_ranged < 0) p->current_ranged = 0;
    p->current_magic -= brew.magic_drain;
    if (p->current_magic < 0) p->current_magic = 0;

    p->current_defence += brew.def_boost;
    int def_cap = p->base_defence + brew.def_boost;
    if (p->current_defence > def_cap) p->current_defence = def_cap;
}

static inline void encounter_apply_stat_restore(Player* p, int (*amount)(int)) {
    int restore = amount(p->base_attack);
    p->current_attack += restore;
    if (p->current_attack > p->base_attack) p->current_attack = p->base_attack;
    restore = amount(p->base_strength);
    p->current_strength += restore;
    if (p->current_strength > p->base_strength) p->current_strength = p->base_strength;
    restore = amount(p->base_defence);
    p->current_defence += restore;
    if (p->current_defence > p->base_defence) p->current_defence = p->base_defence;
    restore = amount(p->base_ranged);
    p->current_ranged += restore;
    if (p->current_ranged > p->base_ranged) p->current_ranged = p->base_ranged;
    restore = amount(p->base_magic);
    p->current_magic += restore;
    if (p->current_magic > p->base_magic) p->current_magic = p->base_magic;
}

static inline void encounter_restore_stats(Player* p) {
    encounter_apply_stat_restore(p, osrs_super_restore_amount);
}

static inline void encounter_sanfew_restore_stats(Player* p) {
    encounter_apply_stat_restore(p, osrs_sanfew_restore_amount);
}

static inline void encounter_apply_consumable_stat_effect(
    Player* p,
    EncounterConsumableStatEffect effect
) {
    switch (effect) {
        case ENCOUNTER_CONSUMABLE_STAT_EFFECT_NONE:
            return;
        case ENCOUNTER_CONSUMABLE_STAT_EFFECT_BREW_DRAIN:
            encounter_brew_drain_stats(p);
            return;
        case ENCOUNTER_CONSUMABLE_STAT_EFFECT_RESTORE:
            encounter_restore_stats(p);
            return;
        case ENCOUNTER_CONSUMABLE_STAT_EFFECT_SANFEW:
            encounter_sanfew_restore_stats(p);
            return;
    }
    abort();
}

static inline void encounter_bastion_boost(Player* p) {
    int rng_boost = osrs_ranging_boost_amount(p->base_ranged);
    int def_boost = osrs_super_combat_boost_amount(p->base_defence);
    p->current_ranged += rng_boost;
    int rng_cap = p->base_ranged + rng_boost;
    if (p->current_ranged > rng_cap) p->current_ranged = rng_cap;
    p->current_defence += def_boost;
    int def_cap = p->base_defence + def_boost;
    if (p->current_defence > def_cap) p->current_defence = def_cap;
}

static inline void encounter_super_combat_boost(Player* p) {
    int boost = osrs_super_combat_boost_amount(p->base_attack);
    p->current_attack += boost;
    if (p->current_attack > p->base_attack + boost) p->current_attack = p->base_attack + boost;
    boost = osrs_super_combat_boost_amount(p->base_strength);
    p->current_strength += boost;
    if (p->current_strength > p->base_strength + boost) p->current_strength = p->base_strength + boost;
    boost = osrs_super_combat_boost_amount(p->base_defence);
    p->current_defence += boost;
    if (p->current_defence > p->base_defence + boost) p->current_defence = p->base_defence + boost;
}

static inline void encounter_ranging_boost(Player* p) {
    int boost = osrs_ranging_boost_amount(p->base_ranged);
    p->current_ranged += boost;
    if (p->current_ranged > p->base_ranged + boost) p->current_ranged = p->base_ranged + boost;
}

static inline void encounter_recompute_loadout_max_hits(
    EncounterLoadoutStats* loadouts, int num_loadouts, Player* p
) {
    for (int i = 0; i < num_loadouts; i++) {
        EncounterLoadoutStats* ls = &loadouts[i];
        if (ls->style == ATTACK_STYLE_RANGED) {
            encounter_update_loadout_level(ls, p->offensive_prayer, p->current_ranged, p->current_ranged);
        } else if (ls->style == ATTACK_STYLE_MAGIC) {
            encounter_update_loadout_level(ls, p->offensive_prayer, p->current_magic, p->current_magic);
        } else {
            encounter_update_loadout_level(ls, p->offensive_prayer, p->current_attack, p->current_strength);
        }
    }
}

static inline int encounter_use_spec(Player* p, int cost) {
    if (p->special_energy < cost) return 0;
    p->special_energy -= cost;
    return 1;
}

static inline void encounter_apply_loadout(
    Player* p, const uint8_t loadout[NUM_GEAR_SLOTS], GearSet gear_set
) {
    encounter_effective_loadout_for_equipment(loadout, p->equipped);
    p->current_gear = gear_set;
    p->visible_gear = gear_set;
    osrs_refresh_player_equipment(p);
}

static void encounter_populate_inventory(
    Player* p,
    const uint8_t* const* loadouts, int num_loadouts,
    const uint8_t extra_items[NUM_GEAR_SLOTS]
) {
    memset(p->inventory, 255 , sizeof(p->inventory));
    memset(p->num_items_in_slot, 0, sizeof(p->num_items_in_slot));

    for (int s = 0; s < NUM_GEAR_SLOTS; s++) {
        int n = 0;
        for (int l = 0; l < num_loadouts && n < MAX_ITEMS_PER_SLOT; l++) {
            uint8_t item = loadouts[l][s];
            if (item == 255 ) continue;
            int dup = 0;
            for (int j = 0; j < n; j++) { if (p->inventory[s][j] == item) { dup = 1; break; } }
            if (dup) continue;
            p->inventory[s][n++] = item;
        }
        if (extra_items && extra_items[s] != 255  && n < MAX_ITEMS_PER_SLOT) {
            int dup = 0;
            for (int j = 0; j < n; j++) { if (p->inventory[s][j] == extra_items[s]) { dup = 1; break; } }
            if (!dup) p->inventory[s][n++] = extra_items[s];
        }
        p->num_items_in_slot[s] = n;
    }
}

static inline void encounter_clear_ammo_inventory_slot(Player* p) {
    for (int i = 0; i < MAX_ITEMS_PER_SLOT; i++)
        p->inventory[GEAR_SLOT_AMMO][i] = ITEM_NONE;
    p->num_items_in_slot[GEAR_SLOT_AMMO] = 0;
}

static inline void encounter_translate_movement(HumanInput* hi, int* actions,
                                                 int head_move,
                                                 void* (*get_entity)(void*, int),
                                                 void* state) {
    if (hi->pending_move_x < 0 || hi->pending_move_y < 0 || head_move < 0) return;
    Player* player = (Player*)get_entity(state, 0);
    if (!player) return;
    int dx = hi->pending_move_x - player->x;
    int dy = hi->pending_move_y - player->y;
    int sx = (dx > 0) ? 1 : (dx < 0) ? -1 : 0;
    int sy = (dy > 0) ? 1 : (dy < 0) ? -1 : 0;
    static const int DX8[9] = { 0, 0, 1, 1, 1, 0, -1, -1, -1 };
    static const int DY8[9] = { 0, 1, 1, 0, -1, -1, -1, 0, 1 };
    for (int m = 1; m < 9; m++) {
        if (DX8[m] == sx && DY8[m] == sy) {
            actions[head_move] = m;
            break;
        }
    }
}

static inline void encounter_translate_prayer(HumanInput* hi, int* actions, int head_prayer) {
    if (hi->pending_prayer < 0 || head_prayer < 0) return;
    actions[head_prayer] = hi->pending_prayer;
}

static inline void encounter_translate_offensive_prayer(
    HumanInput* hi, int* actions, int head_offensive
) {
    if (hi->pending_offensive_prayer < 0 || head_offensive < 0) return;
    actions[head_offensive] = hi->pending_offensive_prayer;
}

static inline int encounter_find_observed_target_slot(
    const int* current_obs_slots,
    int observed_slot_count,
    int raw_npc_slot
) {
    for (int slot = 0; slot < observed_slot_count; slot++)
        if (current_obs_slots[slot] == raw_npc_slot) return slot;
    return -1;
}

typedef struct {
    const char* name;

    int obs_size;
    int num_action_heads;
    const int* action_head_dims;
    int mask_size;

    size_t state_size;
    size_t context_size;
    void (*init_context)(EncounterContext* context);
    void (*destroy_context)(EncounterContext* context);
    void (*init_state)(EncounterState* state, EncounterContext* context);

    EncounterState* (*create)(void);
    void (*destroy)(EncounterState* state);

    void (*reset)(EncounterState* state, EncounterContext* context, uint32_t seed);
    void (*step)(EncounterState* state, EncounterContext* context, const int* actions);
    void (*step_human_commands)(
        EncounterState* state, EncounterContext* context, struct HumanInput* hi);

    size_t (*snapshot_size)(EncounterState* state, EncounterContext* context);
    void (*snapshot)(EncounterState* state, EncounterContext* context, void* out);
    void (*restore)(
        EncounterState* state, EncounterContext* context, const void* data, size_t n);

    void (*write_obs)(EncounterState* state, EncounterContext* context, float* obs_out);
    void (*write_mask)(EncounterState* state, EncounterContext* context, float* mask_out);
    float (*get_reward)(EncounterState* state, EncounterContext* context);
    int (*is_terminal)(EncounterState* state, EncounterContext* context);

    int (*get_entity_count)(EncounterState* state, EncounterContext* context);
    void* (*get_entity)(
        EncounterState* state, EncounterContext* context, int index);

    void (*fill_render_entities)(
        EncounterState* state,
        EncounterContext* context,
        RenderEntity* out,
        int max_entities,
        int* count);

    void (*put_int)(EncounterState* state, EncounterContext* context, const char* key, int value);
    void (*put_float)(
        EncounterState* state, EncounterContext* context, const char* key, float value);
    void (*put_ptr)(
        EncounterState* state, EncounterContext* context, const char* key, void* value);

    int arena_base_x, arena_base_y;
    int arena_width, arena_height;

    void (*translate_human_input)(
        struct HumanInput* hi, int* actions, EncounterState* state, EncounterContext* context);
    int (*is_human_targetable_npc_slot)(
        EncounterState* state, EncounterContext* context, int npc_slot);

    int (*apply_lab_command)(
        EncounterState* state, EncounterContext* context, const char* line);

    int head_move;
    int head_prayer;
    int head_target;

    void (*render_post_tick)(
        EncounterState* state, EncounterContext* context, EncounterOverlay* overlay);

    void* (*get_log)(EncounterState* state, EncounterContext* context);

    int (*get_tick)(EncounterState* state, EncounterContext* context);
    int (*get_winner)(EncounterState* state, EncounterContext* context);
} EncounterDef;

#define MAX_ENCOUNTERS 32

typedef struct {
    const EncounterDef* defs[MAX_ENCOUNTERS];
    int count;
} EncounterRegistry;

/* WARNING: static in header — each TU gets its own copy. only works correctly
   when all encounter headers are included from a single compilation unit. */
static EncounterRegistry g_encounter_registry = { .count = 0 };

static inline void encounter_register(const EncounterDef* def) {
    if (g_encounter_registry.count >= MAX_ENCOUNTERS) {
        fprintf(stderr, "encounter registry capacity exceeded: %d\n", MAX_ENCOUNTERS);
        abort();
    }
    g_encounter_registry.defs[g_encounter_registry.count++] = def;
}

static inline const EncounterDef* encounter_find(const char* name) {
    for (int i = 0; i < g_encounter_registry.count; i++) {
        if (strcmp(g_encounter_registry.defs[i]->name, name) == 0) {
            return g_encounter_registry.defs[i];
        }
    }
    return NULL;
}

#endif
