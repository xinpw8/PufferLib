#line 76 "encounter_colosseum.h"

#define COLO_INVENTORY_CELL_OBS_FEATURES \
    OSRS_SHARED_INVENTORY_CELL_OBS_FEATURES
#define COLO_INVENTORY_CELL_ENCODER_FEATURES \
    OSRS_INVENTORY_CELL_OBS_FEATURES_COMPACT
#define COLO_INVENTORY_OBS_SIZE OSRS_SHARED_INVENTORY_OBS_SIZE
#define COLO_EQUIPPED_OBS_SIZE OSRS_SHARED_EQUIPPED_OBS_SIZE
static_assert(OSRS_ITEM_OBS_TABLE_ROWS == OSRS_ITEM_CONTENT_COUNT,
    "regenerate osrs_item_obs_table.inc: the cell code space changed");
static_assert(OSRS_ITEM_OBS_TABLE_COLS == COLO_INVENTORY_CELL_ENCODER_FEATURES,
    "regenerate osrs_item_obs_table.inc: the expanded record width changed");
static_assert(OSRS_ITEM_OBS_TABLE_BASE_HITPOINTS == MAXED_BASE_HITPOINTS,
    "regenerate osrs_item_obs_table.inc: it bakes hp_heal at this level");
static_assert(OSRS_ITEM_OBS_TABLE_BASE_PRAYER == COLO_PLAYER_PRAYER_LEVEL,
    "regenerate osrs_item_obs_table.inc: it bakes prayer_restore at this level");
static_assert(OSRS_ITEM_OBS_TABLE_BASE_RANGED == MAXED_BASE_RANGED,
    "regenerate osrs_item_obs_table.inc: it bakes offensive_boost at this level");
static_assert(OSRS_ITEM_CONTENT_COUNT <= OSRS_ITEM_OBS_CODE_SCALE,
    "every observed content code must remain exact after scaling");
#define COLO_NPC_CORE_FEATURES    11
#define COLO_NPC_TELLS_FEATURES   10
#define COLO_NPC_VENATOR_FEATURES 1
#define COLO_NPC_TYPE_CODE_FEATURES 1
#define COLO_NPC_TELLS_OFFSET     (COLO_NPC_TYPE_CODE_FEATURES + COLO_NPC_CORE_FEATURES)
#define COLO_FEATURES_PER_NPC \
    (COLO_NPC_TELLS_OFFSET + COLO_NPC_TELLS_FEATURES + COLO_NPC_VENATOR_FEATURES)
#define COLO_NPC_OBS_SIZE         (COLO_OBS_NPCS * COLO_FEATURES_PER_NPC)

#define COLO_EXPECTED_DPT_NORM    15.0f

#define COLO_MODIFIER_FLAGS_OBS_SIZE  (COLO_NUM_REAL_MODIFIERS + 1 + COLO_MODIFIER_DRAFT_OPTIONS)
#define COLO_MODIFIER_HAZARD_OBS_SIZE 42
#define COLO_MODIFIER_OBS_SIZE    (COLO_MODIFIER_FLAGS_OBS_SIZE + COLO_MODIFIER_HAZARD_OBS_SIZE)
#define COLO_WAVE_OBS_SIZE        4

typedef enum {
    COLO_SOL_PARRY_CUE_INACTIVE = 0,
    COLO_SOL_PARRY_CUE_LEAD_TIME,
    COLO_SOL_PARRY_CUE_OVERHEAD_OFF,
    COLO_SOL_PARRY_CUE_FLICK_MELEE,
    COLO_SOL_PARRY_CUE_POISONED,
    COLO_SOL_PARRY_CUE_COUNT,
} ColoSolParryCue;

typedef enum {
    COLO_SOL_HAZARD_AOE = 0,
    COLO_SOL_HAZARD_CRYSTAL_LASER,
    COLO_SOL_HAZARD_MOLTEN_SAND,
    COLO_SOL_HAZARD_SOURCE_COUNT,
} ColoSolHazardSource;

#define COLO_SOL_PARRY_LAND_TICKS_MAX 4
#define COLO_BOSS_OBS_PARRY_CUE_OFFSET 19
#define COLO_BOSS_OBS_PARRY_TICKS_OFFSET \
    (COLO_BOSS_OBS_PARRY_CUE_OFFSET + COLO_SOL_PARRY_CUE_COUNT)
#define COLO_BOSS_OBS_PARRY_DAMAGE_OFFSET \
    (COLO_BOSS_OBS_PARRY_TICKS_OFFSET + COLO_SOL_PARRY_LAND_TICKS_MAX)
#define COLO_BOSS_OBS_PREMOVE_HAZARD_OFFSET 37
#define COLO_SOL_LASER_CRYSTAL_FEATS 3
#define COLO_SOL_LASER_OBS_SIZE \
    (1 + COLO_SOL_MAX_CRYSTALS * COLO_SOL_LASER_CRYSTAL_FEATS)
#define COLO_BOSS_OBS_PREMOVE_TAIL_SIZE (2 + COLO_SOL_LASER_OBS_SIZE)
#define COLO_BOSS_OBS_SIZE \
    (COLO_BOSS_OBS_PREMOVE_HAZARD_OFFSET + COLO_BOSS_OBS_PREMOVE_TAIL_SIZE)

#define COLO_OBS_PENDING_HITS     16
#define COLO_FEATURES_PER_HIT     7
#define COLO_PENDING_HIT_OBS_SIZE (COLO_FEATURES_PER_HIT * COLO_OBS_PENDING_HITS)

#define COLO_THREAT_LOS_OBS_SIZE  3

#define COLO_THRALL_DC_OBS_SIZE   5

#define COLO_SPAWN_PHASE_OBS_SIZE 2
#define COLO_ANCHOR_BLOCKED_OBS_SIZE COLO_NUM_SPAWN_ANCHORS
#define COLO_SPAWN_OBS_SIZE (COLO_SPAWN_PHASE_OBS_SIZE + COLO_ANCHOR_BLOCKED_OBS_SIZE)

#define COLO_NUM_OBS (OSRS_SHARED_OBS_SIZE + COLO_NPC_OBS_SIZE + \
    COLO_MODIFIER_OBS_SIZE + COLO_WAVE_OBS_SIZE + COLO_BOSS_OBS_SIZE + \
    COLO_PENDING_HIT_OBS_SIZE + COLO_THREAT_LOS_OBS_SIZE + \
    COLO_THRALL_DC_OBS_SIZE + COLO_SPAWN_OBS_SIZE)

#define COLO_OBS_INVENTORY_START OSRS_SHARED_SELF_OBS_SIZE
#define COLO_OBS_EQUIPMENT_START \
    (COLO_OBS_INVENTORY_START + OSRS_SHARED_INVENTORY_OBS_SIZE)
#define COLO_OBS_AFTER_SHARED OSRS_SHARED_OBS_SIZE
#define COLO_OBS_AFTER_NPCS (COLO_OBS_AFTER_SHARED + COLO_NPC_OBS_SIZE)
#define COLO_OBS_AFTER_MODS (COLO_OBS_AFTER_NPCS + COLO_MODIFIER_OBS_SIZE)
#define COLO_OBS_AFTER_WAVE (COLO_OBS_AFTER_MODS + COLO_WAVE_OBS_SIZE)
#define COLO_OBS_AFTER_BOSS (COLO_OBS_AFTER_WAVE + COLO_BOSS_OBS_SIZE)
#define COLO_OBS_AFTER_PENDING_HITS \
    (COLO_OBS_AFTER_BOSS + COLO_PENDING_HIT_OBS_SIZE)
#define COLO_OBS_AFTER_THREAT_LOS \
    (COLO_OBS_AFTER_PENDING_HITS + COLO_THREAT_LOS_OBS_SIZE)
#define COLO_OBS_AFTER_THRALL_DC \
    (COLO_OBS_AFTER_THREAT_LOS + COLO_THRALL_DC_OBS_SIZE)
#define COLO_OBS_AFTER_SPAWN \
    (COLO_OBS_AFTER_THRALL_DC + COLO_SPAWN_OBS_SIZE)

static inline float col_clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static inline float col_obs_rel_x(int nx, int px) {
    return (float)(nx - px) / (float)COLO_ARENA_WIDTH;
}

static inline float col_obs_rel_y(int ny, int py) {
    return (float)(ny - py) / (float)COLO_ARENA_HEIGHT;
}


static ColoNPC col_matchup_representative_npc(ColoNpcType type) {
    if (type < 0 || type >= COLO_NUM_NPC_TYPES) abort();
    const ColoNpcStats* ns = &COLO_NPC_STATS[type];
    return (ColoNPC){
        .type = type,
        .hp = ns->hp,
        .max_hp = ns->hp,
        .size = ns->size,
        .active = 1,
        .death_ticks = 0,
    };
}

typedef struct {
    int same_roster;
    uint32_t changed_slots;
} ColoVenatorPreviewDelta;

static int col_collect_venator_preview_monsters(
    const ColosseumState* s,
    OsrsVenatorMonster monsters[COLO_MAX_NPCS]
) {
    int count = 0;
    for (int slot = 0; slot < COLO_MAX_NPCS; slot++) {
        const ColoNPC* npc = &s->npcs[slot];
        if (!col_npc_is_live_enemy(npc)) continue;
        monsters[count++] = col_venator_monster_from_npc(npc, slot);
    }
    return count;
}

static ColoVenatorPreviewDelta col_venator_preview_delta(
    const ColoVenatorPreviewTargets* targets,
    const OsrsVenatorMonster current[COLO_MAX_NPCS],
    int current_count
) {
    if (current_count != targets->count)
        return (ColoVenatorPreviewDelta){0};
    uint32_t changed_slots = 0;
    for (int idx = 0; idx < current_count; idx++) {
        const OsrsVenatorMonster* cached = &targets->monsters[idx];
        const OsrsVenatorMonster* now = &current[idx];
        if (cached->slot != now->slot)
            return (ColoVenatorPreviewDelta){0};
        if (cached->footprint.sw_x != now->footprint.sw_x ||
                cached->footprint.sw_y != now->footprint.sw_y ||
                cached->footprint.size != now->footprint.size ||
                cached->life != now->life)
            changed_slots |= UINT32_C(1) << now->slot;
    }
    return (ColoVenatorPreviewDelta){
        .same_roster = 1,
        .changed_slots = changed_slots,
    };
}

static void col_write_venator_preview_pair(
    ColoVenatorPreviewTargets* targets,
    int sender_idx,
    int target_idx
) {
    OsrsVenatorMonster sender = targets->monsters[sender_idx];
    OsrsVenatorMonster target = targets->monsters[target_idx];
    uint32_t target_bit = UINT32_C(1) << target.slot;
    if (osrs_venator_can_bounce(sender.footprint, target.footprint))
        targets->bounce_masks[sender.slot] |= target_bit;
    else
        targets->bounce_masks[sender.slot] &= ~target_bit;
    int distance = osrs_venator_selection_distance(
        sender.footprint, target.footprint);
    if (distance < 0 || distance > UINT8_MAX) abort();
    targets->selection_distances[sender.slot][target.slot] =
        (uint8_t)distance;
}

static const ColoVenatorPreviewTargets* col_get_venator_preview_targets(
    const ColosseumState* s,
    ColosseumContext* ctx
) {
    if (!s || !ctx) abort();
#ifdef COLO_PROFILE_ENABLED
    if (COLO_PROFILE_ENABLED())
        COLO_PROFILE_ADD(COLO_PROF_VENATOR_REQUESTS, 1.0);
#endif
    OsrsVenatorMonster current[COLO_MAX_NPCS];
    int current_count = col_collect_venator_preview_monsters(s, current);
    ColoVenatorPreviewCache* cache = &ctx->venator_preview_cache;
    ColoVenatorPreviewTargets* targets = &cache->targets;
    ColoVenatorPreviewDelta delta = cache->valid
        ? col_venator_preview_delta(targets, current, current_count)
        : (ColoVenatorPreviewDelta){0};
    if (delta.same_roster && delta.changed_slots == 0) {
#ifdef COLO_PROFILE_ENABLED
        if (COLO_PROFILE_ENABLED())
            COLO_PROFILE_ADD(COLO_PROF_VENATOR_HITS, 1.0);
#endif
        return targets;
    }

#ifdef COLO_PROFILE_ENABLED
    if (COLO_PROFILE_ENABLED())
        COLO_PROFILE_ADD(COLO_PROF_VENATOR_REFRESHES, 1.0);
#endif
    if (delta.same_roster) {
        for (int idx = 0; idx < current_count; idx++) {
            int slot = current[idx].slot;
            if ((delta.changed_slots & (UINT32_C(1) << slot)) != 0)
                targets->monsters[idx] = current[idx];
        }
        for (int sender_idx = 0; sender_idx < current_count; sender_idx++) {
            int sender_slot = current[sender_idx].slot;
            for (int target_idx = 0; target_idx < current_count; target_idx++) {
                int target_slot = current[target_idx].slot;
                if ((delta.changed_slots &
                        ((UINT32_C(1) << sender_slot) |
                         (UINT32_C(1) << target_slot))) != 0)
                    col_write_venator_preview_pair(
                        targets, sender_idx, target_idx);
            }
        }
    } else {
        memset(targets, 0, sizeof(*targets));
        targets->count = current_count;
        memcpy(
            targets->monsters,
            current,
            (size_t)current_count * sizeof(*current));
        for (int sender_idx = 0; sender_idx < current_count; sender_idx++) {
            for (int target_idx = 0; target_idx < current_count; target_idx++)
                col_write_venator_preview_pair(
                    targets, sender_idx, target_idx);
        }
    }
    cache->valid = 1;
    cache->generation++;
    return targets;
}
static int col_venator_best_bounce_slot(
    const ColoVenatorPreviewTargets* targets,
    int range_sender_slot,
    int selection_sender_slot,
    int forbidden_slot
) {
    int best_slot = -1;
    int best_distance = INT_MAX;
    uint32_t bounce_mask = targets->bounce_masks[range_sender_slot];
    for (int candidate_idx = 0;
            candidate_idx < targets->count;
            candidate_idx++) {
        int candidate_slot = targets->monsters[candidate_idx].slot;
        if (candidate_slot == forbidden_slot ||
                (bounce_mask & (UINT32_C(1) << candidate_slot)) == 0)
            continue;
        int distance =
            targets->selection_distances[selection_sender_slot][candidate_slot];
        if (osrs_venator_is_better_candidate(
                distance,
                candidate_slot,
                best_distance,
                best_slot)) {
            best_slot = candidate_slot;
            best_distance = distance;
        }
    }
    return best_slot;
}

static int col_venator_extra_bounce_if_shot(
    const ColosseumState* s,
    const ColoVenatorPreviewTargets* targets,
    int primary_slot
) {
    if (!s || !targets || primary_slot < 0 || primary_slot >= COLO_MAX_NPCS)
        abort();
    const ColoNPC* npc = &s->npcs[primary_slot];
    if (!col_npc_is_live_enemy(npc)) return 0;
    int second_slot = col_venator_best_bounce_slot(
        targets, primary_slot, primary_slot, primary_slot);
    if (second_slot < 0) return 0;
    if (targets->count < 3) return 1;
    int third_slot = col_venator_best_bounce_slot(
        targets, primary_slot, second_slot, second_slot);
    return third_slot >= 0 ? 2 : 1;
}

typedef struct {
    int found;
    int index;
    int x;
    int y;
    int distance;
} ColoNearestHazardTile;

static inline ColoNearestHazardTile col_nearest_hazard_tile_none(void) {
    return (ColoNearestHazardTile){
        .found = 0,
        .index = -1,
        .x = 0,
        .y = 0,
        .distance = 1 << 30,
    };
}

static inline void col_nearest_hazard_tile_consider(
    ColoNearestHazardTile* nearest,
    int index,
    int x,
    int y,
    int player_x,
    int player_y
) {
    int dx = abs(x - player_x);
    int dy = abs(y - player_y);
    int distance = dx > dy ? dx : dy;
    if (distance < nearest->distance) {
        nearest->found = 1;
        nearest->index = index;
        nearest->x = x;
        nearest->y = y;
        nearest->distance = distance;
    }
}

static inline void col_write_hazard_xy_value_if(
    const ColosseumState* s,
    int present,
    int x,
    int y,
    float value,
    float* obs,
    int* idx
) {
    int i = *idx;
    if (present) {
        obs[i++] = col_obs_rel_x(x, s->player.x);
        obs[i++] = col_obs_rel_y(y, s->player.y);
        obs[i++] = value;
    } else {
        obs[i++] = 0.0f;
        obs[i++] = 0.0f;
        obs[i++] = 0.0f;
    }
    *idx = i;
}

static inline void col_write_nearest_hazard_xy_value(
    const ColosseumState* s,
    const ColoNearestHazardTile* nearest,
    float value,
    float* obs,
    int* idx
) {
    col_write_hazard_xy_value_if(
        s, nearest->found, nearest->x, nearest->y, value, obs, idx);
}

static void col_refresh_current_obs_slots(ColosseumState* s) {
    for (int i = 0; i < COLO_OBS_NPCS; i++) s->current_obs_slots[i] = -1;
    int slot = 0;
    for (int i = 0; i < COLO_MAX_NPCS && slot < COLO_OBS_NPCS; i++) {
        if (s->npcs[i].active) s->current_obs_slots[slot++] = i;
    }
}

static int col_obs_slot_is_targetable(const ColosseumState* s, int obs_slot) {
    int npc_idx = s->current_obs_slots[obs_slot];
    if (npc_idx < 0 || npc_idx >= COLO_MAX_NPCS) return 0;
    const ColoNPC* npc = &s->npcs[npc_idx];
    return col_npc_is_live_target(npc);
}

static void col_tell_style_onehot(AttackStyle style, float* out3) {
    out3[0] = (style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
    out3[1] = (style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
    out3[2] = (style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;
}

typedef struct {
    int active;
    AttackStyle style;
    int ticks;
} ColoNpcNextPrayerObs;

static inline int col_prayer_attack_style(AttackStyle style) {
    return style == ATTACK_STYLE_MELEE ||
           style == ATTACK_STYLE_RANGED ||
           style == ATTACK_STYLE_MAGIC;
}

static inline ColoNpcNextPrayerObs col_npc_next_prayer_none(void) {
    return (ColoNpcNextPrayerObs){
        .active = 0,
        .style = ATTACK_STYLE_NONE,
        .ticks = 0,
    };
}

static void col_write_style_onehot_melee_ranged_magic(AttackStyle style, float* obs, int* idx) {
    int i = *idx;
    obs[i++] = (style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;
    obs[i++] = (style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
    obs[i++] = (style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
    *idx = i;
}

static int col_manticore_orbs_visible(const ColoNPC* npc) {
    const ColoManticoreState* mc = &npc->type_state.manticore;
    if (mc->cycle_step < 0) return 0;
    if (mc->phase != COLO_MANTICORE_PHASE_CHARGING) return 1;
    return npc->attack_timer <=
        COLO_MANTICORE_CHARGE_TICKS - COLO_MANTICORE_ARM_ANIMATION_TICKS;
}

static int col_manticore_next_orb_ticks(const ColoNPC* npc) {
    const ColoManticoreState* mc = &npc->type_state.manticore;
    if (mc->cycle_step > 0) return 1;
    if (mc->phase == COLO_MANTICORE_PHASE_CHARGING)
        return npc->attack_timer + 1;
    return npc->attack_timer == 0 ? 1 : npc->attack_timer;
}

static ColoNpcNextPrayerObs col_visible_manticore_next_prayer(const ColoNPC* npc) {
    if (npc->type != COLO_MANTICORE) return col_npc_next_prayer_none();
    if (npc->stun_timer > 0 || npc->frozen_ticks > 0) return col_npc_next_prayer_none();
    const ColoManticoreState* mc = &npc->type_state.manticore;
    if (!col_manticore_orbs_visible(npc)) return col_npc_next_prayer_none();
    int orb = mc->cycle_step >= 0 ? mc->cycle_step : 0;
    if (orb < 0 || orb >= 3) return col_npc_next_prayer_none();
    AttackStyle style = mc->orb_style[orb];
    if (!col_prayer_attack_style(style)) return col_npc_next_prayer_none();
    return (ColoNpcNextPrayerObs){
        .active = 1,
        .style = style,
        .ticks = col_manticore_next_orb_ticks(npc),
    };
}

static int col_npc_current_attack_geometry(
    const ColosseumState* s,
    const ColosseumContext* ctx,
    const ColoNPC* npc,
    const ColoNpcStats* ns
) {
    if (ns->attack_range < 1) return 0;
    if (npc->type == COLO_SOL_HEREDIT) return 0;
    if (col_type_is_warbander(npc->type)) return col_warband_melee_adjacent(s, npc);
    if (npc->type == COLO_MINOTAUR) return col_npc_dist_to_player(s, npc) == 1;
    if (ns->attack_range > 1) {
        int dist = col_npc_dist_to_player(s, npc);
        if (dist < 1 || dist > ns->attack_range) return 0;
        return col_npc_has_los_to_player(s, ctx, npc);
    }
    return col_npc_melee_adjacent_to_player(s, npc);
}

static int col_warband_next_window_wait(const ColosseumState* s, const ColoNPC* npc) {
    int target = col_warband_window_offset(npc->type);
    for (int ticks = 1; ticks <= COLO_WARBAND_CYCLE_TICKS; ticks++) {
        int phase = (s->tick + ticks - s->warband_cycle_anchor) % COLO_WARBAND_CYCLE_TICKS;
        if (phase < 0) phase += COLO_WARBAND_CYCLE_TICKS;
        if (phase == target) return ticks;
    }
    assert(0 && "warband window must recur within one cycle");
    return 0;
}

static ColoNpcNextPrayerObs col_deterministic_npc_next_prayer(
    const ColosseumState* s,
    const ColosseumContext* ctx,
    const ColoNPC* npc,
    const ColoNpcStats* ns
) {
    if (!col_prayer_attack_style(ns->default_style)) return col_npc_next_prayer_none();
    if (npc->stun_timer > 0 || npc->frozen_ticks > 0) return col_npc_next_prayer_none();
    if (npc->type == COLO_MANTICORE) return col_npc_next_prayer_none();
    if (npc->type == COLO_JAVELIN_COLOSSUS &&
            (npc->type_state.javelin.attack_count + 1) % 5 == 0)
        return col_npc_next_prayer_none();
    if (!col_npc_current_attack_geometry(s, ctx, npc, ns))
        return col_npc_next_prayer_none();
    int ticks = npc->attack_timer == 0 ? 1 : npc->attack_timer;
    if (col_type_is_warbander(npc->type)) {
        if (s->warband_cycle_anchor < 0) return col_npc_next_prayer_none();
        ticks = col_warband_next_window_wait(s, npc);
    }
    return (ColoNpcNextPrayerObs){
        .active = 1,
        .style = ns->default_style,
        .ticks = ticks,
    };
}

static ColoNpcNextPrayerObs col_npc_next_prayer_obs(
    const ColosseumState* s,
    const ColosseumContext* ctx,
    const ColoNPC* npc,
    int npc_slot
) {
    (void)npc_slot;
    ColoNpcNextPrayerObs manticore = col_visible_manticore_next_prayer(npc);
    if (manticore.active) return manticore;
    return col_deterministic_npc_next_prayer(
        s, ctx, npc, &COLO_NPC_STATS[npc->type]);
}

static const ColoBeeSwarm* col_obs_bee_state_for_npc_slot(
    const ColosseumState* s,
    int npc_slot
) {
    if (npc_slot < 0 || npc_slot >= COLO_MAX_NPCS) return NULL;
    for (int b = 0; b < COLO_MAX_BEE_SWARMS; b++) {
        const ColoBeeSwarm* bee = &s->bees[b];
        if (bee->phase == COLO_HAZARD_ALIVE && bee->npc_slot == npc_slot) return bee;
    }
    return NULL;
}

static void col_write_npc_tells(const ColosseumState* s, const ColoNPC* npc, float* obs, int* idx) {
    int i = *idx;
    float tells[10] = {0};
    int npc_slot = (int)(npc - s->npcs);
    switch (npc->type) {
        case COLO_MANTICORE: {
            const ColoManticoreState* mc = &npc->type_state.manticore;
            int barrage_started = mc->cycle_step > 0;
            int visible = col_manticore_orbs_visible(npc);
            if (visible) {
                int start = mc->cycle_step;
                int speed = COLO_NPC_STATS[npc->type].attack_speed;

                tells[0] = barrage_started ? 0.0f
                    : col_clamp01((float)npc->attack_timer / (float)(speed > 0 ? speed : 1));
                for (int slot = 0; slot < 3; slot++) {
                    int orb = start + slot;
                    if (orb < 3) col_tell_style_onehot(mc->orb_style[orb], &tells[1 + slot * 3]);
                }
            }
            break;
        }
        case COLO_JAVELIN_COLOSSUS: {
            const ColoJavelinState* jv = &npc->type_state.javelin;
            tells[0] = jv->skyfall_pending ? 1.0f : 0.0f;
            tells[1] = jv->skyfall_pending
                ? col_clamp01((float)jv->skyfall_timer / (float)COLO_JAVELIN_SKYFALL_DELAY)
                : 0.0f;
            tells[2] = jv->skyfall_pending
                ? col_obs_rel_x(jv->skyfall_tile_x, s->player.x)
                : 0.0f;
            tells[3] = jv->skyfall_pending
                ? col_obs_rel_y(jv->skyfall_tile_y, s->player.y)
                : 0.0f;
            break;
        }
        case COLO_BEE_SWARM: {
            const ColoBeeSwarm* bee = col_obs_bee_state_for_npc_slot(s, npc_slot);
            int next_x, next_y;
            int size = col_npc_effective_size(npc);
            col_bee_swarm_next_tile_toward(
                s->player.x,
                s->player.y,
                npc->x,
                npc->y,
                size,
                &next_x,
                &next_y);
            tells[0] = bee
                ? col_clamp01((float)bee->move_timer / (float)COLO_BEE_MOVE_INTERVAL)
                : 0.0f;
            tells[1] = col_bee_swarm_contacts_player(s, npc) ? 1.0f : 0.0f;
            tells[2] = col_obs_rel_x(next_x, s->player.x);
            tells[3] = col_obs_rel_y(next_y, s->player.y);
            tells[4] = encounter_entity_footprints_overlap(
                next_x, next_y, size, s->player.x, s->player.y, 1) ? 1.0f : 0.0f;
            break;
        }
        case COLO_MINOTAUR: {
            int size = col_npc_effective_size(npc);
            int dist = encounter_dist_to_npc(s->player.x, s->player.y, npc->x, npc->y, size);
            tells[0] = (dist > 1) ? 1.0f : 0.0f;
            break;
        }
        case COLO_HEALING_TOTEM: {
            const ColoTotem* tot = col_totem_for_npc_slot(s, npc_slot);
            if (tot) {
                const ColoNPC* owner = &s->npcs[tot->owner_slot];
                tells[0] = tot->projectile_timer > 0
                    ? 0.0f
                    : col_clamp01(
                        (float)tot->heal_timer / (float)COLO_TOTEM_HEAL_INTERVAL);
                tells[1] = col_clamp01(
                    (float)tot->projectile_timer / (float)COLO_TOTEM_PROJECTILE_TICKS);
                tells[2] = owner->max_hp > 0
                    ? col_clamp01(
                        (float)col_totem_heal_amount(owner) / (float)owner->max_hp)
                    : 0.0f;
                tells[3] = col_obs_rel_x(owner->x, s->player.x);
                tells[4] = col_obs_rel_y(owner->y, s->player.y);
            }
            break;
        }
        case COLO_JAGUAR_WARRIOR:
            tells[0] = 1.0f;
            break;
        default:
            break;
    }
    for (int t = 0; t < COLO_NPC_TELLS_FEATURES; t++) obs[i++] = tells[t];
    *idx = i;
}

static void col_write_npc_slot(
    const ColosseumState* s,
    const ColosseumContext* ctx,
    const ColoNPC* npc,
    int npc_slot,
    const ColoVenatorPreviewTargets* venator_targets,
    float* obs,
    int* idx
) {
    int i = *idx;
    int slot_base = i;
    int px = s->player.x, py = s->player.y;
    const ColoNpcStats* ns = &COLO_NPC_STATS[npc->type];
    int size = col_npc_effective_size(npc);

    obs[i++] = (float)(npc->type + 1);
    obs[i++] = col_obs_rel_x(npc->x, px);
    obs[i++] = col_obs_rel_y(npc->y, py);
    obs[i++] = npc->max_hp > 0 ? col_clamp01((float)npc->hp / (float)npc->max_hp) : 0.0f;
    obs[i++] = npc->type == COLO_SOL_HEREDIT
        ? 0.0f : col_clamp01((float)npc->attack_timer / 8.0f);
    int dist = encounter_dist_to_npc(px, py, npc->x, npc->y, size);
    obs[i++] = (ns->attack_range > 0 && dist >= 1 && dist <= ns->attack_range)
        ? 1.0f : 0.0f;

    ColoNpcNextPrayerObs next_prayer =
        col_npc_next_prayer_obs(s, ctx, npc, npc_slot);
    obs[i++] = next_prayer.active ? 1.0f : 0.0f;
    *idx = i;
    col_write_style_onehot_melee_ranged_magic(
        next_prayer.active ? next_prayer.style : ATTACK_STYLE_NONE, obs, idx);
    i = *idx;
    obs[i++] = next_prayer.active ? col_clamp01((float)next_prayer.ticks / 10.0f) : 0.0f;
    obs[i++] = col_npc_has_los_to_player(s, ctx, npc) ? 1.0f : 0.0f;
    assert(i - slot_base == COLO_NPC_TELLS_OFFSET);
    *idx = i;
    col_write_npc_tells(s, npc, obs, idx);
    i = *idx;
    int extra_bounces = col_venator_extra_bounce_if_shot(s, venator_targets, npc_slot);
    obs[i++] = (float)extra_bounces / (float)(OSRS_VENATOR_MAX_CHAIN_HITS - 1);
    assert(i - slot_base == COLO_FEATURES_PER_NPC);
    *idx = i;
}

static int col_sol_premove_aoe_hit(const ColosseumState* s) {
    return s->sol.aoe_attack != COLO_SOL_AOE_NONE &&
        s->sol.aoe_age + 1 == COLO_SOL_AOE_DAMAGE_AGE &&
        col_sol_aoe_tile_is_hazard(&s->sol, s->player.x, s->player.y);
}

static int col_sol_premove_laser_hit(const ColosseumState* s) {
    for (int k = 0; k < s->sol.crystal_count; k++) {
        const ColoSolCrystal* crystal = &s->sol.crystals[k];
        if (crystal->active &&
                crystal->firing_freeze == COLO_SOL_LASER_DAMAGE_FREEZE &&
                col_sol_crystal_hits_tile(crystal, s->player.x, s->player.y))
            return 1;
    }
    return 0;
}

static int col_sol_premove_molten_hit(const ColosseumState* s) {
    if (col_sol_molten_sand_at(&s->sol, s->player.x, s->player.y)) return 1;
    for (int k = 0; k < COLO_SOL_BEAM_MAX; k++) {
        const ColoSolBeam* beam = &s->sol.beams[k];
        if (beam->active && beam->ticks_to_pool <= 1 &&
                beam->x == s->player.x && beam->y == s->player.y)
            return 1;
    }
    return 0;
}

static int col_premove_molten_burn(
    const ColosseumState* s,
    const ColosseumContext* ctx
) {
    int imminent_skyfall = 0;
    if (col_mod_tier(s, COLO_MOD_REENTRY) > 0) {
        for (int n = 0; n < COLO_MAX_NPCS; n++) {
            const ColoNPC* npc = &s->npcs[n];
            if (!col_npc_is_live_target(npc) ||
                    npc->type != COLO_JAVELIN_COLOSSUS) {
                continue;
            }
            const ColoJavelinState* jv = &npc->type_state.javelin;
            if (jv->skyfall_pending && jv->skyfall_timer <= 1) {
                imminent_skyfall = 1;
                break;
            }
        }
    }

    if (!imminent_skyfall) {
        for (int pool = 0; pool < s->molten_count; pool++)
            if (s->molten_x[pool] == s->player.x &&
                    s->molten_y[pool] == s->player.y &&
                    s->molten_burn_timer[pool] == 0)
                return 1;
        return 0;
    }

    ColosseumState forecast;
    forecast.modifiers = s->modifiers;
    forecast.rng_state = s->rng_state;
    forecast.molten_count = s->molten_count;
    size_t molten_bytes = (size_t)s->molten_count * sizeof(s->molten_x[0]);
    memcpy(forecast.molten_x, s->molten_x, molten_bytes);
    memcpy(forecast.molten_y, s->molten_y, molten_bytes);
    memcpy(forecast.molten_kind, s->molten_kind, molten_bytes);
    memcpy(forecast.molten_lifetime, s->molten_lifetime, molten_bytes);
    memcpy(forecast.molten_burn_timer, s->molten_burn_timer, molten_bytes);

    for (int n = 0; n < COLO_MAX_NPCS; n++) {
        const ColoNPC* npc = &s->npcs[n];
        if (!col_npc_is_live_target(npc) ||
                npc->type != COLO_JAVELIN_COLOSSUS) {
            continue;
        }
        const ColoJavelinState* jv = &npc->type_state.javelin;
        if (!jv->skyfall_pending || jv->skyfall_timer > 1) continue;
        col_mod_reentry_on_skyfall(
            &forecast, ctx->route_topology,
            jv->skyfall_tile_x, jv->skyfall_tile_y);
    }

    for (int pool = 0; pool < forecast.molten_count; pool++)
        if (forecast.molten_x[pool] == s->player.x &&
                forecast.molten_y[pool] == s->player.y &&
                forecast.molten_burn_timer[pool] == 0)
            return 1;
    return 0;
}

typedef struct {
    ColoSolParryCue cue;
    int ticks_to_land;
    int damage;
} ColoSolParryTell;

static ColoSolParryTell col_sol_parry_tell(const ColosseumState* s) {
    const SolHereditState* sol = &s->sol;
    int hit = -1;
    for (int h = 0; h < COLO_SOL_PARRY_HITS; h++) {
        if (sol->parry_land_in[h] <= 0) continue;
        if (hit < 0 || sol->parry_land_in[h] < sol->parry_land_in[hit]) hit = h;
    }
    if (hit < 0) {
        return (ColoSolParryTell){
            .cue = COLO_SOL_PARRY_CUE_INACTIVE,
            .ticks_to_land = 0,
            .damage = 0,
        };
    }

    int ticks_to_land = sol->parry_land_in[hit];
    int landing_tick = s->tick + ticks_to_land;
    int lookback = COLO_SOL_PARRY_LOOKBACK[hit];
    int poisoned = 0;
    for (int k = 1; k <= lookback; k++) {
        int history_tick = landing_tick - k;
        if (history_tick < 0 || history_tick > s->tick) continue;
        if (sol->overhead_history[history_tick % COLO_SOL_PRAYER_HISTORY]) {
            poisoned = 1;
            break;
        }
    }

    ColoSolParryCue cue = COLO_SOL_PARRY_CUE_LEAD_TIME;
    if (poisoned)
        cue = COLO_SOL_PARRY_CUE_POISONED;
    else if (ticks_to_land == 1)
        cue = COLO_SOL_PARRY_CUE_FLICK_MELEE;
    else if (ticks_to_land <= lookback + 1)
        cue = COLO_SOL_PARRY_CUE_OVERHEAD_OFF;

    return (ColoSolParryTell){
        .cue = cue,
        .ticks_to_land = ticks_to_land,
        .damage = sol->parry_damage[hit],
    };
}

static void col_write_boss_obs(
    const ColosseumState* s,
    float* obs,
    int* idx,
    int laser_obs_mode
) {
    int i = *idx;
    const SolHereditState* sol = &s->sol;
    int px = s->player.x, py = s->player.y;

    obs[i++] = (float)sol->phase / (float)(COLO_SOL_NUM_PHASES - 1);
    for (int p = 0; p < COLO_SOL_NUM_PHASES; p++)
        obs[i++] = (sol->phase == p) ? 1.0f : 0.0f;

    obs[i++] = (sol->aoe_attack == COLO_SOL_AOE_NONE) ? 1.0f : 0.0f;
    obs[i++] = (sol->aoe_attack == COLO_SOL_AOE_SPEAR1) ? 1.0f : 0.0f;
    obs[i++] = (sol->aoe_attack == COLO_SOL_AOE_SPEAR2) ? 1.0f : 0.0f;
    obs[i++] = (sol->aoe_attack == COLO_SOL_AOE_SHIELD1) ? 1.0f : 0.0f;
    obs[i++] = (sol->aoe_attack == COLO_SOL_AOE_SHIELD2) ? 1.0f : 0.0f;
    obs[i++] = col_clamp01((float)sol->aoe_age / (float)COLO_SOL_AOE_LIFETIME);
    if (sol->boss_idx >= 0) {
        const ColoNPC* boss = &s->npcs[sol->boss_idx];
        int bsz = boss->size > 0 ? boss->size : 5;
        obs[i++] = col_obs_rel_x(px, boss->x + bsz / 2);
        obs[i++] = col_obs_rel_y(py, boss->y + bsz / 2);
    } else {
        obs[i++] = 0.0f; obs[i++] = 0.0f;
    }

    obs[i++] = col_clamp01((float)sol->attack_delay / 12.0f);
    obs[i++] = col_clamp01((float)sol->immobile_ticks / 10.0f);
    obs[i++] = sol->special_cooldown <= 0 ? 1.0f : 0.0f;
    obs[i++] = sol->force_spear ? 1.0f : 0.0f;

    ColoSolParryTell parry = col_sol_parry_tell(s);
    assert(parry.ticks_to_land >= 0 &&
        parry.ticks_to_land <= COLO_SOL_PARRY_LAND_TICKS_MAX);
    for (int cue = 0; cue < COLO_SOL_PARRY_CUE_COUNT; cue++)
        obs[i++] = (int)parry.cue == cue ? 1.0f : 0.0f;
    for (int tick = 1; tick <= COLO_SOL_PARRY_LAND_TICKS_MAX; tick++)
        obs[i++] = parry.ticks_to_land == tick ? 1.0f : 0.0f;
    obs[i++] = col_clamp01((float)parry.damage / 45.0f);

    obs[i++] = sol->grapple_active ? 1.0f : 0.0f;
    for (int g = 0; g < COLO_NUM_GRAPPLE_SLOTS; g++)
        obs[i++] = (sol->grapple_active && sol->grapple_body_slot == g) ? 1.0f : 0.0f;
    obs[i++] = col_clamp01((float)sol->grapple_timer / (float)COLO_SOL_GRAPPLE_WINDOW);
    obs[i++] = sol->next_attack_guaranteed_max ? 1.0f : 0.0f;
    if (col_sol_clamp_active(s)) {
        obs[i++] = col_sol_premove_aoe_hit(s) ? 1.0f : 0.0f;
        obs[i++] = col_sol_premove_molten_hit(s) ? 1.0f : 0.0f;
        if (laser_obs_mode) {
            obs[i++] = col_clamp01((float)sol->laser_cooldown / (float)(
                COLO_SOL_CRYSTAL_COOLDOWN_MIN + COLO_SOL_CRYSTAL_COOLDOWN_RAND - 1));
            for (int crystal = 0; crystal < COLO_SOL_MAX_CRYSTALS; crystal++) {
                if (crystal < sol->crystal_count && sol->crystals[crystal].active) {
                    const ColoSolCrystal* c = &sol->crystals[crystal];
                    obs[i++] = 1.0f;
                    if (c->edge == COLO_SOL_EDGE_NORTH ||
                            c->edge == COLO_SOL_EDGE_SOUTH)
                        obs[i++] = (float)(c->x - px) / (float)COLO_ARENA_WIDTH;
                    else
                        obs[i++] = (float)(c->y - py) / (float)COLO_ARENA_HEIGHT;
                    obs[i++] = c->firing_freeze > 0
                        ? col_clamp01(
                            (float)c->firing_freeze / (float)COLO_SOL_LASER_FREEZE)
                        : 0.0f;
                } else {
                    obs[i++] = 0.0f;
                    obs[i++] = 0.0f;
                    obs[i++] = 0.0f;
                }
            }
        } else {
            memset(&obs[i], 0,
                COLO_SOL_LASER_OBS_SIZE * sizeof(*obs));
            i += COLO_SOL_LASER_OBS_SIZE;
        }
    } else {
        memset(&obs[i], 0,
            COLO_BOSS_OBS_PREMOVE_TAIL_SIZE * sizeof(*obs));
        i += COLO_BOSS_OBS_PREMOVE_TAIL_SIZE;
    }
    *idx = i;
}

static inline int col_pending_hit_ignores_prayer(const EncounterPendingHit* ph) {
    return (!ph->check_prayer && ph->attack_style != ATTACK_STYLE_NONE) ? 1 : 0;
}

static int col_collect_obs_pending_hits(const ColosseumState* s, int* out) {
    int n = 0;
    for (int h = 0; h < s->player_pending_hits.count && h < ENCOUNTER_MAX_PENDING_HITS; h++) {
        if (!s->player_pending_hits.hits[h].active) continue;
        out[n++] = h;
    }
    for (int a = 1; a < n; a++) {
        int key = out[a];
        int kt = s->player_pending_hits.hits[key].ticks_remaining;
        int b = a - 1;
        while (b >= 0 && s->player_pending_hits.hits[out[b]].ticks_remaining > kt) {
            out[b + 1] = out[b];
            b--;
        }
        out[b + 1] = key;
    }
    return n < COLO_OBS_PENDING_HITS ? n : COLO_OBS_PENDING_HITS;
}

#define COLO_MODIFIER_MOLTEN_OBS_COUNT 4

static int col_pending_hit_will_damage_pre_move(
    const EncounterPendingHit* hit,
    OverheadPrayer prayer
) {
    if (!hit->active) return 0;
    if (hit->ticks_remaining > 1) return 0;
    if (hit->damage <= 0) return 0;
    if (hit->attack_style == ATTACK_STYLE_NONE) return 0;
    if (hit->check_prayer &&
            encounter_prayer_correct_for_style(prayer, hit->attack_style))
        return 0;
    return 1;
}

static void col_write_top_modifier_molten(const ColosseumState* s, float* obs, int* idx) {
    ColoNearestHazardTile nearest[COLO_MODIFIER_MOLTEN_OBS_COUNT];
    for (int i = 0; i < COLO_MODIFIER_MOLTEN_OBS_COUNT; i++)
        nearest[i] = col_nearest_hazard_tile_none();
    for (int k = 0; k < s->molten_count; k++) {
        int dx = abs(s->molten_x[k] - s->player.x);
        int dy = abs(s->molten_y[k] - s->player.y);
        int distance = dx > dy ? dx : dy;
        for (int slot = 0; slot < COLO_MODIFIER_MOLTEN_OBS_COUNT; slot++) {
            if (distance >= nearest[slot].distance) continue;
            for (int move = COLO_MODIFIER_MOLTEN_OBS_COUNT - 1; move > slot; move--)
                nearest[move] = nearest[move - 1];
            nearest[slot] = (ColoNearestHazardTile){
                .found = 1,
                .index = k,
                .x = s->molten_x[k],
                .y = s->molten_y[k],
                .distance = distance,
            };
            break;
        }
    }
    for (int k = 0; k < COLO_MODIFIER_MOLTEN_OBS_COUNT; k++) {
        col_write_nearest_hazard_xy_value(s, &nearest[k], 1.0f, obs, idx);
    }
}

static void col_write_nearest_modifier_solarflare(
    const ColosseumState* s,
    float* obs,
    int* idx
) {
    ColoNearestHazardTile nearest = col_nearest_hazard_tile_none();
    if (s->solarflare.active) {
        for (int p = 0; p < COLO_NUM_PILLARS; p++) {
            int x, y;
            col_solarflare_tile(s, p, s->solarflare.step, &x, &y);
            col_nearest_hazard_tile_consider(
                &nearest, p, x, y, s->player.x, s->player.y);
        }
    }
    col_write_nearest_hazard_xy_value(s, &nearest, 1.0f, obs, idx);
    int i = *idx;
    if (nearest.found) {
        int tier = col_mod_tier(s, COLO_MOD_SOLARFLARE);
        int next_step = s->solarflare.step + 1;
        if (next_step == COLO_SOLARFLARE_RING_STEPS) next_step = 0;
        int next_x, next_y;
        col_solarflare_tile(s, nearest.index, next_step, &next_x, &next_y);
        int move_interval = tier >= 3
            ? COLO_SOLARFLARE_MOVE_INTERVAL_T3 : COLO_SOLARFLARE_MOVE_INTERVAL;
        int pause_interval = tier >= 3 ? COLO_SOLARFLARE_CORNER_PAUSE_T3 : 1;
        if (tier == 1) pause_interval = COLO_SOLARFLARE_CORNER_PAUSE;
        obs[i++] = col_clamp01(
            (float)s->solarflare.step / (float)COLO_SOLARFLARE_RING_STEPS);
        obs[i++] = col_clamp01((float)s->solarflare.move_timer / (float)move_interval);
        obs[i++] = col_clamp01((float)s->solarflare.pause_timer / (float)pause_interval);
        obs[i++] = col_obs_rel_x(next_x, s->player.x);
        obs[i++] = col_obs_rel_y(next_y, s->player.y);
    } else {
        for (int k = 0; k < 5; k++) obs[i++] = 0.0f;
    }
    *idx = i;
}

typedef struct {
    int found;
    int x;
    int y;
    int size;
    int ticks;
    int distance;
} ColoVolatilityBlastPreview;

static ColoVolatilityBlastPreview col_soonest_queued_volatility_blast(
    const ColosseumState* s
) {
    ColoVolatilityBlastPreview best = {
        .distance = 1 << 30,
    };
    for (int n = 0; n < COLO_MAX_NPCS; n++) {
        const ColoNPC* npc = &s->npcs[n];
        if (!col_npc_is_live_enemy(npc)) continue;
        for (int h = 0; h < npc->pending_hits.count; h++) {
            const EncounterPendingHit* hit = &npc->pending_hits.hits[h];
            if (!hit->active) continue;
            int ticks = hit->ticks_remaining < 0 ? 0 : hit->ticks_remaining;
            int damage = 0;
            for (int j = 0; j < npc->pending_hits.count; j++) {
                const EncounterPendingHit* other = &npc->pending_hits.hits[j];
                if (!other->active) continue;
                int other_ticks =
                    other->ticks_remaining < 0 ? 0 : other->ticks_remaining;
                if (other_ticks <= ticks) damage += other->damage;
            }
            if (damage < npc->hp) continue;
            int size = col_npc_effective_size(npc);
            int center_x = npc->x + size / 2;
            int center_y = npc->y + size / 2;
            int dx = abs(center_x - s->player.x);
            int dy = abs(center_y - s->player.y);
            int distance = dx > dy ? dx : dy;
            if (best.found &&
                    (ticks > best.ticks ||
                     (ticks == best.ticks && distance >= best.distance)))
                continue;
            best = (ColoVolatilityBlastPreview){
                .found = 1,
                .x = npc->x,
                .y = npc->y,
                .size = size,
                .ticks = ticks,
                .distance = distance,
            };
        }
    }
    return best;
}

static void col_write_volatility_blast_previews(
    const ColosseumState* s,
    float* obs,
    int* idx
) {
    int i = *idx;
    int tier = col_mod_tier(s, COLO_MOD_VOLATILITY);
    int reach = col_volatility_reach_for_tier(tier);
    int target = col_player_target_slot(s);
    if (tier > 0 && target >= 0 && target < COLO_MAX_NPCS &&
            col_npc_is_live_enemy(&s->npcs[target])) {
        const ColoNPC* npc = &s->npcs[target];
        int size = col_npc_effective_size(npc);
        int center_x = npc->x + size / 2;
        int center_y = npc->y + size / 2;
        obs[i++] = col_obs_rel_x(center_x, s->player.x);
        obs[i++] = col_obs_rel_y(center_y, s->player.y);
        obs[i++] = col_clamp01((float)reach / 2.0f);
        obs[i++] = col_volatility_player_in_blast(s, npc->x, npc->y, size, tier)
            ? 1.0f : 0.0f;
    } else {
        for (int k = 0; k < 4; k++) obs[i++] = 0.0f;
    }

    ColoVolatilityBlastPreview queued = col_soonest_queued_volatility_blast(s);
    if (tier > 0 && queued.found) {
        int center_x = queued.x + queued.size / 2;
        int center_y = queued.y + queued.size / 2;
        obs[i++] = col_obs_rel_x(center_x, s->player.x);
        obs[i++] = col_obs_rel_y(center_y, s->player.y);
        obs[i++] = col_clamp01((float)queued.ticks / 10.0f);
        obs[i++] = col_volatility_player_in_blast(
            s, queued.x, queued.y, queued.size, tier) ? 1.0f : 0.0f;
    } else {
        for (int k = 0; k < 4; k++) obs[i++] = 0.0f;
    }
    *idx = i;
}

static int col_count_pre_move_unavoidable_damage_instances(
    const ColosseumState* s,
    int premove_molten_burn
) {
    int count = 0;
    for (int n = 0; n < COLO_MAX_NPCS; n++) {
        const ColoNPC* npc = &s->npcs[n];
        if (!col_npc_is_live_target(npc) || npc->type != COLO_JAVELIN_COLOSSUS)
            continue;
        const ColoJavelinState* jv = &npc->type_state.javelin;
        if (jv->skyfall_pending && jv->skyfall_timer <= 1 &&
                jv->skyfall_damage > 0 &&
                s->player.x == jv->skyfall_tile_x &&
                s->player.y == jv->skyfall_tile_y)
            count++;
    }
    for (int b = 0; b < COLO_MAX_BEE_SWARMS; b++) {
        const ColoBeeSwarm* bee = &s->bees[b];
        if (bee->phase != COLO_HAZARD_ALIVE || bee->npc_slot < 0) continue;
        const ColoNPC* npc = &s->npcs[bee->npc_slot];
        if (!npc->active || npc->type != COLO_BEE_SWARM) continue;
        int contact = 0;
        if (bee->move_timer <= 1) {
            int next_x, next_y;
            int size = col_npc_effective_size(npc);
            col_bee_swarm_next_tile_toward(
                s->player.x,
                s->player.y,
                npc->x,
                npc->y,
                size,
                &next_x,
                &next_y);
            contact = encounter_entity_footprints_overlap(
                next_x, next_y, size, s->player.x, s->player.y, 1);
        } else {
            contact = col_bee_swarm_contacts_player(s, npc);
        }
        if (contact) count++;
    }
    if (s->solarflare.active && col_mod_tier(s, COLO_MOD_SOLARFLARE) > 0) {
        int solarflare_contact_step = s->solarflare.step;
        if (s->solarflare.pause_timer <= 0 && s->solarflare.move_timer <= 1) {
            solarflare_contact_step++;
            if (solarflare_contact_step == COLO_SOLARFLARE_RING_STEPS)
                solarflare_contact_step = 0;
        }
        for (int p = 0; p < COLO_NUM_PILLARS; p++) {
            int x, y;
            col_solarflare_tile(s, p, solarflare_contact_step, &x, &y);
            if (x == s->player.x && y == s->player.y) count++;
        }
    }
    count += premove_molten_burn;
    if (s->player_venom > 0 && s->player_venom_timer <= 1) count++;
    if (s->player_poison > 0 && s->player_poison_timer <= 1) count++;
    int volatility_tier = col_mod_tier(s, COLO_MOD_VOLATILITY);
    if (volatility_tier > 0) {
        ColoVolatilityBlastPreview queued = col_soonest_queued_volatility_blast(s);
        if (queued.found && queued.ticks <= 1 &&
                col_volatility_player_in_blast(
                    s, queued.x, queued.y, queued.size, volatility_tier))
            count++;
    }
    if (col_sol_clamp_active(s)) {
        count += col_sol_premove_aoe_hit(s);
        count += col_sol_premove_laser_hit(s);
        count += col_sol_premove_molten_hit(s);
        if (s->sol.grapple_active &&
                !s->sol.grapple_resolved &&
                s->sol.grapple_timer <= 1) {
            count++;
        }
    }
    return count;
}

static int col_count_pre_move_doom_instances(
    const ColosseumState* s,
    int premove_molten_burn
) {
    int count = col_count_pre_move_unavoidable_damage_instances(
        s, premove_molten_burn);
    for (int h = 0; h < s->player_pending_hits.count; h++) {
        const EncounterPendingHit* hit = &s->player_pending_hits.hits[h];
        if (col_pending_hit_will_damage_pre_move(hit, s->player.prayer)) count++;
    }
    for (int hit = 0; hit < COLO_SOL_PARRY_HITS; hit++) {
        if (s->sol.parry_land_in[hit] != 1 ||
                s->sol.parry_damage[hit] <= 0) {
            continue;
        }
        int early =
            col_sol_overhead_on_in_window(s, COLO_SOL_PARRY_LOOKBACK[hit]);
        int blocked = !early && s->player.prayer == PRAYER_PROTECT_MELEE;
        if (!blocked) count++;
    }
    return count;
}

static void col_write_modifier_hazards(
    const ColosseumState* s,
    int premove_molten_burn,
    float* obs,
    int* idx
) {
    int i = *idx;
    int doom_tier = col_mod_tier(s, COLO_MOD_DOOM);
    int doom_cap = doom_tier > 0 ? COLO_DOOM_CAP[doom_tier] : 0;
    int doom_pending = doom_cap > 0
        ? col_count_pre_move_doom_instances(s, premove_molten_burn)
        : 0;
    obs[i++] = doom_tier > 0 ? 1.0f : 0.0f;
    obs[i++] = doom_cap > 0 ? col_clamp01((float)s->doom_stacks / (float)doom_cap) : 0.0f;
    obs[i++] = doom_cap > 0 && s->doom_stacks + 1 >= doom_cap ? 1.0f : 0.0f;
    obs[i++] = doom_cap > 0 ? col_clamp01((float)doom_pending / (float)doom_cap) : 0.0f;

    obs[i++] = col_clamp01((float)s->player_venom / (float)COLO_VENOM_CAP);
    obs[i++] = s->player_venom > 0
        ? col_clamp01((float)s->player_venom_timer / (float)COLO_VENOM_INTERVAL)
        : 0.0f;
    obs[i++] = col_clamp01(
        (float)s->player_poison / (float)COLO_POISON_BEE_CONTACT_SEVERITY);
    obs[i++] = s->player_poison > 0
        ? col_clamp01((float)s->player_poison_timer / (float)COLO_POISON_INTERVAL)
        : 0.0f;

    ColoNearestHazardTile nearest_bee = col_nearest_hazard_tile_none();
    for (int b = 0; b < COLO_MAX_BEE_SWARMS; b++) {
        if (s->bees[b].phase != COLO_HAZARD_ALIVE) continue;
        const ColoNPC* npc = &s->npcs[s->bees[b].npc_slot];
        if (!npc->active || npc->type != COLO_BEE_SWARM) continue;
        col_nearest_hazard_tile_consider(
            &nearest_bee, b, npc->x, npc->y, s->player.x, s->player.y);
    }
    *idx = i;
    col_write_nearest_hazard_xy_value(s, &nearest_bee, 1.0f, obs, idx);
    i = *idx;

    *idx = i;
    col_write_nearest_modifier_solarflare(s, obs, idx);
    col_write_top_modifier_molten(s, obs, idx);
    col_write_volatility_blast_previews(s, obs, idx);
    i = *idx;
    obs[i++] =
        col_count_pre_move_unavoidable_damage_instances(
            s, premove_molten_burn) > 0 ? 1.0f : 0.0f;

    obs[i++] = col_clamp01(
        (float)s->sol_totem_spawn_timer / (float)COLO_TOTEM_SOL_EXTRA_INTERVAL);
    obs[i++] = col_clamp01(
        (float)col_live_totem_count(s) / (float)COLO_TOTEM_OBS_COUNT_NORM);
    *idx = i;
}

static void col_write_threat_los_counts(
    const ColosseumState* s,
    const ColosseumContext* ctx,
    float* obs,
    int* idx
) {
    int i = *idx;
    int magic, ranged, melee;
    col_live_threat_style_counts(
        s, ctx, &magic, &ranged, &melee);
    obs[i++] = col_clamp01((float)magic / 4.0f);
    obs[i++] = col_clamp01((float)ranged / 4.0f);
    obs[i++] = col_clamp01((float)melee / 4.0f);
    *idx = i;
}

static void col_write_thrall_dc(const ColosseumState* s, float* obs, int* idx) {
    int i = *idx;
    obs[i++] = col_clamp01(
        (float)s->thrall_lifetime_left / (float)s->thrall_lifetime_total);
    obs[i++] = col_clamp01(
        (float)s->thrall_aggro_ticks_left / (float)COLO_THRALL_AGGRO_TICKS);
    obs[i++] = s->death_charge_window_left > 0 ? 1.0f : 0.0f;
    obs[i++] = col_clamp01((float)s->death_charge_window_left / (float)COLO_DEATH_CHARGE_WINDOW);
    obs[i++] = col_clamp01((float)s->death_charge_cd / (float)COLO_DEATH_CHARGE_CD);
    *idx = i;
}



static float col_expected_dpt_for_equipment_vs_npc(
    ColosseumState* s,
    const uint8_t equipment[NUM_GEAR_SLOTS],
    const ColoNPC* npc,
    int use_style_matched_prayer
) {
    if (!s || !equipment || !npc) abort();
    if (!col_npc_is_live_target(npc) || col_type_is_hazard_entity(npc->type))
        return 0.0f;

    uint8_t weapon_item = equipment[GEAR_SLOT_WEAPON];
    AttackStyle style = ATTACK_STYLE_MELEE;
    int item_style = get_item_attack_style(weapon_item);
    if (item_style == 2) style = ATTACK_STYLE_RANGED;
    else if (item_style == 3) style = ATTACK_STYLE_MAGIC;
    else if (item_style != 1) return 0.0f;

    FightStyle fight_style = style == ATTACK_STYLE_RANGED ? FIGHT_STYLE_RAPID
        : style == ATTACK_STYLE_MAGIC ? FIGHT_STYLE_ACCURATE
        : FIGHT_STYLE_AGGRESSIVE;
    int current_att_level;
    int current_str_level;
    col_current_style_levels(
        &s->player, style, &current_att_level, &current_str_level);

    OffensivePrayer dpt_prayer = use_style_matched_prayer
        ? encounter_offensive_prayer_for_style(style)
        : s->player.offensive_prayer;
    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        equipment,
        style,
        dpt_prayer,
        current_att_level,
        fight_style,
        style == ATTACK_STYLE_MAGIC
            ? col_weapon_set_spell_base_damage(s, COLO_GEAR_MAGIC) : 0,
        &stats);
    encounter_update_loadout_level(
        &stats,
        dpt_prayer,
        current_att_level,
        current_str_level);

    OsrsEquipmentEffectProfile effects;
    encounter_derive_loadout_effect_profile(equipment, &effects);
    const ColoNpcStats* ns = &COLO_NPC_STATS[npc->type];
    return col_expected_dpt_vs_npc(
        &stats,
        &effects,
        weapon_item,
        style,
        col_item_melee_style(weapon_item),
        npc,
        ns);
}

#define COLO_BEST_GEAR_MAX_CAND 16

typedef struct {
    uint8_t items[NUM_GEAR_SLOTS][COLO_BEST_GEAR_MAX_CAND];
    int count[NUM_GEAR_SLOTS];
} ColoGearCandidates;

static void col_gear_candidates_add(ColoGearCandidates* c, int slot, uint8_t item) {
    if (item == ITEM_NONE) return;
    for (int k = 0; k < c->count[slot]; k++)
        if (c->items[slot][k] == item) return;
    if (c->count[slot] >= COLO_BEST_GEAR_MAX_CAND) {
        fprintf(stderr, "colosseum best-gear candidate overflow at slot %d\n", slot);
        abort();
    }
    c->items[slot][c->count[slot]++] = item;
}

static void col_build_gear_candidates(const ColosseumState* s, ColoGearCandidates* c) {
    memset(c, 0, sizeof(*c));
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
        col_gear_candidates_add(c, slot, s->player.equipped[slot]);
    for (int cell = 0; cell < COLO_INVENTORY_DISPLAY_SLOTS; cell++) {
        const OsrsItemContentMetadata* metadata =
            osrs_inventory_cell_metadata(&s->player.inventory_cells[cell]);
        uint8_t item = metadata->item_idx;
        if (item == ITEM_NONE) continue;
        int gslot = metadata->gear_slot;
        if (gslot < 0) continue;
        col_gear_candidates_add(c, gslot, item);
    }
}

static int col_item_dominates_empty_for_style(
    uint8_t item, AttackStyle style, MeleeStyle melee_style
) {
    const Item* it = get_item(item);
    if (!it) return 1;
    switch (style) {
        case ATTACK_STYLE_MAGIC:
            return it->attack_magic >= 0 && it->magic_damage >= 0;
        case ATTACK_STYLE_RANGED:
            return it->attack_ranged >= 0 && it->ranged_strength >= 0;
        case ATTACK_STYLE_MELEE: {
            int atk = melee_style == MELEE_STYLE_SLASH ? it->attack_slash
                : melee_style == MELEE_STYLE_CRUSH ? it->attack_crush
                : it->attack_stab;
            return atk >= 0 && it->melee_strength >= 0;
        }
        default:
            return 1;
    }
}

#define COLO_LOADOUT_MEMO_SLOTS 256
typedef struct {
    uint8_t key[COLO_LOADOUT_MEMO_SLOTS][NUM_GEAR_SLOTS];
    EncounterLoadoutStats stats[COLO_LOADOUT_MEMO_SLOTS];
    OsrsEquipmentEffectProfile effects[COLO_LOADOUT_MEMO_SLOTS];
    uint8_t used[COLO_LOADOUT_MEMO_SLOTS];
    int hits;
    int misses;
} ColoLoadoutMemo;

static uint32_t col_loadout_hash(const uint8_t equipment[NUM_GEAR_SLOTS]) {
    uint32_t h = 2166136261u;
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        h ^= equipment[slot];
        h *= 16777619u;
    }
    return h;
}

static void col_memo_loadout_stats(
    ColoLoadoutMemo* memo,
    const uint8_t equipment[NUM_GEAR_SLOTS],
    AttackStyle style,
    OffensivePrayer prayer,
    int current_att_level,
    int current_str_level,
    FightStyle fight_style,
    int spell_base_damage,
    const EncounterLoadoutStats** out_stats,
    const OsrsEquipmentEffectProfile** out_effects
) {
    uint32_t slot = col_loadout_hash(equipment) % COLO_LOADOUT_MEMO_SLOTS;
    for (int probe = 0; probe < COLO_LOADOUT_MEMO_SLOTS; probe++) {
        uint32_t idx = (slot + (uint32_t)probe) % COLO_LOADOUT_MEMO_SLOTS;
        if (memo->used[idx] &&
                memcmp(memo->key[idx], equipment, NUM_GEAR_SLOTS) == 0) {
            memo->hits++;
            *out_stats = &memo->stats[idx];
            *out_effects = &memo->effects[idx];
            return;
        }
        if (!memo->used[idx]) {
            memo->misses++;
            memcpy(memo->key[idx], equipment, NUM_GEAR_SLOTS);
            encounter_compute_loadout_stats(
                equipment, style, prayer, current_att_level, fight_style,
                spell_base_damage, &memo->stats[idx]);
            encounter_update_loadout_level(
                &memo->stats[idx], prayer, current_att_level, current_str_level);
            encounter_derive_loadout_effect_profile(equipment, &memo->effects[idx]);
            memo->used[idx] = 1;
            *out_stats = &memo->stats[idx];
            *out_effects = &memo->effects[idx];
            return;
        }
    }

    static OSRS_THREAD_LOCAL EncounterLoadoutStats overflow_stats;
    static OSRS_THREAD_LOCAL OsrsEquipmentEffectProfile overflow_effects;
    encounter_compute_loadout_stats(
        equipment, style, prayer, current_att_level, fight_style,
        spell_base_damage, &overflow_stats);
    encounter_update_loadout_level(
        &overflow_stats, prayer, current_att_level, current_str_level);
    encounter_derive_loadout_effect_profile(equipment, &overflow_effects);
    *out_stats = &overflow_stats;
    *out_effects = &overflow_effects;
}

static void col_build_best_gear_table(
    ColosseumState* s,
    ColoBestGear best[COLO_NUM_WEAPON_SETS][COLO_NUM_NPC_TYPES]
) {
    if (!s || !best) abort();
    for (int style = 0; style < COLO_NUM_WEAPON_SETS; style++)
        for (int type = 0; type < COLO_NUM_NPC_TYPES; type++) {
            best[style][type].dpt = -1.0f;
            memset(best[style][type].setup, ITEM_NONE, NUM_GEAR_SLOTS);
        }

    ColoGearCandidates cand;
    col_build_gear_candidates(s, &cand);

    ColoNPC rep[COLO_NUM_NPC_TYPES];
    for (int type = 0; type < COLO_NUM_NPC_TYPES; type++)
        if (!col_type_is_hazard_entity((ColoNpcType)type))
            rep[type] = col_matchup_representative_npc((ColoNpcType)type);

    ColoLoadoutMemo memo;
    memset(&memo, 0, sizeof(memo));

    static const int combo_slots[] = {
        GEAR_SLOT_HEAD, GEAR_SLOT_CAPE, GEAR_SLOT_NECK, GEAR_SLOT_AMMO,
        GEAR_SLOT_BODY, GEAR_SLOT_LEGS, GEAR_SLOT_HANDS, GEAR_SLOT_FEET,
        GEAR_SLOT_RING,
    };
    const int n_combo_slots = (int)(sizeof(combo_slots) / sizeof(combo_slots[0]));

    for (int wc = 0; wc < cand.count[GEAR_SLOT_WEAPON]; wc++) {
        uint8_t weapon = cand.items[GEAR_SLOT_WEAPON][wc];
        int wstyle = get_item_attack_style(weapon);
        if (wstyle < 1 || wstyle > 3) continue;
        ColoWeaponSet style_set = (ColoWeaponSet)(wstyle - 1);
        AttackStyle style = col_weapon_set_attack_style(style_set);
        FightStyle fight_style = col_weapon_set_fight_style(style_set);
        OffensivePrayer prayer = encounter_offensive_prayer_for_style(style);
        int spell_base = style == ATTACK_STYLE_MAGIC
            ? col_weapon_set_spell_base_damage(s, COLO_GEAR_MAGIC) : 0;
        int current_att_level, current_str_level;
        col_current_style_levels(&s->player, style, &current_att_level, &current_str_level);
        MeleeStyle melee_style = col_item_melee_style(weapon);
        int two_h = item_is_two_handed(weapon);

        uint8_t shield_items[COLO_BEST_GEAR_MAX_CAND + 1];
        int shield_count;
        if (two_h) {
            shield_items[0] = ITEM_NONE;
            shield_count = 1;
        } else {
            shield_count = 0;
            for (int k = 0; k < cand.count[GEAR_SLOT_SHIELD]; k++)
                shield_items[shield_count++] = cand.items[GEAR_SLOT_SHIELD][k];
            if (shield_count == 0) shield_items[shield_count++] = ITEM_NONE;
        }

        uint8_t slot_opt[16][COLO_BEST_GEAR_MAX_CAND + 1];
        int slot_opt_count[16];
        for (int d = 0; d < n_combo_slots; d++) {
            int slot = combo_slots[d];
            int cnt = cand.count[slot];
            int dominated = 0;
            for (int k = 0; k < cnt; k++) {
                slot_opt[d][k] = cand.items[slot][k];
                if (col_item_dominates_empty_for_style(cand.items[slot][k], style, melee_style))
                    dominated = 1;
            }
            slot_opt_count[d] = cnt;
            if (cnt == 0) {
                slot_opt[d][0] = ITEM_NONE;
                slot_opt_count[d] = 1;
            } else if (!dominated) {
                slot_opt[d][slot_opt_count[d]++] = ITEM_NONE;
            }
        }

        int digit_max[16];
        for (int d = 0; d < n_combo_slots; d++)
            digit_max[d] = slot_opt_count[d];
        int shield_digit = n_combo_slots;
        digit_max[shield_digit] = shield_count;
        int n_digits = n_combo_slots + 1;

        int odo[16];
        for (int d = 0; d < n_digits; d++) odo[d] = 0;

        for (;;) {
            uint8_t equipment[NUM_GEAR_SLOTS];
            memset(equipment, ITEM_NONE, NUM_GEAR_SLOTS);
            equipment[GEAR_SLOT_WEAPON] = weapon;
            for (int d = 0; d < n_combo_slots; d++)
                equipment[combo_slots[d]] = slot_opt[d][odo[d]];
            equipment[GEAR_SLOT_SHIELD] = shield_items[odo[shield_digit]];

            const EncounterLoadoutStats* stats;
            const OsrsEquipmentEffectProfile* effects;
            col_memo_loadout_stats(
                &memo, equipment, style, prayer, current_att_level, current_str_level,
                fight_style, spell_base, &stats, &effects);

            for (int type = 0; type < COLO_NUM_NPC_TYPES; type++) {
                if (col_type_is_hazard_entity((ColoNpcType)type)) continue;
                const ColoNpcStats* ns = &COLO_NPC_STATS[type];
                float dpt = col_expected_dpt_vs_npc(
                    stats, effects, weapon, style, melee_style, &rep[type], ns);
                if (dpt > best[style_set][type].dpt) {
                    best[style_set][type].dpt = dpt;
                    memcpy(best[style_set][type].setup, equipment, NUM_GEAR_SLOTS);
                }
            }

            int d = 0;
            for (; d < n_digits; d++) {
                if (++odo[d] < digit_max[d]) break;
                odo[d] = 0;
            }
            if (d == n_digits) break;
        }
    }
}

static uint64_t col_best_gear_cache_signature(const ColosseumState* s) {
    ColoGearCandidates candidates;
    col_build_gear_candidates(s, &candidates);
    uint64_t h = 1469598103934665603ULL;
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        int count = candidates.count[slot];
        h = (h ^ (uint64_t)(uint32_t)count) * 1099511628211ULL;
        uint8_t sorted[COLO_BEST_GEAR_MAX_CAND];
        memcpy(sorted, candidates.items[slot], (size_t)count);
        for (int i = 1; i < count; i++) {
            uint8_t item = sorted[i];
            int j = i;
            while (j > 0 && sorted[j - 1] > item) {
                sorted[j] = sorted[j - 1];
                j--;
            }
            sorted[j] = item;
        }
        for (int i = 0; i < count; i++)
            h = (h ^ (uint64_t)sorted[i]) * 1099511628211ULL;
    }
    const int scalars[5] = {
        s->player.current_attack, s->player.current_strength,
        s->player.current_ranged, s->player.current_magic,
        (int)s->active_loadout_profile,
    };
    for (int i = 0; i < 5; i++)
        h = (h ^ (uint64_t)(uint32_t)scalars[i]) * 1099511628211ULL;
    return h;
}

static const ColoBestGear (*col_get_best_gear_table(ColosseumState* s))[COLO_NUM_NPC_TYPES] {
    if (!s) abort();
#ifdef COLO_PROFILE_ENABLED
    if (COLO_PROFILE_ENABLED())
        COLO_PROFILE_ADD(COLO_PROF_BEST_GEAR_REQUESTS, 1.0);
#endif
    ColoObsMemoCaches* memos = &s->obs_memos;
    uint64_t sig = col_best_gear_cache_signature(s);
    for (int w = 0; w < COLO_BEST_GEAR_CACHE_WAYS; w++) {
        if (memos->best_gear_valid[w] && memos->best_gear_sig[w] == sig) {
#ifdef COLO_PROFILE_ENABLED
            if (COLO_PROFILE_ENABLED())
                COLO_PROFILE_ADD(COLO_PROF_BEST_GEAR_HITS, 1.0);
#endif
            return memos->best_gear[w];
        }
    }
#ifdef COLO_PROFILE_ENABLED
    if (COLO_PROFILE_ENABLED()) {
        COLO_PROFILE_ADD(COLO_PROF_BEST_GEAR_BUILDS, 1.0);
    }
#endif
    int slot = memos->best_gear_next;
    col_build_best_gear_table(s, memos->best_gear[slot]);
    memos->best_gear_sig[slot] = sig;
    memos->best_gear_valid[slot] = 1;
    memos->best_gear_next = (slot + 1) % COLO_BEST_GEAR_CACHE_WAYS;
    return memos->best_gear[slot];
}

static void col_apply_bis_gear_oracle(ColosseumState* s) {
    int target = col_player_target_slot(s);
    if (target < 0 || target >= COLO_MAX_NPCS) return;
    const ColoNPC* npc = &s->npcs[target];
    if (!col_npc_is_live_target(npc) || col_type_is_hazard_entity(npc->type)) return;

    int melee_ok = col_npc_melee_adjacent_to_player(s, npc);
    const uint8_t* const* loadouts = col_loadouts_for_profile(s->active_loadout_profile);
    float best_dpt = -1.0f;
    ColoWeaponSet best_set = s->weapon_set;
    for (int set = 0; set < COLO_NUM_WEAPON_SETS; set++) {
        if (set == COLO_GEAR_MELEE && !melee_ok) continue;
        float dpt = col_expected_dpt_for_equipment_vs_npc(s, loadouts[set], npc, 1);
        if (dpt > best_dpt) { best_dpt = dpt; best_set = (ColoWeaponSet)set; }
    }

    OffensivePrayer best_prayer =
        encounter_offensive_prayer_for_style(col_weapon_set_attack_style(best_set));
    if (s->player.offensive_prayer != best_prayer) {
        s->player.offensive_prayer = best_prayer;
        col_mark_live_loadout_dirty(s);
    }
    col_apply_weapon_set(s, best_set);
}

static void col_write_spawn_obs(ColosseumState* s, float* obs, int* io) {
    int i = *io;
    float until_spawn = (float)s->wave_spawn_delay;
    float until_attack = s->wave_spawn_delay > 0
        ? (float)s->wave_spawn_delay + (float)COLO_WAVE_ATTACK_GATE_TICKS
        : (float)s->wave_attack_delay;
    obs[i++] = col_clamp01(until_spawn / 12.0f);
    obs[i++] = col_clamp01(until_attack / 16.0f);
    for (int a = 0; a < COLO_NUM_SPAWN_ANCHORS; a++)
        obs[i++] = col_spawn_excluded_near_player(s,
            COLO_SPAWN_ANCHORS[a][0], COLO_SPAWN_ANCHORS[a][1],
            COLO_SPAWN_ZONE_SIZE) ? 1.0f : 0.0f;
    *io = i;
}

static int col_weapon_set_is_argmax_dpt_for_npc(
    ColosseumState* s,
    int npc_slot,
    ColoWeaponSet set,
    const ColoBestGear best[COLO_NUM_WEAPON_SETS][COLO_NUM_NPC_TYPES]
) {
    if (npc_slot < 0 || npc_slot >= COLO_MAX_NPCS) return -1;
    const ColoNPC* npc = &s->npcs[npc_slot];
    if (!col_npc_is_live_target(npc) || col_type_is_hazard_entity(npc->type))
        return -1;
    if (set < 0 || set >= COLO_NUM_WEAPON_SETS)
        col_abort_invalid_weapon_set(set);

    float selected = best[set][npc->type].dpt;
    for (int style = 0; style < COLO_NUM_WEAPON_SETS; style++)
        if (best[style][npc->type].dpt > selected) return 0;
    return 1;
}

static int col_current_set_is_argmax_dpt_for_target(ColosseumState* s) {
    if (!s->tick_scratch.player_attacked) return -1;
    const ColoBestGear (*best)[COLO_NUM_NPC_TYPES] = col_get_best_gear_table(s);
    return col_weapon_set_is_argmax_dpt_for_npc(
        s, col_player_target_slot(s), s->weapon_set, best);
}

static int col_attacked_with_argmax_set(ColosseumState* s) {
    if (!s->tick_scratch.player_attacked) return -1;
    const ColoBestGear (*best)[COLO_NUM_NPC_TYPES] = col_get_best_gear_table(s);
    return col_weapon_set_is_argmax_dpt_for_npc(
        s, s->player_attack_npc_idx, s->weapon_set, best);
}


static void col_write_obs_ctx(
    EncounterState* state,
    EncounterContext* context,
    float* obs
) {
    ColosseumState* s = (ColosseumState*)state;
    ColosseumContext* ctx = col_require_context(context);
    encounter_arena_topology_require_finalized(ctx->route_topology);
#ifdef COLO_PROFILE_ENABLED
    int col_prof_enabled = COLO_PROFILE_ENABLED();
    double col_prof_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
#endif
    col_refresh_current_obs_slots(s);
    int premove_molten_burn = col_premove_molten_burn(s, ctx);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_REFRESH_SLOTS);
#endif
    int i = 0;
    const EncounterLoadoutStats* ls = col_live_loadout_stats(s);

    int colo_spec_cost =
        osrs_spec_cost(s->player.equipped[GEAR_SLOT_WEAPON]);
    OsrsSharedObservationInput shared_input = {
        .player = &s->player,
        .interaction = &s->interaction,
        .arena_min_x = COLO_ARENA_MIN_X,
        .arena_max_x = COLO_ARENA_MAX_X,
        .arena_min_y = COLO_ARENA_MIN_Y,
        .arena_max_y = COLO_ARENA_MAX_Y,
        .attack_style = col_equipped_weapon_attack_style(&s->player),
        .attack_range = ls->attack_range,
        .max_hit = ls->max_hit,
        .attack_speed = ls->attack_speed,
        .defence_stab = ls->def_stab,
        .defence_slash = ls->def_slash,
        .defence_crush = ls->def_crush,
        .defence_magic = ls->def_magic,
        .defence_ranged = ls->def_ranged,
        .effective_level = ls->eff_level,
        .attack_bonus = ls->attack_bonus,
        .strength_bonus = ls->strength_bonus,
        .spell_base_damage = ls->spell_base_damage,
        .special_attack_cost = colo_spec_cost,
    };
    i = osrs_write_shared_observations(obs, &shared_input);
    assert(i == COLO_OBS_AFTER_SHARED);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_PREFIX);
#endif
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_INVENTORY);
#endif

    const ColoVenatorPreviewTargets* venator_targets =
        col_get_venator_preview_targets(s, ctx);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_VENATOR);
#endif

    for (int n = 0; n < COLO_OBS_NPCS; n++) {
        int npc_idx = s->current_obs_slots[n];
        if (npc_idx < 0 || npc_idx >= COLO_MAX_NPCS ||
                !s->npcs[npc_idx].active) {
            memset(&obs[i], 0, COLO_FEATURES_PER_NPC * sizeof(*obs));
            i += COLO_FEATURES_PER_NPC;
            continue;
        }
        col_write_npc_slot(
            s, ctx, &s->npcs[npc_idx], npc_idx,
            venator_targets, obs, &i);
    }
    assert(i == COLO_OBS_AFTER_NPCS);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_NPC_SLOTS);
#endif

    for (int m = 0; m < COLO_NUM_REAL_MODIFIERS; m++)
        obs[i++] = col_clamp01((float)s->modifiers.tier[m] / 3.0f);
    obs[i++] = s->modifiers.draft_pending ? 1.0f : 0.0f;
    for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++) {
        int opt = s->modifiers.draft_options[o];
        obs[i++] = (s->modifiers.draft_pending && opt >= 0)
            ? (float)(opt + 1) / (float)COLO_NUM_MODIFIERS : 0.0f;
    }
    col_write_modifier_hazards(
        s, premove_molten_burn, obs, &i);
    assert(i == COLO_OBS_AFTER_MODS);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_MODIFIERS);
#endif

    int npcs_remaining = 0;
    for (int n = 0; n < COLO_MAX_NPCS; n++)
        if (col_npc_is_live_enemy(&s->npcs[n])) npcs_remaining++;
    obs[i++] = col_clamp01((float)(s->wave + 1) / (float)COLO_NUM_WAVES);
    obs[i++] = col_clamp01((float)npcs_remaining / (float)COLO_MAX_NPCS);
    obs[i++] = (s->reinforcement_timer > 0)
        ? col_clamp01((float)s->reinforcement_timer / (float)COLO_REINFORCEMENT_TICKS) : 0.0f;
    obs[i++] = (s->wave_spawn_delay > 0 || s->wave_ready_delay > 0) ? 1.0f : 0.0f;
    assert(i == COLO_OBS_AFTER_WAVE);

    col_write_boss_obs(s, obs, &i, ctx->config.laser_obs_mode);
    assert(i == COLO_OBS_AFTER_BOSS);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_BOSS);
#endif

    int order[ENCOUNTER_MAX_PENDING_HITS];
    int nhits = col_collect_obs_pending_hits(s, order);
    for (int h = 0; h < COLO_OBS_PENDING_HITS; h++) {
        if (h < nhits) {
            const EncounterPendingHit* ph = &s->player_pending_hits.hits[order[h]];
            obs[i++] = 1.0f;
            obs[i++] = (ph->attack_style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
            obs[i++] = (ph->attack_style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
            obs[i++] = (ph->attack_style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;
            obs[i++] = col_clamp01((float)ph->ticks_remaining / 10.0f);
            obs[i++] = col_clamp01((float)ph->damage / 75.0f);
            obs[i++] = col_pending_hit_ignores_prayer(ph) ? 1.0f : 0.0f;
        } else {
            memset(&obs[i], 0, COLO_FEATURES_PER_HIT * sizeof(*obs));
            i += COLO_FEATURES_PER_HIT;
        }
    }
    assert(i == COLO_OBS_AFTER_PENDING_HITS);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_PENDING_HITS);
#endif

    col_write_threat_los_counts(s, ctx, obs, &i);
    assert(i == COLO_OBS_AFTER_THREAT_LOS);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_THREAT_LOS);
#endif

    col_write_thrall_dc(s, obs, &i);
    assert(i == COLO_OBS_AFTER_THRALL_DC);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_THRALL_DC);
#endif

    col_write_spawn_obs(s, obs, &i);
    assert(i == COLO_OBS_AFTER_SPAWN);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_SPAWN);
#endif

    if (i != COLO_NUM_OBS) {
        fprintf(stderr, "colosseum obs index mismatch: wrote %d, expected %d\n", i, COLO_NUM_OBS);
        abort();
    }
}
