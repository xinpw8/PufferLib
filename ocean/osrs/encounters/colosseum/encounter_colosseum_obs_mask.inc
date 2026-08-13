#line 76 "encounter_colosseum.h"

#define COLO_PLAYER_OBS_SIZE      36
#define COLO_PILLAR_OBS_SIZE      (COLO_NUM_PILLARS * 3)
#define COLO_INVENTORY_CELL_OBS_FEATURES OSRS_INVENTORY_CELL_OBS_FEATURES
#define COLO_INVENTORY_OBS_SIZE \
    (COLO_INVENTORY_DISPLAY_SLOTS * COLO_INVENTORY_CELL_OBS_FEATURES)
#define COLO_EQUIPPED_SELF_OBS_SIZE \
    (NUM_GEAR_SLOTS * OSRS_EQUIPPED_SELF_OBS_FEATURES)
static_assert(COLO_INVENTORY_OBS_SIZE + COLO_EQUIPPED_SELF_OBS_SIZE ==
    COLO_INVENTORY_OBS_CACHE_FLOATS,
    "inventory obs cache float count must match the inventory+equipped block size");
#define COLO_FEATURES_PER_NPC     37
#define COLO_NPC_OBS_SIZE         (COLO_OBS_NPCS * COLO_FEATURES_PER_NPC)

#define COLO_EXPECTED_DPT_NORM    15.0f

#define COLO_MODIFIER_FLAGS_OBS_SIZE  (2 * COLO_NUM_MODIFIERS + 1 + COLO_MODIFIER_DRAFT_OPTIONS)
#define COLO_MODIFIER_HAZARD_OBS_SIZE 38
#define COLO_MODIFIER_OBS_SIZE    (COLO_MODIFIER_FLAGS_OBS_SIZE + COLO_MODIFIER_HAZARD_OBS_SIZE)
#define COLO_WAVE_OBS_SIZE        4
#define COLO_BOSS_OBS_SIZE        51

#define COLO_OBS_PENDING_HITS     16
#define COLO_FEATURES_PER_HIT     7
#define COLO_PENDING_HIT_OBS_SIZE (COLO_FEATURES_PER_HIT * COLO_OBS_PENDING_HITS)

#define COLO_THREAT_LOS_OBS_SIZE  3

#define COLO_THRALL_DC_OBS_SIZE   6

#define COLO_CELL_WEAPON_DPT_OBS_SIZE OSRS_INVENTORY_SIZE
#define COLO_CELL_SPEC_OBS_SIZE       OSRS_INVENTORY_SIZE
#define COLO_WIELDED_DPT_OBS_SIZE     2
#define COLO_WEAPON_CHOICE_OBS_SIZE (COLO_CELL_WEAPON_DPT_OBS_SIZE + \
    COLO_CELL_SPEC_OBS_SIZE + COLO_WIELDED_DPT_OBS_SIZE)
static_assert(COLO_WEAPON_CHOICE_OBS_SIZE == COLO_WEAPON_CHOICE_OBS_CACHE_FLOATS,
    "weapon-choice obs cache float count must match the block size");

#define COLO_SPAWN_PHASE_OBS_SIZE 2
#define COLO_ANCHOR_BLOCKED_OBS_SIZE COLO_NUM_SPAWN_ANCHORS
#define COLO_SPAWN_OBS_SIZE (COLO_SPAWN_PHASE_OBS_SIZE + COLO_ANCHOR_BLOCKED_OBS_SIZE)

#define COLO_THREAT_FIELD_RADIUS 8
#define COLO_THREAT_FIELD_DIM (2 * COLO_THREAT_FIELD_RADIUS + 1)
#define COLO_THREAT_FIELD_TILES (COLO_THREAT_FIELD_DIM * COLO_THREAT_FIELD_DIM)
#define COLO_THREAT_FIELD_OBS_SIZE (2 * COLO_THREAT_FIELD_TILES)
static_assert(COLO_THREAT_FIELD_OBS_SIZE == COLO_THREAT_FIELD_OBS_CACHE_FLOATS,
    "threat-field obs cache float count must match the block size");

#define COLO_NUM_OBS (COLO_PLAYER_OBS_SIZE + COLO_PILLAR_OBS_SIZE + \
    COLO_INVENTORY_OBS_SIZE + COLO_EQUIPPED_SELF_OBS_SIZE + COLO_NPC_OBS_SIZE + \
    COLO_MODIFIER_OBS_SIZE + COLO_WAVE_OBS_SIZE + COLO_BOSS_OBS_SIZE + \
    COLO_PENDING_HIT_OBS_SIZE + COLO_STEP_OUT_FORECAST_OBS_SIZE + \
    COLO_THREAT_LOS_OBS_SIZE + COLO_THRALL_DC_OBS_SIZE + \
    COLO_WEAPON_CHOICE_OBS_SIZE + COLO_SPAWN_OBS_SIZE + \
    COLO_THREAT_FIELD_OBS_SIZE)

#define COLO_OBS_AFTER_PLAYER   COLO_PLAYER_OBS_SIZE
#define COLO_OBS_AFTER_PILLARS  (COLO_OBS_AFTER_PLAYER + COLO_PILLAR_OBS_SIZE)
#define COLO_OBS_AFTER_INVENTORY (COLO_OBS_AFTER_PILLARS + COLO_INVENTORY_OBS_SIZE)
#define COLO_OBS_AFTER_EQUIPPED_SELF \
    (COLO_OBS_AFTER_INVENTORY + COLO_EQUIPPED_SELF_OBS_SIZE)
#define COLO_OBS_AFTER_NPCS     (COLO_OBS_AFTER_EQUIPPED_SELF + COLO_NPC_OBS_SIZE)
#define COLO_OBS_AFTER_MODS     (COLO_OBS_AFTER_NPCS + COLO_MODIFIER_OBS_SIZE)
#define COLO_OBS_AFTER_WAVE     (COLO_OBS_AFTER_MODS + COLO_WAVE_OBS_SIZE)
#define COLO_OBS_AFTER_BOSS     (COLO_OBS_AFTER_WAVE + COLO_BOSS_OBS_SIZE)
#define COLO_OBS_AFTER_PENDING_HITS (COLO_OBS_AFTER_BOSS + COLO_PENDING_HIT_OBS_SIZE)
#define COLO_OBS_AFTER_STEP_OUT_FORECAST \
    (COLO_OBS_AFTER_PENDING_HITS + COLO_STEP_OUT_FORECAST_OBS_SIZE)
#define COLO_OBS_AFTER_THREAT_LOS \
    (COLO_OBS_AFTER_STEP_OUT_FORECAST + COLO_THREAT_LOS_OBS_SIZE)
#define COLO_OBS_AFTER_THRALL_DC \
    (COLO_OBS_AFTER_THREAT_LOS + COLO_THRALL_DC_OBS_SIZE)
#define COLO_OBS_AFTER_WEAPON_CHOICE \
    (COLO_OBS_AFTER_THRALL_DC + COLO_WEAPON_CHOICE_OBS_SIZE)
#define COLO_OBS_AFTER_SPAWN \
    (COLO_OBS_AFTER_WEAPON_CHOICE + COLO_SPAWN_OBS_SIZE)
#define COLO_OBS_AFTER_THREAT_FIELD \
    (COLO_OBS_AFTER_SPAWN + COLO_THREAT_FIELD_OBS_SIZE)

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

typedef struct {
    OsrsVenatorMonster monsters[COLO_MAX_NPCS];
    int count;
} ColoVenatorPreviewTargets;

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

static void col_collect_venator_preview_targets(
    const ColosseumState* s,
    ColoVenatorPreviewTargets* out
) {
    if (!s || !out) abort();
    memset(out, 0, sizeof(*out));
    for (int slot = 0; slot < COLO_MAX_NPCS; slot++) {
        const ColoNPC* npc = &s->npcs[slot];
        if (!col_npc_is_live_enemy(npc)) continue;
        out->monsters[out->count++] = col_venator_monster_from_npc(npc, slot);
    }
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
    OsrsVenatorMonster primary = col_venator_monster_from_npc(npc, primary_slot);
    OsrsVenatorChain chain =
        osrs_venator_resolve_chain(primary, targets->monsters, targets->count);
    int extra = (int)chain.length - 1;
    if (extra < 0 || extra > OSRS_VENATOR_MAX_CHAIN_HITS - 1) abort();
    return extra;
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

static inline void col_write_nearest_hazard_xy_values(
    const ColosseumState* s,
    const ColoNearestHazardTile* nearest,
    float value0,
    float value1,
    float* obs,
    int* idx
) {
    int i = *idx;
    if (nearest->found) {
        obs[i++] = col_obs_rel_x(nearest->x, s->player.x);
        obs[i++] = col_obs_rel_y(nearest->y, s->player.y);
        obs[i++] = value0;
        obs[i++] = value1;
    } else {
        obs[i++] = 0.0f;
        obs[i++] = 0.0f;
        obs[i++] = 0.0f;
        obs[i++] = 0.0f;
    }
    *idx = i;
}

static void col_refresh_current_obs_slots_ctx(ColosseumState* s, const ColosseumContext* ctx) {
    (void)ctx;
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

static int col_pending_hit_prayer_ticks(const EncounterPendingHit* hit) {
    return hit->prayer_check_delay > 0 ? hit->prayer_check_delay : hit->ticks_remaining;
}

static ColoNpcNextPrayerObs col_source_matched_pending_prayer(
    const ColosseumState* s,
    int npc_slot
) {
    ColoNpcNextPrayerObs best = col_npc_next_prayer_none();
    for (int h = 0; h < s->player_pending_hits.count && h < ENCOUNTER_MAX_PENDING_HITS; h++) {
        const EncounterPendingHit* hit = &s->player_pending_hits.hits[h];
        if (!hit->active) continue;
        if (!hit->check_prayer) continue;
        if (hit->source_npc_slot != npc_slot) continue;
        if (!col_prayer_attack_style((AttackStyle)hit->attack_style)) continue;
        int ticks = col_pending_hit_prayer_ticks(hit);
        if (!best.active || ticks < best.ticks) {
            best.active = 1;
            best.style = (AttackStyle)hit->attack_style;
            best.ticks = ticks;
        }
    }
    return best;
}

static int col_manticore_next_orb_ticks(const ColoNPC* npc) {
    const ColoManticoreState* mc = &npc->type_state.manticore;
    if (mc->cycle_step > 0) return 1;
    return npc->attack_timer == 0 ? 1 : npc->attack_timer;
}

static ColoNpcNextPrayerObs col_visible_manticore_next_prayer(const ColoNPC* npc) {
    if (npc->type != COLO_MANTICORE) return col_npc_next_prayer_none();
    if (npc->stun_timer > 0 || npc->frozen_ticks > 0) return col_npc_next_prayer_none();
    const ColoManticoreState* mc = &npc->type_state.manticore;
    int armed = mc->cycle_step >= 0 || mc->orb_style[0] != ATTACK_STYLE_NONE;
    if (!armed) return col_npc_next_prayer_none();
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
        return col_npc_has_los_to_player(s, npc);
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
    const ColoNPC* npc,
    const ColoNpcStats* ns
) {
    if (!col_prayer_attack_style(ns->default_style)) return col_npc_next_prayer_none();
    if (npc->stun_timer > 0 || npc->frozen_ticks > 0) return col_npc_next_prayer_none();
    if (npc->type == COLO_MANTICORE) return col_npc_next_prayer_none();
    if (npc->type == COLO_JAVELIN_COLOSSUS &&
            (npc->type_state.javelin.attack_count + 1) % 5 == 0)
        return col_npc_next_prayer_none();
    if (!col_npc_current_attack_geometry(s, npc, ns)) return col_npc_next_prayer_none();
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
    const ColoNPC* npc,
    int npc_slot
) {
    ColoNpcNextPrayerObs pending = col_source_matched_pending_prayer(s, npc_slot);
    if (pending.active) return pending;
    ColoNpcNextPrayerObs manticore = col_visible_manticore_next_prayer(npc);
    if (manticore.active) return manticore;
    return col_deterministic_npc_next_prayer(s, npc, &COLO_NPC_STATS[npc->type]);
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
            int firing = mc->cycle_step >= 0;
            int armed = firing || mc->orb_style[0] != ATTACK_STYLE_NONE;
            if (armed) {
                int start = firing ? mc->cycle_step : 0;
                int speed = COLO_NPC_STATS[npc->type].attack_speed;

                tells[0] = firing ? 0.0f
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
            col_bee_swarm_next_tile_toward_player(s, npc, &next_x, &next_y);
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
        case COLO_JAGUAR_WARRIOR:
            tells[0] = 1.0f;
            break;
        default:
            break;
    }
    for (int t = 0; t < 10; t++) obs[i++] = tells[t];
    *idx = i;
}

static void col_write_npc_slot(
    const ColosseumState* s,
    const ColoNPC* npc,
    int npc_slot,
    const ColoVenatorPreviewTargets* venator_targets,
    float* obs,
    int* idx
) {
    int i = *idx;
    int px = s->player.x, py = s->player.y;
    const ColoNpcStats* ns = &COLO_NPC_STATS[npc->type];
    int size = col_npc_effective_size(npc);

    for (int t = 0; t < COLO_NUM_NPC_TYPES; t++)
        obs[i++] = (npc->type == t) ? 1.0f : 0.0f;
    obs[i++] = col_obs_rel_x(npc->x, px);
    obs[i++] = col_obs_rel_y(npc->y, py);
    obs[i++] = npc->max_hp > 0 ? col_clamp01((float)npc->hp / (float)npc->max_hp) : 0.0f;
    *idx = i;
    col_write_style_onehot_melee_ranged_magic(npc->attack_style, obs, idx);
    i = *idx;
    obs[i++] = col_clamp01((float)npc->attack_timer / 8.0f);
    int dist = encounter_dist_to_npc(px, py, npc->x, npc->y, size);
    int range = ns->attack_range > 0 ? ns->attack_range : 1;
    obs[i++] = (dist >= 1 && dist <= range) ? 1.0f : 0.0f;

    ColoNpcNextPrayerObs next_prayer = col_npc_next_prayer_obs(s, npc, npc_slot);
    obs[i++] = next_prayer.active ? 1.0f : 0.0f;
    *idx = i;
    col_write_style_onehot_melee_ranged_magic(
        next_prayer.active ? next_prayer.style : ATTACK_STYLE_NONE, obs, idx);
    i = *idx;
    obs[i++] = next_prayer.active ? col_clamp01((float)next_prayer.ticks / 10.0f) : 0.0f;
    obs[i++] = col_npc_has_los_to_player(s, npc) ? 1.0f : 0.0f;
    *idx = i;
    col_write_npc_tells(s, npc, obs, idx);
    i = *idx;
    int extra_bounces = col_venator_extra_bounce_if_shot(s, venator_targets, npc_slot);
    obs[i++] = (float)extra_bounces / (float)(OSRS_VENATOR_MAX_CHAIN_HITS - 1);
    *idx = i;
}

static void col_write_nearest_armed_line(const ColosseumState* s, float* obs, int* idx) {
    ColoNearestHazardTile nearest = col_nearest_hazard_tile_none();
    for (int k = 0; k < s->sol.crystal_count; k++) {
        const ColoSolCrystal* c = &s->sol.crystals[k];
        if (!c->active || c->firing_freeze < COLO_SOL_LASER_DAMAGE_FREEZE) continue;
        if (!col_sol_crystal_in_line(s, c)) continue;
        col_nearest_hazard_tile_consider(
            &nearest, k, c->x, c->y, s->player.x, s->player.y);
    }
    float ticks_frac = nearest.found
        ? col_clamp01((float)(s->sol.crystals[nearest.index].firing_freeze
                              - COLO_SOL_LASER_DAMAGE_FREEZE) /
              (float)(COLO_SOL_LASER_FREEZE - COLO_SOL_LASER_DAMAGE_FREEZE))
        : 0.0f;
    col_write_nearest_hazard_xy_values(s, &nearest, ticks_frac, 1.0f, obs, idx);
}

static void col_write_nearest_molten(const ColosseumState* s, float* obs, int* idx) {
    ColoNearestHazardTile nearest = col_nearest_hazard_tile_none();
    for (int k = 0; k < s->sol.hazard_tile_count; k++) {
        col_nearest_hazard_tile_consider(
            &nearest, k, s->sol.hazard_tile_x[k], s->sol.hazard_tile_y[k],
            s->player.x, s->player.y);
    }
    col_write_nearest_hazard_xy_value(s, &nearest, 1.0f, obs, idx);
}

static int col_player_in_aoe_danger(const ColosseumState* s) {
    const SolHereditState* sol = &s->sol;
    if (sol->aoe_attack == COLO_SOL_AOE_NONE) return 0;
    return col_sol_aoe_tile_is_hazard(sol, s->player.x, s->player.y);
}

static void col_write_nearest_crystal(const ColosseumState* s, float* obs, int* idx) {
    ColoNearestHazardTile nearest = col_nearest_hazard_tile_none();
    for (int k = 0; k < s->sol.crystal_count; k++) {
        const ColoSolCrystal* ck = &s->sol.crystals[k];
        if (!ck->active) continue;
        col_nearest_hazard_tile_consider(
            &nearest, k, ck->x, ck->y, s->player.x, s->player.y);
    }
    const ColoSolCrystal* c = nearest.found ? &s->sol.crystals[nearest.index] : NULL;
    col_write_nearest_hazard_xy_values(
        s,
        &nearest,
        c && c->firing_freeze > 0 ? 1.0f : 0.0f,
        c ? col_clamp01((float)c->firing_freeze / (float)COLO_SOL_LASER_FREEZE) : 0.0f,
        obs,
        idx);
}

static void col_write_nearest_beam(const ColosseumState* s, float* obs, int* idx) {
    ColoNearestHazardTile nearest = col_nearest_hazard_tile_none();
    for (int k = 0; k < COLO_SOL_BEAM_MAX; k++) {
        if (!s->sol.beams[k].active) continue;
        col_nearest_hazard_tile_consider(
            &nearest, k, s->sol.beams[k].x, s->sol.beams[k].y,
            s->player.x, s->player.y);
    }
    float ticks_frac = nearest.found
        ? col_clamp01((float)s->sol.beams[nearest.index].ticks_to_pool /
              (float)COLO_SOL_BEAM_TO_POOL_TICKS)
        : 0.0f;
    col_write_nearest_hazard_xy_values(s, &nearest, ticks_frac, 1.0f, obs, idx);
}

static void col_write_boss_obs(const ColosseumState* s, float* obs, int* idx) {
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
    obs[i++] = col_player_in_aoe_danger(s) ? 1.0f : 0.0f;
    if (sol->boss_idx >= 0) {
        const ColoNPC* boss = &s->npcs[sol->boss_idx];
        int bsz = boss->size > 0 ? boss->size : 5;
        obs[i++] = col_obs_rel_x(px, boss->x + bsz / 2);
        obs[i++] = col_obs_rel_y(py, boss->y + bsz / 2);
    } else {
        obs[i++] = 0.0f; obs[i++] = 0.0f;
    }

    if (sol->aoe_attack == COLO_SOL_AOE_NONE) {
        obs[i++] = 0.0f;
        obs[i++] = 0.0f;
    } else {
        int adx, ady;
        col_sol_aoe_effective_dir(sol, px, py, &adx, &ady);
        obs[i++] = (float)adx;
        obs[i++] = (float)ady;
    }

    obs[i++] = col_clamp01((float)sol->attack_delay / 12.0f);
    obs[i++] = col_clamp01((float)sol->immobile_ticks / 10.0f);
    obs[i++] = sol->special_cooldown <= 0 ? 1.0f : 0.0f;
    obs[i++] = sol->force_spear ? 1.0f : 0.0f;

    int next_land = 0, next_dmg = 0;
    for (int h = 0; h < COLO_SOL_PARRY_HITS; h++) {
        if (sol->parry_land_in[h] <= 0) continue;
        if (next_land == 0 || sol->parry_land_in[h] < next_land) {
            next_land = sol->parry_land_in[h];
            next_dmg = sol->parry_damage[h];
        }
    }
    obs[i++] = sol->parry_hits_left > 0 ? 1.0f : 0.0f;
    obs[i++] = col_clamp01((float)next_land / 10.0f);
    obs[i++] = col_clamp01((float)next_dmg / 45.0f);
    obs[i++] = col_clamp01((float)sol->parry_hits_left / (float)COLO_SOL_PARRY_HITS);
    obs[i++] = sol->phase >= 3 ? 1.0f : 0.0f;

    obs[i++] = sol->grapple_active ? 1.0f : 0.0f;
    for (int g = 0; g < COLO_NUM_GRAPPLE_SLOTS; g++)
        obs[i++] = (sol->grapple_active && sol->grapple_body_slot == g) ? 1.0f : 0.0f;
    obs[i++] = col_clamp01((float)sol->grapple_timer / (float)COLO_SOL_GRAPPLE_WINDOW);
    obs[i++] = sol->next_attack_guaranteed_max ? 1.0f : 0.0f;

    obs[i++] = col_clamp01((float)sol->crystal_count / (float)COLO_SOL_MAX_CRYSTALS);
    *idx = i;
    col_write_nearest_crystal(s, obs, idx);
    col_write_nearest_armed_line(s, obs, idx);

    col_write_nearest_molten(s, obs, idx);

    col_write_nearest_beam(s, obs, idx);
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

typedef struct {
    int found;
    int npc_slot;
    int x;
    int y;
    int size;
    int ticks;
    int distance;
} ColoVolatilityBlastPreview;

static inline ColoVolatilityBlastPreview col_volatility_blast_preview_none(void) {
    return (ColoVolatilityBlastPreview){
        .found = 0,
        .npc_slot = -1,
        .x = 0,
        .y = 0,
        .size = 0,
        .ticks = 0,
        .distance = 1 << 30,
    };
}

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

static ColoVolatilityBlastPreview col_soonest_queued_volatility_blast(
    const ColosseumState* s
) {
    ColoVolatilityBlastPreview best = col_volatility_blast_preview_none();
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
                int other_ticks = other->ticks_remaining < 0 ? 0 : other->ticks_remaining;
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
                .npc_slot = n,
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

static int col_count_pre_move_doom_instances(const ColosseumState* s) {
    int count = 0;
    for (int h = 0; h < s->player_pending_hits.count; h++) {
        const EncounterPendingHit* hit = &s->player_pending_hits.hits[h];
        if (col_pending_hit_will_damage_pre_move(hit, s->player.prayer)) count++;
    }
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
            col_bee_swarm_next_tile_toward_player(s, npc, &next_x, &next_y);
            int size = col_npc_effective_size(npc);
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
    for (int m = 0; m < s->molten_count; m++) {
        if (s->molten_x[m] == s->player.x && s->molten_y[m] == s->player.y) {
            count++;
            break;
        }
    }
    if (s->player_venom > 0 && s->player_venom_timer <= 1) count++;
    int volatility_tier = col_mod_tier(s, COLO_MOD_VOLATILITY);
    if (volatility_tier > 0) {
        ColoVolatilityBlastPreview queued = col_soonest_queued_volatility_blast(s);
        if (queued.found && queued.ticks <= 1 &&
                col_volatility_player_in_blast(
                    s, queued.x, queued.y, queued.size, volatility_tier))
            count++;
    }
    return count;
}

static void col_write_modifier_hazards(const ColosseumState* s, float* obs, int* idx) {
    int i = *idx;
    int doom_tier = col_mod_tier(s, COLO_MOD_DOOM);
    int doom_cap = doom_tier > 0 ? COLO_DOOM_CAP[doom_tier] : 0;
    int doom_pending = doom_cap > 0 ? col_count_pre_move_doom_instances(s) : 0;
    obs[i++] = doom_tier > 0 ? 1.0f : 0.0f;
    obs[i++] = doom_cap > 0 ? col_clamp01((float)s->doom_stacks / (float)doom_cap) : 0.0f;
    obs[i++] = doom_cap > 0 && s->doom_stacks + 1 >= doom_cap ? 1.0f : 0.0f;
    obs[i++] = doom_cap > 0 ? col_clamp01((float)doom_pending / (float)doom_cap) : 0.0f;

    obs[i++] = s->player_venom > 0 ? 1.0f : 0.0f;
    obs[i++] = col_clamp01((float)s->player_venom / (float)COLO_VENOM_CAP);
    obs[i++] = s->player_venom > 0
        ? col_clamp01((float)s->player_venom_timer / (float)COLO_VENOM_INTERVAL)
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
}

static void col_write_threat_los_counts(const ColosseumState* s, float* obs, int* idx) {
    int i = *idx;
    int magic, ranged, melee;
    col_live_threat_style_counts(s, &magic, &ranged, &melee);
    obs[i++] = col_clamp01((float)magic / 4.0f);
    obs[i++] = col_clamp01((float)ranged / 4.0f);
    obs[i++] = col_clamp01((float)melee / 4.0f);
    *idx = i;
}

static uint64_t col_threat_field_obs_signature(const ColosseumState* s) {
    uint64_t h = 1469598103934665603ULL;
    h = (h ^ (uint64_t)(uint32_t)s->player.x) * 1099511628211ULL;
    h = (h ^ (uint64_t)(uint32_t)s->player.y) * 1099511628211ULL;
    for (int n = 0; n < COLO_MAX_NPCS; n++) {
        const ColoNPC* npc = &s->npcs[n];
        if (!col_npc_is_live_enemy(npc)) continue;
        int range = COLO_NPC_STATS[npc->type].attack_range;
        if (range <= 1) continue;
        uint64_t shooter = (uint64_t)(uint32_t)npc->x
            | ((uint64_t)(uint32_t)npc->y << 8)
            | ((uint64_t)(uint32_t)col_npc_effective_size(npc) << 16)
            | ((uint64_t)(uint32_t)range << 24);
        h = (h ^ shooter) * 1099511628211ULL;
    }
    const uint8_t* grid = &s->npc_collision_flags[0][0];
    const size_t grid_bytes = sizeof(s->npc_collision_flags);
    size_t b = 0;
    for (; b + 8 <= grid_bytes; b += 8) {
        uint64_t word;
        memcpy(&word, grid + b, 8);
        h = (h ^ word) * 1099511628211ULL;
    }
    for (; b < grid_bytes; b++)
        h = (h ^ (uint64_t)grid[b]) * 1099511628211ULL;
    return h;
}

static void col_write_threat_field_obs(ColosseumState* s, float* obs, int* idx) {
    int i = *idx;
    if (!COLO_STATIC_LOOKUP_TABLES_BUILT) col_build_static_arena();

    ColoObsMemoCaches* memos = &s->obs_memos;
    uint64_t sig = col_threat_field_obs_signature(s);
    for (int w = 0; w < COLO_THREAT_FIELD_OBS_CACHE_WAYS; w++) {
        if (memos->threat_field_valid[w] && memos->threat_field_sig[w] == sig) {
            memcpy(&obs[i], memos->threat_field[w],
                   sizeof(float) * COLO_THREAT_FIELD_OBS_CACHE_FLOATS);
            *idx = i + COLO_THREAT_FIELD_OBS_CACHE_FLOATS;
            return;
        }
    }
    int block_start = i;

    int shooter_x0[COLO_MAX_NPCS], shooter_y0[COLO_MAX_NPCS];
    int shooter_x1[COLO_MAX_NPCS], shooter_y1[COLO_MAX_NPCS];
    int shooter_ranges[COLO_MAX_NPCS];
    int n_shooters = 0;
    for (int n = 0; n < COLO_MAX_NPCS; n++) {
        const ColoNPC* npc = &s->npcs[n];
        if (!col_npc_is_live_enemy(npc)) continue;
        int range = COLO_NPC_STATS[npc->type].attack_range;
        if (range <= 1) continue;
        int size = col_npc_effective_size(npc);
        shooter_x0[n_shooters] = npc->x;
        shooter_y0[n_shooters] = npc->y;
        shooter_x1[n_shooters] = npc->x + size - 1;
        shooter_y1[n_shooters] = npc->y + size - 1;
        shooter_ranges[n_shooters] = range;
        n_shooters++;
    }
    for (int dy = -COLO_THREAT_FIELD_RADIUS; dy <= COLO_THREAT_FIELD_RADIUS; dy++) {
        for (int dx = -COLO_THREAT_FIELD_RADIUS; dx <= COLO_THREAT_FIELD_RADIUS; dx++) {
            int tx = s->player.x + dx, ty = s->player.y + dy;
            int gx, gy;
            int count = 0;
            if (col_grid_index(tx, ty, &gx, &gy)) {
                for (int k = 0; k < n_shooters; k++) {
                    int sx = tx < shooter_x0[k] ? shooter_x0[k]
                        : (tx > shooter_x1[k] ? shooter_x1[k] : tx);
                    int sy = ty < shooter_y0[k] ? shooter_y0[k]
                        : (ty > shooter_y1[k] ? shooter_y1[k] : ty);
                    int adx = tx - sx; if (adx < 0) adx = -adx;
                    int ady = ty - sy; if (ady < 0) ady = -ady;
                    int dist = adx > ady ? adx : ady;
                    if (dist < 1 || dist > shooter_ranges[k]) continue;
                    if (!col_static_los_get(sx, sy, tx, ty)) continue;
                    count++;
                }
            }
            obs[i++] = col_clamp01((float)count / 4.0f);
        }
    }
    for (int dy = -COLO_THREAT_FIELD_RADIUS; dy <= COLO_THREAT_FIELD_RADIUS; dy++) {
        for (int dx = -COLO_THREAT_FIELD_RADIUS; dx <= COLO_THREAT_FIELD_RADIUS; dx++) {
            int tx = s->player.x + dx, ty = s->player.y + dy;
            int gx, gy;
            int blocked = 1;
            if (col_grid_index(tx, ty, &gx, &gy))
                blocked = col_static_blocked(tx, ty) || s->npc_collision_flags[gx][gy];
            obs[i++] = blocked ? 1.0f : 0.0f;
        }
    }

    int slot = memos->threat_field_next;
    memcpy(memos->threat_field[slot], &obs[block_start],
           sizeof(float) * COLO_THREAT_FIELD_OBS_CACHE_FLOATS);
    memos->threat_field_sig[slot] = sig;
    memos->threat_field_valid[slot] = 1;
    memos->threat_field_next = (slot + 1) % COLO_THREAT_FIELD_OBS_CACHE_WAYS;

    *idx = i;
}

static void col_write_thrall_dc(const ColosseumState* s, float* obs, int* idx) {
    int i = *idx;
    int lifetime_full = COLO_THRALL_LIFETIME_TICKS[s->active_loadout_profile];
    obs[i++] = s->thrall_active ? 1.0f : 0.0f;
    obs[i++] = (s->thrall_active && lifetime_full > 0)
        ? col_clamp01((float)s->thrall_lifetime_left / (float)lifetime_full) : 0.0f;
    obs[i++] = col_clamp01((float)s->thrall_recast_cd / (float)COLO_THRALL_RECAST_CD);
    obs[i++] = s->death_charge_window_left > 0 ? 1.0f : 0.0f;
    obs[i++] = col_clamp01((float)s->death_charge_window_left / (float)COLO_DEATH_CHARGE_WINDOW);
    obs[i++] = col_clamp01((float)s->death_charge_cd / (float)COLO_DEATH_CHARGE_CD);
    *idx = i;
}

static void col_write_pillar_obs(const ColosseumState* s, float* obs, int* idx) {
    int i = *idx;
    for (int p = 0; p < COLO_NUM_PILLARS; p++) {
        obs[i++] = col_obs_rel_x(COLO_PILLARS[p][0], s->player.x);
        obs[i++] = col_obs_rel_y(COLO_PILLARS[p][1], s->player.y);
        obs[i++] = col_clamp01((float)COLO_PILLAR_SIZE / 7.0f);
    }
    *idx = i;
}

static void col_equipment_bonus_projection(
    const uint8_t equipment[NUM_GEAR_SLOTS],
    float out[6]
) {
    int slash = 0;
    int melee_str = 0;
    int ranged_att = 0;
    int ranged_str = 0;
    int magic = 0;
    int defence = 0;
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        uint8_t item_idx = equipment[slot];
        if (item_idx == ITEM_NONE) continue;
        if (item_idx >= NUM_ITEMS) {
            fprintf(stderr, "colosseum obs: invalid equipped item %u\n", item_idx);
            abort();
        }
        const Item* item = &ITEM_DATABASE[item_idx];
        slash += item->attack_slash;
        melee_str += item->melee_strength;
        ranged_att += item->attack_ranged;
        ranged_str += item->ranged_strength;
        magic += item->attack_magic + item->magic_damage;
        defence += item->defence_stab + item->defence_slash + item->defence_crush +
            item->defence_magic + item->defence_ranged;
    }
    out[0] = (float)slash / STAT_NORM_ATTACK;
    out[1] = (float)melee_str / STAT_NORM_STRENGTH;
    out[2] = (float)ranged_att / STAT_NORM_ATTACK;
    out[3] = (float)ranged_str / STAT_NORM_STRENGTH;
    out[4] = (float)magic / (STAT_NORM_ATTACK + STAT_NORM_MAGIC_DMG);
    out[5] = (float)defence / (5.0f * STAT_NORM_DEFENCE);
}

static void col_hypothetical_equipment_after_cell(
    const ColosseumState* s,
    int cell_idx,
    uint8_t out[NUM_GEAR_SLOTS]
) {
    memcpy(out, s->player.equipped, NUM_GEAR_SLOTS);
    const ColoInvCell* cell = &s->inventory_cells[cell_idx];
    if (cell->item_idx == ITEM_NONE) return;
    int gear_slot = osrs_item_gear_slot(cell->item_idx);
    if (gear_slot < 0) return;
    if (gear_slot == GEAR_SLOT_WEAPON && item_is_two_handed(cell->item_idx))
        out[GEAR_SLOT_SHIELD] = ITEM_NONE;
    if (gear_slot == GEAR_SLOT_SHIELD && item_is_two_handed(out[GEAR_SLOT_WEAPON]))
        out[GEAR_SLOT_WEAPON] = ITEM_NONE;
    out[gear_slot] = cell->item_idx;
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
        uint8_t item = s->inventory_cells[cell].item_idx;
        if (item == ITEM_NONE) continue;
        int gslot = osrs_item_gear_slot(item);
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

    static EncounterLoadoutStats overflow_stats;
    static OsrsEquipmentEffectProfile overflow_effects;
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

static uint64_t col_best_gear_signature(const ColosseumState* s) {
    uint64_t h = 1469598103934665603ULL;
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
        h = (h ^ (uint64_t)s->player.equipped[slot]) * 1099511628211ULL;
    for (int cell = 0; cell < COLO_INVENTORY_DISPLAY_SLOTS; cell++)
        h = (h ^ (uint64_t)s->inventory_cells[cell].item_idx) * 1099511628211ULL;
    const int scalars[5] = {
        s->player.current_attack, s->player.current_strength,
        s->player.current_ranged, s->player.current_magic,
        (int)s->active_loadout_profile,
    };
    for (int k = 0; k < 5; k++)
        h = (h ^ (uint64_t)(uint32_t)scalars[k]) * 1099511628211ULL;
    return h;
}

static const ColoBestGear (*col_get_best_gear_table(ColosseumState* s))[COLO_NUM_NPC_TYPES] {
    if (!s) abort();
    ColoObsMemoCaches* memos = &s->obs_memos;
    uint64_t sig = col_best_gear_signature(s);
    for (int w = 0; w < COLO_BEST_GEAR_CACHE_WAYS; w++) {
        if (memos->best_gear_valid[w] && memos->best_gear_sig[w] == sig)
            return memos->best_gear[w];
    }
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

static ColoWeaponSet col_weapon_set_for_item(uint8_t item) {
    int item_style = get_item_attack_style(item);
    if (item_style == 2) return COLO_GEAR_RANGED;
    if (item_style == 3) return COLO_GEAR_MAGIC;
    return COLO_GEAR_MELEE;
}

static uint64_t col_weapon_choice_obs_signature(ColosseumState* s) {
    uint64_t h = col_best_gear_signature(s);
    int has_target = 0, type = 0, def_drained = 0, size = 0;
    int target = col_player_target_slot(s);
    if (target >= 0 && target < COLO_MAX_NPCS) {
        const ColoNPC* npc = &s->npcs[target];
        if (col_npc_is_live_target(npc) && !col_type_is_hazard_entity(npc->type)) {
            has_target = 1;
            type = (int)npc->type;
            def_drained = npc->def_drained;
            size = npc->size;
        }
    }
    const int target_fields[4] = { has_target, type, def_drained, size };
    for (int k = 0; k < 4; k++)
        h = (h ^ (uint64_t)(uint32_t)target_fields[k]) * 1099511628211ULL;
    return h;
}

static void col_write_weapon_choice_obs(ColosseumState* s, float* obs, int* io) {
    int i = *io;

    ColoObsMemoCaches* memos = &s->obs_memos;
    uint64_t sig = col_weapon_choice_obs_signature(s);
    for (int w = 0; w < COLO_WEAPON_CHOICE_OBS_CACHE_WAYS; w++) {
        if (memos->weapon_choice_valid[w] && memos->weapon_choice_sig[w] == sig) {
            memcpy(&obs[i], memos->weapon_choice[w],
                   sizeof(float) * COLO_WEAPON_CHOICE_OBS_CACHE_FLOATS);
            *io = i + COLO_WEAPON_CHOICE_OBS_CACHE_FLOATS;
            return;
        }
    }
    int block_start = i;

    const ColoNPC* npc = NULL;
    int target = col_player_target_slot(s);
    if (target >= 0 && target < COLO_MAX_NPCS) {
        const ColoNPC* cand = &s->npcs[target];
        if (col_npc_is_live_target(cand) && !col_type_is_hazard_entity(cand->type))
            npc = cand;
    }
    const ColoBestGear (*best)[COLO_NUM_NPC_TYPES] = NULL;
    float best_dpt = 0.0f;
    if (npc) {
        best = col_get_best_gear_table(s);
        for (int st = 0; st < COLO_NUM_WEAPON_SETS; st++)
            if (best[st][npc->type].dpt > best_dpt)
                best_dpt = best[st][npc->type].dpt;
    }

    for (int c = 0; c < OSRS_INVENTORY_SIZE; c++) {
        float v = 0.0f;
        uint8_t item = s->inventory_cells[c].item_idx;
        if (npc && item != ITEM_NONE &&
                osrs_item_gear_slot(item) == GEAR_SLOT_WEAPON) {
            ColoWeaponSet set = col_weapon_set_for_item(item);
            const ColoBestGear* bs = &best[set][npc->type];
            if (bs->dpt >= 0.0f) {
                uint8_t setup[NUM_GEAR_SLOTS];
                memcpy(setup, bs->setup, NUM_GEAR_SLOTS);
                setup[GEAR_SLOT_WEAPON] = item;
                if (item_is_two_handed(item))
                    setup[GEAR_SLOT_SHIELD] = ITEM_NONE;
                float dpt = col_expected_dpt_for_equipment_vs_npc(
                    s, setup, npc, 1);
                v = col_clamp01(dpt / COLO_EXPECTED_DPT_NORM);
            }
        }
        obs[i++] = v;
    }

    for (int c = 0; c < OSRS_INVENTORY_SIZE; c++) {
        uint8_t item = s->inventory_cells[c].item_idx;
        obs[i++] = (item != ITEM_NONE && osrs_spec_cost(item) > 0) ? 1.0f : 0.0f;
    }

    float cur = npc
        ? col_expected_dpt_for_equipment_vs_npc(s, s->player.equipped, npc, 1)
        : 0.0f;
    obs[i++] = col_clamp01(cur / COLO_EXPECTED_DPT_NORM);
    obs[i++] = (npc && best_dpt > 0.0f) ? col_clamp01(cur / best_dpt) : 0.0f;

    int slot = memos->weapon_choice_next;
    memcpy(memos->weapon_choice[slot], &obs[block_start],
           sizeof(float) * COLO_WEAPON_CHOICE_OBS_CACHE_FLOATS);
    memos->weapon_choice_sig[slot] = sig;
    memos->weapon_choice_valid[slot] = 1;
    memos->weapon_choice_next = (slot + 1) % COLO_WEAPON_CHOICE_OBS_CACHE_WAYS;

    *io = i;
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
    int target = col_player_target_slot(s);
    if (target < 0 || target >= COLO_MAX_NPCS) return -1;
    const ColoNPC* npc = &s->npcs[target];
    if (!col_npc_is_live_target(npc) || col_type_is_hazard_entity(npc->type))
        return -1;
    const ColoBestGear (*best)[COLO_NUM_NPC_TYPES] = col_get_best_gear_table(s);
    return col_weapon_set_is_argmax_dpt_for_npc(s, target, s->weapon_set, best);
}

static int col_attacked_with_argmax_set(ColosseumState* s) {
    if (!s->tick_scratch.player_attacked) return -1;
    int npc_slot = s->player_attack_npc_idx;
    if (npc_slot < 0 || npc_slot >= COLO_MAX_NPCS) return -1;
    const ColoNPC* npc = &s->npcs[npc_slot];
    if (!col_npc_is_live_target(npc) || col_type_is_hazard_entity(npc->type))
        return -1;
    const ColoBestGear (*best)[COLO_NUM_NPC_TYPES] = col_get_best_gear_table(s);
    return col_weapon_set_is_argmax_dpt_for_npc(s, npc_slot, s->weapon_set, best);
}

static void col_inventory_cell_post_use_deltas(
    const ColosseumState* s,
    int cell_idx,
    const float before[6],
    float out[6]
) {
    for (int i = 0; i < 6; i++) out[i] = 0.0f;
    if (s->inventory_cells[cell_idx].item_idx == ITEM_NONE) return;

    float after[6];
    uint8_t hypothetical[NUM_GEAR_SLOTS];
    col_hypothetical_equipment_after_cell(s, cell_idx, hypothetical);
    col_equipment_bonus_projection(hypothetical, after);
    for (int i = 0; i < 6; i++) out[i] = after[i] - before[i];
}

static uint64_t col_inventory_obs_signature(const ColosseumState* s) {
    uint64_t h = 1469598103934665603ULL;
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
        h = (h ^ (uint64_t)s->player.equipped[slot]) * 1099511628211ULL;
    for (int cell = 0; cell < COLO_INVENTORY_DISPLAY_SLOTS; cell++) {
        const ColoInvCell* c = &s->inventory_cells[cell];
        uint64_t packed = (uint64_t)c->item_idx
            | ((uint64_t)c->raw_osrs_id << 8)
            | ((uint64_t)c->dose << 24);
        h = (h ^ packed) * 1099511628211ULL;
    }
    const int scalars[3] = {
        s->player.base_hitpoints, s->player.base_prayer, s->player.base_attack,
    };
    for (int k = 0; k < 3; k++)
        h = (h ^ (uint64_t)(uint32_t)scalars[k]) * 1099511628211ULL;
    return h;
}

static void col_write_inventory_and_equipped_obs(
    ColosseumState* s,
    float* obs,
    int* idx
) {
    int i = *idx;

    ColoObsMemoCaches* memos = &s->obs_memos;
    uint64_t sig = col_inventory_obs_signature(s);
    for (int w = 0; w < COLO_INVENTORY_OBS_CACHE_WAYS; w++) {
        if (memos->inventory_valid[w] && memos->inventory_sig[w] == sig) {
            memcpy(&obs[i], memos->inventory[w],
                   sizeof(float) * COLO_INVENTORY_OBS_CACHE_FLOATS);
            *idx = i + COLO_INVENTORY_OBS_CACHE_FLOATS;
            return;
        }
    }
    int block_start = i;

    float current_equipment_projection[6];
    col_equipment_bonus_projection(s->player.equipped, current_equipment_projection);
    for (int cell_idx = 0; cell_idx < COLO_INVENTORY_DISPLAY_SLOTS; cell_idx++) {
        float deltas[6];
        const ColoInvCell* cell = &s->inventory_cells[cell_idx];
        col_inventory_cell_post_use_deltas(
            s, cell_idx, current_equipment_projection, deltas);
        osrs_write_inventory_cell_affordance_features(
            &obs[i],
            cell->item_idx,
            cell->raw_osrs_id,
            cell->dose,
            col_inventory_cell_holds_equipped_item(s, cell_idx),
            deltas,
            s->player.base_hitpoints,
            s->player.base_prayer,
            s->player.base_attack);
        i += OSRS_INVENTORY_CELL_OBS_FEATURES;
    }
    assert(i == COLO_OBS_AFTER_INVENTORY);

    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        osrs_write_equipped_self_features(&obs[i], s->player.equipped[slot]);
        i += OSRS_EQUIPPED_SELF_OBS_FEATURES;
    }
    assert(i == COLO_OBS_AFTER_EQUIPPED_SELF);

    int slot = memos->inventory_next;
    memcpy(memos->inventory[slot], &obs[block_start],
           sizeof(float) * COLO_INVENTORY_OBS_CACHE_FLOATS);
    memos->inventory_sig[slot] = sig;
    memos->inventory_valid[slot] = 1;
    memos->inventory_next = (slot + 1) % COLO_INVENTORY_OBS_CACHE_WAYS;

    *idx = i;
}

static void col_write_obs_ctx(EncounterState* state, EncounterContext* context, float* obs) {
    ColosseumState* s = (ColosseumState*)state;
    ColosseumContext* ctx = context ? (ColosseumContext*)context : col_legacy_context();
#ifdef COLO_PROFILE_ENABLED
    int col_prof_enabled = COLO_PROFILE_ENABLED();
    double col_prof_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
#endif
    col_refresh_current_obs_slots_ctx(s, ctx);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_REFRESH_SLOTS);
#endif
    memset(obs, 0, COLO_NUM_OBS * sizeof(float));
    int i = 0;
    int px = s->player.x, py = s->player.y;
    const EncounterLoadoutStats* ls = col_live_loadout_stats(s);

    obs[i++] = col_clamp01((float)s->player.current_hitpoints / 99.0f);
    obs[i++] = (float)(px - COLO_ARENA_MIN_X) / (float)COLO_ARENA_WIDTH;
    obs[i++] = (float)(COLO_ARENA_MAX_X - px) / (float)COLO_ARENA_WIDTH;
    obs[i++] = (float)(py - COLO_ARENA_MIN_Y) / (float)COLO_ARENA_HEIGHT;
    obs[i++] = (float)(COLO_ARENA_MAX_Y - py) / (float)COLO_ARENA_HEIGHT;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_MELEE) ? 1.0f : 0.0f;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_PIETY) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_RIGOUR) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_AUGURY) ? 1.0f : 0.0f;
    obs[i++] = col_clamp01((float)s->player.current_prayer / 99.0f);
    obs[i++] = col_clamp01((float)s->player.run_energy / 10000.0f);
    obs[i++] = col_clamp01((float)s->player.special_energy / 100.0f);
    obs[i++] = s->player.spec_armed ? 1.0f : 0.0f;
    obs[i++] = col_clamp01((float)s->player.attack_timer / 8.0f);
    obs[i++] = col_clamp01((float)s->player.food_timer / 3.0f);
    obs[i++] = col_clamp01((float)s->player.potion_timer / 3.0f);
    obs[i++] = osrs_interaction_active(&s->interaction) ? 1.0f : 0.0f;
    obs[i++] = col_clamp01((float)ls->attack_range / 15.0f);
    obs[i++] = col_clamp01((float)ls->max_hit / 80.0f);
    obs[i++] = col_clamp01((float)ls->attack_speed / 6.0f);
    obs[i++] = col_clamp01((float)s->player.current_defence / 99.0f);
    obs[i++] = col_clamp01((float)ls->def_stab / 300.0f);
    obs[i++] = col_clamp01((float)ls->def_slash / 300.0f);
    obs[i++] = col_clamp01((float)ls->def_crush / 300.0f);
    obs[i++] = col_clamp01((float)ls->def_magic / 300.0f);
    obs[i++] = col_clamp01((float)ls->def_ranged / 300.0f);
    obs[i++] = col_clamp01((float)ls->eff_level / 165.0f);
    obs[i++] = col_clamp01((float)ls->attack_bonus / 200.0f);
    obs[i++] = col_clamp01((float)ls->strength_bonus / 160.0f);
    int colo_spec_cost = osrs_spec_cost(s->player.equipped[GEAR_SLOT_WEAPON]);
    obs[i++] = col_clamp01((float)colo_spec_cost / 100.0f);
    obs[i++] = colo_spec_cost > 0 ? 1.0f : 0.0f;
    obs[i++] = s->active_loadout_profile == COLO_LOADOUT_PROFILE_SPEEDRUN ? 1.0f : 0.0f;
    obs[i++] = s->active_loadout_profile == COLO_LOADOUT_PROFILE_BEGINNER ? 1.0f : 0.0f;
    obs[i++] = col_clamp01((float)ls->spell_base_damage / 40.0f);
    assert(i == COLO_OBS_AFTER_PLAYER);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_PREFIX);
#endif

    col_write_pillar_obs(s, obs, &i);
    assert(i == COLO_OBS_AFTER_PILLARS);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_PILLARS);
#endif

    col_write_inventory_and_equipped_obs(s, obs, &i);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_INVENTORY);
#endif

    ColoVenatorPreviewTargets venator_targets;
    col_collect_venator_preview_targets(s, &venator_targets);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_VENATOR);
#endif

    for (int n = 0; n < COLO_OBS_NPCS; n++) {
        int npc_idx = s->current_obs_slots[n];
        if (npc_idx < 0 || npc_idx >= COLO_MAX_NPCS || !s->npcs[npc_idx].active) {
            i += COLO_FEATURES_PER_NPC;
            continue;
        }
        col_write_npc_slot(
            s, &s->npcs[npc_idx], npc_idx, &venator_targets, obs, &i);
    }
    assert(i == COLO_OBS_AFTER_NPCS);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_NPC_SLOTS);
#endif

    for (int m = 0; m < COLO_NUM_MODIFIERS; m++) {
        obs[i++] = (s->modifiers.active_mask & (1u << m)) ? 1.0f : 0.0f;
        obs[i++] = col_clamp01((float)s->modifiers.tier[m] / 3.0f);
    }
    obs[i++] = s->modifiers.draft_pending ? 1.0f : 0.0f;
    for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++) {
        int opt = s->modifiers.draft_options[o];
        obs[i++] = (s->modifiers.draft_pending && opt >= 0)
            ? (float)(opt + 1) / (float)COLO_NUM_MODIFIERS : 0.0f;
    }
    col_write_modifier_hazards(s, obs, &i);
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

    col_write_boss_obs(s, obs, &i);
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
            i += COLO_FEATURES_PER_HIT;
        }
    }
    assert(i == COLO_OBS_AFTER_PENDING_HITS);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_PENDING_HITS);
#endif

    if (ctx->config.step_out_forecast_obs_enabled) {
        int horizon = ctx->config.forecast_horizon;
        if (horizon < 1) horizon = 1;
        if (horizon > COLO_STEP_OUT_FORECAST_HORIZON)
            horizon = COLO_STEP_OUT_FORECAST_HORIZON;
        ColoStepOutForecast forecast;
        ColoForecastObsSummary summaries[ENCOUNTER_MOVE_ACTIONS];
        col_build_step_out_forecast_horizon_mode_summary(
            s, &forecast, summaries, horizon, ctx->config.forecast_run_tile_mode);
        i = col_write_step_out_forecast_obs_summary(&forecast, summaries, horizon, obs, i);
    } else {
        i += COLO_STEP_OUT_FORECAST_OBS_SIZE;
    }
    assert(i == COLO_OBS_AFTER_STEP_OUT_FORECAST);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_FORECAST);
#endif

    col_write_threat_los_counts(s, obs, &i);
    assert(i == COLO_OBS_AFTER_THREAT_LOS);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_THREAT_LOS);
#endif

    col_write_thrall_dc(s, obs, &i);
    assert(i == COLO_OBS_AFTER_THRALL_DC);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_THRALL_DC);
#endif

    col_write_weapon_choice_obs(s, obs, &i);
    assert(i == COLO_OBS_AFTER_WEAPON_CHOICE);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_WEAPON_CHOICE);
#endif

    col_write_spawn_obs(s, obs, &i);
    assert(i == COLO_OBS_AFTER_SPAWN);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_SPAWN);
#endif

    if (ctx->config.threat_field_obs_enabled) {
        col_write_threat_field_obs(s, obs, &i);
    } else {
        i += COLO_THREAT_FIELD_OBS_SIZE;
    }
    assert(i == COLO_OBS_AFTER_THREAT_FIELD);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_OBS_THREAT_FIELD);
#endif

    if (i != COLO_NUM_OBS) {
        fprintf(stderr, "colosseum obs index mismatch: wrote %d, expected %d\n", i, COLO_NUM_OBS);
        abort();
    }
}
