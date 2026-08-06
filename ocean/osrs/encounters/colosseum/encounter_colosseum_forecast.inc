#line 76 "encounter_colosseum.h"

#define COLO_STYLE_MASK_MELEE  1
#define COLO_STYLE_MASK_RANGED 2
#define COLO_STYLE_MASK_MAGIC  4

typedef struct {
    int valid;
    int land_x;
    int land_y;
} ColoForecastLanding;

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

static int col_step_out_forecast_action_valid(const ColosseumState* s, int action_idx) {
    if (action_idx == 0) return 1;
    if (s->modifiers.draft_pending) return 0;
    int nx = s->player.x + ENCOUNTER_MOVE_TARGET_DX[action_idx];
    int ny = s->player.y + ENCOUNTER_MOVE_TARGET_DY[action_idx];
    return col_player_walkable((void*)s, nx, ny);
}

static ColoForecastLanding col_step_out_forecast_action_landing_ctx(
    const ColosseumState* s,
    int action_idx
) {
    ColoForecastLanding landing = {
        .valid = col_step_out_forecast_action_valid(s, action_idx),
        .land_x = s->player.x,
        .land_y = s->player.y,
    };
    if (action_idx == 0) return landing;

    int target_dx = ENCOUNTER_MOVE_TARGET_DX[action_idx];
    int target_dy = ENCOUNTER_MOVE_TARGET_DY[action_idx];
    int tx = landing.land_x + target_dx;
    int ty = landing.land_y + target_dy;
    int abs_dx = target_dx < 0 ? -target_dx : target_dx;
    int abs_dy = target_dy < 0 ? -target_dy : target_dy;
    int max_steps = abs_dx > abs_dy ? abs_dx : abs_dy;

    for (int step = 0; step < max_steps; step++) {
        if (landing.land_x == tx && landing.land_y == ty) break;
        int dx = 0;
        int dy = 0;
        if (tx > landing.land_x) dx = 1;
        else if (tx < landing.land_x) dx = -1;
        if (ty > landing.land_y) dy = 1;
        else if (ty < landing.land_y) dy = -1;

        int moved = 0;
        if (dx != 0 && dy != 0 &&
                col_player_walkable((void*)s, landing.land_x + dx, landing.land_y + dy) &&
                col_player_walkable((void*)s, landing.land_x + dx, landing.land_y) &&
                col_player_walkable((void*)s, landing.land_x, landing.land_y + dy)) {
            landing.land_x += dx;
            landing.land_y += dy;
            moved = 1;
        } else if (dx != 0 &&
                col_player_walkable((void*)s, landing.land_x + dx, landing.land_y)) {
            landing.land_x += dx;
            moved = 1;
        } else if (dy != 0 &&
                col_player_walkable((void*)s, landing.land_x, landing.land_y + dy)) {
            landing.land_y += dy;
            moved = 1;
        }
        if (!moved) break;
    }

    return landing;
}

static int col_primary_action_valid(
    const ColosseumState* s,
    int primary_action
) {
    if (primary_action < 0 || primary_action >= COLO_PRIMARY_DIM) {
        fprintf(stderr, "colosseum primary action out of range: %d\n", primary_action);
        abort();
    }
    if (primary_action == 0) return 1;
    if (s->modifiers.draft_pending && !s->modifiers.draft_free_movement) return 0;
    if (primary_action < ENCOUNTER_MOVE_ACTIONS) {
        int nx = s->player.x + ENCOUNTER_MOVE_TARGET_DX[primary_action];
        int ny = s->player.y + ENCOUNTER_MOVE_TARGET_DY[primary_action];
        return col_player_walkable((void*)s, nx, ny);
    }

    int obs_slot = col_primary_obs_slot_from_action(primary_action);
    int npc_slot = col_obs_slot_to_npc(s, obs_slot);
    return npc_slot >= 0 && col_npc_is_live_target(&s->npcs[npc_slot]);
}

