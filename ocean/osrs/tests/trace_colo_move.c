#include <stdio.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"

static int nearest_live_obs_target(const ColosseumState* s) {
    int best_slot = -1, best_dist = 1 << 30;
    for (int slot = 0; slot < COLO_OBS_NPCS; slot++) {
        int idx = s->current_obs_slots[slot];
        if (idx < 0 || idx >= COLO_MAX_NPCS) continue;
        const ColoNPC* npc = &s->npcs[idx];
        if (!col_npc_is_live_target(npc)) continue;
        int d = col_npc_dist_to_player(s, npc);
        if (d < best_dist) { best_dist = d; best_slot = slot; }
    }
    return best_slot < 0 ? 0 : col_primary_attack_action_for_obs_slot(best_slot);
}

static int first_walkable_move(const ColosseumState* s) {
    for (int a = 1; a < ENCOUNTER_MOVE_ACTIONS; a++) {
        int nx = s->player.x + ENCOUNTER_MOVE_TARGET_DX[a];
        int ny = s->player.y + ENCOUNTER_MOVE_TARGET_DY[a];
        if (col_player_walkable((void*)s, nx, ny)) return a;
    }
    return 0;
}

int main(void) {
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;
    ctx.config.loadout_profile_mode = COLO_LOADOUT_PROFILE_MODE_MIXED;
    ctx.config.beginner_loadout_fraction = 0.5f;
    ctx.config.step_out_forecast_obs_enabled = 1;
    ctx.config.action_debug_log = 1;

    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 0x1234u);

    static float obs[COLO_NUM_OBS];
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    for (int tick = 0; tick < 80 && !s.episode_over; tick++) {
        col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
        for (int h = 0; h < COLO_NUM_ACTION_HEADS; h++) actions[h] = 0;
        if (s.modifiers.draft_pending) {
            int opt = 0;
            for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++)
                if (s.modifiers.draft_options[o] >= 0) { opt = o; break; }
            actions[COLO_HEAD_MODIFIER_SELECT] = opt + 1;
        } else {
            actions[COLO_HEAD_PRIMARY] = (tick & 1)
                ? first_walkable_move(&s)
                : nearest_live_obs_target(&s);
        }
        col_step_ctx((EncounterState*)&s, (EncounterContext*)&ctx, actions);
    }
    return 0;
}
