#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"

static int scripted_overhead(const ColosseumState* s) {
    int best_style = ATTACK_STYLE_NONE, best_ticks = 1 << 30, best_dmg = -1;
    const EncounterPendingHitQueue* q = &s->player_pending_hits;
    for (int i = 0; i < q->count; i++) {
        const EncounterPendingHit* h = &q->hits[i];
        if (!h->active || !h->check_prayer) continue;
        if (h->ticks_remaining < best_ticks ||
                (h->ticks_remaining == best_ticks && h->damage > best_dmg)) {
            best_ticks = h->ticks_remaining; best_dmg = h->damage; best_style = h->attack_style;
        }
    }

    for (int n = 0; n < COLO_MAX_NPCS; n++) {
        const ColoNPC* npc = &s->npcs[n];
        if (npc->type != COLO_MANTICORE || !col_npc_is_live_enemy(npc)) continue;
        const ColoManticoreState* mc = &npc->type_state.manticore;
        if (mc->cycle_step < 0) continue;
        int orb = mc->cycle_step;
        if (orb >= 3) continue;
        int ticks = (mc->cycle_step == 0) ? npc->attack_timer : 0;
        int dmg = 36;
        if (ticks < best_ticks || (ticks == best_ticks && dmg > best_dmg)) {
            best_ticks = ticks; best_dmg = dmg; best_style = mc->orb_style[orb];
        }
    }
    switch (best_style) {
        case ATTACK_STYLE_MAGIC:  return COLO_OVERHEAD_MAGIC;
        case ATTACK_STYLE_RANGED: return COLO_OVERHEAD_RANGED;
        case ATTACK_STYLE_MELEE:  return COLO_OVERHEAD_MELEE;
        default:                  return COLO_OVERHEAD_NO_CHANGE;
    }
}

static int forecast_action_danger(const ColoStepOutForecastAction* a, int horizon) {
    if (!a->valid) return 1 << 30;
    int danger = a->same_tick_mixed_style_conflict ? 100000 : 0;
    int worst = 0;
    for (int t = 0; t < horizon; t++)
        if (a->ticks[t].max_hit > worst) worst = a->ticks[t].max_hit;
    return danger + worst;
}

static int g_move_mode = 0;

static int scripted_move(ColosseumState* s) {
    if (g_move_mode == 1) return 0;
    ColoStepOutForecast f;
    col_build_step_out_forecast_horizon_mode(s, &f, COLO_STEP_OUT_FORECAST_HORIZON, 0);
    int horizon = COLO_STEP_OUT_FORECAST_HORIZON;
    const ColoStepOutForecastAction* idle = &f.actions[0];
    if (idle->valid && !idle->same_tick_mixed_style_conflict) return 0;
    int best = 0, best_danger = forecast_action_danger(idle, horizon);
    for (int a = 1; a < ENCOUNTER_MOVE_ACTIONS; a++) {
        int d = forecast_action_danger(&f.actions[a], horizon);
        if (d < best_danger) { best_danger = d; best = a; }
    }
    return best;
}

static int scripted_target(const ColosseumState* s) {
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

static int scripted_least_bad_modifier(const ColosseumState* s) {
    for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++)
        if (s->modifiers.draft_options[o] >= 0) return o;
    return 0;
}

static int scripted_heal_cell(const ColosseumState* s) {
    for (int i = 0; i < COLO_INVENTORY_DISPLAY_SLOTS; i++) {
        const ColoInvCell* cell = &s->player.inventory_cells[i];
        OsrsInventoryClickResolution r =
            osrs_inventory_cell_click_interpret(
                cell, OSRS_CLICK_TICK_FIRST);
        if ((r.consumable_kind == OSRS_CONSUMABLE_BREW ||
                r.consumable_kind == OSRS_CONSUMABLE_GUTHIX_REST ||
                r.consumable_kind == OSRS_CONSUMABLE_SHARK_FOOD ||
                r.consumable_kind == OSRS_CONSUMABLE_KARAMBWAN) &&
                col_inventory_cell_actionable(s, i)) return i;
    }
    return -1;
}

static void scripted_policy(ColosseumState* s, int* actions) {
    for (int h = 0; h < COLO_NUM_ACTION_HEADS; h++) actions[h] = 0;
    if (s->modifiers.draft_pending) {
        actions[COLO_HEAD_MODIFIER_SELECT] = scripted_least_bad_modifier(s) + 1;
        return;
    }
    actions[COLO_HEAD_PRAYER] = scripted_overhead(s);
    actions[COLO_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_SET_REFRESH_RIGOUR;
    int move = scripted_move(s);
    actions[COLO_HEAD_PRIMARY] = move > 0 ? move : scripted_target(s);
    if (s->player.current_hitpoints < 55) {
        int heal_cell = scripted_heal_cell(s);
        if (heal_cell >= 0) {
            OsrsInventoryClickResolution r = osrs_inventory_cell_click_interpret(
                &s->player.inventory_cells[heal_cell], OSRS_CLICK_TICK_FIRST);
            int head = r.click_action == OSRS_CLICK_DRINK ? COLO_HEAD_DRINK : COLO_HEAD_EAT;
            actions[head] = heal_cell + 1;
        }
    }
}

typedef int (*PolicyFn)(ColosseumState*, int*, unsigned int*);

static void run_episodes(const char* label, int scripted, int start_wave, int n_eps) {
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = start_wave;
    ctx.config.loadout_profile_mode = COLO_LOADOUT_PROFILE_MODE_MIXED;
    ctx.config.beginner_loadout_fraction = 0.5f;
    ctx.config.step_out_forecast_obs_enabled = 1;
    col_finalize_route_topology(&ctx);

    static float obs[COLO_NUM_OBS];
    ColosseumState s;
    int wins = 0, wave_sum = 0, wave_max = 0;
    int hist[16] = {0};
    double pray_correct = 0, npc_attacks = 0, dmg_recv = 0, dmg_dealt = 0;
    unsigned int rng = 0x9e3779b9u ^ (unsigned)(start_wave * 2654435761u);

    for (int ep = 0; ep < n_eps; ep++) {
        memset(&s, 0, sizeof(s));
        col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, (uint32_t)(rng | 1u));
        int actions[COLO_NUM_ACTION_HEADS] = {0};
        long guard = 0;
        while (!s.episode_over && guard++ < COLO_MAX_TICKS + 16) {
            col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
            if (scripted) {
                scripted_policy(&s, actions);
            } else {
                for (int h = 0; h < COLO_NUM_ACTION_HEADS; h++) {
                    rng = rng * 1103515245u + 12345u;
                    actions[h] = (int)((rng >> 16) % (unsigned)COLO_ACTION_DIMS[h]);
                }
            }
            col_step_ctx((EncounterState*)&s, (EncounterContext*)&ctx, actions);
        }
        int wave = s.wave;
        if (s.winner == COLO_OUTCOME_PLAYER_WON) wins++;
        wave_sum += wave;
        if (wave > wave_max) wave_max = wave;
        if (wave < 16) hist[wave]++;
        pray_correct += s.log.total_prayer_correct;
        npc_attacks  += s.log.total_npc_attacks;
        dmg_recv     += s.log.total_damage_received;
        dmg_dealt    += s.log.total_damage_dealt;
        rng = rng * 1103515245u + 12345u;
    }
    double pray_rate = npc_attacks > 0 ? pray_correct / npc_attacks : 0.0;
    printf("%-22s start_wave=%d  eps=%d  mean_wave=%.2f  max_wave=%d  wins=%d  pray_rate=%.2f  dmg_recv/ep=%.0f  dmg_dealt/ep=%.0f\n",
           label, start_wave, n_eps, (double)wave_sum / n_eps, wave_max, wins,
           pray_rate, dmg_recv / n_eps, dmg_dealt / n_eps);
    printf("    wave histogram (0-based): ");
    for (int w = 0; w < 13; w++) if (hist[w]) printf("w%d:%d ", w, hist[w]);
    printf("\n");
}

int main(void) {
    printf("=== Colosseum winnability probe (scripted vs random, CPU) ===\n");
    printf("(RL baseline for reference: mean_wave ~4.5 0-based, prayer ~0.62)\n");
    run_episodes("RANDOM (sanity)", 0, 0, 64);
    g_move_mode = 1;
    run_episodes("SCRIPTED-stand", 1, 0, 64);
    run_episodes("SCRIPTED-stand@w4", 1, 3, 64);
    g_move_mode = 0;
    run_episodes("SCRIPTED-deconflict", 1, 0, 64);
    run_episodes("SCRIPTED-deconflict@w4", 1, 3, 64);
    return 0;
}
