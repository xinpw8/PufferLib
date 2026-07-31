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

static int scripted_first_modifier(const ColosseumState* s) {
    for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++)
        if (s->modifiers.draft_options[o] >= 0) return o;
    return 0;
}

static int scripted_heal_cell(const ColosseumState* s) {
    for (int i = 0; i < COLO_INVENTORY_DISPLAY_SLOTS; i++) {
        const ColoInvCell* cell = &s->inventory_cells[i];
        OsrsInventoryClickResolution r = osrs_inventory_click_interpret(
            cell->item_idx, cell->raw_osrs_id, OSRS_CLICK_TICK_FIRST);
        if ((r.consumable_kind == OSRS_CONSUMABLE_BREW ||
                r.consumable_kind == OSRS_CONSUMABLE_GUTHIX_REST ||
                r.consumable_kind == OSRS_CONSUMABLE_SHARK_FOOD ||
                r.consumable_kind == OSRS_CONSUMABLE_KARAMBWAN) &&
                col_inventory_cell_actionable(s, i)) return i;
    }
    return -1;
}

static void aggressive_policy(ColosseumState* s, int* actions) {
    for (int h = 0; h < COLO_NUM_ACTION_HEADS; h++) actions[h] = 0;
    if (s->modifiers.draft_pending) {
        actions[COLO_HEAD_MODIFIER_SELECT] = scripted_first_modifier(s) + 1;
        return;
    }
    actions[COLO_HEAD_PRAYER] = scripted_overhead(s);
    actions[COLO_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_SET_REFRESH_RIGOUR;
    actions[COLO_HEAD_PRIMARY] = scripted_target(s);
    if (s->player.current_hitpoints < 55) {
        int heal_cell = scripted_heal_cell(s);
        if (heal_cell >= 0) {
            OsrsInventoryClickResolution r = osrs_inventory_cell_click_interpret(
                &s->inventory_cells[heal_cell], OSRS_CLICK_TICK_FIRST);
            int head = r.click_action == OSRS_CLICK_DRINK ? COLO_HEAD_DRINK : COLO_HEAD_EAT;
            actions[head] = heal_cell + 1;
        }
    }
}

static int count_live_enemies(const ColosseumState* s) {
    int n = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (col_npc_is_live_enemy(&s->npcs[i])) n++;
    return n;
}

static int forcekill_live_enemies(ColosseumState* s) {
    int killed = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (!col_npc_is_live_enemy(npc)) continue;
        npc->hp = 0;
        killed++;
    }
    return killed;
}

enum { MODE_AGGRESSIVE, MODE_FORCEKILL };

static void run_episodes(const char* label, int mode, int n_eps) {
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;
    ctx.config.loadout_profile_mode = COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY;
    ctx.config.beginner_loadout_fraction = 0.0f;
    ctx.config.step_out_forecast_obs_enabled = 1;
    ctx.config.wave_clear_bonus = 1.0f;

    static float obs[COLO_NUM_OBS];
    ColosseumState s;
    unsigned int rng = 0x9e3779b9u;

    int advanced_eps = 0;
    int max_wave_overall = 0;
    int total_clear_bonus_fires = 0;
    int total_waves_cleared = 0;
    int wins = 0;
    long total_spawned_w1 = 0, total_killed_w1 = 0;
    int wave_ended_eps = 0;
    double dmg_dealt = 0, dmg_recv = 0;

    for (int ep = 0; ep < n_eps; ep++) {
        memset(&s, 0, sizeof(s));
        col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, (uint32_t)(rng | 1u));

        int actions[COLO_NUM_ACTION_HEADS] = {0};
        long guard = 0;
        int wave1_spawn_count = -1;
        int max_wave_this_ep = s.wave;
        float clear_bonus_seen = 0.0f;
        int wave_completed_seen = 0;
        int prev_waves_cleared = s.log.waves_cleared;

        while (!s.episode_over && guard++ < COLO_MAX_TICKS + 16) {
            col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);

            if (wave1_spawn_count < 0 && s.wave == 0 && !s.modifiers.draft_pending &&
                    s.wave_spawn_delay == 0 && s.wave_ready_delay == 0) {
                int live = count_live_enemies(&s);
                if (live > 0) wave1_spawn_count = s.current_wave_total_killable;
            }

            if (s.modifiers.draft_pending) {
                aggressive_policy(&s, actions);
            } else if (mode == MODE_FORCEKILL) {

                for (int h = 0; h < COLO_NUM_ACTION_HEADS; h++) actions[h] = 0;
                forcekill_live_enemies(&s);
            } else {
                aggressive_policy(&s, actions);
            }

            float r_before = s.reward;
            (void)r_before;
            col_step_ctx((EncounterState*)&s, (EncounterContext*)&ctx, actions);

            if (s.tick_scratch.wave_completed) {
                wave_completed_seen++;

                clear_bonus_seen += ctx.config.wave_clear_bonus;
            }
            if (s.wave > max_wave_this_ep) max_wave_this_ep = s.wave;
        }

        int waves_cleared = s.log.waves_cleared - prev_waves_cleared;
        total_waves_cleared += waves_cleared;
        total_clear_bonus_fires += wave_completed_seen;
        if (max_wave_this_ep >= 1) advanced_eps++;
        if (max_wave_this_ep > max_wave_overall) max_wave_overall = max_wave_this_ep;
        if (wave_completed_seen > 0) wave_ended_eps++;
        if (s.winner == COLO_OUTCOME_PLAYER_WON) wins++;
        if (wave1_spawn_count > 0) { total_spawned_w1 += wave1_spawn_count; }
        total_killed_w1 += s.log.total_npc_kills;
        dmg_dealt += s.log.total_damage_dealt;
        dmg_recv  += s.log.total_damage_received;

        rng = rng * 1103515245u + 12345u;
    }

    printf("%-14s eps=%d\n", label, n_eps);
    printf("    wave1 spawns (killable)/ep = %.2f   kills/ep = %.2f\n",
           (double)total_spawned_w1 / n_eps, (double)total_killed_w1 / n_eps);
    printf("    episodes that reached wave>=1 (0->1 advance) = %d / %d\n",
           advanced_eps, n_eps);
    printf("    episodes where the wave ever ENDED (wave_completed fired) = %d / %d\n",
           wave_ended_eps, n_eps);
    printf("    total wave_completed fires = %d   total log.waves_cleared = %d\n",
           total_clear_bonus_fires, total_waves_cleared);
    printf("    clear-bonus fires (== wave_completed) = %d   wins = %d\n",
           total_clear_bonus_fires, wins);
    printf("    max wave reached overall = %d\n", max_wave_overall);
    printf("    dmg_dealt/ep = %.0f   dmg_recv/ep = %.0f\n",
           dmg_dealt / n_eps, dmg_recv / n_eps);
    printf("\n");
}

int main(void) {
    printf("=== Colosseum wave-1 -> wave-2 advance probe (CPU, speedrun loadout) ===\n");
    printf("wave index 0 == \"wave 1\"; advance means s->wave reaches 1.\n");
    printf("COLO_NUM_WAVES=%d  COLO_WAVE_BOSS=%d  SPEEDRUN_MAX_TICKS=%d  REINFORCE_TICKS=%d\n\n",
           COLO_NUM_WAVES, COLO_WAVE_BOSS, COLO_SPEEDRUN_MAX_TICKS, COLO_REINFORCEMENT_TICKS);

    run_episodes("AGGRESSIVE", MODE_AGGRESSIVE, 32);
    run_episodes("FORCEKILL", MODE_FORCEKILL, 32);
    return 0;
}
