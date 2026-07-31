#include <stdio.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"

#define col_init_context_typed(ctx_ptr) do { \
    col_init_context_typed(ctx_ptr); \
    (ctx_ptr)->config.late_start_state_mode = 0; \
} while (0)

typedef enum { POLICY_HOLD_MELEE, POLICY_OBS_TELEGRAPH } PrayerPolicy;

static int telegraph_overhead(const ColosseumState* s) {
    AttackStyle best_style = ATTACK_STYLE_NONE;
    int best_ticks = 1 << 30;
    for (int slot = 0; slot < COLO_OBS_NPCS; slot++) {
        int idx = s->current_obs_slots[slot];
        if (idx < 0 || idx >= COLO_MAX_NPCS) continue;
        const ColoNPC* npc = &s->npcs[idx];
        if (!col_npc_is_live_enemy(npc)) continue;
        ColoNpcNextPrayerObs t = col_npc_next_prayer_obs(s, npc, idx);
        if (!t.active) continue;
        if (t.ticks < best_ticks) { best_ticks = t.ticks; best_style = t.style; }
    }
    switch (best_style) {
        case ATTACK_STYLE_MAGIC:  return COLO_OVERHEAD_MAGIC;
        case ATTACK_STYLE_RANGED: return COLO_OVERHEAD_RANGED;
        default:                  return COLO_OVERHEAD_MELEE;
    }
}

static int nearest_target(const ColosseumState* s) {
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

static void run_test(const char* label, PrayerPolicy policy, int start_wave, int n_eps) {
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = start_wave;
    ctx.config.loadout_profile_mode = COLO_LOADOUT_PROFILE_MODE_MIXED;
    ctx.config.beginner_loadout_fraction = 0.5f;
    ctx.config.step_out_forecast_obs_enabled = 1;

    static float obs[COLO_NUM_OBS];
    ColosseumState s;
    int wave_sum = 0;
    double pray_correct = 0, npc_attacks = 0, dmg_recv = 0;
    long tele_total = 0, tele_match = 0;
    AttackStyle predicted[COLO_MAX_NPCS];
    unsigned int rng = 0x51ed270bu ^ (unsigned)(start_wave * 2654435761u + policy);

    for (int ep = 0; ep < n_eps; ep++) {
        memset(&s, 0, sizeof(s));
        col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, (uint32_t)(rng | 1u));
        int actions[COLO_NUM_ACTION_HEADS] = {0};
        long guard = 0;
        while (!s.episode_over && guard++ < COLO_MAX_TICKS + 16) {
            col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
            for (int h = 0; h < COLO_NUM_ACTION_HEADS; h++) actions[h] = 0;
            if (s.modifiers.draft_pending) {
                for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++)
                    if (s.modifiers.draft_options[o] >= 0) { actions[COLO_HEAD_MODIFIER_SELECT] = o + 1; break; }
                col_step_ctx((EncounterState*)&s, (EncounterContext*)&ctx, actions);
                continue;
            }
            actions[COLO_HEAD_PRAYER] = (policy == POLICY_HOLD_MELEE)
                ? COLO_OVERHEAD_MELEE : telegraph_overhead(&s);
            actions[COLO_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_SET_REFRESH_RIGOUR;
            actions[COLO_HEAD_PRIMARY] = nearest_target(&s);

            for (int n = 0; n < COLO_MAX_NPCS; n++) predicted[n] = ATTACK_STYLE_NONE;
            for (int n = 0; n < COLO_MAX_NPCS; n++) {
                const ColoNPC* npc = &s.npcs[n];
                if (!col_npc_is_live_enemy(npc)) continue;
                ColoNpcNextPrayerObs t = col_npc_next_prayer_obs(&s, npc, n);
                if (t.active && t.ticks <= 1) predicted[n] = t.style;
            }

            col_step_ctx((EncounterState*)&s, (EncounterContext*)&ctx, actions);

            for (int n = 0; n < COLO_MAX_NPCS; n++) {
                const ColoNPC* npc = &s.npcs[n];
                if (!npc->attacked_this_tick) continue;
                AttackStyle actual = (AttackStyle)npc->attack_style_this_tick;
                if (actual != ATTACK_STYLE_MELEE && actual != ATTACK_STYLE_RANGED &&
                        actual != ATTACK_STYLE_MAGIC) continue;
                tele_total++;
                if (predicted[n] == actual) tele_match++;
            }
        }
        wave_sum += s.wave;
        pray_correct += s.log.total_prayer_correct;
        npc_attacks  += s.log.total_npc_attacks;
        dmg_recv     += s.log.total_damage_received;
        rng = rng * 1103515245u + 12345u;
    }
    double pray_rate = npc_attacks > 0 ? pray_correct / npc_attacks : 0.0;
    double tele_acc = tele_total > 0 ? (double)tele_match / (double)tele_total : 0.0;
    printf("%-26s wave=%d eps=%d  mean_wave=%.2f  pray_correct=%.2f  offpray_dmg/ep=%.0f  telegraph_acc=%.2f (%ld/%ld)\n",
           label, start_wave, n_eps, (double)wave_sum / n_eps, pray_rate,
           dmg_recv / n_eps, tele_acc, tele_match, tele_total);
}

int main(void) {
    printf("=== Colosseum prayer-wiring probe (stand + attack, prayer policy varies) ===\n");
    run_test("HOLD_MELEE   @w1", POLICY_HOLD_MELEE, 0, 128);
    run_test("OBS_TELEGRAPH@w1", POLICY_OBS_TELEGRAPH, 0, 128);
    run_test("HOLD_MELEE   @w4", POLICY_HOLD_MELEE, 3, 128);
    run_test("OBS_TELEGRAPH@w4", POLICY_OBS_TELEGRAPH, 3, 128);
    return 0;
}
