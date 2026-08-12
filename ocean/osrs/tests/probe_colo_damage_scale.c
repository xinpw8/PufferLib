#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"

static int failures = 0;

static void check(const char* name, int got, int want) {
    if (got != want) {
        printf("FAIL %s: got %d want %d\n", name, got, want);
        failures++;
    } else {
        printf("ok   %s: %d\n", name, got);
    }
}

static int direct_hp_loss(uint32_t seed, int dmg, float scale) {
    static ColosseumContext ctx;
    static ColosseumState s;
    col_init_context_typed(&ctx);
    col_finalize_route_topology(&ctx);
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, seed);
    s.active_player_damage_received_scale = scale;
    int hp0 = s.player.current_hitpoints;
    col_damage_player_from(&s, dmg, COLO_MANTICORE, COLO_DMG_UNPRAYABLE);
    return hp0 - s.player.current_hitpoints;
}

static int queued_hp_loss(uint32_t seed, int dmg, float scale) {
    static ColosseumContext ctx;
    static ColosseumState s;
    col_init_context_typed(&ctx);
    col_finalize_route_topology(&ctx);
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, seed);
    s.active_player_damage_received_scale = scale;
    int hp0 = s.player.current_hitpoints;
    int prayed = 0;

    s.player.prayer = PRAYER_NONE;
    EncounterPendingHit hit = encounter_pending_hit_resolved_at_throw(
        dmg, 1, ATTACK_STYLE_RANGED, s.player.prayer, COLO_SERPENT_SHAMAN, 0, 1, &prayed);
    int landed_raw = hit.damage;
    col_push_player_pending_hit(&s, hit);

    s.tick++;
    col_resolve_player_pending_hits_ctx(&s, &ctx);
    (void)landed_raw;
    return hp0 - s.player.current_hitpoints;
}

static int doom_stacks_after_melee(uint32_t seed, int dmg, float scale) {
    static ColosseumContext ctx;
    static ColosseumState s;
    col_init_context_typed(&ctx);
    col_finalize_route_topology(&ctx);
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, seed);
    s.active_player_damage_received_scale = scale;
    s.modifiers.active_mask |= (1u << COLO_MOD_DOOM);
    s.modifiers.tier[COLO_MOD_DOOM] = 1;
    s.player.prayer = PRAYER_NONE;
    s.doom_stacks = 0;
    col_apply_instant_melee_hit(&s, &ctx, 0, COLO_JAGUAR_WARRIOR, dmg, 1);
    return s.doom_stacks;
}

static int round_half_up(int dmg, float scale) {
    if (scale >= 1.0f) return dmg;
    if (scale <= 0.0f) return 0;
    int v = (int)((float)dmg * scale + 0.5f);
    return v < 0 ? 0 : v;
}

int main(void) {
    const uint32_t SEED = 0xC0DEu;
    const int DMG = 20;

    int d_full = direct_hp_loss(SEED, DMG, 1.0f);
    int d_half = direct_hp_loss(SEED, DMG, 0.5f);
    int d_zero = direct_hp_loss(SEED, DMG, 0.0f);
    check("direct scale=1.0 == raw dmg", d_full, DMG);
    check("direct scale=0.5 == round_half_up", d_half, round_half_up(DMG, 0.5f));
    check("direct scale=0.0 == 0 (invuln)", d_zero, 0);

    int q_full = queued_hp_loss(SEED, DMG, 1.0f);
    int q_half = queued_hp_loss(SEED, DMG, 0.5f);
    int q_zero = queued_hp_loss(SEED, DMG, 0.0f);
    check("queued scale=1.0 landed > 0 (sanity)", q_full > 0 ? 1 : 0, 1);
    check("queued scale=0.5 == round_half_up(landed)", q_half, round_half_up(q_full, 0.5f));
    check("queued scale=0.0 == 0 (invuln)", q_zero, 0);

    int doom_full = doom_stacks_after_melee(SEED, DMG, 1.0f);
    int doom_zero = doom_stacks_after_melee(SEED, DMG, 0.0f);
    check("doom scale=1.0 accrues one stack", doom_full, 1);
    check("doom scale=0.0 accrues NO stack (invuln-equivalent)", doom_zero, 0);

    int identity_ok = 1;
    for (int dmg = 0; dmg <= 255; dmg++) {
        ColosseumState s;
        memset(&s, 0, sizeof(s));
        s.active_player_damage_received_scale = 1.0f;
        if (col_scale_incoming_damage(&s, dmg) != dmg) { identity_ok = 0; break; }
    }
    check("scale=1.0 identity over dmg[0,255]", identity_ok, 1);

    int zero_ok = 1;
    for (int dmg = 0; dmg <= 255; dmg++) {
        ColosseumState s;
        memset(&s, 0, sizeof(s));
        s.active_player_damage_received_scale = 0.0f;
        if (col_scale_incoming_damage(&s, dmg) != 0) { zero_ok = 0; break; }
    }
    check("scale=0.0 zeros dmg[0,255]", zero_ok, 1);

    if (failures) {
        printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("\nALL PASS\n");
    return 0;
}
