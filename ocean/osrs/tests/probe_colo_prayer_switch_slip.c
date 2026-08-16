#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"

static void prep(ColosseumContext* ctx, ColosseumState* s, float fail_prob) {
    col_init_context_typed(ctx);
    memset(s, 0, sizeof(*s));
    s->rng_state = 0xC0FFEEu;

    ENCOUNTER_COLOSSEUM.put_float((EncounterState*)s, (EncounterContext*)ctx,
        "prayer_switch_fail_prob", fail_prob);
    col_finalize_route_topology(ctx);
    s->player.prayer = PRAYER_PROTECT_RANGED;
    s->player.current_prayer = 990;
    s->player.prayer_just_activated = 0;
}

int main(void) {
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    actions[COLO_HEAD_PRAYER] = ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC;

    ColosseumContext ctx_off; ColosseumState s_off;
    prep(&ctx_off, &s_off, 0.0f);
    col_player_pretick(&s_off, &ctx_off, actions);
    int applied = (s_off.player.prayer == PRAYER_PROTECT_MAGIC);

    ColosseumContext ctx_on; ColosseumState s_on;
    prep(&ctx_on, &s_on, 1.0f);
    col_player_pretick(&s_on, &ctx_on, actions);
    int reverted = (s_on.player.prayer == PRAYER_PROTECT_RANGED);
    int flag_clear = (s_on.player.prayer_just_activated == 0);

    printf("p=0.0: prayer=%d (expect MAGIC=%d) applied=%d\n",
        s_off.player.prayer, PRAYER_PROTECT_MAGIC, applied);
    printf("p=1.0: prayer=%d (expect RANGED=%d) reverted=%d just_activated=%d\n",
        s_on.player.prayer, PRAYER_PROTECT_RANGED, reverted,
        s_on.player.prayer_just_activated);
    if (applied && reverted && flag_clear) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL\n");
    return 1;
}
