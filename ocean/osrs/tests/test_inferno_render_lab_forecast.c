#include <stdio.h>
#include <string.h>

#include "ocean/osrs/osrs_pvp_actions.h"
#include "ocean/osrs/encounters/encounter_colosseum.h"
#include "ocean/osrs/encounters/encounter_inferno.h"
#include "ocean/osrs/encounters/encounter_zulrah.h"
#include "ocean/osrs/osrs_render.h"
int main(void) {
    InfernoContext context;
    inf_init_context_typed(&context);
    inf_finalize_route_topology(&context);

    InfernoState state;
    memset(&state, 0, sizeof(state));
    inf_reset_ctx((EncounterState*)&state, (EncounterContext*)&context, 20260812u);

    OsrsEnv env = {0};
    env.encounter_def = &ENCOUNTER_INFERNO;
    env.encounter_state = &state;
    env.encounter_context = &context;

    RenderClient client = {0};
    client.gui.encounter_def = &ENCOUNTER_INFERNO;
    client.gui.encounter_state = &state;
    client.lab_enabled = 1;
    client.lab_show_forecast = 1;

    InfStepOutForecast forecast;
    if (!render_inferno_lab_build_forecast(&client, &env, &forecast)) return 1;
    if (!forecast.actions[0].valid) return 1;

    puts("inferno renderer lab forecast PASS");
    return 0;
}
