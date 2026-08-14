/* Pure C demo file for Squared Continuous. Build it with:
 * bash scripts/build_ocean.sh squared_continuous local (debug)
 * bash scripts/build_ocean.sh squared_continuous fast
 */

#include "squared_continuous.h"
#include "puffercpu.h"

void demo() {
    Squared env = {.size = 11};
    env.agents[0].observations = (unsigned char*)calloc(env.size*env.size, sizeof(unsigned char));
    env.agents[0].actions = (float*)calloc(2, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));

    Weights* weights = load_weights("resources/squared_continuous/squared_continuous_weights.bin");
    int logit_sizes[2] = {1, 1};
    PufferNet* net = make_puffernet(weights, 1, 121, 128, 1, logit_sizes, 2);

    puf_reset(&env);
    puf_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.agents[0].actions[0] = 0.0f;
            env.agents[0].actions[1] = 0.0f;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.agents[0].actions[0] = -1.0f;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.agents[0].actions[0] = 1.0f;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.agents[0].actions[1] = -1.0f;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.agents[0].actions[1] = 1.0f;
        } else {
            float obs_f[121];
            for(int i=0; i<121; i++) obs_f[i] = (float)env.agents[0].observations[i];
            forward_puffernet(net, obs_f, env.agents[0].actions);
        }
        puf_step(&env);
        puf_render(&env);
    }
    
    free_puffernet(net);
    free(weights);

    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
}

int main() {
    demo();
    return 0;
}
