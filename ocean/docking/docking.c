#include "docking.h"
#include "puffercpu.h"

int main() {
    Weights* weights = load_weights("resources/docking/docking_weights.bin");
    int logit_sizes[1] = {5};
    PufferNet* net = make_puffernet(weights, 1, DOCKING_OBS_SIZE, 128, 1, logit_sizes, 1);

    Docking env = {0};
    env.width = 256;
    env.height = 192;
    env.max_ticks = 1024;
    env.max_speed = 6.0f;
    env.turn_rate = 0.10f;
    env.accel = 0.55f;
    env.drag = 0.90f;
    env.dock_radius = 18.0f;
    env.dock_speed_threshold = 0.72f;
    env.dock_heading_threshold = 0.28f;
    env.step_penalty = -0.01f;
    env.progress_reward_scale = 0.25f;

    env.agents[0].observations = (float*)calloc(DOCKING_OBS_SIZE, sizeof(float));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));

    c_init(&env);
    puf_reset(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.agents[0].actions[0] = DOCK_NOOP;
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                env.agents[0].actions[0] = DOCK_TURN_LEFT;
            } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                env.agents[0].actions[0] = DOCK_TURN_RIGHT;
            } else if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                env.agents[0].actions[0] = DOCK_THRUST;
            } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                env.agents[0].actions[0] = DOCK_BRAKE;
            }
        } else {
            forward_puffernet(net, env.agents[0].observations, env.agents[0].actions);
        }

        if (IsKeyPressed(KEY_R)) {
            puf_reset(&env);
        } else {
            puf_step(&env);
        }
        puf_render(&env);
    }

    free_puffernet(net);
    free(weights);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
    return 0;
}
