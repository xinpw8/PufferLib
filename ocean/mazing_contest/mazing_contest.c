#include "mazing_contest.h"
#include <time.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

void generate_dummy_action(MazingContest* env) {
    env->agents[0].actions[0] = (float)(rand_r(&env->rng) % TOTAL_ACTIONS);
}

#ifdef __EMSCRIPTEN__
typedef struct {
    MazingContest *env;
} WebRenderArgs;

void emscriptenStep(void *e) {
    WebRenderArgs *args = (WebRenderArgs *)e;
    MazingContest *env = args->env;
    generate_dummy_action(env);
    puf_step(env);
    puf_render(env);
}

WebRenderArgs *web_args = NULL;
#endif

int main() {
    MazingContest *env = (MazingContest*)calloc(1, sizeof(MazingContest));
    env->rng = (unsigned int)time(NULL);
    env->num_agents = 1;
    env->client = NULL;
    init(env);
    env->agents[0].observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env->agents[0].actions = (float*)calloc(NUM_ATNS, sizeof(float));
    env->agents[0].rewards = (float*)calloc(1, sizeof(float));
    env->agents[0].terminals = (float*)calloc(1, sizeof(float));
    puf_reset(env);

#ifdef __EMSCRIPTEN__
    WebRenderArgs *args = (WebRenderArgs*)calloc(1, sizeof(WebRenderArgs));
    args->env = env;
    web_args = args;
    emscripten_set_main_loop_arg(emscriptenStep, args, 0, true);
#else
    puf_render(env);
    while (!WindowShouldClose()) {
        generate_dummy_action(env);
        puf_step(env);
        puf_render(env);
    }
    puf_close(env);
    free(env->agents[0].observations);
    free(env->agents[0].actions);
    free(env->agents[0].rewards);
    free(env->agents[0].terminals);
    free(env);
#endif
    return 0;
}
