#include "matsci.h"

int main() {
    int num_agents = 16;
    Matsci env = {.num_agents=num_agents};
    init(&env);

    puf_reset(&env);
    puf_render(&env);
    while (!WindowShouldClose()) {
	for (int i=0; i<num_agents; i++) {
            env.agents[i].actions[0] = rndf(-1.0f, 1.0f);
            env.agents[i].actions[1] = rndf(-1.0f, 1.0f);
            env.agents[i].actions[2] = rndf(-1.0f, 1.0f);
	}
        puf_step(&env);
        puf_render(&env);
    }
    puf_close(&env);
}

