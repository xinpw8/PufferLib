#include "checkers.h"

int main() {
  Checkers env = {.size = BOARD_SIZE, .num_agents = 1};
  env.agents[0].observations =
      (obs_t *)calloc(env.size * env.size, sizeof(obs_t));
  env.agents[0].actions = (float *)calloc(1, sizeof(float));
  env.agents[0].rewards = (float *)calloc(1, sizeof(float));
  env.agents[0].terminals = (float *)calloc(1, sizeof(float));

  puf_reset(&env);
  puf_render(&env);
  while (!WindowShouldClose()) {
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      env.agents[0].actions[0] = 0;
      if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
        env.agents[0].actions[0] = 1;
      if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
        env.agents[0].actions[0] = 2;
      if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
        env.agents[0].actions[0] = 3;
      if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
        env.agents[0].actions[0] = 4;
    } else {
      env.agents[0].actions[0] = (float)(rand() % (env.size * env.size * 8));
    }
    puf_step(&env);
    puf_render(&env);
  }
  free(env.agents[0].observations);
  free(env.agents[0].actions);
  free(env.agents[0].rewards);
  free(env.agents[0].terminals);
  puf_close(&env);
}
