#include "asteroids.h"

int main() {
  Asteroids env = {.size = 500, .frameskip = 1, .num_agents = 1};
  env.agents[0].observations = (obs_t *)calloc(OBS_SIZE, sizeof(obs_t));
  env.agents[0].actions = (float *)calloc(1, sizeof(float));
  env.agents[0].rewards = (float *)calloc(1, sizeof(float));
  env.agents[0].terminals = (float *)calloc(1, sizeof(float));

  puf_reset(&env);
  puf_render(&env);
  while (!WindowShouldClose()) {
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) {
        env.agents[0].actions[0] = 0;
      } else if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) {
        env.agents[0].actions[0] = 1;
      } else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) {
        env.agents[0].actions[0] = 2;
      } else if (IsKeyDown(KEY_SPACE)) {
        env.agents[0].actions[0] = 3;
      } else {
        env.agents[0].actions[0] = -1;
      }
    } else {
      env.agents[0].actions[0] = (float)(rand() % 4);
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
