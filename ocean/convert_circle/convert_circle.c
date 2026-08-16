#include "convert_circle.h"
#include "puffercpu.h"
#include <stdlib.h>
#include <time.h>

int main() {
  ConvertCircle env = {
      .width = 1920,
      .height = 1080,
      .num_agents = 128,
      .num_factories = 16,
      .num_resources = 8,
      .equidistant = 1,
      .radius = 400,
  };
  srand(time(NULL));
  init(&env);

  for (int i = 0; i < env.num_agents; i++) {
    env.agents[i].observations = calloc(OBS_SIZE, sizeof(float));
    env.agents[i].actions = (float *)calloc(NUM_ATNS, sizeof(float));
    env.agents[i].rewards = (float *)calloc(1, sizeof(float));
    env.agents[i].terminals = (float *)calloc(1, sizeof(float));
    env.agents[i].action_mask = NULL;
    env.agents[i].policy = 0;
  }

  Weights *weights =
      load_weights("resources/convert/convert_weights.bin");
  int logit_sizes[2] = {9, 5};
  LinearLSTM *net =
      make_linearlstm(weights, env.num_agents, OBS_SIZE, logit_sizes, 2);

  // Pack flat buffers for net if needed; demo uses random actions
  puf_reset(&env);
  puf_render(&env);

  while (!WindowShouldClose()) {
    for (int i = 0; i < env.num_agents; i++) {
      env.agents[i].actions[0] = (float)(rand() % 9);
      env.agents[i].actions[1] = (float)(rand() % 5);
    }

    puf_step(&env);
    puf_render(&env);
  }

  free_linearlstm(net);
  free(weights);
  for (int i = 0; i < env.num_agents; i++) {
    free(env.agents[i].observations);
    free(env.agents[i].actions);
    free(env.agents[i].rewards);
    free(env.agents[i].terminals);
  }
  puf_close(&env);
}
