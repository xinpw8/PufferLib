#include <raylib.h>
#include <unistd.h>
#include "cpr.h"
#include "puffercpu.h"
#include "shared_pool.h"

int main() {
  CCpr env = {
      .num_agents = 4,
      .width = 32,
      .height = 32,
      .vision = 3,
      .reward_food = 1.0f,
      .interactive_food_reward = 5.0f,
      .food_base_spawn_rate = 2e-3,
  };
  allocate_ccpr(&env);
  puf_reset(&env);
  puf_render(&env);

  Weights* weights = load_weights("resources/cpr/cpr_weights.bin");
  int logit_sizes[] = {5};
  LinearLSTM* net = make_linearlstm(weights, env.num_agents, 49, logit_sizes, 1);
 
  while (!WindowShouldClose()) {
    // User can take control of the first puffer
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
        env.agents[0].actions[0] = 0;
      if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
        env.agents[0].actions[0] = 1;
      if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
        env.agents[0].actions[0] = 2;
      if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
        env.agents[0].actions[0] = 3;

      printf("Getting user input %d\n", env.agents[0].actions[0]);
      sleep(2);
    } else {
        for (int i = 0; i < env.num_agents*49; i++) {
            net->obs[i] = env.agents[0].observations[i];
        }
        forward_linearlstm(net, net->obs, env.agents[0].actions);
    }

    puf_step(&env);
    puf_render(&env);
  }
  //close_renderer(renderer);
  free_CCpr(&env);

  return 0;
}
