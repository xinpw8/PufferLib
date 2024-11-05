#ifndef TRASH_PICKUP_H
#define TRASH_PICKUP_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

typedef enum {
    EMPTY = 0,
    TRASH = 1,
    TRASH_BIN = 2,
    AGENT = 3
} GridState;

typedef struct {
    int grid_size;
    int num_agents;
    int num_trash;
    int num_bins;
    int max_steps;
    int current_step;

    float positive_reward;
    float negative_reward;
    float total_episode_reward;

    // Dynamic grid allocation
    int** grid;

    // Agent positions and states
    int (*agent_positions)[2];
    int* agent_carrying;

    // PufferLib interface arrays
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;

} CTrashPickupEnv;

// Function declarations
void initialize_env(CTrashPickupEnv* env, int grid_size, int num_agents, int num_trash, int num_bins, int max_steps);
void reset_env(CTrashPickupEnv* env);
void step_env(CTrashPickupEnv* env);
bool is_episode_over(CTrashPickupEnv* env);
void get_observations(CTrashPickupEnv* env);
void free_env(CTrashPickupEnv* env);

#endif // TRASH_PICKUP_H
