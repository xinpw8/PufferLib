#include "trash_pickup.h"

// Helper function prototypes
static void place_random_items(CTrashPickupEnv* env, int count, int item_type);
static void move_agent(CTrashPickupEnv* env, int agent_idx, int action);

// Initialize the environment with dynamic memory allocation
void initialize_env(CTrashPickupEnv* env, int grid_size, int num_agents, int num_trash, int num_bins, int max_steps) {
    env->grid_size = grid_size;
    env->num_agents = num_agents;
    env->num_trash = num_trash;
    env->num_bins = num_bins;
    env->max_steps = max_steps;
    env->current_step = 0;

    env->positive_reward = 0.5f / num_trash; // This makes it so that +1 is the total reward if all trash is picked up
    env->negative_reward = 1.0f / (max_steps * num_agents); // This makes it so that -1 is the total reward if max steps is reached
    env->total_episode_reward = 0.0f;

    // Allocate memory for grid
    env->grid = (int**)malloc(grid_size * sizeof(int*));
    for (int i = 0; i < grid_size; i++) {
        env->grid[i] = (int*)malloc(grid_size * sizeof(int));
    }

    // Allocate memory for agent positions and carrying states
    env->agent_positions = (int(*)[2])malloc(num_agents * sizeof(int[2]));
    env->agent_carrying = (int*)malloc(num_agents * sizeof(int));

    // Allocate memory for PufferLib interface arrays
    env->observations = (float*)malloc(num_agents * (3 + grid_size * grid_size) * sizeof(float));
    env->actions = (int*)malloc(num_agents * sizeof(int));
    env->rewards = (float*)malloc(num_agents * sizeof(float));
    env->dones = (unsigned char*)malloc(num_agents * sizeof(unsigned char));

    reset_env(env);
}

// Reset the environment
void reset_env(CTrashPickupEnv* env) {
    env->current_step = 0;
    env->total_episode_reward = 0.0f;

    // Clear grid
    for (int x = 0; x < env->grid_size; x++) {
        for (int y = 0; y < env->grid_size; y++) {
            env->grid[x][y] = EMPTY;
        }
    }

    // Place trash
    place_random_items(env, env->num_trash, TRASH);

    // Place bins
    place_random_items(env, env->num_bins, TRASH_BIN);

    // Place agents
    place_random_items(env, env->num_agents, AGENT);
    for (int i = 0; i < env->num_agents; i++) {
        env->agent_carrying[i] = 0;
    }

    get_observations(env);
}

// Check if the episode is over
bool is_episode_over(CTrashPickupEnv* env) {
    // Check if all trash is collected and no agents are carrying trash
    bool no_trash_left = true;
    for (int x = 0; x < env->grid_size; x++) {
        for (int y = 0; y < env->grid_size; y++) {
            if (env->grid[x][y] == TRASH) {
                no_trash_left = false;
                break;
            }
        }
        if (!no_trash_left) break;
    }

    bool no_agent_carrying = true;
    for (int i = 0; i < env->num_agents; i++) {
        if (env->agent_carrying[i] != 0) {
            no_agent_carrying = false;
            break;
        }
    }

    return no_trash_left && no_agent_carrying;
}

// Step the environment
void step_env(CTrashPickupEnv* env) {
    // Reset rewards and dones for each agent at the start of the step
    for (int i = 0; i < env->num_agents; i++) {
        env->rewards[i] = 0.0f;
        env->dones[i] = 0;
    }

    // Process each agent's action
    for (int i = 0; i < env->num_agents; i++) {
        move_agent(env, i, env->actions[i]);
    }

    // Update the step count and check if the episode is over
    env->current_step++;
    if (env->current_step >= env->max_steps || is_episode_over(env)) {
        for (int i = 0; i < env->num_agents; i++) {
            env->dones[i] = 1;
        }
    }

    // Update observations after each step
    get_observations(env);
}

// Get observations for each agent
void get_observations(CTrashPickupEnv* env) {
    int obs_idx = 0;
    for (int i = 0; i < env->num_agents; i++) {
        // Agent's position
        env->observations[obs_idx++] = env->agent_positions[i][0];
        env->observations[obs_idx++] = env->agent_positions[i][1];
        
        // Whether the agent is carrying trash
        env->observations[obs_idx++] = env->agent_carrying[i];

        // Flattened grid representation
        for (int x = 0; x < env->grid_size; x++) {
            for (int y = 0; y < env->grid_size; y++) {
                env->observations[obs_idx++] = (float)env->grid[x][y];
            }
        }
    }
}

// Free dynamically allocated memory in the environment
void free_env(CTrashPickupEnv* env) {
    // Free grid memory
    for (int i = 0; i < env->grid_size; i++) {
        free(env->grid[i]);
    }
    free(env->grid);

    // Free agent memory
    free(env->agent_positions);
    free(env->agent_carrying);

    // Free PufferLib interface arrays
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->dones);
}

// Helper functions

static void place_random_items(CTrashPickupEnv* env, int count, int item_type) {
    int placed = 0;
    while (placed < count) {
        int x = rand() % env->grid_size;
        int y = rand() % env->grid_size;

        // Check if position is empty
        if (env->grid[x][y] == EMPTY) {
            env->grid[x][y] = item_type;
            if (item_type == AGENT) {
                int agent_idx = placed;
                env->agent_positions[agent_idx][0] = x;
                env->agent_positions[agent_idx][1] = y;
            }
            placed++;
        }
    }
}

static void move_agent(CTrashPickupEnv* env, int agent_idx, int action) {
    int x = env->agent_positions[agent_idx][0];
    int y = env->agent_positions[agent_idx][1];
    int carrying = env->agent_carrying[agent_idx];

    int new_x = x;
    int new_y = y;

    // Map actions to movements
    if (action == 0) {        // UP
        new_y -= 1;
    } else if (action == 1) { // DOWN
        new_y += 1;
    } else if (action == 2) { // LEFT
        new_x -= 1;
    } else if (action == 3) { // RIGHT
        new_x += 1;
    }

    // Check grid boundaries
    if (new_x < 0 || new_x >= env->grid_size || new_y < 0 || new_y >= env->grid_size) {
        new_x = x;
        new_y = y;
    }

    int cell_state = env->grid[new_x][new_y];

    if (cell_state == EMPTY) {
        // Move agent
        env->agent_positions[agent_idx][0] = new_x;
        env->agent_positions[agent_idx][1] = new_y;
    } else if (cell_state == TRASH && carrying == 0) {
        // Pick up trash
        env->agent_carrying[agent_idx] = 1;
        env->grid[new_x][new_y] = EMPTY;
        env->agent_positions[agent_idx][0] = new_x;
        env->agent_positions[agent_idx][1] = new_y;
        env->rewards[agent_idx] += env->positive_reward;
        env->total_episode_reward += env->positive_reward;
    } else if (cell_state == TRASH_BIN) {
        if (carrying == 1) {
            // Deposit trash
            env->agent_carrying[agent_idx] = 0;
            env->rewards[agent_idx] += env->positive_reward;
            env->total_episode_reward += env->positive_reward;
            // Agent does not move onto the bin
        } else {
            // Attempt to push bin
            // Check if the space after the bin is within bounds and empty
            if (new_x >= 0 && new_x < env->grid_size && new_y >= 0 && new_y < env->grid_size && env->grid[new_x][new_y] == EMPTY) 
            {
                // Move bin
                env->grid[x][y] = EMPTY;
                env->grid[new_x][new_y] = TRASH_BIN;
            }
        }
    } else {
        // Invalid move or occupied cell: stay in place
    }

    // Apply negative reward to encourage time / step efficiency
    env->rewards[agent_idx] -= env->negative_reward;
    env->total_episode_reward -= env->negative_reward;
}
