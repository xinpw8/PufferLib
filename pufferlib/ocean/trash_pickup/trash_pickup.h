#ifndef TRASH_PICKUP_H
#define TRASH_PICKUP_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "raylib.h"

#define EMPTY 0
#define TRASH 1
#define TRASH_BIN 2
#define AGENT 3

#define ACTION_UP 0
#define ACTION_DOWN 1
#define ACTION_LEFT 2
#define ACTION_RIGHT 3

#define AGENT_IS_CARRYING 1
#define AGENT_IS_NOT_CARRYING 0

#define LOG_BUFFER_SIZE 1024

// Helper macro for 1D indexing
#define INDEX(env, x, y) ((y) * (env)->grid_size + (x))

typedef struct Log {
    float episode_return;
    float episode_length;
    float trash_collected;
} Log;

typedef struct LogBuffer {
    Log* logs;
    int length;
    int idx;
} LogBuffer;

// LogBuffer functions
LogBuffer* allocate_logbuffer(int size) {
    LogBuffer* logs = (LogBuffer*)calloc(1, sizeof(LogBuffer));
    logs->logs = (Log*)calloc(size, sizeof(Log));
    logs->length = size;
    logs->idx = 0;
    return logs;
}

void free_logbuffer(LogBuffer* buffer) {
    free(buffer->logs);
    free(buffer);
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
        log.trash_collected += logs->logs[i].trash_collected;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.trash_collected /= logs->idx;
    logs->idx = 0;
    return log;
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
}

typedef struct {
    int type;      // Entity type: EMPTY, TRASH, TRASH_BIN, AGENT
    int index;     // Index in the positions array (-1 if not applicable)
} GridCell;

typedef struct {
    // Interface for PufferLib
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;

    int grid_size;
    int num_agents;
    int num_trash;
    int num_bins;
    int max_steps;
    int current_step;

    float positive_reward;
    float negative_reward;
    float total_episode_reward;

    GridCell* grid; // 1D array for grid
    int (*agent_positions)[2]; // Agent positions
    int* agent_carrying; // Carrying status of agents
    int (*trash_positions)[2];  // Positions of trash
    int* trash_presence;        // Presence indicator for each piece of trash
    int (*bin_positions)[2];    // Positions of trash bins

    int obs_size;
} CTrashPickupEnv;

void compute_observations(CTrashPickupEnv* env) {
    float* obs = env->observations;
    float norm_factor = 1.0f / env->grid_size;

    int obs_per_agent = env->obs_size / env->num_agents;

    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        int obs_index = agent_idx * obs_per_agent;

        // Add the observing agent's own position and carrying status
        obs[obs_index++] = (float) (env->agent_positions[agent_idx][0]) * norm_factor;
        obs[obs_index++] = (float) (env->agent_positions[agent_idx][1]) * norm_factor;
        obs[obs_index++] = env->agent_carrying[agent_idx] ? 1.0f : 0.0f;

        // Add positions of other agents
        for (int other_idx = 0; other_idx < env->num_agents; other_idx++) {
            if (other_idx == agent_idx) continue; // Skip self
            obs[obs_index++] = (float) (env->agent_positions[other_idx][0]) * norm_factor;
            obs[obs_index++] = (float) (env->agent_positions[other_idx][1]) * norm_factor;
            obs[obs_index++] = env->agent_carrying[other_idx] ? 1.0f : 0.0f;
        }

        // Add shared trash data
        for (int i = 0; i < env->num_trash; i++) {
            if (env->trash_presence[i]) {
                obs[obs_index++] = (float) (env->trash_positions[i][0]) * norm_factor;
                obs[obs_index++] = (float) (env->trash_positions[i][1]) * norm_factor;
                obs[obs_index++] = 1.0f; // Trash present
            } else {
                obs[obs_index++] = -1.0f; // Placeholder for x
                obs[obs_index++] = -1.0f; // Placeholder for y
                obs[obs_index++] = 0.0f; // Trash not present
            }
        }

        // Add shared bin data
        for (int i = 0; i < env->num_bins; i++) {
            obs[obs_index++] = env->bin_positions[i][0] * norm_factor;
            obs[obs_index++] = env->bin_positions[i][1] * norm_factor;
        }
    }
}

// Helper functions
void place_random_items(CTrashPickupEnv* env, int count, int item_type) {
    int placed = 0;
    while (placed < count) 
    {
        int x = rand() % env->grid_size;
        int y = rand() % env->grid_size;

        if (env->grid[INDEX(env, x, y)].type == EMPTY) {
            env->grid[INDEX(env, x, y)].type = item_type;

            if (item_type == AGENT) {
                env->grid[INDEX(env, x, y)].index = placed;
                env->agent_positions[placed][0] = x;
                env->agent_positions[placed][1] = y;
            } else if (item_type == TRASH) {
                env->grid[INDEX(env, x, y)].index = placed; // Set trash index
                env->trash_positions[placed][0] = x;
                env->trash_positions[placed][1] = y;
                env->trash_presence[placed] = 1;
            } else if (item_type == TRASH_BIN) {
                env->grid[INDEX(env, x, y)].index = placed; // Set bin index
                env->bin_positions[placed][0] = x;
                env->bin_positions[placed][1] = y;
            }

            placed++;
        }
    }
}

void move_agent(CTrashPickupEnv* env, int agent_idx, int action) {
    int x = env->agent_positions[agent_idx][0];
    int y = env->agent_positions[agent_idx][1];
    int carrying = env->agent_carrying[agent_idx];

    int move_dir_x = 0;
    int move_dir_y = 0;
    if (action == ACTION_UP) move_dir_y = -1;
    else if (action == ACTION_DOWN) move_dir_y = 1;
    else if (action == ACTION_LEFT) move_dir_x = -1;
    else if (action == ACTION_RIGHT) move_dir_x = 1;
    else printf("Undefined action: %d", action);

    int new_x = x + move_dir_x;
    int new_y = y + move_dir_y;

    if (new_x >= 0 && new_x < env->grid_size && new_y >= 0 && new_y < env->grid_size)
    {
        int cell_state_type = env->grid[INDEX(env, new_x, new_y)].type;
        if (cell_state_type == EMPTY) 
        {
            env->agent_positions[agent_idx][0] = new_x;
            env->agent_positions[agent_idx][1] = new_y;
            env->grid[INDEX(env, x, y)].type = EMPTY;
            env->grid[INDEX(env, x, y)].index = -1;
            env->grid[INDEX(env, new_x, new_y)].type = AGENT;
            env->grid[INDEX(env, new_x, new_y)].index = agent_idx;
        } 
        else if (cell_state_type == TRASH && carrying == AGENT_IS_NOT_CARRYING) 
        {
            int trash_index = env->grid[INDEX(env, new_x, new_y)].index;
            env->trash_presence[trash_index] = 0; // Mark as not present
            env->trash_positions[trash_index][0] = -1;
            env->trash_positions[trash_index][1] = -1;

            env->agent_carrying[agent_idx] = AGENT_IS_CARRYING;
            env->grid[INDEX(env, x, y)].type = EMPTY;
            env->grid[INDEX(env, x, y)].index = -1;
            env->grid[INDEX(env, new_x, new_y)].type = AGENT;
            env->grid[INDEX(env, new_x, new_y)].index = agent_idx;
            env->agent_positions[agent_idx][0] = new_x;
            env->agent_positions[agent_idx][1] = new_y;
            env->rewards[agent_idx] += env->positive_reward;
            env->total_episode_reward += env->positive_reward;
        } 
        else if (cell_state_type == TRASH_BIN) 
        {
            if (carrying == AGENT_IS_CARRYING)
            {
                env->agent_carrying[agent_idx] = AGENT_IS_NOT_CARRYING;
                env->rewards[agent_idx] += env->positive_reward;
                env->total_episode_reward += env->positive_reward;
            }
            else
            {
                bool try_pull_bin = false;
                int new_bin_x = new_x + move_dir_x;
                int new_bin_y = new_y + move_dir_y;
                if (new_bin_x >= 0 && new_bin_x < env->grid_size && new_bin_y >= 0 && new_bin_y < env->grid_size)
                {
                    int new_bin_cell_state_type = env->grid[INDEX(env, new_bin_x, new_bin_y)].type;
                    if (new_bin_cell_state_type == EMPTY)
                    {
                        int bin_index = env->grid[INDEX(env, new_x, new_y)].index;
                        env->bin_positions[bin_index][0] = new_bin_x;
                        env->bin_positions[bin_index][1] = new_bin_y;

                        env->grid[INDEX(env, new_x, new_y)].type = EMPTY;
                        env->grid[INDEX(env, new_x, new_y)].index = -1;
                        env->grid[INDEX(env, new_bin_x, new_bin_y)].type = TRASH_BIN;
                        env->grid[INDEX(env, new_bin_x, new_bin_y)].index = bin_index;
                    }
                    else{
                        try_pull_bin = true;
                    }
                }
                else
                {
                    try_pull_bin = true;
                }

                if (try_pull_bin)
                {
                    // pull the bin if on map border instead of push
                    int new_agent_x = new_x - move_dir_x * 2;
                    int new_agent_y = new_y - move_dir_y * 2;
                    if (new_agent_x >= 0 && new_agent_x < env->grid_size && new_agent_y >= 0 && new_agent_y < env->grid_size)
                    {
                        // Verify there isn't another agent already occupying this slot
                        // If its trash, lets just say the agent grabs it and puts it in the trash, env logic should handle this no problem.
                        if (env->grid[INDEX(env, new_agent_x, new_agent_y)].type == EMPTY || env->grid[INDEX(env, new_agent_x, new_agent_y)].type == TRASH) 
                        {
                            if (env->grid[INDEX(env, new_agent_x, new_agent_y)].type == TRASH)
                            { 
                                env->rewards[agent_idx] += env->positive_reward;
                                env->total_episode_reward += env->positive_reward;

                                int trash_index = env->grid[INDEX(env, new_agent_x, new_agent_y)].index;
                                env->trash_presence[trash_index] = 0; // Mark as not present
                                env->trash_positions[trash_index][0] = -1;
                                env->trash_positions[trash_index][1] = -1;
                            }

                            int bin_index = env->grid[INDEX(env, new_x, new_y)].index;
                            env->bin_positions[bin_index][0] = x;
                            env->bin_positions[bin_index][1] = y;

                            env->agent_positions[agent_idx][0] = new_agent_x;
                            env->agent_positions[agent_idx][1] = new_agent_y;

                            env->grid[INDEX(env, new_x, new_y)].type = EMPTY; // Remove trash bin from current cell
                            env->grid[INDEX(env, new_x, new_y)].index = -1;
                            env->grid[INDEX(env, x, y)].type = TRASH_BIN; // Move trash bin to agent's old position
                            env->grid[INDEX(env, x, y)].index = bin_index;
                            env->grid[INDEX(env, new_agent_x, new_agent_y)].type = AGENT; // Move agent to new position.
                            env->grid[INDEX(env, new_agent_x, new_agent_y)].index = agent_idx;
                        }
                    }
                }
            }
        }
    }

    env->rewards[agent_idx] -= env->negative_reward;
    env->total_episode_reward -= env->negative_reward;
}

bool is_episode_over(CTrashPickupEnv* env) {
    for (int i = 0; i < env->num_agents; i++) 
    {
        if (env->agent_carrying[i] == AGENT_IS_CARRYING) 
            return false;
    }

    for (int i = 0; i < env->num_trash; i++) 
    {
        if (env->trash_presence[i] == 1)
            return false;
    }

    return true;
}

void reset(CTrashPickupEnv* env) {
    env->current_step = 0;
    env->total_episode_reward = 0;

    memset(env->grid, EMPTY, env->grid_size * env->grid_size * sizeof(GridCell));

    // Place trash, bins, and agents randomly across the grid.
    place_random_items(env, env->num_trash, TRASH);
    place_random_items(env, env->num_bins, TRASH_BIN);
    place_random_items(env, env->num_agents, AGENT);

    for (int i = 0; i < env->num_agents; i++) {
        env->agent_carrying[i] = 0;
    }

    compute_observations(env);
}

// Environment functions
void initialize_env(CTrashPickupEnv* env) {
    env->current_step = 0;

    env->positive_reward = 0.5f / env->num_trash;
    env->negative_reward = 1.0f / (env->max_steps * env->num_agents);

    env->grid = (GridCell*)calloc(env->grid_size * env->grid_size, sizeof(GridCell));
    env->agent_positions = (int(*)[2])calloc(env->num_agents, sizeof(int[2]));
    env->agent_carrying = (int*)calloc(env->num_agents, sizeof(int));
    env->trash_positions = (int(*)[2])calloc(env->num_trash, sizeof(int[2]));
    env->trash_presence = (int*)calloc(env->num_trash, sizeof(int));
    env->bin_positions = (int(*)[2])calloc(env->num_bins, sizeof(int[2]));

    reset(env);
}

// Helper function to count occurrences of a grid state
int get_grid_count(CTrashPickupEnv* env, int type) {
    int count = 0;
    for (int i = 0; i < env->grid_size * env->grid_size; i++) {
        if (env->grid[i].type == type) {
            count++;
        }
    }
    return count;
}

void allocate(CTrashPickupEnv* env) {
    env->obs_size = env->num_agents * ((env->num_trash * 3) + (env->num_bins * 2) + (env->num_agents * 3)); // See .py file for why this is calculated this way

    env->observations = (float*)calloc(env->obs_size * env->num_agents, sizeof(float));
    env->actions = (int*)calloc(env->num_agents, sizeof(int));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->dones = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);

    for (int i = 0; i < env->num_agents; i++) {
        env->rewards[i] = 0;
    }

    initialize_env(env);
}

void step(CTrashPickupEnv* env) {
    // Reset reward for each agent
    for (int i = 0; i < env->num_agents; i++) {
        env->rewards[i] = 0;
        env->dones[i] = 0;
    }

    for (int i = 0; i < env->num_agents; i++) {
        move_agent(env, i, env->actions[i]);
    }

    env->current_step++;
    if (env->current_step >= env->max_steps || is_episode_over(env)) 
    {
        for (int i = 0; i < env->num_agents; i++) {
            env->dones[i] = 1;
        }

        Log log = {0};

        log.episode_length = env->current_step;
        log.episode_return = env->total_episode_reward;

        int total_trash_not_collected = 0;
        for (int i = 0; i < env->num_trash; i++){
            total_trash_not_collected += env->trash_presence[i];
        }

        log.trash_collected = (float) (env->num_trash - total_trash_not_collected);

        add_log(env->log_buffer, &log);

        reset(env);
    }

    // printf("current step: %d | rewards: %f %f %f \n", env->current_step, env->rewards[0], env->rewards[1], env->rewards[2]);

    compute_observations(env);
}

void free_initialized(CTrashPickupEnv* env) {
    free(env->grid);
    free(env->agent_positions);
    free(env->agent_carrying);
    free(env->trash_positions);
    free(env->trash_presence);
    free(env->bin_positions);
}

void free_allocated(CTrashPickupEnv* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->dones);
    free_logbuffer(env->log_buffer);
    free_initialized(env);
}

typedef struct Client {
    int window_width;
    int window_height;
    int header_offset;
    int cell_size;
    Texture2D agent_texture;
} Client;

// Initialize a rendering client
Client* make_client(CTrashPickupEnv* env) {
    const int CELL_SIZE = 40;
    Client* client = (Client*)malloc(sizeof(Client));
    client->cell_size = CELL_SIZE;
    client->header_offset = 60;
    client->window_width = env->grid_size * CELL_SIZE;
    client->window_height = client->window_width + client->header_offset;

    InitWindow(client->window_width, client->window_height, "Trash Pickup Environment");
    SetTargetFPS(3);

    client->agent_texture = LoadTexture("resources/puffers_128.png");

    return client;
}

// Render the TrashPickup environment
void render(Client* client, CTrashPickupEnv* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        free(client);
        exit(0);
    }

    BeginDrawing();
    ClearBackground(RAYWHITE);

    // Draw header with current step and total episode reward
    int total_trash_not_collected = 0;
    for (int i = 0; i < env->num_trash; i++){
        total_trash_not_collected += env->trash_presence[i];
    }
    DrawText(
        TextFormat(
            "Step: %d\nTotal Episode Reward: %.2f\nTrash Collected: %d/%d",
            env->current_step,
            env->total_episode_reward,
            env->num_trash - total_trash_not_collected,
            env->num_trash
        ),
        5, 2, 10, BLACK
    );

    // Draw the grid and its elements
    for (int x = 0; x < env->grid_size; x++) {
        for (int y = 0; y < env->grid_size; y++) {
            int cell_type =  env->grid[INDEX(env, x, y)].type;
            int screen_x = x * client->cell_size;
            int screen_y = y * client->cell_size + client->header_offset;

            Rectangle cell_rect = {
                .x = screen_x,
                .y = screen_y,
                .width = client->cell_size,
                .height = client->cell_size
            };

            // Draw grid cell border
            DrawRectangleLines((int)cell_rect.x, (int)cell_rect.y, (int)cell_rect.width, (int)cell_rect.height, LIGHTGRAY);

            // Draw grid cell content
            if (cell_type == EMPTY) {
                // Do nothing for empty cells
            } else if (cell_type == TRASH) {
                DrawRectangle(
                    screen_x + client->cell_size / 4,
                    screen_y + client->cell_size / 4,
                    client->cell_size / 2,
                    client->cell_size / 2,
                    BROWN
                );
            } else if (cell_type == TRASH_BIN) {
                DrawRectangle(
                    screen_x + client->cell_size / 8,
                    screen_y + client->cell_size / 8,
                    3 * client->cell_size / 4,
                    3 * client->cell_size / 4,
                    BLUE
                );
            } else if (cell_type == AGENT) {
                DrawTexturePro(
                    client->agent_texture, 
                    (Rectangle) {0, 0, 128, 128},
                    (Rectangle) {
                        screen_x + client->cell_size / 2, 
                        screen_y + client->cell_size / 2,
                        client->cell_size,
                        client->cell_size
                        },
                    (Vector2){client->cell_size / 2, client->cell_size / 2},
                    0,
                    WHITE
                );

                // Display if the agent is carrying trash
                for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
                    if (env->agent_positions[agent_idx][0] == x &&
                        env->agent_positions[agent_idx][1] == y) {
                        if (env->agent_carrying[agent_idx]) {
                            DrawRectangle(
                                screen_x + client->cell_size / 2,
                                screen_y + client->cell_size / 2,
                                client->cell_size / 4,
                                client->cell_size / 4,
                                BROWN
                            );
                        }

                        // Display agent rewards on the grid
                        DrawText(
                            TextFormat("%.2f", env->rewards[agent_idx]),
                            screen_x + client->cell_size / 4,
                            screen_y + client->cell_size / 4,
                            10, ORANGE
                        );
                    }
                }
            }
        }
    }

    EndDrawing();
}

// Cleanup and free the rendering client
void close_client(Client* client) {
    UnloadTexture(client->agent_texture);
    CloseWindow();
    free(client);
}

#endif // TRASH_PICKUP_H
