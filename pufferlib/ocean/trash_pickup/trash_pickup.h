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
    if (logs->idx < logs->length) {
        logs->logs[logs->idx++] = *log;
    }
}

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

    int* grid; // 1D array for grid
    int (*agent_positions)[2]; // Agent positions
    int* agent_carrying; // Carrying status of agents

    int obs_size;

    // Interface for PufferLib
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;

    LogBuffer* log_buffer;
    Log log;
} CTrashPickupEnv;

// Helper functions
void place_random_items(CTrashPickupEnv* env, int count, int item_type) {
    int placed = 0;
    int agent_id = 0;
    while (placed < count) {
        int x = rand() % env->grid_size;
        int y = rand() % env->grid_size;

        if (env->grid[INDEX(env, x, y)]  == EMPTY) {
            env->grid[INDEX(env, x, y)] = item_type;
            if (item_type == AGENT) {
                int agent_idx = agent_id; // KEHOE - why would we assign this to 'placed'???
                env->agent_positions[agent_idx][0] = x;
                env->agent_positions[agent_idx][1] = y;
                agent_id += 1;
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
        int cell_state = env->grid[INDEX(env, new_x, new_y)];
        if (cell_state == EMPTY) 
        {
            env->agent_positions[agent_idx][0] = new_x;
            env->agent_positions[agent_idx][1] = new_y;
            env->grid[INDEX(env, x, y)] = EMPTY;
            env->grid[INDEX(env, new_x, new_y)] = AGENT;
        } 
        else if (cell_state == TRASH && carrying == AGENT_IS_NOT_CARRYING) 
        {
            env->agent_carrying[agent_idx] = AGENT_IS_CARRYING;
            env->grid[INDEX(env, x, y)] = EMPTY;
            env->grid[INDEX(env, new_x, new_y)] = AGENT;
            env->agent_positions[agent_idx][0] = new_x;
            env->agent_positions[agent_idx][1] = new_y;
            env->rewards[agent_idx] += env->positive_reward;
            env->total_episode_reward += env->positive_reward;
        } 
        else if (cell_state == TRASH_BIN) 
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
                    int new_bin_cell_state = env->grid[INDEX(env, new_bin_x, new_bin_y)];
                    if (new_bin_cell_state == EMPTY)
                    {
                        env->grid[INDEX(env, new_x, new_y)] = EMPTY;
                        env->grid[INDEX(env, new_bin_x, new_bin_y)] = TRASH_BIN;
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
                        if (env->grid[INDEX(env, new_agent_x, new_agent_y)] != AGENT) 
                        {
                            env->grid[INDEX(env, new_x, new_y)] = EMPTY; // Remove trash bin from current cell
                            env->grid[INDEX(env, x, y)] = TRASH_BIN; // Move trash bin to agent's old position
                            env->grid[INDEX(env, new_agent_x, new_agent_y)] = AGENT; // Move agent to new position.

                            env->agent_positions[agent_idx][0] = new_agent_x;
                            env->agent_positions[agent_idx][1] = new_agent_y;
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

    for (int i = 0; i < env->grid_size * env->grid_size; i++) 
    {
        if (env->grid[i] == TRASH)
            return false;
    }

    return true;
}

void reset(CTrashPickupEnv* env) {
    env->current_step = 0;
    env->total_episode_reward = 0.0f;

    memset(env->grid, EMPTY, env->grid_size * env->grid_size * sizeof(int));

    place_random_items(env, env->num_trash, TRASH);
    place_random_items(env, env->num_bins, TRASH_BIN);
    place_random_items(env, env->num_agents, AGENT);

    for (int i = 0; i < env->num_agents; i++) {
        env->agent_carrying[i] = 0;
    }
}

// Environment functions
void initialize_env(CTrashPickupEnv* env) {
    env->current_step = 0;

    env->obs_size = (env->num_trash * 3) + (env->num_bins * 2) + (env->num_agents * 3); // See .py file for why this is calculated this way

    env->positive_reward = 0.5f / env->num_trash;
    env->negative_reward = 1.0f / (env->max_steps * env->num_agents);
    env->total_episode_reward = 0.0f;

    env->grid = (int*)calloc(env->grid_size * env->grid_size, sizeof(int));
    env->agent_positions = (int(*)[2])calloc(env->num_agents, sizeof(int[2]));
    env->agent_carrying = (int*)calloc(env->num_agents, sizeof(int));

    reset(env);
}

void allocate(CTrashPickupEnv* env) {
    initialize_env(env);
    env->observations = (float*)calloc(env->obs_size, sizeof(float));
    env->actions = (int*)calloc(env->num_agents, sizeof(int));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->dones = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);

    for (int i = 0; i < env->num_agents; i++) {
        env->rewards[i] = 0;
    }
}

void step(CTrashPickupEnv* env) {
    // Reset reward for each agent
    for (int i = 0; i < env->num_agents; i++) {
        env->rewards[i] = 0;
    }

    for (int i = 0; i < env->num_agents; i++) {
        move_agent(env, i, env->actions[i]);
    }

    env->current_step++;
    if (env->current_step >= env->max_steps || is_episode_over(env)) {
        for (int i = 0; i < env->num_agents; i++) {
            env->dones[i] = 1;
        }
    }
}

void free_initialized(CTrashPickupEnv* env) {
    free(env->grid);
    free(env->agent_positions);
    free(env->agent_carrying);
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
    SetTargetFPS(4);

    client->agent_texture = LoadTexture("resources/puffers_128.png");

    return client;
}

// Helper function to count occurrences of a grid state
int get_grid_count(CTrashPickupEnv* env, int state) {
    int count = 0;
    for (int i = 0; i < env->grid_size * env->grid_size; i++) {
        if (env->grid[i] == state) {
            count++;
        }
    }
    return count;
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
    DrawText(
        TextFormat(
            "Step: %d\nTotal Episode Reward: %.2f\nTrash Collected: %d/%d",
            env->current_step,
            env->total_episode_reward,
            env->num_trash - (int)get_grid_count(env, TRASH),
            env->num_trash
        ),
        5, 2, 10, BLACK
    );

    // Draw the grid and its elements
    for (int x = 0; x < env->grid_size; x++) {
        for (int y = 0; y < env->grid_size; y++) {
            int cell_value =  env->grid[INDEX(env, x, y)];
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
            if (cell_value == EMPTY) {
                // Do nothing for empty cells
            } else if (cell_value == TRASH) {
                DrawRectangle(
                    screen_x + client->cell_size / 4,
                    screen_y + client->cell_size / 4,
                    client->cell_size / 2,
                    client->cell_size / 2,
                    BROWN
                );
            } else if (cell_value == TRASH_BIN) {
                DrawRectangle(
                    screen_x + client->cell_size / 8,
                    screen_y + client->cell_size / 8,
                    3 * client->cell_size / 4,
                    3 * client->cell_size / 4,
                    BLUE
                );
            } else if (cell_value == AGENT) {
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
