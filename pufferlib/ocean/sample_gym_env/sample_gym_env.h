/* sample_gym_env.h
 * SampleGymEnv: A custom grid-based collector environment
 * ALL environment logic is contained in this header file
 */

#ifndef SAMPLE_GYM_ENV_H
#define SAMPLE_GYM_ENV_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "raylib.h"

// Action constants
const unsigned char NOOP = 0;
const unsigned char UP = 1;
const unsigned char DOWN = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

// Grid cell types
const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char ITEM = 2;
const unsigned char WALL = 3;

// Environment parameters
const int MAX_ITEMS = 5;
const int MAX_STEPS = 200;

// Required logging struct - only use floats!
typedef struct {
    float perf;           // 0-1 normalized performance metric
    float score;          // Unnormalized score
    float episode_return; // Sum of rewards over episode
    float episode_length; // Number of steps in episode
    float n;              // Required as last field
} Log;

// Main environment struct
typedef struct {
    Log log;                    // Required logging field
    unsigned char* observations; // Grid observations
    int* actions;               // Agent actions
    float* rewards;             // Step rewards
    unsigned char* terminals;   // Episode termination flags
    
    // Environment state
    int size;                   // Grid size (size x size)
    int agent_row;              // Agent position
    int agent_col;
    int items_collected;        // Number of items collected
    int total_items;            // Total items in environment
    int step_count;             // Current step count
    int item_positions[5][2];   // Item coordinates (MAX_ITEMS = 5)
} SampleGymEnv;

// Add episode data to logs
void add_log(SampleGymEnv* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1.0f : 0.0f;
    env->log.score += env->rewards[0];
    env->log.episode_return += env->rewards[0];
    env->log.episode_length += env->step_count;
    env->log.n++;
}

// Helper function to place items randomly
void place_items(SampleGymEnv* env) {
    env->total_items = 3 + (rand() % 3); // 3-5 items
    
    for (int i = 0; i < env->total_items; i++) {
        int row, col;
        do {
            row = rand() % env->size;
            col = rand() % env->size;
        } while ((row == env->agent_row && col == env->agent_col) ||
                 env->observations[row * env->size + col] != EMPTY);
        
        env->item_positions[i][0] = row;
        env->item_positions[i][1] = col;
        env->observations[row * env->size + col] = ITEM;
    }
}

// Required function: reset environment
void c_reset(SampleGymEnv* env) {
    int total_cells = env->size * env->size;
    
    // Clear grid
    memset(env->observations, EMPTY, total_cells * sizeof(unsigned char));
    
    // Reset state
    env->agent_row = env->size / 2;
    env->agent_col = env->size / 2;
    env->items_collected = 0;
    env->step_count = 0;
    
    // Place agent
    env->observations[env->agent_row * env->size + env->agent_col] = AGENT;
    
    // Add some walls around the border
    for (int i = 0; i < env->size; i++) {
        env->observations[0 * env->size + i] = WALL;           // Top wall
        env->observations[(env->size-1) * env->size + i] = WALL; // Bottom wall
        env->observations[i * env->size + 0] = WALL;           // Left wall
        env->observations[i * env->size + (env->size-1)] = WALL; // Right wall
    }
    
    // Re-place agent (in case it was overwritten by walls)
    env->observations[env->agent_row * env->size + env->agent_col] = AGENT;
    
    // Place items
    place_items(env);
}

// Required function: step environment forward
void c_step(SampleGymEnv* env) {
    env->step_count++;
    
    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
    
    // Clear agent's old position
    env->observations[env->agent_row * env->size + env->agent_col] = EMPTY;
    
    // Calculate new position
    int new_row = env->agent_row;
    int new_col = env->agent_col;
    
    switch (action) {
        case UP:    new_row--; break;
        case DOWN:  new_row++; break;
        case LEFT:  new_col--; break;
        case RIGHT: new_col++; break;
        case NOOP:  break;
    }
    
    // Check bounds and walls
    if (new_row >= 0 && new_row < env->size && 
        new_col >= 0 && new_col < env->size &&
        env->observations[new_row * env->size + new_col] != WALL) {
        
        // Check if there's an item at new position
        if (env->observations[new_row * env->size + new_col] == ITEM) {
            env->items_collected++;
            env->rewards[0] = 10.0f; // Reward for collecting item
        }
        
        // Move agent
        env->agent_row = new_row;
        env->agent_col = new_col;
    } else {
        // Invalid move - small penalty
        env->rewards[0] = -0.1f;
    }
    
    // Place agent at new position
    env->observations[env->agent_row * env->size + env->agent_col] = AGENT;
    
    // Check termination conditions
    if (env->items_collected >= env->total_items) {
        // Collected all items - success!
        env->terminals[0] = 1;
        env->rewards[0] += 50.0f; // Bonus for completing episode
        add_log(env);
        c_reset(env);
    } else if (env->step_count >= MAX_STEPS) {
        // Timeout - failure
        env->terminals[0] = 1;
        env->rewards[0] -= 10.0f; // Penalty for timeout
        add_log(env);
        c_reset(env);
    } else {
        // Small negative reward for each step to encourage efficiency
        env->rewards[0] -= 0.01f;
    }
}

// Required function: render environment
void c_render(SampleGymEnv* env) {
    if (!IsWindowReady()) {
        InitWindow(32 * env->size, 32 * env->size + 60, "SampleGymEnv");
        SetTargetFPS(10);
    }
    
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    
    BeginDrawing();
    ClearBackground((Color){20, 20, 20, 255});
    
    int cell_size = 32;
    
    // Draw grid
    for (int row = 0; row < env->size; row++) {
        for (int col = 0; col < env->size; col++) {
            int x = col * cell_size;
            int y = row * cell_size;
            int cell_type = env->observations[row * env->size + col];
            
            Color color = (Color){40, 40, 40, 255}; // Default empty
            
            switch (cell_type) {
                case AGENT:
                    color = (Color){0, 200, 0, 255}; // Green agent
                    break;
                case ITEM:
                    color = (Color){255, 255, 0, 255}; // Yellow item
                    break;
                case WALL:
                    color = (Color){100, 100, 100, 255}; // Gray wall
                    break;
            }
            
            DrawRectangle(x, y, cell_size, cell_size, color);
            DrawRectangleLines(x, y, cell_size, cell_size, (Color){60, 60, 60, 255});
        }
    }
    
    // Draw info
    char info_text[100];
    sprintf(info_text, "Items: %d/%d  Steps: %d", 
            env->items_collected, env->total_items, env->step_count);
    DrawText(info_text, 10, env->size * cell_size + 10, 20, WHITE);
    
    DrawText("WASD/Arrow keys to move, ESC to exit", 10, env->size * cell_size + 35, 16, GRAY);
    
    EndDrawing();
}

// Required function: cleanup
void c_close(SampleGymEnv* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

#endif // SAMPLE_GYM_ENV_H