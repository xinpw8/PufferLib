#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

#define SIZE 4
#define EMPTY 0
#define UP 1
#define DOWN 2
#define LEFT 3
#define RIGHT 4

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

// Required struct for env_binding.h compatibility
typedef struct {
    Log log;                        // Required
    unsigned char* observations;            // Required (flattened 256 floats: 16 tiles * 16 one-hot)
    int* actions;                   // Required
    float* rewards;                 // Required
    unsigned char* terminals;       // Required
    int score;
    int tick;
    unsigned char grid[SIZE][SIZE];         // Store tile values directly as floats
    float episode_reward;           // Accumulate episode reward
} Game;

// --- Logging ---
void add_log(Game* game);

// --- Required functions for env_binding.h ---
void c_reset(Game* env);
void c_step(Game* env);
void c_render(Game* env);
void c_close(Game* env);

// Update the observation vector to be one-hot encoded for all tiles
static void update_observations(Game* game) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            game->observations[i * SIZE + j] = game->grid[i][j];
        }
    }
}

// --- Implementation ---

void add_log(Game* game) {
    int max_tile = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int tile = (int)(game->grid[i][j]);
            if (tile > max_tile) max_tile = tile;
        }
    }
    game->log.score = (float)pow(2,max_tile);
    game->log.perf += (game->rewards[0] > 0) ? 1 : 0;
    game->log.episode_length += game->tick;
    game->log.episode_return += game->episode_reward;
    game->log.n += 1;
}

void c_reset(Game* game) {
    memset(game->grid, EMPTY, sizeof(game->grid));

    game->score = 0;
    game->tick = 0;
    game->episode_reward = 0;
    if (game->terminals) game->terminals[0] = 0;
    // Add two random tiles at the start
    int added = 0;
    while (added < 2) {
        int i = rand() % SIZE;
        int j = rand() % SIZE;
        if ((int)(game->grid[i][j]) == EMPTY) {
            game->grid[i][j] = (rand() % 10 == 0) ? 2 : 1;
            added++;
        }
    }
    update_observations(game);
}

void add_random_tile(Game* game) {
    int empty_cells[SIZE * SIZE][2];
    int count = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if ((int)(game->grid[i][j]) == EMPTY) {
                empty_cells[count][0] = i;
                empty_cells[count][1] = j;
                count++;
            }
        }
    }
    if (count > 0) {
        int random_index = rand() % count;
        int value = (rand() % 10 == 0) ? 4 : 2;
        game->grid[empty_cells[random_index][0]][empty_cells[random_index][1]] = (float)value;
    }
    update_observations(game);
}

void print_grid(Game* game) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%4.0f ", game->grid[i][j]);
        }
        printf("\n");
    }
    printf("Score: %d\n", game->score);
}

bool slide_and_merge_row(float* row, float* reward) {
    bool moved = false;
    // Slide left
    for (int i = 0; i < SIZE - 1; i++) {
        if ((int)row[i] != EMPTY) {
            continue;
        }
        for (int j = i + 1; j < SIZE; j++) {
            if ((int)row[j] != EMPTY) {
                row[i] = row[j];
                row[j] = EMPTY;
                moved = true;
                break;
            }
        }
    }
    // Merge
    for (int i = 0; i < SIZE - 1; i++) {
        if ((int)row[i] != EMPTY && (int)row[i] == (int)row[i + 1]) {
            row[i] += 1;
            //*reward += (row[i] / 11.0f); 
            row[i + 1] = EMPTY;
            moved = true;
        }
    }
    // Slide again
    for (int i = 0; i < SIZE - 1; i++) {
        if ((int)row[i] != EMPTY) {
            continue;
        }
        for (int j = i + 1; j < SIZE; j++) {
            if ((int)row[j] != EMPTY) {
                row[i] = row[j];
                row[j] = EMPTY;
                moved = true;
                break;
            }
        }
    }
    return moved;
}

bool move(Game* game, int direction, float* reward) {
    bool moved = false;
    if (direction == UP || direction == DOWN) {
        for (int col = 0; col < SIZE; col++) {
            float temp[SIZE];
            for (int row = 0; row < SIZE; row++) {
                temp[row] = game->grid[row][col];
            }
            if (direction == DOWN) {
                for (int i = 0; i < SIZE / 2; i++) {
                    float tmp = temp[i];
                    temp[i] = temp[SIZE - 1 - i];
                    temp[SIZE - 1 - i] = tmp;
                }
            }
            moved |= slide_and_merge_row(temp, reward);
            if (direction == DOWN) {
                for (int i = 0; i < SIZE / 2; i++) {
                    float tmp = temp[i];
                    temp[i] = temp[SIZE - 1 - i];
                    temp[SIZE - 1 - i] = tmp;
                }
            }
            for (int row = 0; row < SIZE; row++) {
                game->grid[row][col] = temp[row];
            }
        }
    } else if (direction == LEFT || direction == RIGHT) {
        for (int row = 0; row < SIZE; row++) {
            float temp[SIZE];
            for (int col = 0; col < SIZE; col++) {
                temp[col] = game->grid[row][col];
            }
            if (direction == RIGHT) {
                for (int i = 0; i < SIZE / 2; i++) {
                    float tmp = temp[i];
                    temp[i] = temp[SIZE - 1 - i];
                    temp[SIZE - 1 - i] = tmp;
                }
            }
            moved |= slide_and_merge_row(temp, reward);
            if (direction == RIGHT) {
                for (int i = 0; i < SIZE / 2; i++) {
                    float tmp = temp[i];
                    temp[i] = temp[SIZE - 1 - i];
                    temp[SIZE - 1 - i] = tmp;
                }
            }
            for (int col = 0; col < SIZE; col++) {
                game->grid[row][col] = temp[col];
            }
        }
    }
    if (!moved) {
        *reward = -0.5;
    }
    return moved;
}

bool is_game_over(Game* game) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if ((int)(game->grid[i][j]) == EMPTY) return false;
            if (i < SIZE - 1 && (int)(game->grid[i][j]) == (int)(game->grid[i + 1][j])) return false;
            if (j < SIZE - 1 && (int)(game->grid[i][j]) == (int)(game->grid[i][j + 1])) return false;
        }
    }
    return true;
}

int calc_score(Game* game) {
    int max_tile = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int tile = (int)(game->grid[i][j]);
            if (tile > max_tile) max_tile = tile;
        }
    }
    return max_tile;
}

void c_step(Game* game) {
    float reward = 0.;
    bool did_move = move(game, game->actions[0]+1, &reward);
    game->tick += 1;
    if (did_move) {
        add_random_tile(game);
        game->score = calc_score(game);
    }
    
    game->terminals[0] = is_game_over(game) ? 1 : 0;
    if (game->terminals[0] == 1) {
        reward = -1.0; // punish for losing
    }
    game->rewards[0] = reward;
    game->episode_reward += reward;

    update_observations(game);

    if (game->terminals[0]) {
        add_log(game);
        c_reset(game);
    }
}

void c_render(Game* game) {
    if (!IsWindowReady()) {
        InitWindow(100 * SIZE, 100 * SIZE, "2048");
        SetTargetFPS(10);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }

    BeginDrawing();
    ClearBackground(RAYWHITE);

    int px = 100;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int val = (int)((game->grid[i][j]));
            val = pow(2, val);
            if (val == 1) {
                val = 0;
            }
            Color color = LIGHTGRAY;
            if (val == 2) color = (Color){238, 228, 218, 255};
            else if (val == 4) color = (Color){237, 224, 200, 255};
            else if (val == 8) color = (Color){242, 177, 121, 255};
            else if (val == 16) color = (Color){245, 149, 99, 255};
            else if (val == 32) color = (Color){246, 124, 95, 255};
            else if (val == 64) color = (Color){246, 94, 59, 255};
            else if (val == 128) color = (Color){237, 207, 114, 255};
            else if (val == 256) color = (Color){237, 204, 97, 255};
            else if (val == 512) color = (Color){237, 200, 80, 255};
            else if (val == 1024) color = (Color){237, 197, 63, 255};
            else if (val == 2048) color = (Color){237, 194, 46, 255};
            else if (val > 0) color = (Color){60, 60, 60, 255};
            DrawRectangle(j * px, i * px, px - 5, px - 5, color);
            if (val > 0) {
                DrawText(TextFormat("%d", val), j * px + 30, i * px + 40, 32, BLACK);
            }
        }
    }
    DrawText(TextFormat("Score: %d", game->score), 10, 10, 24, DARKGRAY);
    EndDrawing();
}

void c_close(Game* game) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
