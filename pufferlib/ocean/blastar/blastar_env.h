#ifndef BLASTAR_ENV_H
#define BLASTAR_ENV_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h> // for calloc, free
#include "raylib.h"

// Include the renderer header
#include "blastar_renderer.h"

#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
#define LOG_BUFFER_SIZE 4096
#define MAX_EPISODE_STEPS 2800
#define PLAYER_MAX_LIVES 5
#define REWARD_BUFFER_SIZE 200

// Log structure
typedef struct Log {
    float episode_return;
    float episode_length;
    float score;
    float lives;
    float bullet_travel_rew;
    float fired_bullet_rew;
    float bullet_distance_to_enemy_rew;
    float gradient_penalty_rew;
    float flat_below_enemy_rew;
    float danger_zone_penalty_rew;
    float crashing_penalty_rew;
    float hit_enemy_with_bullet_rew;
    float hit_by_enemy_bullet_penalty_rew;
    int enemy_crossed_screen;
} Log;

// LogBuffer structure
typedef struct LogBuffer {
    Log* logs;
    int length;
    int idx;
} LogBuffer;

typedef struct RewardBuffer {
    float* rewards; // Sliding window for reward smoothing
    int size;       // Size of the buffer
    int idx;        // Current index in the buffer
} RewardBuffer;

// Bullet structure
typedef struct Bullet {
    float x, y;
    float last_x, last_y;
    bool active;
    double travel_time;
} Bullet;

// Enemy structure
typedef struct Enemy {
    float x, y;
    float last_x, last_y;
    bool active;
    bool attacking;
    int direction; // Movement direction (-1, 0, 1)
    int width;
    int height;
    int crossed_screen;
    Bullet bullet;
} Enemy;

// Player structure
typedef struct Player {
    float x, y;
    float last_x, last_y;
    int score;
    int lives;
    Bullet bullet;
    bool bulletFired;
    bool playerStuck; // Player status (stuck in beam or not)
    float explosion_timer; // Timer for player explosion effect
} Player;

// Blastar environment structure
typedef struct BlastarEnv {
    int screen_width;
    int screen_height;
    float player_width;
    float player_height;
    float enemy_width;
    float enemy_height;
    float player_bullet_width;
    float player_bullet_height;
    float enemy_bullet_width;
    float enemy_bullet_height;
    float last_bullet_distance;
    bool game_over;
    int tick;
    int playerExplosionTimer;  // Timer for player explosion effect
    int enemyExplosionTimer;   // Timer for enemy explosion effect
    int max_score;
    int bullet_travel_time;
    bool bullet_crossed_enemy_y; // Reset on bullet deactivation
    int kill_streak;
    Player player;
    Enemy enemy;               // Singular enemy
    Bullet bullet;
    RewardBuffer* reward_buffer;
    // RL fields
    float* observations;       // [6]
    int* actions;              // [1]
    float* rewards;            // [1]
    unsigned char* terminals;  // [1]
    LogBuffer* log_buffer;
    Log log;
} BlastarEnv;


// Function declarations

// Log buffer functions
LogBuffer* allocate_logbuffer(int size);
void free_logbuffer(LogBuffer* buffer);
void add_log(LogBuffer* logs, Log* log);
Log aggregate_and_clear(LogBuffer* logs);

// Reward buffer
RewardBuffer* allocate_reward_buffer(int size);
void free_reward_buffer(RewardBuffer* buffer);

// RL memory allocation
void allocate_env(BlastarEnv* env);
void free_allocated_env(BlastarEnv* env);

// Initialization, reset, and cleanup
void init_blastar(BlastarEnv* env);
void reset_blastar(BlastarEnv* env);
void close_blastar(BlastarEnv* env);

// Observation computation
void compute_observations(BlastarEnv* env);

// RL step function
void c_step(BlastarEnv* env);

#endif // BLASTAR_ENV_H