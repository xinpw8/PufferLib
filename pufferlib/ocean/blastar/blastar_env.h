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
#define LOG_BUFFER_SIZE 1024
#define MAX_EPISODE_STEPS 81920
#define PLAYER_MAX_LIVES 5

// Log structure
typedef struct Log {
    float episode_return;
    float episode_length;
    float score;
    float lives;
} Log;

// LogBuffer structure
typedef struct LogBuffer {
    Log* logs;
    int length;
    int idx;
} LogBuffer;

// Bullet structure
typedef struct Bullet {
    float x, y;
    float last_x, last_y;
    bool active;
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
    bool game_over;
    int tick;
    int playerExplosionTimer;  // Timer for player explosion effect
    int enemyExplosionTimer;   // Timer for enemy explosion effect
    int max_score;
    int bullet_travel_time;
    Player player;
    Enemy enemy;               // Singular enemy
    Bullet bullet;
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