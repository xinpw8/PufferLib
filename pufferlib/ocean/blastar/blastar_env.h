// blastar_env.h
#ifndef BLASTAR_ENV_H
#define BLASTAR_ENV_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h> // for calloc, free
#include "raylib.h"

// Include the renderer header
#include "blastar_renderer.h"

#define MAX_BULLETS 10
#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
#define LOG_BUFFER_SIZE 1024

// Struct Definitions
typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float score;
};

typedef struct LogBuffer LogBuffer;
struct LogBuffer {
    Log* logs;
    int length;
    int idx;
};

// Bullet structure
typedef struct {
    float x, y;
    bool active;
} Bullet;

// Enemy structure
typedef struct {
    float x, y;
    bool active;
    bool attacking;
    int direction;
    int width;
    int height;
    Bullet bullets[MAX_BULLETS];
} Enemy;

// Player structure
typedef struct {
    float x, y;
    int score;
    int lives;
    Bullet bullets[MAX_BULLETS];
    bool bulletFired;
    bool playerStuck;
} Player;

// BlastarEnv structure
typedef struct BlastarEnv {
    bool gameOver;
    int tick;
    int playerExplosionTimer;  // Timer for player explosion
    int enemyExplosionTimer;   // Timer for enemy explosion
    Player player;
    Enemy enemy;

    // RL fields
    float* observations;       // [6]
    int* actions;              // [1]
    float* rewards;            // [1]
    unsigned char* terminals;  // [1]
    LogBuffer* log_buffer;
    Log log;
} BlastarEnv;

// Function declarations
LogBuffer* allocate_logbuffer(int size);
void free_logbuffer(LogBuffer* buffer);
void add_log(LogBuffer* logs, Log* log);
Log aggregate_and_clear(LogBuffer* logs);

// RL allocation
void allocate_env(BlastarEnv* env);
void free_allocated_env(BlastarEnv* env);

// Initialization, reset, close
void init_blastar(BlastarEnv *env);
void reset_blastar(BlastarEnv *env);
void close_blastar(BlastarEnv* env);

// Compute observations
void compute_observations(BlastarEnv* env);

// Combined step function
void c_step(BlastarEnv *env);

#endif // BLASTAR_ENV_H
