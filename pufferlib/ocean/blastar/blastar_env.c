// blastar_env.c
#include "blastar_env.h"
#include <stdlib.h>

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

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
    //printf("Log: %f, %f, %f\n", log->episode_return, log->episode_length, log->score);
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
        log.score += logs->logs[i].score;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    logs->idx = 0;
    return log;
}

// RL allocation
void allocate_env(BlastarEnv* env) {
    if (env) {
        env->observations = (float*)calloc(12, sizeof(float));
        env->actions = (int*)calloc(1, sizeof(int));
        env->rewards = (float*)calloc(1, sizeof(float));
        env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
        env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
    }
}

void free_allocated_env(BlastarEnv* env) {
    if (env) {
        if (env->observations) {
            free(env->observations);
            env->observations = NULL;
        }
        if (env->actions) {
            free(env->actions);
            env->actions = NULL;
        }
        if (env->rewards) {
            free(env->rewards);
            env->rewards = NULL;
        }
        if (env->terminals) {
            free(env->terminals);
            env->terminals = NULL;
        }
        if (env->log_buffer) {
            free_logbuffer(env->log_buffer);
            env->log_buffer = NULL;
        }
    }
}

// Initialization, reset, close
void init_blastar(BlastarEnv *env) {
    if (env) {
        env->gameOver = false;
        env->tick = 0;
        env->playerExplosionTimer = 0;
        env->enemyExplosionTimer = 0;

        // Initialize player
        env->player.x = SCREEN_WIDTH / 2 - 17 / 2;
        env->player.y = SCREEN_HEIGHT - 60;
        env->player.score = 0;
        env->player.lives = 5;
        env->player.bulletFired = false;
        env->player.playerStuck = false;
        for (int i = 0; i < MAX_BULLETS; i++) {
            env->player.bullets[i].active = false;
        }

        // Initialize enemy
        env->enemy.x = -30;
        env->enemy.y = 50;
        env->enemy.active = true;
        env->enemy.attacking = false;
        env->enemy.direction = 1;
        env->enemy.width = 16;
        env->enemy.height = 17;
        for (int i = 0; i < MAX_BULLETS; i++) {
            env->enemy.bullets[i].active = false;
        }

        // memset(&env->log, 0, sizeof(Log)); // Initialize log
    }
}

void reset_blastar(BlastarEnv *env) {
    if (env) {
        init_blastar(env);
        if (env->log_buffer) {
            env->log = (Log){0};
        }
    }
}

void close_blastar(BlastarEnv* env) {
    if (env) {
        free_allocated_env(env);
    }
}

void compute_observations(BlastarEnv* env) {
    if (env && env->observations) {
        // Normalize player and enemy positions
        env->observations[0] = env->player.x / SCREEN_WIDTH;   // Normalized player x
        env->observations[1] = env->player.y / SCREEN_HEIGHT; // Normalized player y
        env->observations[2] = env->enemy.x / SCREEN_WIDTH;   // Normalized enemy x
        env->observations[3] = env->enemy.y / SCREEN_HEIGHT;  // Normalized enemy y

        // Player bullet location and status
        env->observations[4] = env->player.bullets[0].active ? env->player.bullets[0].x / SCREEN_WIDTH : 0.0f; // Normalized x
        env->observations[5] = env->player.bullets[0].active ? env->player.bullets[0].y / SCREEN_HEIGHT : 0.0f; // Normalized y
        env->observations[6] = env->player.bullets[0].active ? -3.0f / 10.0f : 0.0f; // Player bullet speed normalized (-3.0 is hardcoded speed)

        // Enemy bullet location and status
        bool enemyBulletActive = false;
        for (int i = 0; i < MAX_BULLETS; i++) {
            if (env->enemy.bullets[i].active) {
                env->observations[7] = env->enemy.bullets[i].x / SCREEN_WIDTH;   // Normalized x
                env->observations[8] = env->enemy.bullets[i].y / SCREEN_HEIGHT; // Normalized y
                env->observations[9] = 2.0f / 10.0f; // Enemy bullet speed normalized (2.0 is hardcoded speed)
                enemyBulletActive = true;
                break;
            }
        }
        if (!enemyBulletActive) {
            env->observations[7] = 0.0f; // No active enemy bullet
            env->observations[8] = 0.0f;
            env->observations[9] = 0.0f;
        }

        // Additional observations for player score and lives
        env->observations[10] = env->player.score / 100.0f;  // Score normalized to [0, 1] assuming max score 100
        env->observations[11] = env->player.lives / 5.0f;    // Lives normalized to [0, 1] assuming max lives 5
    }
}

// Combined step function
void c_step(BlastarEnv *env) {
    if (env == NULL) return;

    if (!env->actions || !env->rewards || !env->terminals) {
        // If RL arrays not allocated, run the game logic without RL info
    }

    if (env->gameOver) {
        if (env->terminals) env->terminals[0] = 1;
        add_log(env->log_buffer, &env->log);
        return;
    }

    env->tick++;
    if (env->log_buffer) env->log.episode_length += 1;

    float playerSpeed = 2.0f;
    float enemySpeed = 1.0f;
    float playerBulletSpeed = 3.0f;
    float enemyBulletSpeed = 2.0f;

    int action = 0;
    if (env->actions) action = env->actions[0];

    // Handle player explosion
    if (env->playerExplosionTimer > 0) {
        env->playerExplosionTimer--;
        if (env->playerExplosionTimer == 0) {
            env->player.playerStuck = false;
            env->player.bullets[0].active = false;
        }
        goto compute_obs; // Skip further logic while exploding
    }

    // Handle enemy explosion
    if (env->enemyExplosionTimer > 0) {
        env->enemyExplosionTimer--;
        if (env->enemyExplosionTimer == 0) {
            if (rand() % 2 == 0) {
                env->enemy.x = -env->enemy.width;
                env->enemy.y = rand() % (SCREEN_HEIGHT - env->enemy.height);
            }
            env->enemy.active = true;
            env->enemy.attacking = false;
        }
        goto compute_obs; // Skip further logic while exploding
    }

    // Player movement if not stuck
    if (!env->player.playerStuck) {
        if (action == 1 && env->player.x > 0) env->player.x -= playerSpeed;
        if (action == 2 && env->player.x < SCREEN_WIDTH - 17) env->player.x += playerSpeed;
        if (action == 3 && env->player.y > 0) env->player.y -= playerSpeed;
        if (action == 4 && env->player.y < SCREEN_HEIGHT - 17) env->player.y += playerSpeed;
    }

    // Fire player bullet
    if (action == 5 && !env->player.playerStuck) {
        // If a bullet is already active
        if (env->player.bullets[0].active) {
            env->player.bullets[0].active = false;
        }
        env->player.bullets[0].active = true;
        env->player.bullets[0].x = env->player.x;
        env->player.bullets[0].y = env->player.y;
    }

    // Update player bullet
    if (env->player.bullets[0].active) {
        env->player.bullets[0].y -= playerBulletSpeed;
        if (env->player.bullets[0].y < 0) {
            env->player.bullets[0].active = false;
        }
    }

    // Enemy movement
    if (!env->enemy.attacking) {
        env->enemy.x += enemySpeed;
        if (env->enemy.x > SCREEN_WIDTH) {
            env->enemy.x = -env->enemy.width; // Respawn off-screen
        }
    }

    float playerCenterX = env->player.x + 8.5f;
    float enemyCenterX = env->enemy.x + env->enemy.width / 2.0f;

    // Enemy attack logic
    if (fabs(playerCenterX - enemyCenterX) < 1.0f && !env->enemy.attacking && env->enemy.active) {
        env->enemy.attacking = true;
        for (int i = 0; i < MAX_BULLETS; i++) {
            if (!env->enemy.bullets[i].active) {
                env->enemy.bullets[i].active = true;
                env->enemy.bullets[i].x = enemyCenterX - 5.0f;
                env->enemy.bullets[i].y = env->enemy.y + env->enemy.height;
                // Disable active player bullet
                env->player.bullets[0].active = false;
                // Player stuck
                env->player.playerStuck = true;
                break;
            }
        }
    }

    // Update enemy bullets
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (env->enemy.bullets[i].active) {
            env->enemy.bullets[i].y += enemyBulletSpeed;
            if (env->enemy.bullets[i].y > SCREEN_HEIGHT) {
                env->enemy.bullets[i].active = false;
                env->player.playerStuck = false;
                env->enemy.attacking = false;
            }
        }
    }

    // Collision detection
    Rectangle playerHitbox = {env->player.x, env->player.y, 17, 17};
    Rectangle enemyHitbox = {env->enemy.x, env->enemy.y, env->enemy.width, env->enemy.height};

    // Player-Enemy Collision
    if (CheckCollisionRecs(playerHitbox, enemyHitbox)) {
        env->player.lives--;
        env->enemy.active = false;
        env->enemyExplosionTimer = 60;

        // Respawn enemy
        env->enemy.x = -env->enemy.width;
        env->enemy.y = rand() % (SCREEN_HEIGHT - env->enemy.height);

        env->playerExplosionTimer = 60;
        env->player.playerStuck = false;

        // Reward/Penalty
        if (env->rewards) {
            env->rewards[0] = -1.0f; // Penalty
            if (env->log_buffer) env->log.episode_return -= 1.0f;
        }

        if (env->player.lives <= 0) {
            env->player.lives = 0;
            env->gameOver = true;
            if (env->terminals) env->terminals[0] = 1;
            add_log(env->log_buffer, &env->log);
        }
        goto compute_obs;
    }

    // Player bullet hits enemy
    if (env->player.bullets[0].active) {
        Rectangle bulletHitbox = {env->player.bullets[0].x, env->player.bullets[0].y, 17, 6};
        if (CheckCollisionRecs(bulletHitbox, enemyHitbox) && env->enemy.active) {
            env->player.bullets[0].active = false;
            env->enemy.active = false;
            env->player.score++;
            env->enemyExplosionTimer = 60;
            env->rewards[0] = 1.0f; // Reward for hitting enemy
            if (env->log_buffer) env->log.episode_return += 1.0f;
            add_log(env->log_buffer, &env->log);
        }
    }

    // Enemy bullet hits player
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (env->enemy.bullets[i].active) {
            Rectangle bulletHitbox = {env->enemy.bullets[i].x, env->enemy.bullets[i].y, 10, 12};
            if (CheckCollisionRecs(bulletHitbox, playerHitbox)) {
                env->enemy.bullets[i].active = false;
                env->player.lives--;
                env->playerExplosionTimer = 60;
                env->player.playerStuck = false;
                env->enemy.attacking = false;
                env->enemy.x = -env->enemy.width;
                env->enemy.y = rand() % (SCREEN_HEIGHT - env->enemy.height);

                env->rewards[0] = -1.0f;
                if (env->log_buffer) env->log.episode_return -= 1.0f;

                if (env->player.lives <= 0) {
                    env->player.lives = 0;
                    env->gameOver = true;
                    if (env->terminals) env->terminals[0] = 1;
                    add_log(env->log_buffer, &env->log);
                }
                break;
            }
        }
    }

compute_obs:
    // Compute observations after step
    compute_observations(env);
}
