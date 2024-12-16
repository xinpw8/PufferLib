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
        log.lives += logs->logs[i].lives;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    log.lives /= logs->idx;
    logs->idx = 0;
    return log;
}

// RL allocation
void allocate_env(BlastarEnv* env) {
    if (env) {
        env->observations = (float*)calloc(22, sizeof(float));
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
        env->game_over = false;
        env->tick = 0;
        env->playerExplosionTimer = 0;
        env->enemyExplosionTimer = 0;
        env->screen_height = SCREEN_HEIGHT;
        env->screen_width = SCREEN_WIDTH;


        // Initialize player
        env->player.x = SCREEN_WIDTH / 2 - 17 / 2;
        env->player.y = SCREEN_HEIGHT - 60;
        env->player_width = 17;
        env->player_height = 17;
        env->player.last_x = env->player.x;
        env->player.last_y = env->player.y;
        env->player.score = 0;
        env->player.lives = 5;
        env->player.bulletFired = false;
        env->player.playerStuck = false;
        env->player.bullet.active = false;
        env->player.bullet.x = env->player.x;
        env->player.bullet.y = env->player.y;
        env->player.bullet.last_x = env->player.bullet.x;
        env->player.bullet.last_y = env->player.bullet.y;

        // Initialize enemy
        env->enemy.x = -30;
        env->enemy.y = 50;
        env->enemy_width = 16;
        env->enemy_height = 17;
        env->enemy.last_x = env->enemy.x;
        env->enemy.last_y = env->enemy.y;
        env->enemy.active = true;
        env->enemy.attacking = false;
        env->enemy.direction = 1;
        env->enemy.width = 16;
        env->enemy.height = 17;
        env->enemy.bullet.active = false;
        env->enemy.bullet.x = env->enemy.x;
        env->enemy.bullet.y = env->enemy.y;
        env->enemy.bullet.last_x = env->enemy.bullet.x;
        env->enemy.bullet.last_y = env->enemy.bullet.y;

        env->player_bullet_width = 17;
        env->player_bullet_height = 6;
        env->enemy_bullet_width = 10;
        env->enemy_bullet_height = 12;
    }
}

void reset_blastar(BlastarEnv *env) {
    init_blastar(env);
    env->log = (Log){0};
}

void close_blastar(BlastarEnv* env) {
    free_allocated_env(env);
}

void compute_observations(BlastarEnv* env) {
    if (env && env->observations) {
        // Normalize player and enemy positions
        env->observations[0] = env->player.x / SCREEN_WIDTH;   // Normalized player x
        env->observations[1] = env->player.y / SCREEN_HEIGHT; // Normalized player y
        env->observations[2] = env->enemy.x / SCREEN_WIDTH;   // Normalized enemy x
        env->observations[3] = env->enemy.y / SCREEN_HEIGHT;  // Normalized enemy y

        // Player bullet location and status
        env->observations[4] = env->player.bullet.active ? env->player.bullet.x / SCREEN_WIDTH : 0.0f; // Normalized x
        env->observations[5] = env->player.bullet.active ? env->player.bullet.y / SCREEN_HEIGHT : 0.0f; // Normalized y
        env->observations[6] = env->player.bullet.active ? -3.0f / -3.0f : 0.0f; // Player bullet speed normalized (-3.0 is hardcoded speed)

        // Bullet closeness to enemy (Euclidean distance)
        if (env->player.bullet.active) {
            float dx = env->player.bullet.x - env->enemy.x;
            float dy = env->player.bullet.y - env->enemy.y;
            float distance = sqrtf(dx * dx + dy * dy);
            // Normalize the distance to [0, 1]
            env->observations[22] = 1.0f - (distance / sqrtf(SCREEN_WIDTH * SCREEN_WIDTH + SCREEN_HEIGHT * SCREEN_HEIGHT));
        } else {
            env->observations[22] = 0.0f; // No bullet
        }

        // Enemy bullet location and status
        bool enemyBulletActive = false;
        if (env->enemy.bullet.active) {
            env->observations[7] = env->enemy.bullet.x / SCREEN_WIDTH;   // Normalized x
            env->observations[8] = env->enemy.bullet.y / SCREEN_HEIGHT; // Normalized y
            env->observations[9] = 2.0f / 2.0f; // Enemy bullet speed normalized (2.0 is hardcoded speed)
            enemyBulletActive = true;
        }
        if (!enemyBulletActive) {
            env->observations[7] = 0.0f; // No active enemy bullet
            env->observations[8] = 0.0f;
            env->observations[9] = 0.0f;
        }

        // Additional observations for player score and lives
        env->observations[10] = env->player.score / 10.0f;  // Score normalized to [0, 1] assuming max score 100
        env->observations[11] = env->player.lives / 5.0f;    // Lives normalized to [0, 1] assuming max lives 5

        // Enemy speed
        env->observations[12] = 1.0f / 2.0f; // Enemy speed normalized (1.0 is hardcoded speed)

        // Player speed
        env->observations[13] = 2.0f / 2.0f; // Player speed normalized (2.0 is hardcoded speed)

        // Enemy last known position
        env->observations[14] = env->enemy.last_x / SCREEN_WIDTH;   // Normalized enemy x
        env->observations[15] = env->enemy.last_y / SCREEN_HEIGHT;  // Normalized enemy y

        // Player last known position
        env->observations[16] = env->player.last_x / SCREEN_WIDTH;   // Normalized player x
        env->observations[17] = env->player.last_y / SCREEN_HEIGHT;  // Normalized player y

        // Enemy bullet last location
        env->observations[18] = env->enemy.bullet.active ? env->enemy.bullet.last_x / SCREEN_WIDTH : 0.0f; // Normalized x
        env->observations[19] = env->enemy.bullet.active ? env->enemy.bullet.last_y / SCREEN_HEIGHT : 0.0f; // Normalized y

        // Player bullet last location
        env->observations[20] = env->player.bullet.active ? env->player.bullet.last_x / SCREEN_WIDTH : 0.0f; // Normalized x
        env->observations[21] = env->player.bullet.active ? env->player.bullet.last_y / SCREEN_HEIGHT : 0.0f; // Normalized y
    }
}

// Combined step function
void c_step(BlastarEnv *env) {
    if (env == NULL) return;

    if (!env->actions || !env->rewards || !env->terminals) {
        // If RL arrays not allocated, run the game logic without RL info
    }

    if (env->game_over) {
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
            env->player.bullet.active = false;
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

    // Last enemy and player positions
    env->enemy.last_x = env->enemy.x;
    env->enemy.last_y = env->enemy.y;
    env->player.last_x = env->player.x;
    env->player.last_y = env->player.y;

    // Player movement if not stuck
    if (!env->player.playerStuck) {
        if (action == 1 && env->player.x > 0) env->player.x -= playerSpeed;
        if (action == 2 && env->player.x < SCREEN_WIDTH - 17) env->player.x += playerSpeed;
        if (action == 3 && env->player.y > 0) env->player.y -= playerSpeed;
        if (action == 4 && env->player.y < SCREEN_HEIGHT - 17) env->player.y += playerSpeed;
    }

    // Fire player bullet
    if (action == 5 && !env->player.playerStuck) {
        // Reward for firing a bullet
        env->rewards[0] = 0.0001f;
        env->log.episode_return += 0.0001f;

        // If a bullet is already active
        if (env->player.bullet.active) {
            env->player.bullet.active = false;
            env->rewards[0] = -0.01f; // Bullet can't hit enemy if constantly replacing w/ new bullet
        }
        env->player.bullet.active = true;
        env->player.bullet.x = env->player.x;
        env->player.bullet.y = env->player.y;
    }

    // Update player bullet
    if (env->player.bullet.active) {
        // Reward for keeping the bullet in the air
        float bullet_reward_per_step = 0.01f;
        env->rewards[0] += bullet_reward_per_step; 
        env->log.episode_return += bullet_reward_per_step;

        // Update bullet position
        env->player.bullet.y -= playerBulletSpeed;

        // If the bullet goes off-screen, deactivate it
        if (env->player.bullet.y < 0) {
            env->player.bullet.active = false;
        }
    }

    if (env->player.bullet.active) {
    // Calculate the distance between the bullet and the enemy
    float dx = env->player.bullet.x - env->enemy.x;
    float dy = env->player.bullet.y - env->enemy.y;
    float distance = sqrtf(dx * dx + dy * dy);
    
    // Invert and normalize the distance to get a reward signal
    float max_distance = sqrtf(SCREEN_WIDTH * SCREEN_WIDTH + SCREEN_HEIGHT * SCREEN_HEIGHT);
    float distance_reward = (max_distance - distance) / max_distance;
    
    // Apply the distance-based reward
    env->rewards[0] += 0.01f * distance_reward; // Modify distance-to-enemy reward here
    env->log.episode_return += 0.01f * distance_reward;

    // Update bullet position
    env->player.bullet.y -= playerBulletSpeed;

        // If the bullet goes off-screen, deactivate it
        if (env->player.bullet.y < 0) {
            env->player.bullet.active = false;
        }
    }

    // Player bullet hits enemy
    if (env->player.bullet.active) {
        Rectangle bulletHitbox = {env->player.bullet.x, env->player.bullet.y, 17, 6};
        Rectangle enemyHitbox = {env->enemy.x, env->enemy.y, env->enemy.width, env->enemy.height};
        if (CheckCollisionRecs(bulletHitbox, enemyHitbox) && env->enemy.active) {
            env->player.bullet.active = false; // Deactivate the bullet
            env->enemy.active = false; // Mark enemy as inactive
            env->enemyExplosionTimer = 60; // Start explosion animation

            // Calculate the total travel reward for the bullet
            float total_travel_reward = (env->screen_height / playerBulletSpeed) * 0.01f;

            // Apply reward: total travel reward + bonus for hitting enemy
            float hit_bonus = 1.0f; // Bonus for hitting the enemy
            env->rewards[0] += total_travel_reward + hit_bonus;
            env->log.episode_return += total_travel_reward + hit_bonus;

            env->player.score++; // Increment player's score
            env->log.score += 1.0f;
        }
    }


    // Last player bullet location
    env->player.bullet.last_x = env->player.bullet.x;
    env->player.bullet.last_y = env->player.bullet.y;

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
            if (!env->enemy.bullet.active) {
                env->enemy.bullet.active = true;
                env->enemy.bullet.x = enemyCenterX - 5.0f;
                env->enemy.bullet.y = env->enemy.y + env->enemy.height;
                // Disable active player bullet
                env->player.bullet.active = false;
                // Player stuck
                env->player.playerStuck = true;
            }
    }

    // Update enemy bullets
        if (env->enemy.bullet.active) {
            env->enemy.bullet.y += enemyBulletSpeed;
            if (env->enemy.bullet.y > SCREEN_HEIGHT) {
                env->enemy.bullet.active = false;
                env->player.playerStuck = false;
                env->enemy.attacking = false;
            }
    }

    // Last enemy bullet location
    env->enemy.bullet.last_x = env->enemy.bullet.x;
    env->enemy.bullet.last_y = env->enemy.bullet.y;

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
            env->game_over = true;
            if (env->terminals) env->terminals[0] = 1;
            add_log(env->log_buffer, &env->log);
        }
        goto compute_obs;
    }

    // Player bullet hits enemy
    if (env->player.bullet.active) {
        Rectangle bulletHitbox = {env->player.bullet.x, env->player.bullet.y, 17, 6};
        if (CheckCollisionRecs(bulletHitbox, enemyHitbox) && env->enemy.active) {
            env->player.bullet.active = false;
            env->enemy.active = false;
            env->player.score++;
            env->log.score += 1.0f;
            env->enemyExplosionTimer = 60;
            env->rewards[0] = 1.0f; // Reward for hitting enemy
            env->log.episode_return += 1.0f;
            add_log(env->log_buffer, &env->log);
        }
    }

    // Enemy bullet hits player
    if (env->enemy.bullet.active) {
        Rectangle bulletHitbox = {env->enemy.bullet.x, env->enemy.bullet.y, 10, 12};
        if (CheckCollisionRecs(bulletHitbox, playerHitbox)) {
            env->enemy.bullet.active = false;
            env->player.lives--;
            env->log.lives = env->player.lives;
            env->playerExplosionTimer = 60;
            env->player.playerStuck = false;
            env->enemy.attacking = false;
            env->enemy.x = -env->enemy.width;
            env->enemy.y = rand() % (SCREEN_HEIGHT - env->enemy.height);

            env->rewards[0] = -1.0f;
            if (env->log_buffer) env->log.episode_return -= 1.0f;

            if (env->player.lives <= 0) {
                env->player.lives = 0;
                env->game_over = true;
                if (env->terminals) env->terminals[0] = 1;
                add_log(env->log_buffer, &env->log);
            }
        }
    }

compute_obs:
    // Compute observations after step
    compute_observations(env);
}
