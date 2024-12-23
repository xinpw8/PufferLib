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
    if (buffer) {
        if (buffer->logs) {
            free(buffer->logs);
            buffer->logs = NULL;
        }
        free(buffer);
    }
}

void free_reward_buffer(RewardBuffer* buffer) {
    if (buffer) {
        if (buffer->rewards) {
            free(buffer->rewards);
            buffer->rewards = NULL;
        }
        free(buffer);
    }
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
        log.bullet_travel_rew += logs->logs[i].bullet_travel_rew;
        log.fired_bullet_rew += logs->logs[i].fired_bullet_rew;
        log.bullet_distance_to_enemy_rew += logs->logs[i].bullet_distance_to_enemy_rew;
        log.gradient_penalty_rew += logs->logs[i].gradient_penalty_rew;
        log.flat_below_enemy_rew += logs->logs[i].flat_below_enemy_rew;
        log.danger_zone_penalty_rew += logs->logs[i].danger_zone_penalty_rew;
        log.crashing_penalty_rew += logs->logs[i].crashing_penalty_rew;
        log.hit_enemy_with_bullet_rew += logs->logs[i].hit_enemy_with_bullet_rew;
        log.hit_by_enemy_bullet_penalty_rew += logs->logs[i].hit_by_enemy_bullet_penalty_rew;
        log.enemy_crossed_screen += logs->logs[i].enemy_crossed_screen;
        log.bad_guy_score += logs->logs[i].bad_guy_score;
        log.avg_score_difference += logs->logs[i].avg_score_difference;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    log.lives /= logs->idx;
    log.bullet_travel_rew /= logs->idx;
    log.fired_bullet_rew /= logs->idx;
    log.bullet_distance_to_enemy_rew /= logs->idx;
    log.gradient_penalty_rew /= logs->idx;
    log.flat_below_enemy_rew /= logs->idx;
    log.danger_zone_penalty_rew /= logs->idx;
    log.crashing_penalty_rew /= logs->idx;
    log.hit_enemy_with_bullet_rew /= logs->idx;
    log.hit_by_enemy_bullet_penalty_rew /= logs->idx;
    log.enemy_crossed_screen /= logs->idx;
    log.bad_guy_score /= logs->idx;
    log.avg_score_difference /= logs->idx;
    logs->idx = 0;
    return log;
}

RewardBuffer* allocate_reward_buffer(int size) {
    // assert(size > 0 && "Reward buffer size must be greater than 0.");
    RewardBuffer* buffer = (RewardBuffer*)calloc(1, sizeof(RewardBuffer));
    // assert(buffer != NULL && "Failed to allocate RewardBuffer.");
    buffer->rewards = (float*)calloc(size, sizeof(float));
    // assert(buffer->rewards != NULL && "Failed to allocate RewardBuffer's rewards array.");
    buffer->size = size;
    buffer->idx = 0;
    return buffer;
}

float update_and_get_smoothed_reward(RewardBuffer* buffer, float reward) {
    buffer->rewards[buffer->idx % buffer->size] = reward;
    buffer->idx++;

    float sum = 0.0f;
    int count = (buffer->idx < buffer->size) ? buffer->idx : buffer->size;

    for (int i = 0; i < count; i++) {
        sum += buffer->rewards[i];
    }

    return sum / count;
}

// RL allocation
void allocate_env(BlastarEnv* env) {
    if (env) {
        env->observations = (float*)calloc(31, sizeof(float));
        env->actions = (int*)calloc(1, sizeof(int));
        env->rewards = (float*)calloc(1, sizeof(float));
        env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
        env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
        env->reward_buffer = allocate_reward_buffer(REWARD_BUFFER_SIZE);
    }
}

int is_valid_pointer(void* ptr) {
    return ptr != NULL;
}

void free_allocated_env(BlastarEnv* env) {
    if (env) {
        if (is_valid_pointer(env->observations)) {
            free(env->observations);
            env->observations = NULL;
        }
        if (is_valid_pointer(env->actions)) {
            free(env->actions);
            env->actions = NULL;
        }
        if (is_valid_pointer(env->rewards)) {
            free(env->rewards);
            env->rewards = NULL;
        }
        if (is_valid_pointer(env->terminals)) {
            free(env->terminals);
            env->terminals = NULL;
        }
        if (is_valid_pointer(env->log_buffer)) {
            free_logbuffer(env->log_buffer);
            env->log_buffer = NULL;
        }
        if (is_valid_pointer(env->reward_buffer)) {
            free_reward_buffer(env->reward_buffer);
            env->reward_buffer = NULL;
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

        // Max score var
        env->max_score = 5 * PLAYER_MAX_LIVES;

        // Initialize player
        // Randomize player x position
        env->player.x = (float)(rand() % (SCREEN_WIDTH - 17));
        // Randomize player y position
        env->player.y = (float)(rand() % (SCREEN_HEIGHT - 17));
        env->player_width = 17;
        env->player_height = 17;
        env->player.last_x = env->player.x;
        env->player.last_y = env->player.y;
        env->player.score = 0;
        env->bad_guy_score = 0.0f;
        env->player.lives = PLAYER_MAX_LIVES;
        env->player.bulletFired = false;
        env->player.playerStuck = false;
        env->player.bullet.active = false;
        env->player.bullet.x = env->player.x;
        env->player.bullet.y = env->player.y;
        env->player.bullet.last_x = env->player.bullet.x;
        env->player.bullet.last_y = env->player.bullet.y;
        env->bullet_travel_time = 0;
        env->last_bullet_distance = 0;
        env->kill_streak = 0;

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

void reset_blastar(BlastarEnv* env) {
    if (!env) return;
    env->log = (Log){0};
    env->tick = 0;
    env->game_over = false;
    init_blastar(env);
}

void close_blastar(BlastarEnv* env) {
    free_allocated_env(env);
}

void compute_observations(BlastarEnv* env) {
    if (env && env->observations) {

        // // Infinite lives
        // if (env->player.lives < 5) {
        //     env->player.lives = 5;
        // }

        env->log.lives = env->player.lives;
        env->log.score = env->player.score;
        env->log.bad_guy_score = env->bad_guy_score;
        env->log.enemy_crossed_screen = env->enemy.crossed_screen;

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
        env->observations[10] = env->player.score / env->max_score;  // Score normalized to [0, 1] assuming max score 100
        env->observations[11] = env->player.lives / PLAYER_MAX_LIVES;    // Lives normalized to [0, 1]; MAX_LIVES is macro

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

        // Danger zone observation
        // Danger Zone (distance from player to enemy)
        float px = env->player.x + env->player_width / 2.0f;  // Player center
        float py = env->player.y + env->player_height / 2.0f;

        float ex = env->enemy.x + env->enemy_width / 2.0f;    // Enemy center
        float ey = env->enemy.y + env->enemy_height / 2.0f;

        float player_enemy_dx = px - ex;
        float player_enemy_dy = py - ey;
        float player_enemy_distance = sqrtf(player_enemy_dx * player_enemy_dx + player_enemy_dy * player_enemy_dy);

        // Normalize the player-enemy distance to [0, 1]
        float max_possible_distance = sqrtf(SCREEN_WIDTH * SCREEN_WIDTH + SCREEN_HEIGHT * SCREEN_HEIGHT);
        env->observations[23] = 1.0f - (player_enemy_distance / max_possible_distance);  // Closer = higher value
        // Danger zone flag: 1 if player too close, else 0
        float danger_threshold = 50.0f;  // Example threshold distance
        env->observations[24] = (player_enemy_distance < danger_threshold) ? 1.0f : 0.0f;
        
        // "Below enemy ship" observation: 1.0 if player is below enemy, 0.0 otherwise
        env->observations[25] = (env->player.y > env->enemy.y + env->enemy.height) ? 1.0f : 0.0f;

        // Enemy crossed screen observation (count)
        if (env->enemy.crossed_screen > 0 && env->player.score > 0) {
            env->observations[26] = (float)env->enemy.crossed_screen / (float)env->player.score;
        } else {
            env->observations[26] = 0.0f;
        }

        // Bad guy score minus player score observation
        // Player score observation normalized to [0, 1]
        // Bad guy score observation normalized to [0, 1]
        float total_score = (float)env->player.score + (float)env->bad_guy_score;
        if (total_score > 0.0f) {
            env->observations[27] = ((float)env->bad_guy_score - (float)env->player.score) / total_score;
            env->observations[28] = (float)env->player.score / total_score;
            env->observations[29] = (float)env->bad_guy_score / total_score;
        } else {
            env->observations[27] = 0.0f; // Default to zero if denominator is zero
            env->observations[28] = 0.0f; // Default to zero if denominator is zero
            env->observations[29] = 0.0f; // Default to zero if denominator is zero
        }

        // Enemy crossed screen observation normalized to [0, 1]
        if (total_score > 0.0f) {
            env->observations[30] = (float)env->enemy.crossed_screen / total_score;
        } else {
            env->observations[30] = 0.0f; // Default to zero if denominator is zero
        }
    }
}

// Combined step function
void c_step(BlastarEnv *env) {
    if (env == NULL) return;

    if (!env->actions || !env->rewards || !env->terminals) {
        // Empty
    }

    if (env->game_over) {
        if (env->terminals) env->terminals[0] = 1;
        add_log(env->log_buffer, &env->log);
        reset_blastar(env);
        return;
    }

    env->tick++;
    env->log.episode_length += 1;

    float speed_scale = 4.0f;
    float playerSpeed = 2.0f;
    float enemySpeed = 1.0f;
    float playerBulletSpeed = 3.0f;
    float enemyBulletSpeed = 2.0f;

    playerSpeed *= speed_scale;
    enemySpeed *= speed_scale;
    playerBulletSpeed *= speed_scale;
    enemyBulletSpeed *= speed_scale;

    // Zero out rewards and env variables
    float rew = 0.0f;
    env->rewards[0] = rew;
    float score = 0.0f;
    float bad_guy_score = 0.0f;
    float fired_bullet_rew = 0.0f;
    float bullet_travel_rew = 0.0f;
    float bullet_distance_to_enemy_rew = 0.0f;
    float gradient_penalty_rew = 0.0f;
    float flat_below_enemy_rew = 0.0f;
    float danger_zone_penalty_rew = 0.0f;
    float crashing_penalty_rew = 0.0f;
    float hit_enemy_with_bullet_rew = 0.0f;
    float hit_by_enemy_bullet_penalty_rew = 0.0f;
    int crossed_screen = 0;
    float flat_reward = 0.0f;
    int action = 0;
    action = env->actions[0];

    // Handle player explosion
    if (env->playerExplosionTimer > 0) {
        env->playerExplosionTimer--;
        env->kill_streak = 0;
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
            env->enemy.crossed_screen = 0;
            // Rarely respawn in the same place
            float respawn_bias = 0.1f; // 10% chance to respawn in the same place
            if ((float)rand() / (float)RAND_MAX > respawn_bias) {
                // Respawn in a new position
                env->enemy.x = -env->enemy.width;
                env->enemy.y = rand() % (SCREEN_HEIGHT - env->enemy.height);
            }
            // Otherwise, respawn in the same place as a rare event
            env->enemy.active = true;
            env->enemy.attacking = false;
        }
        goto compute_obs; // Skip further logic while exploding
    }

    // Keep enemy far enough from bottom of the screen
    if (env->enemy.y > (SCREEN_HEIGHT - (env->enemy.height * 3.5f))) {
        env->enemy.y = (SCREEN_HEIGHT - (env->enemy.height * 3.5f));
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
    if (action == 5 && (!env->enemy.bullet.active)) {
        // If a bullet is already active, replace it with the new one
        if (env->player.bullet.active) {
            env->player.bullet.active = false; // Deactivate the existing bullet
        } else {
            // Reward for firing a single bullet, if it hits enemy
            fired_bullet_rew += 0.002f;
        }

        // Activate and initialize the new bullet
        env->player.bullet.active = true;
        env->player.bullet.x = env->player.x + env->player_width / 2 - env->player_bullet_width / 2;
        env->player.bullet.y = env->player.y;
    }

    // Update player bullet
    if (env->player.bullet.active) {
        // Update bullet position
        env->player.bullet.y -= playerBulletSpeed;

        // Deactivate bullet if off-screen
        if (env->player.bullet.y < 0) {
            env->player.bullet.active = false;
            env->bullet_travel_time = 0;
        }
    }

    float playerCenterX = env->player.x + env->player_width / 2.0f;
    float enemyCenterX = env->enemy.x + env->enemy.width / 2.0f;

    // Last player bullet location
    env->player.bullet.last_x = env->player.bullet.x;
    env->player.bullet.last_y = env->player.bullet.y;

    // Enemy movement
    if (!env->enemy.attacking) {
        env->enemy.x += enemySpeed;
        if (env->enemy.x > SCREEN_WIDTH) {
            env->enemy.x = -env->enemy.width; // Respawn off-screen
            crossed_screen += 1.0f;
        }
    }

    // Enemy attack logic
    if (fabs(playerCenterX - enemyCenterX) < speed_scale && !env->enemy.attacking && env->enemy.active && env->enemy.y < env->player.y - (env->enemy_height / 2)) {
        // 50% chance of attacking
        if (rand() % 2 == 0) {
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
        } else {
            env->enemy.attacking = false;
            env->enemy.x += enemySpeed; // Avoid attack lock
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
        env->enemyExplosionTimer = 30;

        // Respawn enemy
        env->enemy.x = -env->enemy.width;
        env->enemy.y = rand() % (SCREEN_HEIGHT - env->enemy.height);

        env->playerExplosionTimer = 30;
        env->player.playerStuck = false;

        if (env->player.lives <= 0) {
            env->player.lives = 0;
            env->game_over = true;
            if (env->terminals) env->terminals[0] = 1;
            // env->rewards[0] = rew;
            compute_observations(env);
            add_log(env->log_buffer, &env->log);
            reset_blastar(env);
        }
        goto compute_obs;
    }

    // Player bullet hits enemy
    if (env->player.bullet.active && env->player.y > env->enemy.y + env->enemy.height) {
        Rectangle bulletHitbox = {env->player.bullet.x, env->player.bullet.y, 17, 6};
        if (CheckCollisionRecs(bulletHitbox, enemyHitbox) && env->enemy.active) {
            env->player.bullet.active = false;
            env->enemy.active = false;
            env->kill_streak += 1;
            fired_bullet_rew += 1.5f;
            env->player.score += 1.0f;
            env->log.score += 1.0f;
            env->enemyExplosionTimer = 30;
            if (crossed_screen == 0) {
                hit_enemy_with_bullet_rew += 2.5f; // Big reward for quick kill
            } else {
                hit_enemy_with_bullet_rew += 1.5f - (0.1f * env->enemy.crossed_screen); // Less rew if enemy crossed screen
            }
        } else {
        }
    }

    // Enemy bullet hits player
    if (env->enemy.bullet.active) {
        Rectangle bulletHitbox = {env->enemy.bullet.x, env->enemy.bullet.y, 10, 12};
        if (CheckCollisionRecs(bulletHitbox, playerHitbox)) {
            env->enemy.bullet.active = false;
            env->player.lives--;
            bad_guy_score += 1.0f;
            env->playerExplosionTimer = 30;
            env->player.playerStuck = false;
            env->enemy.attacking = false;
            env->enemy.x = -env->enemy.width;
            env->enemy.y = rand() % (SCREEN_HEIGHT - env->enemy.height);

            if (env->player.lives <= 0) {
                env->player.lives = 0;
                env->game_over = true;
                if (env->terminals) env->terminals[0] = 1;
                // env->rewards[0] = rew;
                compute_observations(env);
                add_log(env->log_buffer, &env->log);
                reset_blastar(env);
            }
        }
    }
    if (env->player.y > env->enemy.y + env->enemy.height) {
        flat_reward = 0.01f; // Flat positive reward for being below the enemy            
    } else {
        flat_reward = -0.01f; // Penalty for being above the enemy
    }

    if (env->player.y > env->enemy.y + env->enemy.height) { 
        float vertical_closeness = 1.0f - ((env->player.y - env->enemy.y) / SCREEN_HEIGHT);
        rew += 0.01f * vertical_closeness; 
    }

    env->log.score += score;
    env->log.bad_guy_score += bad_guy_score;
    env->bad_guy_score += bad_guy_score;

    float avg_score_difference = 0.0f;
    if (env->player.score + env->bad_guy_score > 0) {
        int score_difference = env->player.score - env->bad_guy_score;
        avg_score_difference = (float)score_difference / (env->player.score + env->bad_guy_score);
    }

    env->log.avg_score_difference = avg_score_difference;
    env->log.fired_bullet_rew = fired_bullet_rew;
    env->log.bullet_travel_rew = bullet_travel_rew;
    env->log.bullet_distance_to_enemy_rew = bullet_distance_to_enemy_rew;
    env->log.gradient_penalty_rew = gradient_penalty_rew;
    env->log.flat_below_enemy_rew = flat_below_enemy_rew;
    env->log.danger_zone_penalty_rew = danger_zone_penalty_rew;
    env->log.crashing_penalty_rew = crashing_penalty_rew;
    env->log.hit_enemy_with_bullet_rew = hit_enemy_with_bullet_rew;
    env->log.hit_by_enemy_bullet_penalty_rew = hit_by_enemy_bullet_penalty_rew;
    env->log.flat_below_enemy_rew = flat_reward;
    env->enemy.crossed_screen = crossed_screen;

    // Reward player only if below enemy
    if (env->player.y > env->enemy.y + env->enemy.height) {
        rew += score + fired_bullet_rew + bullet_travel_rew + bullet_distance_to_enemy_rew +
            flat_below_enemy_rew + hit_enemy_with_bullet_rew - danger_zone_penalty_rew;
        rew *= (1.0f + env->kill_streak * 0.1f);  // Reward scaling based on kill streak
        env->rewards[0] = rew;
        env->log.episode_return += rew;
    } else {
        env->rewards[0] = 0.0f; // No reward if above enemy
        env->log.episode_return += 0.0f;
    }

    if (env->bad_guy_score > 100.0f) {
        // env->player.lives = 0;
        env->game_over = true;
        env->terminals[0] = 1;
        compute_observations(env);
        add_log(env->log_buffer, &env->log);
        reset_blastar(env);
    }

compute_obs:
    compute_observations(env);
    add_log(env->log_buffer, &env->log);
}
