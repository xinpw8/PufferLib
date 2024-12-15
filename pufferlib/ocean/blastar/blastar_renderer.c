// blastar_renderer.c
#include "blastar_env.h"
#include "blastar_renderer.h"
#include <stdlib.h> // For calloc and free
#include <stdio.h>  // For fprintf

// Initialize the renderer
void init_renderer(BlastarRenderer *renderer) {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Blastar");
    SetTargetFPS(60);

    // Load textures for game assets
    renderer->playerTexture = LoadTexture("pufferlib/resources/blastar/player_ship.png");
    renderer->enemyTexture = LoadTexture("pufferlib/resources/blastar/enemy_ship.png");
    renderer->playerBulletTexture = LoadTexture("pufferlib/resources/blastar/player_bullet.png");
    renderer->enemyBulletTexture = LoadTexture("pufferlib/resources/blastar/enemy_bullet.png");
    renderer->playerDeathExplosion = LoadTexture("pufferlib/resources/blastar/player_death_explosion.png");
}

// Render the Blastar environment
void render_blastar(const BlastarRenderer *renderer, const BlastarEnv *env) {
    BeginDrawing();
    ClearBackground(BLACK);

    if (env->gameOver) {
        DrawText("GAME OVER", SCREEN_WIDTH / 2 - 60, SCREEN_HEIGHT / 2 - 10, 30, RED);
        DrawText(TextFormat("FINAL SCORE: %d", env->player.score), SCREEN_WIDTH / 2 - 80, SCREEN_HEIGHT / 2 + 30, 20, GREEN);
        EndDrawing();
        return;
    }

    // Draw player or explosion on player death
    if (env->playerExplosionTimer > 0) {
        DrawTexture(renderer->playerDeathExplosion, env->player.x, env->player.y, WHITE);
    } else if (env->player.lives > 0) {
        DrawTexture(renderer->playerTexture, env->player.x, env->player.y, WHITE);
    }

    // Draw enemy or explosion on enemy death
    if (env->enemyExplosionTimer > 0) {
        DrawTexture(renderer->playerDeathExplosion, env->enemy.x, env->enemy.y, WHITE);
    } else if (env->enemy.active) {
        DrawTexture(renderer->enemyTexture, env->enemy.x, env->enemy.y, WHITE);
    }

    // Draw player bullet
    if (env->player.bullets[0].active) {
        DrawTexture(renderer->playerBulletTexture, env->player.bullets[0].x, env->player.bullets[0].y, WHITE);
    }

    // Draw enemy bullets
    for (int i = 0; i < 10; i++) { // Replaced MAX_BULLETS with 10
        if (env->enemy.bullets[i].active) {
            DrawTexture(renderer->enemyBulletTexture, env->enemy.bullets[i].x, env->enemy.bullets[i].y, WHITE);
        }
    }

    // Draw status beam indicator
    if (env->player.playerStuck) {
        DrawText("Status Beam", SCREEN_WIDTH - 150, SCREEN_HEIGHT / 3, 20, RED);
    }

    // Draw score and lives
    DrawText(TextFormat("SCORE %d", env->player.score), 10, 10, 20, GREEN);
    DrawText(TextFormat("LIVES %d", env->player.lives), SCREEN_WIDTH - 100, 10, 20, GREEN);

    EndDrawing();
}

// Close the renderer and unload textures
void close_renderer(BlastarRenderer *renderer) {
    // Unload all textures
    UnloadTexture(renderer->playerTexture);
    UnloadTexture(renderer->enemyTexture);
    UnloadTexture(renderer->playerBulletTexture);
    UnloadTexture(renderer->enemyBulletTexture);
    UnloadTexture(renderer->playerDeathExplosion);
    CloseWindow();
}

Client* make_client(BlastarEnv* env) {
    printf("make_client called with env at %p\n", (void*)env);
    Client* client = (Client*)calloc(1, sizeof(Client));
    if (client == NULL) {
        fprintf(stderr, "Failed to allocate memory for Client.\n");
        return NULL;
    }
    init_renderer(&client->renderer);
    printf("Client created with renderer at %p\n", (void*)&client->renderer);
    return client;
}

// Close and free a Client instance
void close_client(Client* client) {
    if (client) {
        close_renderer(&client->renderer);
        free(client);
    }
}

// Render the Blastar environment using the Client
void render_blastar_client(Client* client, BlastarEnv* env) {
    if (client && env) {
        printf("Rendering BlastarEnv at %p using Client at %p\n", (void*)env, (void*)client);
        render_blastar(&client->renderer, env);
    } else {
        printf("Client or Env is NULL.\n");
    }
}
