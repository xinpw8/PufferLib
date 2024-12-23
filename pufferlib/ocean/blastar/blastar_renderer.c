#include "blastar_env.h"
#include "blastar_renderer.h"
#include <stdlib.h> // For calloc and free
#include <stdio.h>  // For fprintf

#include <stdio.h> // For printf and fprintf

// debug
#include <unistd.h> // For getcwd
#include <stdio.h>  // For printf

// Initialize the renderer with debugging
Client* make_client(BlastarEnv* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));

    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Current working directory: %s\n", cwd);
    } else {
        perror("getcwd() error");
    }
    
    // Set screen dimensions
    client->screen_width = env->screen_width;
    client->screen_height = env->screen_height;
    
    printf("Initializing window: %fx%f\n", client->screen_width, client->screen_height);
    InitWindow(client->screen_width, client->screen_height, "Blastar");
    SetTargetFPS(60);

    // Debugging: Attempt to load textures
    printf("Attempting to load textures:\n");

    client->player_texture = LoadTexture("./pufferlib/resources/blastar/player_ship.png");
    if (client->player_texture.id == 0) {
        fprintf(stderr, "Failed to load texture: player_ship.png\n");
    } else {
        printf("Successfully loaded texture: player_ship.png\n");
    }

    client->enemy_texture = LoadTexture("./pufferlib/resources/blastar/enemy_ship.png");
    if (client->enemy_texture.id == 0) {
        fprintf(stderr, "Failed to load texture: enemy_ship.png\n");
    } else {
        printf("Successfully loaded texture: enemy_ship.png\n");
    }

    client->player_bullet_texture = LoadTexture("./pufferlib/resources/blastar/player_bullet.png");
    if (client->player_bullet_texture.id == 0) {
        fprintf(stderr, "Failed to load texture: player_bullet.png\n");
    } else {
        printf("Successfully loaded texture: player_bullet.png\n");
    }

    client->enemy_bullet_texture = LoadTexture("./pufferlib/resources/blastar/enemy_bullet.png");
    if (client->enemy_bullet_texture.id == 0) {
        fprintf(stderr, "Failed to load texture: enemy_bullet.png\n");
    } else {
        printf("Successfully loaded texture: enemy_bullet.png\n");
    }

    client->explosion_texture = LoadTexture("./pufferlib/resources/blastar/player_death_explosion.png");
    if (client->explosion_texture.id == 0) {
        fprintf(stderr, "Failed to load texture: player_death_explosion.png\n");
    } else {
        printf("Successfully loaded texture: player_death_explosion.png\n");
    }

    // Set default colors
    client->player_color = WHITE;
    client->enemy_color = WHITE;
    client->bullet_color = WHITE;
    client->explosion_color = WHITE;

    client->player_width = 17;
    client->player_height = 17;
    client->enemy_width = 16;
    client->enemy_height = 17;

    return client;
}

// Close and free a Client instance
void close_client(Client* client) {
    CloseWindow();
    free(client);
}

// Render the Blastar environment
void c_render(Client* client, BlastarEnv* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(BLACK);

    if (env->game_over) {
        DrawText("GAME OVER", client->screen_width / 2 - 60, client->screen_height / 2 - 10, 30, RED);
        DrawText(TextFormat("FINAL SCORE: %d", env->player.score), client->screen_width / 2 - 80, client->screen_height / 2 + 30, 20, GREEN);
        EndDrawing();
        return;
    }

    // Draw player or explosion on player death
    if (env->playerExplosionTimer > 0) {
        DrawTexture(client->explosion_texture, env->player.x, env->player.y, client->explosion_color);
    } else if (env->player.lives > 0) {
        DrawTexture(client->player_texture, env->player.x, env->player.y, client->player_color);
    }

    // Draw enemy or explosion on enemy death
    if (env->enemyExplosionTimer > 0) {
        DrawTexture(client->explosion_texture, env->enemy.x, env->enemy.y, client->explosion_color);
    } else if (env->enemy.active) {
        DrawTexture(client->enemy_texture, env->enemy.x, env->enemy.y, client->enemy_color);
    }

    // Draw player bullet
    if (env->player.bullet.active) {
        DrawTexture(client->player_bullet_texture, env->player.bullet.x, env->player.bullet.y, client->bullet_color);
    }

    // Draw enemy bullet
    if (env->enemy.bullet.active) {
        DrawTexture(client->enemy_bullet_texture, env->enemy.bullet.x, env->enemy.bullet.y, client->bullet_color);
    }

    // Draw status beam indicator
    if (env->player.playerStuck) {
        DrawText("Status Beam", client->screen_width - 150, client->screen_height / 3, 20, RED);
    }

    // Draw score and lives
    DrawText(TextFormat("BAD GUY SCORE %d", (int)env->bad_guy_score), 240, 10, 20, GREEN);
    DrawText(TextFormat("PLAYER SCORE %d", env->player.score), 10, 10, 20, GREEN);
    DrawText(TextFormat("LIVES %d", env->player.lives), client->screen_width - 100, 10, 20, GREEN);

    EndDrawing();
}

// Close the renderer and unload textures
void close_renderer(Client *client) {
    UnloadTexture(client->player_texture);
    UnloadTexture(client->enemy_texture);
    UnloadTexture(client->player_bullet_texture);
    UnloadTexture(client->enemy_bullet_texture);
    UnloadTexture(client->explosion_texture);
    CloseWindow();
}
