#include "raylib.h"

#define PLAYABLE_AREA_TOP 0
#define PLAYABLE_AREA_BOTTOM 154
#define PLAYABLE_AREA_LEFT 8
#define PLAYABLE_AREA_RIGHT 160
#define SCREEN_WIDTH 160
#define SCREEN_HEIGHT 208

#define ACTION_HEIGHT (PLAYABLE_AREA_BOTTOM - PLAYABLE_AREA_TOP)
#define HORIZON_Y 51
#define MOUNTAIN_HEIGHT 6

int main(void)
{
    // Initialization
    //--------------------------------------------------------------------------------------

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Mountain Shapes with Sprites");

    SetTargetFPS(60);
    //--------------------------------------------------------------------------------------

    // Load mountain sprites
    Texture2D background = LoadTexture("resources/enduro_clone/0_bg.png");
    Texture2D mountains = LoadTexture("resources/enduro_clone/0_mtns.png");

    // Set positions for the mountains
    Vector2 backgroundPos = { 0, 0}; // Adjust as needed
    Vector2 mountainsPos = { PLAYABLE_AREA_LEFT+30, (PLAYABLE_AREA_TOP + HORIZON_Y - MOUNTAIN_HEIGHT)}; // Adjust as needed

    // Main game loop
    while (!WindowShouldClose())
    {
        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

        ClearBackground(RAYWHITE);

        // Draw playable area boundary
        DrawRectangleLines(
            PLAYABLE_AREA_LEFT,
            PLAYABLE_AREA_TOP,
            PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT,
            PLAYABLE_AREA_BOTTOM - PLAYABLE_AREA_TOP,
            LIGHTGRAY);

        // Draw mountain sprites
        DrawTexture(background, backgroundPos.x, backgroundPos.y, WHITE);
        DrawTexture(mountains, mountainsPos.x, mountainsPos.y, WHITE);

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    UnloadTexture(background);
    UnloadTexture(mountains);
    CloseWindow();
    //--------------------------------------------------------------------------------------

    return 0;
}
