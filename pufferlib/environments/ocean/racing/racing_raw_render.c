#include "racing.h"
#include <stdio.h>
#include "raylib.h"

// Function declarations
void render_sky(CRacing* env);
void render_background(CRacing* env);
void render_left_road(CRacing* env);
void render_right_road(CRacing* env);
void render_car(CRacing* env);
void render_scoreboard(CRacing* env);

// Debugging function to print key press info
void print_debug_info(CRacing* env) {
    if (IsKeyDown(KEY_LEFT)) {
        printf("Key LEFT pressed\n");
    }
    if (IsKeyDown(KEY_RIGHT)) {
        printf("Key RIGHT pressed\n");
    }
    printf("Car position: X = %f\n", env->car_x);  // Display the car's X position
}

// Function to update the game state (step function)
void step(CRacing* env) {
    // Handle input for moving the car only horizontally
    if (IsKeyDown(KEY_LEFT)) {
        env->car_x -= env->car_speed;
        // Left boundary check: stop at the left wall
        if (env->car_x < 47) env->car_x = 47;  // 37 is the left boundary of the road
    }
    if (IsKeyDown(KEY_RIGHT)) {
        env->car_x += env->car_speed;
        // Right boundary check: stop at the right wall
        if (env->car_x > 120) env->car_x = 120;  // 137 is the right boundary of the road
    }
}

// Main render function
void render(Client* client, CRacing* env) {
    BeginDrawing();
    ClearBackground(BLACK);  // Clear the previous frame

    render_sky(env);         // Render sky
    render_background(env);  // Render grass (background)
    render_left_road(env);   // Render left road
    render_right_road(env);  // Render right road
    render_car(env);         // Render the player's car
    render_scoreboard(env);  // Render scoreboard

    EndDrawing();
}

// Render the sky (blue from y=0 to y=51)
void render_sky(CRacing* env) {
    Color darkBlue = {0, 128, 128};  // Darker blue
    for (int i = 0; i <= 51; i++) {
        for (int j = 8; j < env->width; j++) {
            DrawPixel(j, i, darkBlue);
        }
    }
}

// Render the grass (green from y=52 to y=155)
void render_background(CRacing* env) {
    Color darkGreen = {100, 100, 0};  // Darker green
    for (int i = 52; i <= 155; i++) {
        for (int j = 8; j < env->width; j++) {
            DrawPixel(j, i, darkGreen);
        }
    }
}

// Render the left side of the road
void render_left_road(CRacing* env) {
    int leftRoadPixels[][2] = {
        {87, 52}, {87, 53}, {87, 54}, {86, 55}, {86, 56}, {86, 57},
        {85, 58}, {85, 59}, {85, 60}, {84, 61}, {84, 62}, {83, 63},
        {83, 64}, {82, 65}, {81, 66}, {81, 67}, {80, 68}, {79, 69},
        {79, 70}, {78, 71}, {78, 72}, {77, 73}, {77, 74}, {76, 75},
        {76, 76}, {75, 77}, {75, 78}, {74, 79}, {74, 80}, {73, 81},
        {73, 82}, {72, 83}, {72, 84}, {71, 85}, {71, 86}, {70, 87},
        {70, 88}, {69, 89}, {69, 90}, {68, 91}, {68, 92}, {67, 93},
        {67, 94}, {66, 95}, {66, 96}, {65, 97}, {65, 98}, {64, 99},
        {64, 100}, {63, 101}, {63, 102}, {62, 103}, {62, 104}, {61, 105},
        {61, 106}, {60, 107}, {60, 108}, {59, 109}, {59, 110}, {58, 111},
        {58, 112}, {57, 113}, {57, 114}, {56, 115}, {56, 116}, {55, 117},
        {55, 118}, {54, 119}, {54, 120}, {53, 121}, {53, 122}, {52, 123},
        {52, 124}, {51, 125}, {51, 126}, {50, 127}, {50, 128}, {49, 129},
        {49, 130}, {48, 131}, {48, 132}, {47, 133}, {47, 134}, {46, 135},
        {46, 136}, {45, 137}, {45, 138}, {44, 139}, {44, 140}, {43, 141},
        {43, 142}, {42, 143}, {42, 144}, {41, 145}, {41, 146}, {40, 147},
        {40, 148}, {39, 149}, {39, 150}, {38, 151}, {38, 152}, {37, 153},
        {37, 154}
    };

    // Render the left side of the road (no movement on Y-axis anymore)
    for (int i = 0; i < sizeof(leftRoadPixels) / sizeof(leftRoadPixels[0]); i++) {
        int newY = leftRoadPixels[i][1];
        int newX = leftRoadPixels[i][0];
        if (newY >= 52 && newY <= 155 && newX >= 8 && newX <= 160) {  // Only draw within the valid game area
            DrawPixel(newX, newY, WHITE);
        }
    }
}

// Render the right side of the road
void render_right_road(CRacing* env) {
    int rightRoadPixels[][2] = {
        {88, 54}, {88, 55}, {89, 56}, {89, 57}, {90, 58}, {90, 59}, {90, 60},
        {91, 61}, {91, 62}, {92, 63}, {92, 64}, {93, 65}, {94, 66}, {94, 67},
        {95, 68}, {95, 69}, {96, 70}, {96, 71}, {97, 72}, {97, 73}, {98, 74},
        {98, 75}, {99, 76}, {99, 77}, {100, 78}, {100, 79}, {101, 80}, {101, 81},
        {102, 82}, {102, 83}, {103, 84}, {103, 85}, {104, 86}, {104, 87}, {105, 88},
        {105, 89}, {106, 90}, {106, 91}, {107, 92}, {107, 93}, {108, 94}, {108, 95},
        {109, 96}, {109, 97}, {110, 98}, {110, 99}, {111, 100}, {111, 101}, {112, 102},
        {112, 103}, {113, 104}, {113, 105}, {114, 106}, {114, 107}, {115, 108}, {115, 109},
        {116, 110}, {116, 111}, {117, 112}, {117, 113}, {118, 114}, {118, 115}, {119, 116},
        {119, 117}, {120, 118}, {120, 119}, {121, 120}, {121, 121}, {122, 122}, {122, 123},
        {123, 124}, {123, 125}, {124, 126}, {124, 127}, {125, 128}, {125, 129}, {126, 130},
        {126, 131}, {127, 132}, {127, 133}, {128, 134}, {128, 135}, {129, 136}, {129, 137},
        {130, 138}, {130, 139}, {131, 140}, {131, 141}, {132, 142}, {132, 143}, {133, 144},
        {133, 145}, {134, 146}, {134, 147}, {135, 148}, {135, 149}, {136, 150}, {136, 151},
        {137, 152}, {137, 153}, {138, 154}
    };

    // Render the right side of the road (no movement on Y-axis anymore)
    for (int i = 0; i < sizeof(rightRoadPixels) / sizeof(rightRoadPixels[0]); i++) {
        int newY = rightRoadPixels[i][1];
        int newX = rightRoadPixels[i][0];
        if (newY >= 52 && newY <= 155 && newX >= 8 && newX <= 160) {  // Only draw within the valid game area
            DrawPixel(newX, newY, WHITE);
        }
    }
}

// Render the player's car (fully, including both halves)
void render_car(CRacing* env) {
    int carPixels[][2] = {
        {env->car_x - 6, 147}, {env->car_x - 6, 149}, {env->car_x - 6, 151}, {env->car_x - 6, 153},
        {env->car_x - 5, 147}, {env->car_x - 5, 149}, {env->car_x - 5, 151}, {env->car_x - 5, 153},
        {env->car_x - 4, 144}, {env->car_x - 4, 145}, {env->car_x - 4, 146}, {env->car_x - 4, 148}, 
        {env->car_x - 4, 150}, {env->car_x - 4, 152}, {env->car_x - 4, 154},
        {env->car_x - 3, 144}, {env->car_x - 3, 145}, {env->car_x - 3, 146}, {env->car_x - 3, 148}, 
        {env->car_x - 3, 150}, {env->car_x - 3, 152}, {env->car_x - 3, 154},
        {env->car_x - 2, 145}, {env->car_x - 2, 146}, {env->car_x - 2, 148}, {env->car_x - 2, 149}, 
        {env->car_x - 2, 150}, {env->car_x - 2, 151}, {env->car_x - 2, 152}, {env->car_x - 2, 153},
        {env->car_x - 1, 145}, {env->car_x - 1, 146}, {env->car_x - 1, 148}, {env->car_x - 1, 149}, 
        {env->car_x - 1, 150}, {env->car_x - 1, 151}, {env->car_x - 1, 152}, {env->car_x - 1, 153},
        // {env->car_x, 144}, {env->car_x, 145}, {env->car_x, 146}, {env->car_x, 147}, 
        // {env->car_x, 148}, {env->car_x, 149}, {env->car_x, 150}, {env->car_x, 151}, 
        // {env->car_x, 152}, {env->car_x, 153}, {env->car_x, 154},
        {env->car_x + 1, 144}, {env->car_x + 1, 145}, {env->car_x + 1, 146}, {env->car_x + 1, 147}, 
        {env->car_x + 1, 148}, {env->car_x + 1, 149}, {env->car_x + 1, 150}, {env->car_x + 1, 151}, 
        {env->car_x + 1, 152}, {env->car_x + 1, 153}, {env->car_x + 1, 154},
        {env->car_x + 2, 144}, {env->car_x + 2, 145}, {env->car_x + 2, 146}, {env->car_x + 2, 147}, 
        {env->car_x + 2, 148}, {env->car_x + 2, 149}, {env->car_x + 2, 150}, {env->car_x + 2, 151}, 
        {env->car_x + 2, 152}, {env->car_x + 2, 153}, {env->car_x + 2, 154},
        {env->car_x + 3, 144}, {env->car_x + 3, 145}, {env->car_x + 3, 146}, {env->car_x + 3, 147}, 
        {env->car_x + 3, 148}, {env->car_x + 3, 149}, {env->car_x + 3, 150}, {env->car_x + 3, 151}, 
        {env->car_x + 3, 152}, {env->car_x + 3, 153}, {env->car_x + 3, 154},
        {env->car_x + 4, 144}, {env->car_x + 4, 145}, {env->car_x + 4, 146}, {env->car_x + 4, 147}, 
        {env->car_x + 4, 148}, {env->car_x + 4, 149}, {env->car_x + 4, 150}, {env->car_x + 4, 151}, 
        {env->car_x + 4, 152}, {env->car_x + 4, 153}, {env->car_x + 4, 154},
        {env->car_x + 5, 144}, {env->car_x + 5, 145}, {env->car_x + 5, 146}, {env->car_x + 5, 147}, 
        {env->car_x + 5, 148}, {env->car_x + 5, 149}, {env->car_x + 5, 150}, {env->car_x + 5, 151}, 
        {env->car_x + 5, 152}, {env->car_x + 5, 153}, {env->car_x + 5, 154},
{env->car_x - 6, 147}, {env->car_x - 6, 149}, {env->car_x - 6, 151}, {env->car_x - 6, 153},
        {env->car_x - 5, 147}, {env->car_x - 5, 149}, {env->car_x - 5, 151}, {env->car_x - 5, 153},
        {env->car_x - 4, 144}, {env->car_x - 4, 145}, {env->car_x - 4, 146}, {env->car_x - 4, 148}, 
        {env->car_x - 4, 150}, {env->car_x - 4, 152}, {env->car_x - 4, 154},
        {env->car_x - 3, 144}, {env->car_x - 3, 145}, {env->car_x - 3, 146}, {env->car_x - 3, 148}, 
        {env->car_x - 3, 150}, {env->car_x - 3, 152}, {env->car_x - 3, 154},
        {env->car_x - 2, 145}, {env->car_x - 2, 146}, {env->car_x - 2, 148}, {env->car_x - 2, 149}, 
        {env->car_x - 2, 150}, {env->car_x - 2, 151}, {env->car_x - 2, 152}, {env->car_x - 2, 153},
        {env->car_x - 1, 145}, {env->car_x - 1, 146}, {env->car_x - 1, 148}, {env->car_x - 1, 149}, 
        {env->car_x - 1, 150}, {env->car_x - 1, 151}, {env->car_x - 1, 152}, {env->car_x - 1, 153},
        // {env->car_x, 144}, {env->car_x, 145}, {env->car_x, 146}, {env->car_x, 147}, 
        {env->car_x, 148}, {env->car_x, 149}, {env->car_x, 150}, {env->car_x, 151}, 
        // {env->car_x, 152}, {env->car_x, 153}, {env->car_x, 154},
        {env->car_x + 1, 144}, {env->car_x + 1, 145}, {env->car_x + 1, 146}, {env->car_x + 1, 147}, 
        {env->car_x + 1, 148}, {env->car_x + 1, 149}, {env->car_x + 1, 150}, {env->car_x + 1, 151}, 
        {env->car_x + 1, 152}, {env->car_x + 1, 153}, {env->car_x + 1, 154},
        {env->car_x + 2, 144}, {env->car_x + 2, 145}, {env->car_x + 2, 146}, {env->car_x + 2, 147}, 
        {env->car_x + 2, 148}, {env->car_x + 2, 149}, {env->car_x + 2, 150}, {env->car_x + 2, 151}, 
        {env->car_x + 2, 152}, {env->car_x + 2, 153}, {env->car_x + 2, 154},
        {env->car_x + 3, 144}, {env->car_x + 3, 145}, {env->car_x + 3, 146}, {env->car_x + 3, 147}, 
        {env->car_x + 3, 148}, {env->car_x + 3, 149}, {env->car_x + 3, 150}, {env->car_x + 3, 151}, 
        {env->car_x + 3, 152}, {env->car_x + 3, 153}, {env->car_x + 3, 154},
        {env->car_x + 4, 144}, {env->car_x + 4, 145}, {env->car_x + 4, 146}, {env->car_x + 4, 147}, 
        {env->car_x + 4, 148}, {env->car_x + 4, 149}, {env->car_x + 4, 150}, {env->car_x + 4, 151}, 
        {env->car_x + 4, 152}, {env->car_x + 4, 153}, {env->car_x + 4, 154},
        {env->car_x + 5, 144}, {env->car_x + 5, 145}, {env->car_x + 5, 146}, {env->car_x + 5, 147}, 
        {env->car_x + 5, 148}, {env->car_x + 5, 149}, {env->car_x + 5, 150}, {env->car_x + 5, 151}, 
        {env->car_x + 5, 152}, {env->car_x + 5, 153}, {env->car_x + 5, 154},
        {env->car_x + 6, 146}, {env->car_x + 6, 147}, {env->car_x + 6, 148}, {env->car_x + 6, 149}, 
        {env->car_x + 6, 150}, {env->car_x + 6, 151}, {env->car_x + 6, 152}, {env->car_x + 6, 153},
        {env->car_x + 6, 154}, {env->car_x + 7, 144}, {env->car_x + 7, 145}, {env->car_x + 7, 146}, 
        {env->car_x + 7, 147}, {env->car_x + 7, 148}, {env->car_x + 7, 149}, {env->car_x + 7, 150},
        {env->car_x + 7, 151}, {env->car_x + 7, 152}, {env->car_x + 7, 153}, {env->car_x + 7, 154},
        {env->car_x + 8, 145}, {env->car_x + 8, 146}, {env->car_x + 8, 148}, {env->car_x + 8, 149},
        {env->car_x + 8, 150}, {env->car_x + 8, 151}, {env->car_x + 8, 152}, {env->car_x + 8, 153},
        {env->car_x + 9, 145}, {env->car_x + 9, 146}, {env->car_x + 9, 148}, {env->car_x + 9, 149},
        {env->car_x + 9, 150}, {env->car_x + 9, 151}, {env->car_x + 9, 152}, {env->car_x + 9, 153},
        {env->car_x + 10, 144}, {env->car_x + 10, 145}, {env->car_x + 10, 146}, {env->car_x + 10, 147},
        {env->car_x + 10, 149}, {env->car_x + 10, 151}, {env->car_x + 10, 153}, {env->car_x + 11, 144},
        {env->car_x + 11, 145}, {env->car_x + 11, 146}, {env->car_x + 11, 147}, {env->car_x + 11, 149},
        {env->car_x + 11, 151}, {env->car_x + 11, 153}, {env->car_x + 12, 148}, {env->car_x + 12, 150},
        {env->car_x + 12, 152}, {env->car_x + 12, 154}, {env->car_x + 13, 148}, {env->car_x + 13, 150},
        {env->car_x + 13, 152}, {env->car_x + 13, 154}
    };

    // Draw the car
    for (int i = 0; i < sizeof(carPixels) / sizeof(carPixels[0]); i++) {
        DrawPixel(carPixels[i][0], carPixels[i][1], WHITE);
    }
}

// Render the scoreboard
void render_scoreboard(CRacing* env) {
    // Example scoreboard rendering with a red rectangle
    for (int i = 160; i < 210; i++) {
        for (int j = 8; j < 152; j++) {
            DrawPixel(j, i, RED);
        }
    }
}

int main() {
    const int screenWidth = 160;
    const int screenHeight = 210;

    InitWindow(screenWidth, screenHeight, "Enduro Racing");

    CRacing env = {
        .car_x = 105,       // Set car X position to 105
        .car_y = 97,        // Set car Y position to 97 (doesn't matter since car won't move vertically)
        .car_speed = 2,     // Movement speed
        .width = screenWidth,
        .height = screenHeight
    };

    SetTargetFPS(60);

    Client client = {screenWidth, screenHeight};

    while (!WindowShouldClose()) {
        print_debug_info(&env);  // Show car position for debugging
        step(&env);              // Update car position
        render(&client, &env);   // Render the updated scene
    }

    CloseWindow();

    return 0;
}
