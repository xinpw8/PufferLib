#include "racing.h"
#include <stdio.h>
#include <math.h>  // For fmod()
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
    if (IsKeyDown(KEY_UP)) {
        printf("Key UP pressed\n");
    }
    if (IsKeyDown(KEY_DOWN)) {
        printf("Key DOWN pressed\n");
    }
    if (IsKeyDown(KEY_LEFT)) {
        printf("Key LEFT pressed\n");
    }
    if (IsKeyDown(KEY_RIGHT)) {
        printf("Key RIGHT pressed\n");
    }
    printf("Car position: X = %f, Y = %f\n", env->car_x, env->car_y);  // Use %f for floats
}

// Function to update the game state (step function)
void step(CRacing* env) {
    // Handle input for moving the car or the environment
    if (IsKeyDown(KEY_UP)) {
        env->road_taper += env->car_speed;  // Move the road under the car (vertically)
    }
    if (IsKeyDown(KEY_DOWN)) {
        env->road_taper -= env->car_speed;  // Move the road in the opposite direction (vertically)
    }
    if (IsKeyDown(KEY_LEFT)) {
        env->road_shift += env->car_speed;  // Move the road right (car moves left)
        if (env->road_shift > 50) env->road_shift = 50;  // Limit to how far the road can shift
    }
    if (IsKeyDown(KEY_RIGHT)) {
        env->road_shift -= env->car_speed;  // Move the road left (car moves right)
        if (env->road_shift < -50) env->road_shift = -50;  // Limit to how far the road can shift
    }
}

// Main render function
void render(Client* client, CRacing* env) {
    BeginDrawing();
    ClearBackground(BLACK);

    render_sky(env);
    render_background(env);
    render_left_road(env);
    render_right_road(env);  // Now rendering the right side of the road
    render_car(env);
    render_scoreboard(env);

    EndDrawing();
}

// Render the sky (blue from y=0 to y=51)
void render_sky(CRacing* env) {
    Color darkBlue = {0, 0, 128};  // Darker blue
    for (int i = 0; i <= 51; i++) {
        for (int j = 8; j < env->width; j++) {
            DrawPixel(j, i, darkBlue);
        }
    }
}

// Render the grass (green from y=52 to y=155)
void render_background(CRacing* env) {
    Color darkGreen = {0, 100, 0};  // Darker green
    for (int i = 52; i <= 155; i++) {
        for (int j = 8; j < env->width; j++) {
            DrawPixel(j, i, darkGreen);
        }
    }
}

// Render the left side of the road with proper tapering and horizontal shift
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

    int road_y_offset = fmod(env->road_taper, 160);  // Use fmod for float modulus
    int road_x_shift = env->road_shift;  // Horizontal shift

    // Render left side of the road with tapering and shifting
    for (int i = 0; i < sizeof(leftRoadPixels) / sizeof(leftRoadPixels[0]); i++) {
        int newY = leftRoadPixels[i][1] + road_y_offset;
        int newX = leftRoadPixels[i][0] + road_x_shift;  // Apply horizontal shift
        if (newY >= 52 && newY <= 155 && newX >= 8 && newX <= 160) {  // Only draw within the valid game area
            DrawPixel(newX, newY, WHITE);
        }
    }
}

// Render the right side of the road with horizontal shift
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

    int road_y_offset = fmod(env->road_taper, 160);  // Use fmod for float modulus
    int road_x_shift = env->road_shift;  // Horizontal shift

    // Render right side of the road with tapering and shifting
    for (int i = 0; i < sizeof(rightRoadPixels) / sizeof(rightRoadPixels[0]); i++) {
        int newY = rightRoadPixels[i][1] + road_y_offset;
        int newX = rightRoadPixels[i][0] + road_x_shift;  // Apply horizontal shift
        if (newY >= 52 && newY <= 155 && newX >= 8 && newX <= 160) {  // Only draw within the valid game area
            DrawPixel(newX, newY, WHITE);
        }
    }
}

// Render the player's car (fully, including both halves)
void render_car(CRacing* env) {
    int carPixels[][2] = {
        {77, 147}, {77, 149}, {77, 151}, {77, 153}, {78, 147}, {78, 149}, {78, 151}, {78, 153},
        {79, 144}, {79, 145}, {79, 146}, {79, 148}, {79, 150}, {79, 152}, {79, 154},
        {80, 144}, {80, 145}, {80, 146}, {80, 148}, {80, 150}, {80, 152}, {80, 154},
        {81, 145}, {81, 146}, {81, 148}, {81, 149}, {81, 150}, {81, 151}, {81, 152}, {81, 153},
        {82, 145}, {82, 146}, {82, 148}, {82, 149}, {82, 150}, {82, 151}, {82, 152}, {82, 153},
        {83, 144}, {83, 145}, {83, 146}, {83, 147}, {83, 148}, {83, 149}, {83, 150}, {83, 151},
        {83, 152}, {83, 153}, {83, 154}, {84, 144}, {84, 145}, {84, 146}, {84, 147}, {84, 148},
        {84, 149}, {84, 150}, {84, 151}, {84, 152}, {84, 153}, {84, 154}, {85, 144}, {85, 145},
        {85, 146}, {85, 147}, {85, 148}, {85, 149}, {85, 150}, {85, 151}, {85, 152}, {85, 153},
        {85, 154}, {86, 144}, {86, 145}, {86, 146}, {86, 147}, {86, 148}, {86, 149}, {86, 150},
        {86, 151}, {86, 152}, {86, 153}, {86, 154}, {87, 145}, {87, 146}, {87, 148}, {87, 149},
        {87, 150}, {87, 151}, {87, 152}, {87, 153}, {88, 145}, {88, 146}, {88, 148}, {88, 149},
        {88, 150}, {88, 151}, {88, 152}, {88, 153}, {89, 144}, {89, 145}, {89, 146}, {89, 147},
        {89, 149}, {89, 151}, {89, 153}, {90, 144}, {90, 145}, {90, 146}, {90, 147}, {90, 149},
        {90, 151}, {90, 153}, {91, 148}, {91, 150}, {91, 152}, {91, 154}, {92, 148}, {92, 150},
        {92, 152}, {92, 154}
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
        .car_y = 97,        // Set car Y position to 97
        .car_speed = 2,     // Movement speed
        .road_taper = 0,    // Initial taper (vertical shift)
        .road_shift = 0,    // Initial shift (horizontal shift)
        .width = screenWidth,
        .height = screenHeight
    };

    SetTargetFPS(60);

    Client client = {screenWidth, screenHeight};

    while (!WindowShouldClose()) {
        print_debug_info(&env);
        step(&env);
        render(&client, &env);  // Pass both the client and env structs
    }

    CloseWindow();

    return 0;
}
