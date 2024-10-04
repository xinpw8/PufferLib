// racing.c
#include "racing.h"

typedef struct EnemyCar {
    float rear_bumper_y;
    int lane;
    int active;
} EnemyCar;

typedef struct CEnduro CEnduro;
struct CEnduro {
    float* observations;
    unsigned char* actions;
    float* rewards;
    unsigned char* terminals;
    float* player_x_y;
    float* other_cars_x_y;
    int* other_cars_active;
    unsigned int* score_day;
    int frameskip;

    float width;
    float height;
    float player_width;
    float player_height;
    float other_car_width;
    float other_car_height;
    float player_speed;
    float base_car_speed;
    float max_player_speed;
    float min_player_speed;
    float speed_increment;
    int max_score;
    float front_bumper_y;
    float left_distance_to_edge;
    float right_distance_to_edge;
    float speed;

    EnemyCar enemy_cars[NUM_ENEMY_CARS];
    int dayNumber;
    int carsRemainingLo;
    int carsRemainingHi;
    int throttleValue;
    float reward;
    int done;


    // Internal game state
    int player_lane;
    float score;
    int cars_passed;
    int day;
    int cars_to_pass;
    int num_other_cars;
};

void init_env(CEnduro *env) {
    env->front_bumper_y = 0.0f;
    env->left_distance_to_edge = (ROAD_WIDTH - CAR_WIDTH) / 2.0f;
    env->right_distance_to_edge = (ROAD_WIDTH - CAR_WIDTH) / 2.0f;
    env->speed = MIN_SPEED;

    for (int i = 0; i < NUM_ENEMY_CARS; i++) {
        env->enemy_cars[i].rear_bumper_y = rand() % (int)(MAX_Y_POSITION) + 100.0f;
        env->enemy_cars[i].lane = rand() % 3; // Lanes: 0, 1, 2
        env->enemy_cars[i].active = true;
    }

    env->reward = 0.0f;
    env->done = false;
    env->dayNumber = 0;  // Initialize the day number
    env->carsRemainingLo = 200;  // Initialize Cars Passed counter for Day 1
}

void allocate(CEnduro* env) {
    env->observations = (float*)calloc(5, sizeof(float));
    env->actions = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->player_x_y = (float*)calloc(2, sizeof(float));
    env->other_cars_x_y = (float*)calloc(2 * NUM_ENEMY_CARS, sizeof(float));
    env->other_cars_active = (int*)calloc(NUM_ENEMY_CARS, sizeof(int));
    env->score_day = (unsigned int*)calloc(2, sizeof(unsigned int));
}

void free_initialized(CEnduro* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->player_x_y);
    free(env->other_cars_x_y);
    free(env->other_cars_active);
    free(env->score_day);
}

void free_allocated(CEnduro* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    free(env->player_x_y);
    free(env->other_cars_x_y);
    free(env->other_cars_active);
    free(env->score_day);
    free_initialized(env);
}

// Should follow this format
// void compute_observations(CEnduro* env) {
//     env->observations[0] = env->player_x_y[0];
//     env->observations[1] = env->player_x_y[1];
//     env->observations[0] = env->other_cars_x_y[0];
//     env->observations[1] = env->other_cars_x_y[1];
//     env->observations[2] = env->player_speed;
//     env->observations[3] = env->day;
//     env->observations[4] = env->cars_passed;
// }

void compute_observations(CEnduro* env) {
    int obs_index = 0;
    env->observations[obs_index++] = env->player_x_y[0];
    env->observations[obs_index++] = env->player_x_y[1];

    for (int i = 0; i < NUM_ENEMY_CARS; i++) {
        env->observations[obs_index++] = env->other_cars_x_y[2 * i];
        env->observations[obs_index++] = env->other_cars_x_y[2 * i + 1];
    }

    env->observations[obs_index++] = env->player_speed;
    env->observations[obs_index++] = (float)env->day;
    env->observations[obs_index++] = (float)env->cars_passed;
}


// Function to enforce enemy car spawn rules
void spawn_enemy_cars(CEnduro *env) {
    int num_active_cars = rand() % (NUM_ENEMY_CARS - 2) + 1;  // Allow 1-3 enemy cars

    int lanes[3] = { false, false, false };  // Track which lanes are occupied

    for (int i = 0; i < NUM_ENEMY_CARS; i++) {
        if (i < num_active_cars) {
            int lane;
            do {
                lane = rand() % 3;  // Randomly pick a lane
            } while (lanes[lane]);  // Ensure no duplicates in the same lane

            // Assign lane to car
            env->enemy_cars[i].lane = lane;
            lanes[lane] = true;  // Mark lane as occupied

            // Assign vertical position and ensure minimum distance between adjacent lane cars
            int valid_position;
            float rear_bumper_y;
            do {
                rear_bumper_y = rand() % (int)(MAX_Y_POSITION) + 100.0f;
                valid_position = true;

                // Ensure cars in adjacent lanes have enough vertical space
                for (int j = 0; j < i; j++) {
                    if (abs(env->enemy_cars[j].lane - lane) == 1 &&  // Adjacent lane check
                        fabs(env->enemy_cars[j].rear_bumper_y - rear_bumper_y) < MIN_DISTANCE_BETWEEN_CARS) {
                        valid_position = false;
                        break;
                    }
                }
            } while (!valid_position);

            env->enemy_cars[i].rear_bumper_y = rear_bumper_y;
            env->enemy_cars[i].active = true;
        } else {
            env->enemy_cars[i].active = false;  // Deactivate extra cars
        }
    }
}

void step(CEnduro *env) {
    // Read user input for controlling the car
    if (IsKeyDown(KEY_UP)) {
        env->actions[0] = ACTION_ACCEL;
    } else if (IsKeyDown(KEY_DOWN)) {
        env->actions[0] = ACTION_DECEL;
    } else if (IsKeyDown(KEY_LEFT)) {
        env->actions[0] = ACTION_LEFT;
    } else if (IsKeyDown(KEY_RIGHT)) {
        env->actions[0] = ACTION_RIGHT;
    } else {
        env->actions[0] = ACTION_NOOP;
    }

    // Update player's speed based on action
    if (env->actions[0] == ACTION_ACCEL) {
        env->speed += SPEED_INCREMENT;
        if (env->speed > MAX_SPEED)
            env->speed = MAX_SPEED;
    } else if (env->actions[0] == ACTION_DECEL) {
        env->speed -= SPEED_INCREMENT;
        if (env->speed < MIN_SPEED)
            env->speed = MIN_SPEED;
    }

    // Scale lateral movement speed with acceleration (max lateral movement = 5 pixels at max speed)
    float lateral_speed = 1 + (env->speed / MAX_SPEED * 5);  // Scale between 1 and 5 pixels

    // Update player's lateral position based on action
    if (env->actions[0] == ACTION_LEFT) {
        env->left_distance_to_edge -= lateral_speed;  // Move left, scaled by speed
        env->right_distance_to_edge += lateral_speed;
        if (env->left_distance_to_edge < 0) {
            env->left_distance_to_edge = 0;  // Prevent moving past the left edge
            env->right_distance_to_edge = ROAD_WIDTH - CAR_WIDTH;
        }
    } else if (env->actions[0] == ACTION_RIGHT) {
        env->left_distance_to_edge += lateral_speed;  // Move right, scaled by speed
        env->right_distance_to_edge -= lateral_speed;
        if (env->right_distance_to_edge < 0) {
            env->right_distance_to_edge = 0;  // Prevent moving past the right edge
            env->left_distance_to_edge = ROAD_WIDTH - CAR_WIDTH;
        }
    }

    // Update enemy cars
    for (int i = 0; i < NUM_ENEMY_CARS; i++) {
        if (env->enemy_cars[i].active) {
            // Move enemy cars relative to player's speed
            env->enemy_cars[i].rear_bumper_y -= env->speed * 0.1f;

            // Remove enemy cars after they pass beyond
            // Remove enemy cars after they pass beyond the player's Y-position by more than PASS_THRESHOLD
            if (env->enemy_cars[i].rear_bumper_y < env->front_bumper_y - PASS_THRESHOLD && env->enemy_cars[i].active) {
                env->enemy_cars[i].active = false;  // Mark enemy car as passed and remove it
                env->carsRemainingLo--;  // Decrement the Cars Passed counter
            }

            // If enemy car passes behind the player, reset it
            if (env->enemy_cars[i].rear_bumper_y < MIN_Y_POSITION) {
                env->enemy_cars[i].rear_bumper_y = MAX_Y_POSITION + rand() % 500;
                env->enemy_cars[i].lane = rand() % 3;
            }

            // Collision detection
            float enemy_lane_center = (env->enemy_cars[i].lane + 0.5f) * (ROAD_WIDTH / 3.0f);
            float enemy_left_edge = enemy_lane_center - (CAR_WIDTH / 2);
            float enemy_right_edge = enemy_lane_center + (CAR_WIDTH / 2);
            float player_left_edge = env->left_distance_to_edge;
            float player_right_edge = ROAD_WIDTH - env->right_distance_to_edge;

            int lateral_overlap = !(player_right_edge <= enemy_left_edge || player_left_edge >= enemy_right_edge);
            float enemy_front_bumper_y = env->enemy_cars[i].rear_bumper_y + ENEMY_CAR_LENGTH;
            int longitudinal_overlap = !(env->front_bumper_y + PLAYER_CAR_LENGTH <= env->enemy_cars[i].rear_bumper_y || env->front_bumper_y >= enemy_front_bumper_y);

            if (lateral_overlap && longitudinal_overlap) {
                // Gradually reduce speed over 30 frames (half a second at 60 FPS) on collision
                env->speed -= (env->speed - MIN_SPEED) / 30.0f;
                if (env->speed < MIN_SPEED)
                    env->speed = MIN_SPEED;
                env->reward = -10.0f;
                env->rewards[0] = env->reward;
                env->done = true;
                env->terminals[0] = 1;
                compute_observations(env);
                return;
            }
        }
    }

    // Update reward for moving forward
    env->reward = env->speed * 0.01f;
    env->rewards[0] = env->reward;

    // Handle day progression and reset car counter
    if (env->carsRemainingLo <= 0) {
        env->dayNumber++;  // Increment day number
        env->carsRemainingLo = 200;  // Reset Cars Passed counter for the new day
    }

    // Apply placeholder weather conditions
    apply_weather_conditions(env);

    // Update observations
    compute_observations(env);
}

void reset_env(CEnduro *env) {
    init_env(env);
    compute_observations(env);
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
}

typedef struct Client Client;
struct Client {
    float width;
    float height;
    float player_width;
    float player_height;
    float other_car_width;
    float other_car_height;
    float player_speed;
    float base_car_speed;
    float max_player_speed;
    float min_player_speed;
    float speed_increment;
    int max_score;
};

Client* make_client(CEnduro* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    client->player_width = env->player_width;
    client->player_height = env->player_height;
    client->other_car_width = env->other_car_width;
    client->other_car_height = env->other_car_height;
    client->player_speed = env->player_speed;
    client->base_car_speed = env->base_car_speed;
    client->max_player_speed = env->max_player_speed;
    client->min_player_speed = env->min_player_speed;
    client->speed_increment = env->speed_increment;
    client->max_score = env->max_score;

    InitWindow(TOTAL_SCREEN_WIDTH, TOTAL_SCREEN_HEIGHT, "Enduro Racing");
    SetTargetFPS(60);

    return client;
}

void render(Client* client, CEnduro* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    
    // Define a dark green background color
    Color darkGreen = { 0, 100, 0, 255 }; // RGB(0, 100, 0)
    
    // Set the background color to dark green
    ClearBackground(darkGreen);

    // Draw road (centered, with side margins)
    float lane_width = ROAD_WIDTH / 3.0f;
    float road_center_x = ACTION_SCREEN_WIDTH / 2.0f;
    float road_left_edge = road_center_x - ROAD_WIDTH / 2.0f;

    // Draw the road in gray
    DrawRectangle(road_left_edge, 0, ROAD_WIDTH, ACTION_SCREEN_HEIGHT, GRAY);
    
    // Draw lane dividers
    DrawLine(road_left_edge + lane_width, 0, road_left_edge + lane_width, ACTION_SCREEN_HEIGHT, WHITE);
    DrawLine(road_left_edge + 2 * lane_width, 0, road_left_edge + 2 * lane_width, ACTION_SCREEN_HEIGHT, WHITE);
    
    // Draw the player's car directly from its calculated position in step
    float player_car_x = road_left_edge + env->left_distance_to_edge;
    float player_car_y = env->front_bumper_y;  // No manual adjustment needed, just use env->front_bumper_y
    
    DrawRectangle(player_car_x, player_car_y, CAR_WIDTH, PLAYER_CAR_LENGTH, BLUE);

    // Draw enemy cars
    for (int i = 0; i < NUM_ENEMY_CARS; i++) {
        if (env->enemy_cars[i].active) {
            float enemy_lane_center = road_left_edge + (env->enemy_cars[i].lane + 0.5f) * lane_width;
            float enemy_car_x = enemy_lane_center - CAR_WIDTH / 2.0f;
            float enemy_car_y = env->enemy_cars[i].rear_bumper_y;

            // Render the enemy car directly from its position
            DrawRectangle(enemy_car_x, enemy_car_y, CAR_WIDTH, ENEMY_CAR_LENGTH, RED);
        }
    }

    render_hud(env);  // Render HUD elements like cars passed and day number

    EndDrawing();
}

void render_hud(CEnduro *env) {
    // Draw the scoreboard background
    DrawRectangle(SCOREBOARD_X_START, SCOREBOARD_Y_START, SCOREBOARD_WIDTH, SCOREBOARD_HEIGHT, RED);

    // Render the score
    char score[6];
    snprintf(score, sizeof(score), "%05d", env->throttleValue);  // Using throttle as a score placeholder
    DrawText(score, 56, 162, 10, BLACK);

    // Render cars left
    char cars_left[5];
    int totalCarsRemaining = (env->carsRemainingHi << 8) | env->carsRemainingLo;
    snprintf(cars_left, sizeof(cars_left), "%d", totalCarsRemaining);
    DrawText(cars_left, CARS_LEFT_X_START, CARS_LEFT_Y_START, 10, BLACK);

    // Render current day
    char day[2];
    snprintf(day, sizeof(day), "%d", env->dayNumber);
    DrawText(day, DAY_X_START, DAY_Y_START, 10, BLACK);
}

void apply_weather_conditions(CEnduro *env) {
    // Placeholder for future weather conditions
    // No effect for now
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

int main() {
    CEnduro env = {
        .width = TOTAL_SCREEN_WIDTH,
        .height = TOTAL_SCREEN_HEIGHT,
        .player_width = CAR_WIDTH,
        .player_height = PLAYER_CAR_LENGTH,
        .other_car_width = CAR_WIDTH,
        .other_car_height = ENEMY_CAR_LENGTH,
        .player_speed = 0.0f,
        .base_car_speed = 0.0f,
        .max_player_speed = MAX_SPEED,
        .min_player_speed = MIN_SPEED,
        .speed_increment = SPEED_INCREMENT,
        .max_score = 0,
    };
    allocate(&env);

    Client* client = make_client(&env);

    reset_env(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = 0;
        if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
            env.actions[0] = ACTION_LEFT;
        } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
            env.actions[0] = ACTION_RIGHT;
        } else if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
            env.actions[0] = ACTION_ACCEL;
        } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
            env.actions[0] = ACTION_DECEL;
        }

        step(&env);
        render(client, &env);
    }

    close_client(client);
    free_allocated(&env);
}