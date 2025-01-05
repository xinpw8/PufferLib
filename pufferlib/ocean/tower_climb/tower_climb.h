#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
#define NOOP -1
#define UP 3
#define LEFT 2
#define RIGHT 0
#define DOWN 1
#define GRAB 5
#define DROP 6
#define DEFAULT 0
#define HANGING 1
#define HOLDING_BLOCK 2
#define NUM_DIRECTIONS 4

static const int level_one[288] = {
   // floor 1 
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,1,1,1,1,0,
   0,0,0,0,0,0,
   // floor 2
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,1,1,1,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   // floor 3
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,1,1,1,1,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   // floor 4
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,1,1,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   // floor 5
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,1,1,1,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   // floor 6 
   0,0,0,0,0,0,
   0,0,0,1,1,0,
   0,1,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   // floor 7
   0,0,0,0,0,0,
   0,0,0,1,1,0,
   0,1,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   // floor 8
   0,0,0,0,0,0,
   0,0,0,0,2,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
};
static const int DIRECTIONS[NUM_DIRECTIONS] = {0, 1, 2, 3};
static const int DIRECTION_VECTORS[NUM_DIRECTIONS][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

#define LOG_BUFFER_SIZE 1024

static inline int max(int a, int b) {
    return (a > b) ? a : b;
}
	
typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float rows_cleared;
};


typedef struct LogBuffer LogBuffer;
struct LogBuffer {
    Log* logs;
    int length;
    int idx;
};

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
    	log.rows_cleared += logs->logs[i].rows_cleared;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.rows_cleared /= logs->idx;
    logs->idx = 0;
    return log;
}


typedef struct TowerClimb TowerClimb;
struct TowerClimb {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log log;
    float score;
    int width;
    int height;
    int map_choice;
    int robot_position; 
    int robot_direction;
    int robot_state;
    int* board_state; 
    int block_grabbed;
};

int get_direction(TowerClimb* env, int action) {
    // For reference: 
    // 0 = right (initial), 1 = down, 2 = left, 3 = up
    int current_direction = env->robot_direction;
    if (env->block_grabbed != -1){
        // Check if direction is opposite (differs by 2) or perpendicular (differs by 1 or 3)
        int diff = abs(current_direction - action);
        if(diff == 1 || diff == 3){
            env->block_grabbed = -1;
        }
        if(diff == 2){
            return current_direction;
        }
    }
    return action;
}


void init(TowerClimb* env) {
    // int map_size = map_sizes[env->map_choice - 1];

    env->board_state = (int*)calloc(288, sizeof(int));
    env->block_grabbed = -1;
    memcpy(env->board_state, level_one, 288 * sizeof(int));
}

void allocate(TowerClimb* env) {
    init(env);
    env->observations = (float*)calloc(128, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_initialized(TowerClimb* env) {
}

void free_allocated(TowerClimb* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_logbuffer(env->log_buffer);
    free_initialized(env);
}

void compute_observations(TowerClimb* env) {
    
}

void reset(TowerClimb* env) {
    env->log = (Log){0};
	env->dones[0] = 0;
    env->robot_position = 64;
    env->robot_direction =2;
    env->robot_state = DEFAULT;
    env->block_grabbed = -1;
    memcpy(env->board_state, level_one, 288 * sizeof(int));
    compute_observations(env);
}

void handle_grab_block(TowerClimb *env){
    int current_floor = env->robot_position / 36;
    int grid_pos = env->robot_position % 36;
    int x = grid_pos % 6;
    int z = grid_pos / 6;

    int dx = DIRECTION_VECTORS[env->robot_direction][0];
    int dz = DIRECTION_VECTORS[env->robot_direction][1];

    int next_x = x + dx;
    int next_z = z + dz;
    if (env->robot_state == HANGING){
        printf("can't grab while hanging\n");
        return;
    }
    if (next_x < 0 || next_x >= 6 || next_z < 0 || next_z >= 6 ) {
        // Attempting to move outside the 4x4 grid on this floor - do nothing
        printf("can't grab out of bounds\n");
        return;
    }
    int next_index = current_floor * 36 + next_z * 6 + next_x;
    int next_cell = env->board_state[next_index];
    if (next_cell!=1){
        printf("can't grab non-block\n");
        return;
    }
    if(env->block_grabbed == next_index){
        env->block_grabbed = -1;
    } else {
        printf("grabbing block\n");
        env->block_grabbed = next_index;
    }
}

int is_next_to_block(TowerClimb* env, int target_position){
    printf("target_position: %d\n", target_position);
    int current_floor = target_position / 36;
    int grid_pos = target_position % 36;
    int x = grid_pos % 6;
    int z = grid_pos / 6;

    // Check each adjacent cell for a block
    for (int i = 0; i < NUM_DIRECTIONS; i++) {
        int dx = DIRECTION_VECTORS[i][0];
        int dz = DIRECTION_VECTORS[i][1];
        int next_x = x + dx;
        int next_z = z + dz;
        
        // Skip if out of bounds
        if (next_x < 0 || next_x >= 6 || next_z < 0 || next_z >= 6) {
            continue;
        }
        
        int next_index = current_floor * 36 + next_z * 6 + next_x;
        printf("next_index: %d\n", next_index);
        int next_cell = env->board_state[next_index];
        printf("next_cell: %d\n", next_cell);
        if (next_cell == 1) {
            return 1;
        }
    }
    return 0;
}

void handle_move_forward(TowerClimb* env, int action) {
    printf("handle_move_forward\n");
    // Calculate current floor, x, z from the robot_position
    int current_floor = env->robot_position / 36;
    int grid_pos = env->robot_position % 36;
    int x = grid_pos % 6;
    int z = grid_pos / 6;

    // Determine the offset for the next cell based on env->robot_direction
    int dx = DIRECTION_VECTORS[env->robot_direction][0];
    int dz = DIRECTION_VECTORS[env->robot_direction][1];
    
    int front_dx = dx;
    int front_dz = dz;
    int front_x = x + front_dx;
    int front_z = z + front_dz;
    int front_index = current_floor * 36 + front_z * 6 + front_x;
    int front_cell = env->board_state[front_index];
    // Calculate the next potential cell’s x, z
    int next_x = x + dx;
    int next_z = z + dz;
    if(env->block_grabbed != -1 && abs(env->robot_direction - action) == 2){
        // inverse direction based on direction vectors 
        dx = DIRECTION_VECTORS[action][0];
        dz = DIRECTION_VECTORS[action][1];
        next_x = x + dx;
        next_z = z + dz;
    }
    
    // Out-of-bounds check (within 4x4 grid for the current floor)
    if (next_x < 0 || next_x >= 6 || next_z < 0 || next_z >= 6) {
        // Attempting to move outside the 4x4 grid on this floor - do nothing
        return;
    }

    // Convert next x,z to a linear index for the same floor
    int next_index = current_floor * 36 + next_z * 6 + next_x;
    int next_cell = env->board_state[next_index];

    // Check if the cell in front is free, the goal, or blocked
    if (next_cell == 0 && env->block_grabbed == -1) {
        int below_index = (current_floor - 1) * 36 + next_z * 6 + next_x;
        int below_cell = env->board_state[below_index];

        int below_next_index = below_index - 36;
        int below_next_cell = -1;
        if (is_next_to_block(env, next_index) == 0 && env->robot_state == HANGING){
            return;
        };
        if(below_next_index > 0){
            below_next_cell = env->board_state[below_next_index];
        }
        
        if ((below_cell == 1 && env->robot_state == DEFAULT) ){
            env->robot_position = next_index;
            env->robot_state = DEFAULT;
        }
        else if (below_cell == 0 && env->robot_state == HANGING){
            env->robot_position = next_index;
            env->robot_state = HANGING;
        }

        else if (below_cell == 0 && below_next_cell <= 0){
            env->robot_position = below_index;
            env->robot_state = HANGING;
        }
        else if(below_cell == 1 && env->robot_state == HANGING){
            env->robot_position = next_index;
        }
        else {
            env->robot_position = below_index;
            env->robot_state = DEFAULT;
        }
    }
    else if ((next_cell == 1 || next_cell == 2) && env->block_grabbed == -1) {
        // There's a block in front. Check if we can climb up to the floor above it
        if (current_floor < 8) { // we have space above if current_floor < 8
            int above_index = (current_floor + 1) * 36 + next_z * 6 + next_x;
            int above_cell = env->board_state[above_index];
            printf("above_index: %d, above_cell: %d\n", above_index, above_cell);
            // If the above cell is free (0) or the goal (2), climb onto it
            if (above_cell == 0 || above_cell == 2) {
                env->robot_position = above_index;
                env->robot_state = DEFAULT; // set hanging to indicate we climbed a block
            }
            else {
                // If there's also a cube (1) above, we cannot climb
                // do nothing, remain in place
            }
        }
    }
    else if (front_cell == 1 && env->block_grabbed != -1) {
        env->board_state[front_index] = 0;
        // Calculate block position based on direction
        int block_offset = (env->robot_direction == 3) ? -6 :  // North
                          (env->robot_direction == 1) ? 6 :    // South
                          (env->robot_direction == 0) ? 1 :    // East
                          (env->robot_direction == 2) ? -1 :   // West
                          -6;  // Default to north behavior for other directions
        env->board_state[next_index + block_offset] = 1;
        env->block_grabbed = next_index + block_offset;
        int below_index = (current_floor - 1) * 36 + next_z * 6 + next_x;
        int below_cell = env->board_state[below_index];
        if (abs(env->robot_direction - action) == 2){
            if (below_cell == 0 && env->robot_state == DEFAULT){
                env->robot_position = below_index;
                env->robot_state = HANGING;
            } else {
                env->robot_position = next_index;
            }
        }
        if (env->robot_position != next_index){
            env->block_grabbed = -1;
        }
        printf("block_grabbed: %d\n", env->block_grabbed);
        printf("robot_position: %d\n", env->robot_position);
        printf("board_state: %d\n", env->board_state[next_index + block_offset]);
    }
}


void step(TowerClimb* env) {
    int action = env->actions[0];
    
    if (action != NOOP) {
        if (action == LEFT || action == RIGHT || action == DOWN || action == UP) {
            int direction = get_direction(env, action);
            int moving_block = (env->block_grabbed != -1 && abs(env->robot_direction - action) == 2);
            if (direction == env->robot_direction || moving_block || env->robot_state == HANGING){
                env->robot_direction = direction;
                handle_move_forward(env, action);
                int below_index = env->robot_position - 36;
                // check if goal is below current position
                if (env->board_state[below_index] == 2){
                    env->dones[0] = 1;
                    env->log.episode_return = 1;
                    add_log(env->log_buffer, &env->log);
                    reset(env);
                }
            }
            else {
                env->robot_direction = direction;
            }

        }
        else if (action == GRAB){
            handle_grab_block(env);
        }
    }
    compute_observations(env);
}

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_GREY = (Color){128, 128, 128, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
    Camera3D camera;
    Model robot;
};

Client* make_client(TowerClimb* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    InitWindow(env->width, env->height, "PufferLib Ray Tower Climb");
    SetTargetFPS(60);
    client->puffers = LoadTexture("resources/puffers_128.png");
    client->camera = (Camera3D){ 0 };
    client->camera.position = (Vector3){ 0.0f, 15.0f, 12.0f };  // Move camera further back and higher up
    client->camera.target = (Vector3){ 2.0f, 4.0f, 2.0f };     // Keep looking at same target point
    client->camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    // client->robot = LoadModel("resources/robot.glb"); 
  return client;
}

void render(Client* client, TowerClimb* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);    
    BeginMode3D(client->camera);

    for(int i= 0; i < 288; i++){
        if(env->board_state[i] > 0){
            int floor = i / 36;
            int grid_pos = i % 36;
            int x = grid_pos % 6;
            int z = grid_pos / 6;
            Color cubeColor = (env->board_state[i] == 1) ? STONE_GRAY : PUFF_CYAN;  // Gray for blocks, Cyan for goal
            DrawCube(
                (Vector3){x * 1.0f, floor * 1.0f, z * 1.0f},
                1.0f, 1.0f, 1.0f,
                cubeColor
            );
            if(i == env->block_grabbed){
                DrawCubeWires(
                (Vector3){x * 1.0f, floor * 1.0f, z * 1.0f},
                1.0f, 1.0f, 1.0f,
                RED);
            }else {
                DrawCubeWires(
                (Vector3){x * 1.0f, floor * 1.0f, z * 1.0f},
                1.0f, 1.0f, 1.0f,
                BLACK
            );
            }
            
        }
    }

    // calculate robot position
    int floor = env->robot_position / 36;
    int grid_pos = env->robot_position % 36;
    int x = grid_pos % 6;
    int z = grid_pos / 6;
    
    Vector3 spherePos = (Vector3){ 
        x * 1.0f,
        floor * 1.0f,  // One unit above platform
        (z * 1.0f)
    };

    // Draw sphere character
    DrawSphere(spherePos, 0.3f, YELLOW);  // 0.3 radius yellow sphere
    // Draw direction arrow
    float arrowLength = 0.5f;
    Vector3 arrowStart = (Vector3){
        spherePos.x,
        spherePos.y + 0.4f,  // Start slightly above sphere
        spherePos.z
    };
    
    // Calculate arrow end based on direction
    Vector3 arrowEnd = arrowStart;
    switch(env->robot_direction) {
        case 0: // right
            arrowEnd.x += arrowLength;
            break;
        case 1: // down
            arrowEnd.z += arrowLength;
            break;
        case 2: // left
            arrowEnd.x -= arrowLength;
            break;
        case 3: // up
            arrowEnd.z -= arrowLength;
            break;
    }

    // Draw arrow shaft (thin cylinder)
    if (env->robot_state == DEFAULT){
        DrawCylinderEx(arrowStart, arrowEnd, 0.05f, 0.05f, 8, RED);
    } else {
        DrawCylinderEx(arrowStart, arrowEnd, 0.05f, 0.05f, 8, PURPLE);
    }
    
    // Draw arrow head (thicker, shorter cylinder)
    Vector3 headStart = arrowEnd;
    Vector3 headEnd = arrowEnd;
    switch(env->robot_direction) {
        case 0: // right
            headEnd.x += 0.2f;
            break;
        case 1: // down
            headEnd.z += 0.2f;
            break;
        case 2: // left
            headEnd.x -= 0.2f;
            break;
        case 3: // up
            headEnd.z -= 0.2f;
            break;
    }
    if (env->robot_state == DEFAULT){
        DrawCylinderEx(headStart, headEnd, 0.1f, 0.0f, 8, RED);  // Tapered cylinder for arrow head
    } else {
        DrawCylinderEx(headStart, headEnd, 0.1f, 0.0f, 8, PURPLE);  // Tapered cylinder for arrow head
    }

    EndMode3D();
    EndDrawing();
}
void close_client(Client* client) {
    CloseWindow();
    UnloadModel(client->robot);
    free(client);
}
