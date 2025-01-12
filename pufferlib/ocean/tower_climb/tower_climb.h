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


static const int level_two[400] = {
    // floor 1

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
    int robot_orientation;
    int* board_state; 
    int* blocks_to_move;
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
    env->blocks_to_move = (int*)calloc(6, sizeof(int));
    env->robot_orientation = UP;
    for (int i = 0; i < 6; i++){
        env->blocks_to_move[i] = -1;
    }
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
    free(env->blocks_to_move);
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


int get_local_direction(TowerClimb* env, int action) {
    // We assume action can be LEFT or RIGHT
    // Orientation is one of (RIGHT=0, DOWN=1, LEFT=2, UP=3).

    if (action == LEFT) {
        switch(env->robot_orientation) {
            case UP:    return LEFT;  // 3 -> 2
            case LEFT:  return DOWN;  // 2 -> 1
            case DOWN:  return RIGHT; // 1 -> 0
            case RIGHT: return UP;    // 0 -> 3
        }
    } 
    if (action == RIGHT) {
        switch(env->robot_orientation) {
            case UP:    return RIGHT; // 3 -> 0
            case RIGHT: return DOWN;  // 0 -> 1
            case DOWN:  return LEFT;  // 1 -> 2
            case LEFT:  return UP;    // 2 -> 3
        }
    }

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
        int next_cell = env->board_state[next_index];
        printf("next_index: %d, next_cell: %d\n", next_index, next_cell);
        if (next_cell == 1) {
            return 1;
        }
    }
    return 0;
}

void add_blocks_to_move(TowerClimb* env, int interval){
    int start_index = env->blocks_to_move[0];
    for (int i = 0; i < 6; i++){
        if(env->blocks_to_move[i-1] == -1 && i != 0){
            break;
        }
        if(env->board_state[start_index + interval * i] == 1){
            env->blocks_to_move[i] = start_index + interval * i;
        }
        
        printf("blocks_to_move[%d]: %d\n", i, env->blocks_to_move[i]);
    }
}

void move_blocks(TowerClimb* env, int interval){
    for (int i = 0; i < 6; i++){
        if(env->blocks_to_move[i] != -1){
            if(i==0){
                env->board_state[env->blocks_to_move[i]] = 0;
                
            }
            env->board_state[env->blocks_to_move[i] + interval] = 1;
        }
    }
}

void shimmy(TowerClimb* env, int action, int current_floor, int x, int z, int x_mod, int z_mod, int x_direction_mod, int z_direction_mod, int final_orienation){
    int next_block = current_floor *36 + (z+z_mod) * 6 + (x+x_mod);
    int destination_block = current_floor *36 + (z+z_direction_mod) * 6 + (x+x_direction_mod);
    printf("current block: %d\n", env->robot_position);
    printf("shimmying to index: %d\n", next_block);
    printf("board state: %d\n", env->board_state[next_block]);
    if (env->board_state[next_block] == 1){
        int above_destination = destination_block + 36;
        // cant wrap shimmy with block above.
        if(env->board_state[above_destination] == 1){
            return;
        }
        env->robot_position = destination_block;
        env->robot_orientation = final_orienation;
    }
    return;
}

void handle_climb(TowerClimb* env, int action, int current_floor,int x, int z, int next_z, int next_x, int next_index, int next_cell){
    if ((next_cell == 1 || next_cell == 2) && env->block_grabbed == -1) {
        // There's a block in front. Check if we can climb up to the floor above it
        if (current_floor < 8) { // we have space above if current_floor < 8
            int climb_index = (current_floor + 1) * 36 + next_z * 6 + next_x;
            int climb_cell = env->board_state[climb_index];
            
            int direct_above = env->robot_position + 36;
            int direct_above_cell = env->board_state[direct_above];
            // If the above cell is free (0) or the goal (2), climb onto it
            if (climb_cell == 0 && direct_above_cell == 0) {
                env->robot_position = climb_index;
                env->robot_state = DEFAULT; // set hanging to indicate we climbed a block
                printf("robot_orientation: %d\n", env->robot_orientation);
            }
            
        }
    }
}


void handle_down(TowerClimb* env, int action, int current_floor,int x, int z, int next_z, int next_x, int next_index, int next_cell){
    int below_index = (current_floor - 1) * 36 + next_z * 6 + next_x;
    int below_cell = env->board_state[below_index];

    int below_next_index = below_index - 36;
    int below_next_cell = env->board_state[below_next_index];

    // Default state cases
    if (below_cell == 1 && env->robot_state == DEFAULT) {
        env->robot_position = next_index;
        env->robot_state = DEFAULT;
        return;
    }

    if (below_cell == 0 && below_next_cell == 0 && env->robot_state == DEFAULT) {
        env->robot_position = below_index;
        env->robot_state = HANGING;
        if(env->robot_direction == RIGHT){
            env->robot_orientation = LEFT;
        }
        if(env->robot_direction == LEFT){
            env->robot_orientation = RIGHT;
        }
        if(env->robot_direction == DOWN){
            env->robot_orientation = UP;
        }
        if(env->robot_direction == UP){
            env->robot_orientation = DOWN;
        }
        printf("robot_orientation: %d\n", env->robot_orientation);
        return;
    }

    if (below_cell == 0 && below_next_cell == 1 && env->robot_state == DEFAULT){
        env->robot_position = below_index;
        env->robot_state = DEFAULT;
        // env->robot_orientation = UP;
    }
}

void handle_left_right(TowerClimb* env, int action, int current_floor,int x, int z, int next_z, int next_x, int next_index, int next_cell){
    // Check if the cell in front is free, the goal, or blocked
    int below_index = (current_floor - 1) * 36 + next_z * 6 + next_x;
    int below_cell = env->board_state[below_index];
    // shimmying
    if (env->robot_state == HANGING){
        int local_direction = get_local_direction(env, action);
        int local_next_dx = DIRECTION_VECTORS[local_direction][0];
        int local_next_dz = DIRECTION_VECTORS[local_direction][1];
        int local_next_x = x + local_next_dx;
        int local_next_z = z + local_next_dz;
        int local_next_index = current_floor * 36 + local_next_z * 6 + local_next_x;

        if (is_next_to_block(env, local_next_index)){
            // standard shimmy
            int above_index = local_next_index + 36;
            int above_cell = env->board_state[above_index];
            // cant shimmy with block above
            if (above_cell == 1){
                return;
            }
            env->robot_position = local_next_index;
            return;
        }
        
        int current_floor = env->robot_position / 36;
        int found_wrap = 0;

        if (env->robot_orientation == UP) {  // Hanging on north face
            if (action == RIGHT) {
                shimmy(env, action, current_floor, x, z, 0, -1, 1, -1, LEFT);
                env->robot_direction = RIGHT;   
            }
            else if (action == LEFT) {
                shimmy(env, action, current_floor, x, z, 0,-1, -1, -1, RIGHT);
                env->robot_direction = LEFT;
            }
        }
        // Similar logic for other directions (DOWN, LEFT, RIGHT)
        else if (env->robot_orientation == DOWN) {
            if (action == RIGHT) {
                shimmy(env, action, current_floor, x, z, 0, 1, -1, 1, RIGHT);
                env->robot_direction = LEFT;
            }
            if (action == LEFT) {
                shimmy(env, action, current_floor, x, z, 0, 1, 1, 1, LEFT);
                env->robot_direction = RIGHT;   
            }
        }
        else if (env->robot_orientation == RIGHT) {
            if (action == LEFT) {
                // Check block to our right
                shimmy(env, action, current_floor, x, z, 1, 0, 1, -1, DOWN);
                env->robot_direction = LEFT;
            }
            else if (action == RIGHT) {
                // Check block to our right
                shimmy(env, action, current_floor, x, z, 1, 0, 1, 1, UP);
                env->robot_direction = RIGHT;
            }
        }
        else if (env->robot_orientation == LEFT) {
            if (action == RIGHT) {
                shimmy(env,action, current_floor, x, z, -1, 0, -1, -1, DOWN);
                env->robot_direction = RIGHT;
            }
            if (action == LEFT) {
                shimmy(env,action, current_floor, x, z, -1, 0, -1, 1, UP);
                env->robot_direction = LEFT;
            }
        }
    }

    // walking left or right
    if (next_x < 0 || next_x >= 6 || next_z < 0 || next_z >= 6) {
        // Attempting to move outside the 4x4 grid on this floor - do nothing
        return;
    }
    if ((next_cell ==1 || next_cell ==2) && env->robot_state == DEFAULT){
        handle_climb(env, action, current_floor, x, z, next_z, next_x, next_index, next_cell);
    }
    if (next_cell == 0 && below_cell == 1 && env->robot_state == DEFAULT){
        env->robot_position = next_index;
        env->robot_state = DEFAULT;
        env->robot_orientation = env->robot_direction;
    }
    
    // dropping to hang state
    if (next_cell == 0 && below_cell == 0 && env->robot_state == DEFAULT){
        handle_down(env, action, current_floor, x, z, next_z, next_x, next_index, next_cell);
    }
    
    return;
}





void handle_move_forward(TowerClimb* env, int action) {
    // Calculate current floor, x, z from the robot_position
    int current_floor = env->robot_position / 36;
    int grid_pos = env->robot_position % 36;
    int x = grid_pos % 6;
    int z = grid_pos / 6;

    // Determine the offset for the next cell based on env->robot_direction
    int dx = DIRECTION_VECTORS[env->robot_direction][0];
    int dz = DIRECTION_VECTORS[env->robot_direction][1];
    
    int orient_dx = DIRECTION_VECTORS[env->robot_orientation][0];
    int orient_dz = DIRECTION_VECTORS[env->robot_orientation][1];

    // based on orient dx and action, determien a new dx and dz

    int front_x = x + orient_dx;
    int front_z = z + orient_dz;
    int front_index = current_floor * 36 + front_z * 6 + front_x;
    int front_cell = env->board_state[front_index];
    int front_below_index = front_index - 36;
    int front_below_cell = env->board_state[front_below_index];
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
    


    // Convert next x,z to a linear index for the same floor
    int next_index = current_floor * 36 + next_z * 6 + next_x;
    int next_cell = env->board_state[next_index];
    if((action == LEFT || action == RIGHT) && env->block_grabbed == -1){
        handle_left_right(env, action, current_floor, x, z, next_z, next_x, next_index, next_cell);
    }

    if (action == UP && env->block_grabbed == -1){
        if(front_cell ==1 || front_cell ==2 ){
            handle_climb(env, action, current_floor, x, z, front_z, front_x, front_index, front_cell);
        }

        if(front_cell == 0 && front_below_cell == 1){
            env->robot_position = front_index;
            env->robot_state = DEFAULT;
            env->robot_orientation = UP;
        }

        if(front_cell == 0 && front_below_cell == 0){
            env->robot_position = front_below_index;
            env->robot_state = HANGING;
            env->robot_orientation = DOWN;
        }
    }

    if (action == DOWN && env->block_grabbed == -1){
        handle_down(env, action, current_floor, x, z, next_z, next_x, next_index, next_cell);
    }
    
    if (front_cell == 1 && env->block_grabbed != -1) {
        env->blocks_to_move[0] = front_index;
        // Calculate block position based on direction
        int block_offset = (env->robot_direction == 3) ? -6 :  // North
                          (env->robot_direction == 1) ? 6 :    // South
                          (env->robot_direction == 0) ? 1 :    // East
                          (env->robot_direction == 2) ? -1 :   // West
                          -6;  // Default to north behavior for other direction
        // Pushing
        int push_condition = abs(env->robot_direction - action) == 0;
        if(push_condition){
            add_blocks_to_move(env, block_offset);
            move_blocks(env, block_offset);
        }
        // Pulling
        int pull_condition = abs(env->robot_direction - action) == 2;
        if (pull_condition){
            int below_index = (current_floor - 1) * 36 + next_z * 6 + next_x;
            int below_cell = env->board_state[below_index];
            env->board_state[front_index] = 0;
            env->board_state[next_index + block_offset] = 1;
            env->block_grabbed = next_index + block_offset;
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
        memset(env->blocks_to_move, -1, 6 * sizeof(int));
        
    }
}

void handle_drop(TowerClimb* env){
    int next_position = env->robot_position - 36;
    if (env->robot_position < 36){
        env->dones[0] = 1;
        env->log.episode_return = -1;
        add_log(env->log_buffer, &env->log);
        reset(env);
        return;
    }
    while (env->board_state[next_position] != 1){
        if(next_position < 36){
            env->dones[0] = 1;
            env->log.episode_return = -1;
            add_log(env->log_buffer, &env->log);
            reset(env);
            return;
        } 
        next_position = next_position - 36;  
    }
    env->robot_position = next_position + 36;
    env->robot_state = DEFAULT;
    env->robot_orientation = env->robot_direction;
}

void step(TowerClimb* env) {
    int action = env->actions[0];
    
    if (action != NOOP) {
        if (action == LEFT || action == RIGHT || action == DOWN || action == UP) {
            int direction = get_direction(env, action);
            int moving_block = (env->block_grabbed != -1 && abs(env->robot_direction - action) == 2);
            if (direction == env->robot_orientation || moving_block || env->robot_state == HANGING){
                env->robot_direction = direction;
                printf("orientation: %d\n", env->robot_orientation);
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
                env->robot_orientation = direction;
            }

        }
        else if (action == GRAB){
            handle_grab_block(env);
        }
        else if (action == DROP){
            env->block_grabbed = -1;
            if(env->robot_state == HANGING){
                handle_drop(env);
            }
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
    switch(env->robot_orientation) {
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
    switch(env->robot_orientation) {
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
