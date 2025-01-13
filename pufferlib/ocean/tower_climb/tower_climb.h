#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
#include "levels.h"
#define NOOP -1
#define UP 3
#define LEFT 2
#define RIGHT 0
#define DOWN 1
#define GRAB 4
#define DROP 5
#define DEFAULT 0
#define HANGING 1
#define HOLDING_BLOCK 2
#define NUM_DIRECTIONS 4
#define LEVEL_MAX_SIZE 432




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

typedef struct CTowerClimb CTowerClimb;
struct CTowerClimb {
    int* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log log;
    float score;
    int map_choice;
    int robot_position; 
    int robot_direction;
    int robot_state;
    int robot_orientation;
    int* board_state; 
    int* blocks_to_move;
    int* blocks_to_fall;
    int block_grabbed;
    int rows_cleared;
    const Level levels[3];
    Level level;
};

int get_direction(CTowerClimb* env, int action) {
    // For reference: 
    // 0 = right (initial), 1 = down, 2 = left, 3 = up
    int current_direction = env->robot_direction;
    if (env->block_grabbed != -1){
        // Check if direction is opposite (differs by 2) or perpendicular (differs by 1 or 3)
        int diff = abs(current_direction - action);
        if(diff == 1 || diff == 3){
            env->block_grabbed = -1;
        }
        else if(diff == 2){
            return current_direction;
        }
    }
    return action;
}


void init(CTowerClimb* env) {
    env->level = levels[0];
    memcpy(env->board_state, env->level.map, env->level.total_length * sizeof(int));
    memset(env->blocks_to_fall, -1, LEVEL_MAX_SIZE * sizeof(int));
    env->board_state = (int*)calloc(LEVEL_MAX_SIZE, sizeof(int));
    env->block_grabbed = -1;
    env->blocks_to_move = (int*)calloc(env->level.rows, sizeof(int));
    env->blocks_to_fall = (int*)calloc(LEVEL_MAX_SIZE, sizeof(int));
    env->rows_cleared = 0;
    env->robot_orientation = UP;
    for (int i = 0; i < env->level.rows; i++){
        env->blocks_to_move[i] = -1;
    }
}

void allocate(CTowerClimb* env) {
    init(env);
    env->observations = (int*)calloc(LEVEL_MAX_SIZE, sizeof(int)); // make this unsigned char
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_initialized(CTowerClimb* env) {
    free(env->blocks_to_move);
    free(env->blocks_to_fall);
    free(env->board_state);
}

void free_allocated(CTowerClimb* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_logbuffer(env->log_buffer);
    free_initialized(env);
}

void compute_observations(CTowerClimb* env) {
    
}

void reset(CTowerClimb* env) {
    env->log = (Log){0};
    env->dones[0] = 0;
    env->robot_position = 64;
    env->robot_direction = 2;
    env->robot_state = DEFAULT;
    env->block_grabbed = -1;
    env->level = levels[0];
    memcpy(env->board_state, env->level.map, env->level.total_length * sizeof(int));
    compute_observations(env);
}

int get_local_direction(CTowerClimb* env, int action) {
    // We assume action can be LEFT or RIGHT
    // Orientation is one of (RIGHT=0, DOWN=1, LEFT=2, UP=3).
    //  c mod robot orientation
    if (action == LEFT) {
        return (env->robot_orientation + 3) % 4;
    } 
    if (action == RIGHT) {
        return (env->robot_orientation + 1) % 4;
    }
    return env->robot_orientation;
}

void handle_grab_block(CTowerClimb *env){
    int sz = env->level.size;
    int cols = env->level.cols;
    int rows = env->level.rows;
    int current_floor = env->robot_position / sz;
    int grid_pos = env->robot_position % sz;
    int x = grid_pos % cols;
    int z = grid_pos / cols;

    int dx = DIRECTION_VECTORS[env->robot_orientation][0]; // possibly make dx and dz 1D arrays
    int dz = DIRECTION_VECTORS[env->robot_orientation][1];

    int next_x = x + dx;
    int next_z = z + dz;
    if (env->robot_state == HANGING){
        return;
    }
    if (next_x < 0 || next_x >= cols || next_z < 0 || next_z >= rows ) {
        // Attempting to move outside the 4x4 grid on this floor - do nothing
        return;
    }
    int next_index = current_floor*sz + next_z*cols + next_x;
    int next_cell = env->board_state[next_index];
    if (next_cell!=1){
        return;
    }
    if(env->block_grabbed == next_index){
        env->block_grabbed = -1;
    } else {
        env->block_grabbed = next_index;
    }
}

int is_next_to_block(CTowerClimb* env, int target_position){
    int sz = env->level.size;
    int cols = env->level.cols;
    int current_floor = target_position / sz;
    int grid_pos = target_position % sz;
    int x = grid_pos % cols;
    int z = grid_pos / cols;
    int dx = DIRECTION_VECTORS[env->robot_orientation][0];
    int dz = DIRECTION_VECTORS[env->robot_orientation][1];
    int next_x = x + dx;
    int next_z = z + dz;    
    int next_index = current_floor*sz + next_z*cols + next_x;  // semantically group multiplication
    int next_cell = env->board_state[next_index];

    if (next_cell == 1) {
        return 1;
    }
    return 0;
}

void add_blocks_to_move(CTowerClimb* env, int interval){
    int start_index = env->blocks_to_move[0];
    int cols = env->level.cols;
    for (int i = 0; i < cols; i++){
        int b_address = start_index + interval * i;
        if(env->blocks_to_move[i-1] == -1 && i != 0){
            break;
        }
        if(env->board_state[b_address] == 1){
            env->blocks_to_move[i] = b_address;
        }
    }
}

// Helper function to check if position is within bounds
static inline int is_valid_position(int pos) {
    return (pos >= 0 && pos < 288);
}

// Helper function to check block stability
static int is_block_stable(CTowerClimb* env, int position) {
    int sz = env->level.size;
    int cols = env->level.cols;
    // Check bottom support first
    int bottom = position - sz;
    if (is_valid_position(bottom) && env->board_state[bottom] == 1) {
        return 1;
    }

    // Check edge supports only if no bottom support
    // Left+Right support below
    int left_support = position - sz - 1;
    int right_support = position - sz + 1;
    if (is_valid_position(left_support) && is_valid_position(right_support) &&
        (env->board_state[left_support] == 1 || env->board_state[right_support] == 1)) {
        return 1;
    }

    // Front+Back support below
    int front_support = position - sz - cols;
    int back_support = position - sz + cols;
    if (is_valid_position(front_support) && is_valid_position(back_support) &&
        (env->board_state[front_support] == 1 || env->board_state[back_support] == 1)) {
        return 1;
    }

    return 0;
}

void add_blocks_to_fall(CTowerClimb* env){
    int sz = env->level.size;
    int cols = env->level.cols;
    int total_length = env->level.total_length;
// Create queue as simple arrays with front/rear indices
    int queue[total_length];  // Max possible blocks
    int front = 0;
    int rear = 0;
    for(int i= 0;i< total_length;i++){
        if(env->blocks_to_fall[i] == -1){
            break;
        }
        queue[rear] = env->blocks_to_fall[i];
        rear++;
    }
    // Helper to add block to queue if it's a valid block
    for(int i = 0; i < env->level.cols; i++) {
        // invert conditional
        if(env->blocks_to_move[i] != -1) {
            // Add block directly above
            int cell_above = env->blocks_to_move[i] + sz;
            
            // If valid block above, add to queue
            if (cell_above < total_length && env->board_state[cell_above] == 1) {
                queue[rear++] = cell_above;
            }
            
            // Check edge-supported blocks
            int edge_blocks[4] = {
                cell_above - 1,  // left
                cell_above + 1,  // right
                cell_above - cols,  // front
                cell_above + cols   // back
            };
            
            // Add valid edge blocks to queue
            for(int j = 0; j < 4; j++) {
                if (edge_blocks[j] >= 0 && edge_blocks[j] < total_length && 
                    env->board_state[edge_blocks[j]] == 1) {
                    queue[rear++] = edge_blocks[j];
                }
            }
        }
    }

    // Process queue until empty
    while(front < rear) {
        int current = queue[front++];
        int falling_position = current;
        int found_support = 0;
        
        // Remove block from current position
        env->board_state[current] = 0;
        
        // Keep moving down until support found or bottom reached
        while (!found_support && falling_position >= sz) {  // Check if we still have space below            
            // Place block temporarily to check stability
            env->board_state[falling_position] = 1;
            if (is_block_stable(env, falling_position)) {
                found_support = 1;
            } else {
                // Remove block if not stable
                env->board_state[falling_position] = 0;
            }
            if (!found_support) {
                falling_position -= sz;  // Move down one level
            }
        }
        
        // If we found support (didn't fall off bottom), check affected blocks
        if (found_support) {
            // Add blocks that might be affected by this fall
            int original_above = current + sz;  // Block directly above original position
            if (original_above < total_length && env->board_state[original_above] == 1) {
                queue[rear++] = original_above;
            }
            
            int edge_check[4] = {
                original_above - 1,  // Left of original
                original_above + 1,  // Right of original
                original_above - cols,  // Front of original
                original_above + cols   // Back of original
            };
            
            for(int i = 0; i < 4; i++) {
                if (edge_check[i] >= 0 && edge_check[i] < total_length && 
                    env->board_state[edge_check[i]] == 1) {
                    queue[rear++] = edge_check[i];
                }
            }
        }
        // If no support found, block disappears (already set to 0)
    }
    memset(env->blocks_to_fall, -1, LEVEL_MAX_SIZE * sizeof(int));
    memset(env->blocks_to_move, -1, cols * sizeof(int));
}
void move_blocks(CTowerClimb* env, int interval){
    int count = 0;
    int cols = env->level.cols;
    int sz = env->level.size;
    // cache reused variables 
    for (int i = 0; i < cols; i++){
        // invert conditional 
        int b_index = env->blocks_to_move[i];
        if(b_index != -1){
            if(i==0){
                env->board_state[b_index] = 0;
            }
            if(b_index % sz % cols + interval >= 0 && b_index % sz % cols + interval <= cols-1) {
                env->board_state[b_index + interval] = 1;
                env->blocks_to_fall[count] = b_index + interval;
                count++;
            }
        }
    }
}

void shimmy(CTowerClimb* env, int action, int current_floor, int x, int z, int x_mod, int z_mod, int x_direction_mod, int z_direction_mod, int final_orienation){
    int sz = env->level.size;
    int cols = env->level.cols;
    int next_block = current_floor*sz + (z+z_mod)*cols + (x+x_mod);
    int destination_block = current_floor*sz + (z+z_direction_mod)*cols + (x+x_direction_mod);
    if (env->board_state[next_block] == 1){
        int above_destination = destination_block + sz;
        // cant wrap shimmy with block above.
        if(env->board_state[above_destination] == 1){
            return;
        }
        env->robot_position = destination_block;
        env->robot_orientation = final_orienation;
    }
    return;
}

void handle_climb(CTowerClimb* env, int action, int current_floor,int x, int z, int next_z, int next_x, int next_index, int next_cell){
    if (!(next_cell == 1 || next_cell == 2) || env->block_grabbed != -1) {
        return;
    }
    // if ((next_cell == 1 || next_cell == 2) && env->block_grabbed == -1) {
        // There's a block in front. Check if we can climb up to the floor above it
    if (current_floor >=8){
        return;
    }
    int sz = env->level.size;
    int cols = env->level.cols;
    int climb_index = (current_floor + 1) * sz + next_z * cols + next_x;
    int climb_cell = env->board_state[climb_index];
    int direct_above = env->robot_position + sz;
    int direct_above_cell = env->board_state[direct_above];
    if (climb_cell == 0 && direct_above_cell == 0) {
        env->robot_position = climb_index;
        env->robot_state = DEFAULT; // set hanging to indicate we climbed a block
        if ((env->robot_position - sz) / sz > env->rows_cleared){
            env->rows_cleared = (env->robot_position - sz) / sz;
            env->log.rows_cleared = env->rows_cleared;
            env->rewards[0] = .1;
            env->log.episode_return += .1;
        }
    }
    // }
}


void handle_down(CTowerClimb* env, int action, int current_floor,int x, int z, int next_z, int next_x, int next_index, int next_cell){
    int sz = env->level.size;
    int cols = env->level.cols;
    int below_index = (current_floor - 1) * sz + next_z * cols + next_x;
    int below_cell = env->board_state[below_index];
    int below_next_index = below_index - sz;
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
        //mod ti thing
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
        return;
    }
    if (below_cell == 0 && below_next_cell == 1 && env->robot_state == DEFAULT){
        env->robot_position = below_index;
        env->robot_state = DEFAULT;
    }
}

void handle_left_right(CTowerClimb* env, int action, int current_floor,int x, int z, int next_z, int next_x, int next_index, int next_cell){
    // Check if the cell in front is free, the goal, or blocked
    int sz = env->level.size;
    int cols = env->level.cols;
    int below_index = (current_floor - 1) * sz + next_z * cols + next_x;
    int below_cell = env->board_state[below_index];
    // shimmying
    if (env->robot_state == HANGING){
        int local_direction = get_local_direction(env, action);
        int local_next_dx = DIRECTION_VECTORS[local_direction][0];
        int local_next_dz = DIRECTION_VECTORS[local_direction][1];
        int local_next_x = x + local_next_dx;
        int local_next_z = z + local_next_dz;
        int local_next_index = current_floor * sz + local_next_z * cols + local_next_x;
        int local_next_cell = env->board_state[local_next_index];
        if (is_next_to_block(env, local_next_index) && local_next_cell == 0){
            // standard shimmy
            int above_index = local_next_index + sz;
            int above_cell = env->board_state[above_index];
            // cant shimmy with block above
            if (above_cell == 1 ){
                return;
            }
            env->robot_position = local_next_index;
            return;
        }

        if(local_next_cell == 1){
            if (env->robot_orientation == RIGHT && action == LEFT){
                env->robot_orientation = UP;
            }
            else if (env->robot_orientation == RIGHT && action == RIGHT){
                env->robot_orientation = DOWN;
            }
            else if (env->robot_orientation == UP && action == LEFT){
                env->robot_orientation = LEFT;
            }
            else if(env->robot_orientation == UP && action == RIGHT){
                env->robot_orientation = RIGHT;
            }
            else if (env->robot_orientation == LEFT && action == RIGHT){
                env->robot_orientation = UP;
            }
            else if (env->robot_orientation == LEFT && action == LEFT){
                env->robot_orientation = DOWN;
            }
            else if (env->robot_orientation == DOWN && action == LEFT){
                env->robot_orientation = RIGHT;
            }
            else if (env->robot_orientation == DOWN && action == RIGHT){
                env->robot_orientation = LEFT;
            }
            return;
        }
        
        int current_floor = env->robot_position / sz;
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
}

void handle_move_forward(CTowerClimb* env, int action) {
    int sz = env->level.size;
    int cols = env->level.cols;
    // Calculate current floor, x, z from the robot_position
    int current_floor = env->robot_position / sz;
    int grid_pos = env->robot_position % sz;
    int x = grid_pos % cols;
    int z = grid_pos / cols;

    // Determine the offset for the next cell based on env->robot_direction
    // make this 1D
    int dx = DIRECTION_VECTORS[env->robot_direction][0];
    int dz = DIRECTION_VECTORS[env->robot_direction][1];
    
    int orient_dx = DIRECTION_VECTORS[env->robot_orientation][0];
    int orient_dz = DIRECTION_VECTORS[env->robot_orientation][1];

    int front_x = x + orient_dx;
    int front_z = z + orient_dz;
    int front_index = current_floor * sz + front_z * cols + front_x;
    int front_cell = env->board_state[front_index];
    int front_below_index = front_index - sz;
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
    int next_index = current_floor * sz + next_z * cols + next_x;
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
        int block_offset = (env->robot_direction == 3) ? -cols :  // North
                          (env->robot_direction == 1) ? cols :    // South
                          (env->robot_direction == 0) ? 1 :    // East
                          (env->robot_direction == 2) ? -1 :   // West
                          -cols;  // Default to north behavior for other direction
        // Pushing
        if(abs(env->robot_direction - action) == 0){
            add_blocks_to_move(env, block_offset);
            move_blocks(env, block_offset);
            add_blocks_to_fall(env);
        }
        // Pulling
        if (abs(env->robot_direction - action) == 2){
            int below_index = (current_floor - 1) * sz + next_z * cols + next_x;
            int below_cell = env->board_state[below_index];
            env->blocks_to_move[0] = front_index;
            env->board_state[front_index] = 0;
            env->board_state[next_index + block_offset] = 1;
            env->block_grabbed = next_index + block_offset;
            add_blocks_to_fall(env);
            env->blocks_to_move[0] = -1;
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
        memset(env->blocks_to_move, -1, cols * sizeof(int));
    }
}

void handle_drop(CTowerClimb* env){
    int sz = env->level.size;
    int next_position = env->robot_position - sz;
    if (env->robot_position < sz){
        env->dones[0] = 1;
        env->rewards[0] = -1;
        env->log.episode_return -= -1;
        reset(env);
        return;
    }
    while (env->board_state[next_position] != 1){
        if(next_position < sz){
            env->dones[0] = 1;
            env->rewards[0] = -1;
            env->log.episode_return -= 1;
            reset(env);
            return;
        } 
        next_position = next_position - sz;  
    }
    env->robot_position = next_position + sz;
    env->robot_state = DEFAULT;
    env->robot_orientation = env->robot_direction;
}

void step(CTowerClimb* env) {
    env->log.episode_length += 1;
    env->rewards[0] = 0.0;
    int action = env->actions[0];
    if (action == LEFT || action == RIGHT || action == DOWN || action == UP) {
        int direction = get_direction(env, action);
        int moving_block = (env->block_grabbed != -1 && abs(env->robot_direction - action) == 2);
        if (direction == env->robot_orientation || moving_block || env->robot_state == HANGING){
            env->robot_direction = direction;
            handle_move_forward(env, action);
            int sz = env->level.size;
            int below_index = env->robot_position - sz;
            // check if goal is below current position
            if (env->board_state[below_index] == 2){
                env->dones[0] = 1;
                env->rewards[0] = 1;
                env->log.episode_return += 1;
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

Client* make_client(CTowerClimb* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 1000;
    client->height = 1000;
    InitWindow(client->width, client->height, "PufferLib Ray Tower Climb");
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

void render(Client* client, CTowerClimb* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);    
    BeginMode3D(client->camera);
    int cols = env->level.cols;
    int sz = env->level.size;
    int total_length = env->level.total_length;
    for(int i= 0; i < total_length; i++){
        if(env->board_state[i] > 0){
            int floor = i / sz;
            int grid_pos = i % sz;
            int x = grid_pos % cols;
            int z = grid_pos / cols;
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
    int floor = env->robot_position / sz;
    int grid_pos = env->robot_position % sz;
    int x = grid_pos % cols;
    int z = grid_pos / cols;
    
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
