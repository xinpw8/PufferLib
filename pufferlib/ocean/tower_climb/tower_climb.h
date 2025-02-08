#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
#include "levels.h"
#include "rlgl.h"
#if defined(PLATFORM_DESKTOP)
    #define GLSL_VERSION            330
#else   // PLATFORM_ANDROID, PLATFORM_WEB
    #define GLSL_VERSION            100

#endif
#define RLIGHTS_IMPLEMENTATION

#include "rlights.h"

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
#define LEVEL_MAX_SIZE 1000
#define PLAYER_OBS 4
#define OBS_VISION 225
#define MAX_FLOORS 9

static const int DIRECTIONS[NUM_DIRECTIONS] = {0, 1, 2, 3};
static const int DIRECTION_VECTORS_X[NUM_DIRECTIONS] = {1, 0, -1, 0};
static const int DIRECTION_VECTORS_Z[NUM_DIRECTIONS] = {0, 1, 0, -1};


#define LOG_BUFFER_SIZE 1024

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float rows_cleared;
    float levels_completed;
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
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) return log;  // Avoid division by zero

    for (int i = 0; i < logs->idx; i++) {
        log.episode_return  += logs->logs[i].episode_return  / logs->idx;
        log.episode_length  += logs->logs[i].episode_length  / logs->idx;
        log.rows_cleared    += logs->logs[i].rows_cleared    / logs->idx;
	log.levels_completed += logs->logs[i].levels_completed / logs->idx;
    }

    logs->idx = 0;
    return log;
}

typedef struct CTowerClimb CTowerClimb;
struct CTowerClimb {
    float* observations;
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
    int animation_state;
    int previous_position;
    Level level;
    int level_number;
    int distance_to_goal;
    float reward_climb_row;
    float reward_fall_row;
    float reward_illegal_move;
    float reward_move_block;
    float reward_distance;
};

float get_distance_to_goal(CTowerClimb* env) {
	int sz = env->level.size;
	int cols = env->level.cols;
		    
		    // Get robot position coordinates
	int robot_floor = env->robot_position / sz;
	int robot_grid = env->robot_position % sz;
	int robot_x = robot_grid % cols;
	int robot_z = robot_grid / cols;
		    //                         // Get goal position coordinates using goal_location
	int goal_floor = env->level.goal_location / sz;
	int goal_grid = env->level.goal_location % sz;
	int goal_x = goal_grid % cols;
	int goal_z = goal_grid / cols;
	// Calculate Manhattan distance
	int x_dist = abs(robot_x - goal_x);
	int z_dist = abs(robot_z - goal_z);
	int floor_dist = abs(robot_floor - goal_floor);    
	return x_dist + z_dist + floor_dist;
}


int get_direction(CTowerClimb* env, int action) {
    // For reference: 
    // 0 = right (initial), 1 = down, 2 = left, 3 = up
    int current_direction = env->robot_orientation;
    if (env->block_grabbed == -1){
        return action;
    }
    // Check if direction is opposite (differs by 2) or perpendicular (differs by 1 or 3)
    int diff = abs(current_direction - action);
    if(diff == 1 || diff == 3){
        env->block_grabbed = -1;
    }
    else if(diff == 2){
        return current_direction;
    }
    return action;
}


void init(CTowerClimb* env) {
    env->level_number = 0;
    env->level = levels[env->level_number];
    env->board_state = (int*)calloc(LEVEL_MAX_SIZE, sizeof(int));
    env->block_grabbed = -1;
    env->blocks_to_move = (int*)calloc(10, sizeof(int));
    env->blocks_to_fall = (int*)calloc(LEVEL_MAX_SIZE, sizeof(int));
    env->rows_cleared = 0;
    env->robot_orientation = UP;
    env->robot_position = env->level.spawn_location;
    env->distance_to_goal = get_distance_to_goal(env);
    memcpy(env->board_state, env->level.map, env->level.total_length * sizeof(int));
    memset(env->blocks_to_fall, -1, LEVEL_MAX_SIZE * sizeof(int));
    for (int i = 0; i < env->level.cols; i++){
        env->blocks_to_move[i] = -1;
    }
}

void allocate(CTowerClimb* env) {
    init(env);
    env->observations = (float*)calloc(OBS_VISION+PLAYER_OBS, sizeof(float)); // make this unsigned char
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


void print_observation_window(CTowerClimb* env) {
    const int WIDTH = 9;    // x dimension
    const int HEIGHT = 5;   // y dimension
    const int DEPTH = 5;    // z dimension
    
    printf("\nObservation Window (Height layers from bottom to top):\n");
    
    // For each height level
    for (int h = 0; h < HEIGHT; h++) {
        printf("\nLayer %d:\n", h);
        
        // For each depth row
        for (int d = 0; d < DEPTH; d++) {
            printf("  ");  // Indent
            
            // For each width position
            for (int w = 0; w < WIDTH; w++) {
                // Get value from observation array
                // Since observations are in x,z,y format:
                int obs_idx = w + (d * WIDTH) + (h * WIDTH * DEPTH);
                float val = env->observations[obs_idx];
                
                // Print appropriate symbol
                if (val == -1.0f) printf("· ");        // Out of bounds
                else if (val == 0.0f) printf("□ ");    // Empty
                else if (val == 1.0f) printf("■ ");    // Block
                else if (val == 2.0f) printf("G ");    // Goal
                else printf("%.0f ", val);
            }
            printf("\n");
        }
    }
    
    // Print player state
    int state_start = WIDTH * DEPTH * HEIGHT;
    printf("\nPlayer State:\n");
    printf("Orientation: %.0f\n", env->observations[state_start]);
    printf("State: %.0f\n", env->observations[state_start + 1]);
    printf("Block grabbed: %.0f\n", env->observations[state_start + 2]);
    printf("Holding: %.0f\n", env->observations[state_start + 3]);
    printf("Current floor: %.0f\n", env->observations[state_start + 4]);
    printf("\n");
}

void compute_observations(CTowerClimb* env) {
    int sz = env->level.size;
    int cols = env->level.cols;
    int rows = env->level.rows;
    int max_floors = env->level.total_length / sz;
    
    // Get player position
    int current_floor = env->robot_position / sz;
    int grid_pos = env->robot_position % sz;
    int player_x = grid_pos % cols;
    int player_z = grid_pos / cols;
    
    // Calculate window bounds
    int window_width = 9;
    int window_height = 5;
    int window_depth = 5;
    
    // Calculate y (floor) bounds centered on player but adjusted for boundaries
    int y_center = current_floor + 1;
    int half_height = window_height / 2;
    
    // Try to center on player
    int y_start = y_center - half_height;
    int y_end = y_center + half_height + 1;  // +1 because half_height rounds down
    
    // Adjust if too close to bottom
    if (y_start < 0) {
        y_start = 0;
        y_end = window_height;
    }
    
    // Adjust if too close to top
    if (y_end > max_floors) {
        y_end = max_floors;
        y_start = y_end - window_height;
        if (y_start < 0) y_start = 0;
    }
    
    // Calculate x bounds centered on player but adjusted for boundaries
    int half_width = window_width / 2;
    int x_start = player_x - half_width;
    int x_end = x_start + window_width;
    
    // Adjust if too close to left edge
    if (window_width > cols){
        x_start = 0;
        x_end = cols;
    }
    else {
        if (x_start < 0) {
            x_start = 0;
            x_end = window_width;
        }
        
        // Adjust if too close to right edge
        if (x_end > cols) {
            x_end = cols;
            x_start = x_end - window_width;
            if (x_start < 0) x_start = 0;
        }
    }
        
    // Calculate z bounds centered on player but adjusted for boundaries
    int half_depth = window_depth / 2;
    int z_start = player_z - half_depth;
    int z_end = z_start + window_depth;
    
    // Adjust if too close to front edge
    if (z_start < 0) {
        z_start = 0;
        z_end = window_depth;
    }
    
    // Adjust if too close to back edge
    if (z_end > rows) {
        z_end = rows;
        z_start = z_end - window_depth;
        if (z_start < 0) z_start = 0;
    }

    // Fill in observations
    for (int y = 0; y < window_height; y++) {  // Always iterate through full window height
        int world_y = y + y_start;
        for (int z = 0; z < window_depth; z++) {  // Always iterate through full window depth
            int world_z = z + z_start;
            for (int x = 0; x < window_width; x++) {  // Always iterate through full window width
                int world_x = x + x_start;
                int obs_idx = x + z * window_width + y * (window_width * window_depth);
                // Check if position is out of bounds
                int board_idx = world_y * sz + world_z * cols + world_x;
                if (world_x < 0 || world_x >= cols || 
                    world_z < 0 || world_z >= rows || 
                    world_y < 0 || world_y >= max_floors || 
                    board_idx >= env->level.total_length) {
                    env->observations[obs_idx] = -1.0f;
                    continue;
                }
                // Position is in bounds, set observation
                if (board_idx == env->robot_position) {
                    env->observations[obs_idx] = 3.0f;
                    continue;
                }
                env->observations[obs_idx] = (float)env->board_state[board_idx];
            }
        }
    }
    
    // Add player state information at the end
    int state_start = window_width * window_depth * window_height;
    env->observations[state_start] = (float)env->robot_orientation;
    env->observations[state_start + 1] = (float)env->robot_state;
    env->observations[state_start + 2] = (float)env->block_grabbed;
    env->observations[state_start + 3] = (float)(env->block_grabbed != -1);
}

void reset(CTowerClimb* env) {
    env->log = (Log){0};
    env->dones[0] = 0;
    env->robot_orientation = UP;
    env->robot_state = DEFAULT;
    env->block_grabbed = -1;
    env->rows_cleared = 0;
    env->level_number = 0;
    env->level = levels[env->level_number];
    env->robot_position = env->level.spawn_location;
    env->distance_to_goal = get_distance_to_goal(env);
    memcpy(env->board_state, env->level.map, env->level.total_length * sizeof(int));
    if(env->level.total_length < LEVEL_MAX_SIZE){
	    memset(env->board_state + env->level.total_length, 0, (LEVEL_MAX_SIZE-env->level.total_length) * sizeof(int));
    }
    memset(env->blocks_to_move, -1, env->level.cols * sizeof(int));
    memset(env->blocks_to_fall, -1, env->level.total_length * sizeof(int));
    compute_observations(env);
}


void illegal_move(CTowerClimb* env){
    env->rewards[0] = env->reward_illegal_move;
    env->log.episode_return += env->reward_illegal_move;
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

    int dx = DIRECTION_VECTORS_X[env->robot_orientation]; // possibly make dx and dz 1D arrays
    int dz = DIRECTION_VECTORS_Z[env->robot_orientation];

    int next_x = x + dx;
    int next_z = z + dz;
    if (env->robot_state == HANGING){
        illegal_move(env);
        return;
    }
    if (next_x < 0 || next_x >= cols || next_z < 0 || next_z >= rows ) {
        // Attempting to move outside the 4x4 grid on this floor - do nothing
        return;
    }
    int next_index = sz*current_floor + cols*next_z + next_x;
    int next_cell = env->board_state[next_index];
    if (next_cell!=1){
        illegal_move(env);
        return;
    }
    if(env->block_grabbed == next_index){
        env->block_grabbed = -1;
    } else {
        env->block_grabbed = next_index;
    }
    //env->rewards[0] = .2;
    //env->log.episode_return += 0.2;
}

int is_next_to_block(CTowerClimb* env, int target_position){
    int sz = env->level.size;
    int cols = env->level.cols;
    int current_floor = target_position / sz;
    int grid_pos = target_position % sz;
    int x = grid_pos % cols;
    int z = grid_pos / cols;
    int dx = DIRECTION_VECTORS_X[env->robot_orientation];
    int dz = DIRECTION_VECTORS_Z[env->robot_orientation];
    int next_x = x + dx;
    int next_z = z + dz;    
    int next_index = sz*current_floor + cols*next_z + next_x;  // semantically group multiplication
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
        int b_address = start_index + i*interval;
        if(i!=0 && env->blocks_to_move[i-1] == -1){
            break;
        }
        if(env->board_state[b_address] == 1){
            env->blocks_to_move[i] = b_address;
        }
    }
}

// Helper function to check if position is within bounds
static inline int is_valid_position(int pos) {
    return (pos >= 0 && pos < LEVEL_MAX_SIZE);
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

int  add_blocks_to_fall(CTowerClimb* env){
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
        if(env->blocks_to_move[i] == -1){
            continue;
        }
        // Add block directly above
        int cell_above = env->blocks_to_move[i] + sz;
        
        // If valid block above and unstable, add to queue
        if (cell_above < total_length && (env->board_state[cell_above] == 1 || env->board_state[cell_above] ==2)
        && !is_block_stable(env, cell_above)) {
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
                (env->board_state[edge_blocks[j]] == 1 || env->board_state[edge_blocks[j]] ==2)
                && !is_block_stable(env, edge_blocks[j])) {
                queue[rear++] = edge_blocks[j];
            }
        }
    }

    // Process queue until empty
    while(front < rear) {
        int current = queue[front++];
        int falling_position = current;
        int found_support = 0;
        if(env->board_state[current]==2){
            return 0;
        }
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
          
        // Add blocks that might be affected by this fall
        int original_above = current + sz;  // Block directly above original position
        if (original_above < total_length && (env->board_state[original_above] == 1 || env->board_state[original_above] ==2) && !is_block_stable(env, original_above)) {
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
                (env->board_state[edge_check[i]] == 1 || env->board_state[edge_check[i]] ==2)
                && !is_block_stable(env, edge_check[i])) {
                queue[rear++] = edge_check[i];
            }
        }
        
        // If no support found, block disappears (already set to 0)
    }
    // death by block crush
    memset(env->blocks_to_fall, -1, LEVEL_MAX_SIZE * sizeof(int));
    memset(env->blocks_to_move, -1, cols * sizeof(int));
    return 1;
}
void move_blocks(CTowerClimb* env, int interval){
    int count = 0;
    int cols = env->level.cols;
    int rows = env->level.rows;
    int sz = env->level.size;
    // cache reused variables 
    for (int i = 0; i < cols; i++){
        // invert conditional 
        int b_index = env->blocks_to_move[i];
        if (b_index == -1){
            continue;
        }
        if(i==0){
            env->board_state[b_index] = 0;
        }
        int grid_pos = b_index % sz;
        int x = grid_pos % cols;
        int z = grid_pos / cols;
        // Check if movement would cross floor boundaries
        if ((x == 0 && interval == -1) ||           // Don't move left off floor
            (x == cols-1 && interval == 1) ||       // Don't move right off floor
            (z == 0 && interval == -cols) ||        // Don't move forward off floor
            (z == rows-1 && interval == cols)) {    // Don't move back off floor
            return;
        }

        // If we get here, movement is safe
        env->board_state[b_index + interval] = 1;
        env->blocks_to_fall[count] = b_index + interval;
        count++;
    }
}

void shimmy(CTowerClimb* env, int action, int current_floor, int x, int z, int x_mod, int z_mod, int x_direction_mod, int z_direction_mod, int final_orienation){
    int sz = env->level.size;
    int cols = env->level.cols;
    int next_block = sz*current_floor + (z+z_mod)*cols + (x+x_mod);
    int destination_block = sz*current_floor + (z+z_direction_mod)*cols + (x+x_direction_mod);
    if (env->board_state[next_block] == 0){
        illegal_move(env);
        return;
    }
    int above_destination = destination_block + sz;
    // cant wrap shimmy with block above.
    if(env->board_state[above_destination] == 1){
        illegal_move(env);
        return;
    }
    env->robot_position = destination_block;
    env->robot_orientation = final_orienation;
    return;
}

void handle_climb(CTowerClimb* env, int action, int current_floor,int x, int z, int next_z, int next_x, int next_index, int next_cell){
    if (!(next_cell == 1 || next_cell == 2) || env->block_grabbed != -1) {
        illegal_move(env);
        return;
    }
    if (current_floor >= MAX_FLOORS){
        return;
    }
    int sz = env->level.size;
    int cols = env->level.cols;
    int climb_index = sz*(current_floor + 1) + cols*next_z + next_x;
    int climb_cell = env->board_state[climb_index];
    int direct_above = env->robot_position + sz;
    int direct_above_cell = env->board_state[direct_above];
    if ((climb_cell != 0 || direct_above_cell != 0) && next_cell != 2){
        illegal_move(env);
        return;
    }
    env->robot_position = climb_index;
    env->robot_state = DEFAULT; // set hanging to indicate we climbed a block
    int previous_row = (env->robot_position - sz) / sz;
    if(previous_row <= env->rows_cleared){
        return;
    }
    env->rows_cleared = previous_row;
    env->rewards[0] = env->reward_climb_row;
    env->log.episode_return += env->reward_climb_row;  
}


void handle_down(CTowerClimb* env, int action, int current_floor,int x, int z, int next_z, int next_x, int next_index, int next_cell){
    int sz = env->level.size;
    int cols = env->level.cols;
    int below_index = sz*(current_floor - 1) + cols*next_z + next_x;
    int below_cell;
    if(below_index >= 0){
	   below_cell =  env->board_state[below_index];
    } else {
	    below_cell = -1;
    }
    int below_next_index = below_index - sz;
    int below_next_cell;
    if (below_next_index >= 0){
	    below_next_cell = env->board_state[below_next_index];
    } else{
    	below_next_cell = 0;
    }
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
        env->rewards[0] = env->reward_fall_row;
        env->log.episode_return += env->reward_fall_row;
    }
}

void handle_left_right(CTowerClimb* env, int action, int current_floor,int x, int z, int next_z, int next_x, int next_index, int next_cell){
    // Check if the cell in front is free, the goal, or blocked
    int sz = env->level.size;
    int cols = env->level.cols;
    int rows = env->level.rows;
    int below_index = sz*(current_floor - 1) + cols*next_z + next_x;
    int below_cell;
    if (below_index >=0){
	below_cell = env->board_state[below_index];
    } else {
	    below_cell = -1;
    }
    // shimmying
    if (env->robot_state == HANGING){
        int local_direction = get_local_direction(env, action);
        int local_next_dx = DIRECTION_VECTORS_X[local_direction];
        int local_next_dz = DIRECTION_VECTORS_Z[local_direction];
        int local_next_x = x + local_next_dx;
        int local_next_z = z + local_next_dz;
        if(local_next_x < 0 || local_next_z < 0 || local_next_x >= cols || local_next_z >= rows){
            illegal_move(env);
            return;
        }
        int local_next_index = sz*current_floor + cols*local_next_z + local_next_x;
        int local_next_cell = env->board_state[local_next_index];
        if (is_next_to_block(env, local_next_index) && local_next_cell == 0){
            // standard shimmy
            int above_index = local_next_index + sz;
            int above_cell = env->board_state[above_index];
            // cant shimmy with block above
            if (above_cell == 1 ){
                illegal_move(env);
                return;
            }
            env->robot_position = local_next_index;
            return;
        }

        if(local_next_cell == 1){
            static const int LEFT_TURNS[] = {3, 0, 1, 2};   // RIGHT->UP, DOWN->RIGHT, LEFT->DOWN, UP->LEFT
            static const int RIGHT_TURNS[] = {1, 2, 3, 0};  // RIGHT->DOWN, DOWN->LEFT, LEFT->UP, UP->RIGHT

            env->robot_orientation = (action == LEFT) ? 
                LEFT_TURNS[env->robot_orientation] : 
                RIGHT_TURNS[env->robot_orientation];
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
void handle_unstable_goal(CTowerClimb* env){
	env->rewards[0] = -1;
	env->log.episode_return -= 1;
	add_log(env->log_buffer, &env->log);
	reset(env);
}

void handle_move_forward(CTowerClimb* env, int action) {
    int sz = env->level.size;
    int cols = env->level.cols;
    int rows = env->level.rows;
    // Calculate current floor, x, z from the robot_position
    int current_floor = env->robot_position / sz;
    int grid_pos = env->robot_position % sz;
    int x = grid_pos % cols;
    int z = grid_pos / cols;

    // Determine the offset for the next cell based on env->robot_direction
    // make this 1D
    int dx = DIRECTION_VECTORS_X[env->robot_direction];
    int dz = DIRECTION_VECTORS_Z[env->robot_direction];
    
    int orient_dx = DIRECTION_VECTORS_X[env->robot_orientation];
    int orient_dz = DIRECTION_VECTORS_Z[env->robot_orientation];

    int front_x = x + orient_dx;
    int front_z = z + orient_dz;
    int front_index = sz*current_floor + cols*front_z + front_x;
    int front_cell = env->board_state[front_index];
    int front_below_index = front_index - sz;
    int front_below_cell;
    if(front_below_index >= 0){
	    front_below_cell = env->board_state[front_below_index];
    } else {
	    front_below_cell = -1;
    }
    int front_below_below_index = front_below_index - sz;
    int front_below_below_cell;
    if(front_below_below_index>=0){
        front_below_below_cell = env->board_state[front_below_below_index];
    } else {
        front_below_below_cell = -1;
    }
    // Calculate the next potential cell's x, z
    int next_x = x + dx;
    int next_z = z + dz;
    if(env->block_grabbed != -1 && abs(env->robot_direction - action) == 2){
        // inverse direction based on direction vectors 
        dx = DIRECTION_VECTORS_X[action];
        dz = DIRECTION_VECTORS_Z[action];
        next_x = x + dx;
        next_z = z + dz;
    }

    if (front_x < 0 || front_z < 0 || front_x >= cols || front_z >= rows){
        illegal_move(env);
        return;
    }
    
    // Convert next x,z to a linear index for the same floor
    int next_index = sz*current_floor + cols*next_z + next_x;
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
        if(front_cell == 0 && front_below_cell == 0 && front_below_below_cell == 0){
            env->robot_position = front_below_index;
            env->robot_state = HANGING;
            env->robot_orientation = DOWN;
        }
        if(front_cell == 0 && front_below_cell == 0 && front_below_below_cell == 1){
            env->robot_position = front_below_index;
            env->robot_state = DEFAULT;
        }
    }
    if (action == DOWN && env->block_grabbed == -1){
        if (env->robot_state == HANGING){
            illegal_move(env);
            return;
        }
        if (front_cell == 1 || front_cell == 2){
            handle_climb(env, action, current_floor, x, z, front_z, front_x, front_index, front_cell);
        } else {
            handle_down(env, action, current_floor, x, z, next_z, next_x, next_index, next_cell);
        }
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
        if(abs(env->robot_orientation - action) == 0){
            add_blocks_to_move(env, block_offset);
            move_blocks(env, block_offset);
            int stable = add_blocks_to_fall(env);
            if (stable == 0){
                handle_unstable_goal(env);
            }
            //env->rewards[0] = 0.25;
            //env->log.episode_return += 0.25;
        }
        // Pulling
        if (abs(env->robot_orientation - action) == 2){
            int below_index = sz*(current_floor - 1) + cols*next_z + next_x;
            int below_cell = env->board_state[below_index];
            if(next_cell == 1){
                illegal_move(env);
                return;
            }
            env->blocks_to_move[0] = front_index;
            env->board_state[front_index] = 0;
            env->board_state[next_index + block_offset] = 1;
            env->block_grabbed = next_index + block_offset;
            int stable = add_blocks_to_fall(env);
            if(stable ==0){
		    handle_unstable_goal(env);
		    return;
	    }
	    env->blocks_to_move[0] = -1;
            if (below_cell == 0 && env->robot_state == DEFAULT){
                env->robot_position = below_index;
                env->robot_state = HANGING;
            } else {
                env->robot_position = next_index;
            }
            //env->rewards[0] = 0.25;
            //env->log.episode_return +=0.25;
        }
        if (env->robot_position != next_index){
            env->block_grabbed = -1;
        }
        memset(env->blocks_to_move, -1, cols * sizeof(int));
    }
}

void handle_drop(CTowerClimb* env){
    env->robot_state = DEFAULT;
    int sz = env->level.size;
    int next_position = env->robot_position - sz;
    // fallen off the map
    if (env->robot_position < sz){
        env->rewards[0] = -1;
        env->log.episode_return -= 1;
        add_log(env->log_buffer, &env->log);
        reset(env);
        return;
    }
    while (env->board_state[next_position] != 1){
        if(next_position < sz){
            env->rewards[0] = -1;
            env->log.episode_return -= 1;
            add_log(env->log_buffer, &env->log);
            reset(env);
            return;
        } 
        next_position = next_position - sz;  
    }
    env->robot_position = next_position + sz;
}

void next_level(CTowerClimb* env){
    if(env->level_number == 2){
        env->rewards[0] = 1;
        env->log.episode_return += 1;
        reset(env);
        return;
    }
    env->rows_cleared= 0;
    env->level_number += 1;
    env->level = levels[env->level_number];
    env->robot_position = env->level.spawn_location;
    env->distance_to_goal = get_distance_to_goal(env);
    env->robot_state = DEFAULT;
    env->robot_orientation = UP;
    env->robot_direction = UP;
    env->block_grabbed = -1;
    memcpy(env->board_state, env->level.map, env->level.total_length * sizeof(int));
    memset(env->blocks_to_move, -1, env->level.cols * sizeof(int));
    memset(env->blocks_to_fall, -1, env->level.total_length * sizeof(int));
}

void step(CTowerClimb* env) {
    env->log.episode_length += 1.0;
    env->rewards[0] = 0.0;
    // if(env->log.episode_length >200){
	//        env->rewards[0] = 0;
	//        //env->log.episode_return +=0;
	//        add_log(env->log_buffer, &env->log);
	//        reset(env);
    // }
    int action = env->actions[0];
    if (action == LEFT || action == RIGHT || action == DOWN || action == UP) {
        int direction = get_direction(env, action);
        int moving_block = (env->block_grabbed != -1 && abs(env->robot_orientation - action) == 2);
        if(moving_block){
            env->rewards[0] = env->reward_move_block;
            env->log.episode_return += env->reward_move_block;
        }
        if (direction == env->robot_orientation || moving_block || env->robot_state == HANGING){
            env->robot_direction = direction;
            handle_move_forward(env, action);
            int sz = env->level.size;
            int below_index = env->robot_position - sz;
            // check if goal is below current position
            if(below_index < 0){
                env->log.rows_cleared = env->rows_cleared;
                compute_observations(env);
                int distance = get_distance_to_goal(env);
                int delta_distance = env->distance_to_goal - distance;
                float distance_reward = delta_distance * env->reward_distance;
                env->rewards[0] += distance_reward;
                env->log.episode_return += distance_reward;
	        }
            else if (env->board_state[below_index] == 2){
                env->rewards[0] = 1;
                env->log.episode_return += 1;
                env->log.rows_cleared = env->rows_cleared;
                env->log.levels_completed = env->level_number + 1;
                add_log(env->log_buffer, &env->log);
                next_level(env);
                //reset(env);
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
        if(env->robot_state == HANGING) handle_drop(env);
    }
    // check if block fell on you
    if(env->board_state[env->robot_position] == 1){
        env->rewards[0] = -1;
        env->log.episode_return -= 1;
        add_log(env->log_buffer, &env->log);
        reset(env);
    }
    env->log.rows_cleared = env->rows_cleared;
    int distance = get_distance_to_goal(env);
    int delta_distance = env->distance_to_goal - distance;
    float distance_reward = delta_distance * env->reward_distance;
    env->rewards[0] += distance_reward;
    env->log.episode_return += distance_reward;
    env->distance_to_goal = distance;
    compute_observations(env);
    // if(action != NOOP){
    //     print_observation_window(env);
    // }
}

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_GREY = (Color){128, 128, 128, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};

typedef enum {
    ANIM_IDLE,
    ANIM_RUNNING,
    ANIM_CLIMBING,
    ANIM_HANGING,
    ANIM_START_GRABBING,
    ANIM_GRABBING,
    ANIM_SHIMMY_RIGHT,
    ANIM_SHIMMY_LEFT,
} AnimationState;

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
    Texture2D background;
    Camera3D camera;
    Model robot;
    Light lights[MAX_LIGHTS];
    Shader shader; 
    ModelAnimation* animations;
    int animFrameCounter;
    AnimationState animState;
    int previousRobotPosition;
    Vector3 visualPosition;
    Vector3 targetPosition;
    bool isMoving;
    float moveProgress;
    Model cube;
    float scale;
};

Client* make_client(CTowerClimb* env) {
    printf("Raylib version: %s\n", RAYLIB_VERSION);
    printf("OpenGL version: %d\n", rlGetVersion());

    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 1600;
    client->height = 900;
    SetConfigFlags(FLAG_MSAA_4X_HINT);  // Enable MSAA
    InitWindow(client->width, client->height, "PufferLib Ray Tower Climb");
    SetTargetFPS(60);
    client->camera = (Camera3D){ 0 };
    client->camera.position = (Vector3){ 0.0f, 25.0f, 20.0f };  // Move camera further back and higher up
    client->camera.target = (Vector3){ 2.0f, 4.0f, 2.0f };     // Keep looking at same target point
    client->camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    // load background
    client->background = LoadTexture("resources/tower_climb/space2.jpg");
    client->puffers = LoadTexture("resources/puffers_128.png");


    // load robot
    client->robot = LoadModel("resources/tower_climb/small_astro.glb");
    client->cube = LoadModel("resources/tower_climb/spacerock.glb");
    BoundingBox bounds = GetModelBoundingBox(client->cube);
    float cubeSize = bounds.max.x - bounds.min.x;
    float scale = 1.0f / cubeSize;
    client->scale = scale; 
    int animCount = 0;
    client->animations = LoadModelAnimations("resources/tower_climb/small_astro.glb", &animCount);
    for (int i = 0; i < animCount; i++) {
        if (client->robot.meshes[0].boneCount != client->animations[i].boneCount) {
            printf("Warning: Animation %d bone count (%d) doesn't match model bone count (%d)\n",
                   i, client->animations[i].boneCount, client->robot.meshes[0].boneCount);
        }
    }
    printf("Loaded %d animations\n", animCount);
 
    
    client->animState = ANIM_IDLE;
    client->animFrameCounter = 0;
    UpdateModelAnimation(client->robot, client->animations[4], 0); 
    // Load and configure shader
    char vsPath[256];
    char fsPath[256];
    sprintf(vsPath, "resources/tower_climb/shaders/gls%i/lighting.vs", GLSL_VERSION);
    sprintf(fsPath, "resources/tower_climb/shaders/gls%i/lighting.fs", GLSL_VERSION);
    client->shader = LoadShader(vsPath, fsPath);
    // Get shader locations
    client->shader.locs[SHADER_LOC_VECTOR_VIEW] = GetShaderLocation(client->shader, "viewPos");
    // Set up ambient light
    int ambientLoc = GetShaderLocation(client->shader, "ambient");
    float ambient[4] = { 0.1f, 0.1f, 0.1f, 1.0f };
    SetShaderValue(client->shader, ambientLoc, ambient, SHADER_UNIFORM_VEC4);
    
    // apply lighting shader
    client->robot.materials[0].shader = client->shader;
    client->cube.materials[0].shader = client->shader;
    // Create lights with much brighter colors and closer positions
    client->lights[0] = CreateLight(LIGHT_POINT, 
        (Vector3){ 10.0f, 7.0f, 5.0f },  // Front
        Vector3Zero(), 
        WHITE,  // ~12% intensity
        client->shader);
    
    // Very dim fill lights
    client->lights[1] = CreateLight(LIGHT_POINT, 
        (Vector3){ 3.0f, 5.0f, 8.0f },  // Right side
        Vector3Zero(), 
        (Color){ 40, 40, 40, 255 },  // ~15% intensity
        client->shader);
    
    client->lights[2] = CreateLight(LIGHT_POINT, 
        (Vector3){ 10.0f, 7.0f, 5.0f },  // Front
        Vector3Zero(), 
        (Color){ 30, 30, 30, 255 },  // ~12% intensity
        client->shader);
    
    client->lights[3] = CreateLight(LIGHT_POINT, 
        (Vector3){ 10.0f, 6.0f, -5.0f },  // Left side
        Vector3Zero(), 
        (Color){ 20, 20, 20, 255 },  // ~8% intensity
        client->shader);

    // Make sure both models' materials use the lighting shader
    for (int i = 0; i < client->robot.materialCount; i++) {
        client->robot.materials[i].shader = client->shader;
    }
    for (int i = 0; i < client->cube.materialCount; i++) {
        client->cube.materials[i].shader = client->shader;
    }
    client->animState = ANIM_IDLE;
    client->previousRobotPosition = env->robot_position;
    
    // Initialize visual position to match starting robot position
    int floor = env->robot_position / env->level.size;
    int grid_pos = env->robot_position % env->level.size;
    int x = grid_pos % env->level.cols;
    int z = grid_pos / env->level.cols;
    
    client->visualPosition = (Vector3){ 
        x * 1.0f,
        floor * 1.0f,
        z * 1.0f
    };
    client->targetPosition = client->visualPosition;  // Initialize target to match
    
    return client;
}

void orient_hang_offset(Client* client, CTowerClimb* env, int reverse){

    client->visualPosition.y -= 0.2f * reverse;
    if (env->robot_orientation == 0) { // Facing +x
        client->visualPosition.x += 0.4f * reverse;
    } else if (env->robot_orientation == 1) { // Facing +z
        client->visualPosition.z += 0.4f * reverse;
    } else if (env->robot_orientation == 2) { // Facing -x
        client->visualPosition.x -= 0.4f * reverse;
    } else if (env->robot_orientation == 3) { // Facing -z
        client->visualPosition.z -= 0.4f * reverse;
    }
}


void render(Client* client, CTowerClimb* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    static int stored_orientation = -1;  // Store original orientation during climb
    int cols = env->level.cols;
    int sz = env->level.size;
    int floor = env->robot_position / sz;
    int grid_pos = env->robot_position % sz;
    int x = grid_pos % cols;
    int z = grid_pos / cols;

    if (env->robot_state == DEFAULT && client->animState == ANIM_HANGING) {
        client->animState = ANIM_IDLE;
        client->animFrameCounter = 0;
        UpdateModelAnimation(client->robot, client->animations[4], 0);
        client->isMoving = false;
        client->visualPosition = client->targetPosition;
    }

    if (env->block_grabbed != -1 && client->animState != ANIM_GRABBING && client->animState != ANIM_START_GRABBING)
    {
        client->animState = ANIM_START_GRABBING;
        client->isMoving = true;
        client->animFrameCounter = 0;
        UpdateModelAnimation(client->robot, client->animations[3], 0); 
        client->visualPosition = client->targetPosition;

    } else if (env->block_grabbed == -1 && client->animState == ANIM_GRABBING) {
        // Reset to idle when block is released
        client->animState = ANIM_IDLE;
        client->animFrameCounter = 0;
        UpdateModelAnimation(client->robot, client->animations[4], 0);
    }

    // Check if robot position has changed
    if (env->robot_position != client->previousRobotPosition) {
        if (client->isMoving) {
            client->visualPosition = client->targetPosition;
        }
        
        client->isMoving = true;
            
        // Calculate new target position
        int floor = env->robot_position / env->level.size;
        int grid_pos = env->robot_position % env->level.size;
        int x = grid_pos % env->level.cols;
        int z = grid_pos / env->level.cols;
        
        client->targetPosition = (Vector3){
            x * 1.0f,
            floor * 1.0f,
            z * 1.0f
        };

        // Choose animation based on vertical movement
        if (client->targetPosition.y - client->visualPosition.y > 0.5) {
            // Add offset based on robot orientation
            if(client->animState == ANIM_HANGING){
                orient_hang_offset(client, env,0);
            } else {
                orient_hang_offset(client, env, 1);
            }
            client->animState = ANIM_CLIMBING;  // Need to add this to AnimationState enum
            UpdateModelAnimation(client->robot, client->animations[1], 0);  // Assuming climbing is animation 4
            client->animFrameCounter = 6;
        } else if (env->robot_state == HANGING){
            client->isMoving = true;
            // Check for wrap shimmy case - when both x and z change
            bool is_wrap_shimmy = fabs(client->targetPosition.x - client->visualPosition.x) > 0.5f && 
                                fabs(client->targetPosition.z - client->visualPosition.z) > 0.5f;

            if (is_wrap_shimmy) {
                // Skip animation and move directly to target for wrap case
                client->isMoving = false;
                client->animState = ANIM_HANGING;
                client->animFrameCounter = 0;
                client->visualPosition = client->targetPosition;
                orient_hang_offset(client, env, 1);
                UpdateModelAnimation(client->robot, client->animations[2], 0);
            } else {
                // Regular shimmy animation logic
                bool moving_right = false;
                if (env->robot_orientation == UP) {  // Facing up (-z)
                    moving_right = client->targetPosition.x > client->visualPosition.x;
                } else if (env->robot_orientation == DOWN) {  // Facing down (+z)
                    moving_right = client->targetPosition.x < client->visualPosition.x;
                } else if (env->robot_orientation == RIGHT) {  // Facing right (+x)
                    moving_right = client->targetPosition.z < client->visualPosition.z;
                } else if (env->robot_orientation == LEFT) {  // Facing left (-x)
                    moving_right = client->targetPosition.z > client->visualPosition.z;
                }
                
                client->animFrameCounter = 0;
                if (client->targetPosition.y < client->visualPosition.y) {
                    client->animState = ANIM_HANGING;
                    orient_hang_offset(client, env, 1);
                    UpdateModelAnimation(client->robot, client->animations[1], 0);
                } else if (moving_right) {
                    client->animState = ANIM_SHIMMY_RIGHT;
                    orient_hang_offset(client, env, 0);
                    UpdateModelAnimation(client->robot, client->animations[1], 0);
                } else {
                    client->animState = ANIM_SHIMMY_LEFT;
                    orient_hang_offset(client, env, 0);
                    UpdateModelAnimation(client->robot, client->animations[1], 0);
                }
                client->animFrameCounter = 0;
            }
        }
        else {
            if (client->targetPosition.y < client->visualPosition.y) {
                client->animState = ANIM_IDLE;
                client->animFrameCounter = 0;
                UpdateModelAnimation(client->robot, client->animations[4], 0);
                client->isMoving = false;
                client->visualPosition = client->targetPosition;
            } else{
                client->animState = ANIM_RUNNING;
                UpdateModelAnimation(client->robot, client->animations[5], 0);
                client->animFrameCounter = 0;
            }
            
        }
        client->previousRobotPosition = env->robot_position;
    }

    // Update animation if moving
    if (client->isMoving) {
        // Use appropriate animation based on state
        if (client->animState == ANIM_CLIMBING) {
            if (stored_orientation == -1) {
                stored_orientation = env->robot_orientation;
            }
            int current_orientation = env->robot_orientation;
            env->robot_orientation = stored_orientation;
            client->animFrameCounter += 6;
            UpdateModelAnimation(client->robot, client->animations[1], client->animFrameCounter);
            if (client->animFrameCounter >= client->animations[1].frameCount) {  // Adjust frame count for climbing animation
                client->isMoving = false;
                client->animState = ANIM_IDLE;
                client->animFrameCounter = 0;
                UpdateModelAnimation(client->robot, client->animations[4], client->animFrameCounter);
                client->visualPosition = client->targetPosition;
                stored_orientation = -1;
                env->robot_orientation = current_orientation;
            }
        } else if (client->animState == ANIM_HANGING){
            client->animFrameCounter = 0;
            UpdateModelAnimation(client->robot, client->animations[2], client->animFrameCounter);
            client->isMoving = false;
            client->visualPosition = client->targetPosition;
            orient_hang_offset(client, env, 1);

        } else if (client->animState == ANIM_SHIMMY_RIGHT || client->animState == ANIM_SHIMMY_LEFT) {
            client->animFrameCounter += 2;
            // Use animation 23 for right shimmy, 19 for left shimmy
            int animIndex = (client->animState == ANIM_SHIMMY_RIGHT) ? 7 : 6;
            UpdateModelAnimation(client->robot, client->animations[animIndex], client->animFrameCounter);
                        // Calculate lerp progress (0 to 1)
            float progress = (float)client->animFrameCounter / 87.0f;
            if (progress > 1.0f) progress = 1.0f;
            
            // Lerp position based on orientation
            if (env->robot_orientation == UP) {  // Facing -z
                client->visualPosition.x = Lerp(client->visualPosition.x, client->targetPosition.x, progress);
            } else if (env->robot_orientation == DOWN) {  // Facing +z
                client->visualPosition.x = Lerp(client->visualPosition.x, client->targetPosition.x, progress);
            } else if (env->robot_orientation == RIGHT) {  // Facing +x
                client->visualPosition.z = Lerp(client->visualPosition.z, client->targetPosition.z, progress);
            } else if (env->robot_orientation == LEFT) {  // Facing -x
                client->visualPosition.z = Lerp(client->visualPosition.z, client->targetPosition.z, progress);
            }
            if (client->animFrameCounter >= 87) {
                client->isMoving = false;
                client->animState = ANIM_HANGING;
                client->animFrameCounter = 0;
                UpdateModelAnimation(client->robot, client->animations[2], client->animFrameCounter);
                client->visualPosition = client->targetPosition;
                orient_hang_offset(client, env, 1);
            }
        } else if (client->animState == ANIM_START_GRABBING) {
            if (client->animFrameCounter >= client->animations[3].frameCount - 2) {
                // Lock at second-to-last frame to prevent animation loop
                client->animFrameCounter = client->animations[3].frameCount - 2;
                UpdateModelAnimation(client->robot, client->animations[3], client->animFrameCounter);
                client->isMoving = false;
                client->animState = ANIM_GRABBING;
            } else {
                UpdateModelAnimation(client->robot, client->animations[3], client->animFrameCounter);
                client->animFrameCounter += 6;
            }
        }
        
        else {  // ANIM_RUNNING
            client->animFrameCounter += 4;
            UpdateModelAnimation(client->robot, client->animations[5], client->animFrameCounter);
            if (client->animFrameCounter >= client->animations[5].frameCount) {
                client->isMoving = false;
                client->animState = ANIM_IDLE;
                client->animFrameCounter = 0;
                UpdateModelAnimation(client->robot, client->animations[4], client->animFrameCounter);
                client->visualPosition = client->targetPosition;
            }
        }
    }else if (client->animState == ANIM_HANGING){
        client->animFrameCounter = 0;
        UpdateModelAnimation(client->robot, client->animations[2], client->animFrameCounter);
        client->visualPosition = client->targetPosition;
        orient_hang_offset(client, env, 1);

    } else if (client->animState == ANIM_IDLE) {
        // Idle animation code remains the same
        client->animFrameCounter++;
        UpdateModelAnimation(client->robot, client->animations[4], client->animFrameCounter);
        if (client->animFrameCounter >= client->animations[4].frameCount) {
            client->animFrameCounter = 0;
        }
    } else if (client->animState == ANIM_GRABBING) {
        UpdateModelAnimation(client->robot, client->animations[3], client->animations[3].frameCount - 2);
        client->visualPosition = client->targetPosition;
    }

    // Camera update code remains the same
    int cameraFloor = (floor-1) *0.5;
    float targetCameraY = cameraFloor * 1.0f + 7.0f;
    float targetLookY = cameraFloor * 1.0f;
    float smoothSpeed = 0.025f;
    client->camera.position.y = Lerp(client->camera.position.y, targetCameraY, smoothSpeed);
    client->camera.target.y = Lerp(client->camera.target.y, targetLookY, smoothSpeed);
    client->camera.position.x = env->level.cols * 0.5;
    client->camera.position.z = 15.0f;
    client->camera.target.x = 4.0f;
    client->camera.target.z = 1.0f;

    BeginDrawing();
    ClearBackground(BLACK);
    

    // Center vertically    
    EndShaderMode();
    // Calculate scaling to maintain aspect ratio and fill screen
    float scaleWidth = (float)client->width / client->background.width;
    float scaleHeight = (float)client->height / client->background.height;
    float scale = fmax(scaleWidth, scaleHeight);  // Use larger scale to fill screen
    
    float newWidth = client->background.width * scale;
    float newHeight = client->background.height * scale;
    
    // Center the scaled image
    float img_x = (client->width - newWidth) * 0.5f;
    float img_y = (client->height - newHeight) * 0.5f;
    
    DrawTexturePro(client->background,
        (Rectangle){ 0, 0, (float)client->background.width, (float)client->background.height },
        (Rectangle){ img_x, img_y, newWidth, newHeight },
        (Vector2){ 0, 0 },
        0.0f,
        WHITE);
    BeginShaderMode(client->shader);
    BeginMode3D(client->camera);
    float cameraPos[3] = { 
        client->camera.position.x, 
        client->camera.position.y, 
        client->camera.position.z 
    };
    SetShaderValue(client->shader, client->shader.locs[SHADER_LOC_VECTOR_VIEW], cameraPos, SHADER_UNIFORM_VEC3);

    BeginBlendMode(BLEND_ALPHA);
    int total_length = env->level.total_length;
    for(int i= 0; i < total_length; i++){
        if(env->board_state[i] > 0){
            int floor = i / sz;
            int grid_pos = i % sz;
            int x = grid_pos % cols;
            int z = grid_pos / cols;
            if (env->board_state[i] == 2){
                EndShaderMode();
                DrawCube(
                (Vector3){x * 1.0f, floor * 1.0f, z * 1.0f},
                1.0f, 1.0f, 1.0f,
                PUFF_CYAN
                );
                BeginShaderMode(client->shader);

            } else {
                DrawModel(client->cube, (Vector3){x * 1.0f, floor * 1.0f, z * 1.0f}, client->scale, WHITE);
            }
            
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
    EndBlendMode();

    // calculate robot position
    Vector3 spherePos = (Vector3){ 
        x * 1.0f,
        floor * 1.0f,  // One unit above platform
        z * 1.0f
    };

    // Draw sphere character
    // DrawSphere(spherePos, 0.3f, YELLOW);  // 0.3 radius yellow sphere
    spherePos = client->visualPosition;  // Use visual position instead of calculating from env
    spherePos.y -= 0.5f;
    rlPushMatrix();
    rlTranslatef(spherePos.x, spherePos.y, spherePos.z);
    rlRotatef(90.0f, 1, 0, 0);
    rlRotatef(-90.0f + env->robot_orientation * 90.0f, 0, 0, 1);
    DrawModel(client->robot, (Vector3){ 0, 0, 0}, 0.5f, WHITE);
    rlPopMatrix();

    
    // Draw light debug spheres
    // for (int i = 0; i < MAX_LIGHTS; i++) {
    //     if (client->lights[i].enabled) {
    //         DrawSphereEx(client->lights[i].position, 0.2f, 8, 8, client->lights[i].color);
    //     } 
    // }
    EndMode3D();
    EndDrawing();
}
void close_client(Client* client) {
    UnloadShader(client->shader);
    CloseWindow();
    UnloadModel(client->robot);
    free(client);
}

