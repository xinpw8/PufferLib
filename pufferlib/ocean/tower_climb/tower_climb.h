#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
// #include "levels.h"
#include "rlgl.h"
#include <time.h>

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
// robot state
#define DEFAULT 0
#define HANGING 1
// observation space
#define PLAYER_OBS 4
#define OBS_VISION 225
// PLG VS ENV
#define PLG_MODE 0
#define RL_MODE 1
//logs
#define LOG_BUFFER_SIZE 1024
// level size
#define row_max 10
#define col_max 10
#define depth_max 10
// block bytes
#define BLOCK_BYTES 125
// FNV-1a hash function
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL
// moves
#define MOVE_ILLEGAL 0
#define MOVE_SUCCESS 1
#define MOVE_DEATH 2
// bitmask operations
#define SET_BIT(mask, i)    ( (mask)[(i)/8] |=  (1 << ((i)%8)) )
#define CLEAR_BIT(mask, i)  ( (mask)[(i)/8] &= ~(1 << ((i)%8)) )
#define TEST_BIT(mask, i)   ( ((mask)[(i)/8] & (1 << ((i)%8))) != 0 )

// BFS
#define MAX_BFS_SIZE 70000000
#define MAX_NEIGHBORS 6 // based on action space

// hash table 
#define TABLE_SIZE 70000003

// actions
#define ACTION_RIGHT 0
#define ACTION_DOWN 1
#define ACTION_LEFT 2
#define ACTION_UP 3
#define ACTION_GRAB 4
#define ACTION_DROP 5

// direction vectors
#define NUM_DIRECTIONS 4
static const int BFS_DIRECTION_VECTORS_X[NUM_DIRECTIONS] = {1, 0, -1, 0};
static const int BFS_DIRECTION_VECTORS_Z[NUM_DIRECTIONS] = {0, 1, 0, -1};
// shimmy wrap constants 
static const int wrap_x[4][2] = {
    {1,1},
    {1,-1},
    {-1,-1},
    {-1, 1}
};
static const int wrap_z[4][2] = {
    {-1, 1},
    {1, 1},
    {1, -1},
    {-1, -1}
};
static const int wrap_orientation[4][2] = {
    {ACTION_DOWN,ACTION_UP},
    {ACTION_LEFT, ACTION_RIGHT},
    {ACTION_UP, ACTION_DOWN},
    {ACTION_RIGHT, ACTION_LEFT}
};

typedef struct Level Level;
struct Level {
    const int* map;
    int rows;
    int cols;
    int size;
    int total_length;
    int goal_location;
    int spawn_location;
};
static Level levels[3];  // Array to store the actual level objects


typedef struct PuzzleState PuzzleState;
struct PuzzleState {
    unsigned char blocks[BLOCK_BYTES];
    int robot_position;
    int robot_orientation;
    int robot_state;
    int block_grabbed;
};

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
    Level level;
    PuzzleState state;  // Contains blocks bitmask, position, orientation, etc.
    int level_number;
    int distance_to_goal;
    float reward_climb_row;
    float reward_fall_row;
    float reward_illegal_move;
    float reward_move_block;
    float reward_distance;
};

void levelToPuzzleState(const Level* level, PuzzleState* state) {
    // 1) Clear the entire bitmask to 0
    memset(state->blocks, 0, BLOCK_BYTES);

    // 2) For each cell i in [0 .. total_length-1]:
    //    if map[i] == 1 => set the bit in state->blocks
    //    if map[i] == 0 or 2 => do nothing (they're empty or goal)
    for (int i = 0; i < level->total_length; i++) {
        if (level->map[i] == 1) {
            SET_BIT(state->blocks, i);
        }
    }

    // 3) Copy spawn location into the puzzle state
    state->robot_position = level->spawn_location;

    // 4) Initialize other fields as needed
    state->robot_orientation = 3;  // e.g., "facing up" or whichever default
    state->robot_state = 0;        // or define a known 'DEFAULT' constant
    state->block_grabbed = -1;     // no block held initially
}


void init(CTowerClimb* env) {
    env->level_number = 0;
    env->level = levels[env->level_number];
    
    // Initialize PuzzleState
    levelToPuzzleState(&env->level, &env->state);
    printf("state: %d\n", env->state.robot_position);   
    printf("level: %d\n", env->level.goal_location);
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
    int current_floor = env->state.robot_position / sz;
    int grid_pos = env->state.robot_position % sz;
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
    for (int y = 0; y < window_height; y++) {
        int world_y = y + y_start;
        for (int z = 0; z < window_depth; z++) {
            int world_z = z + z_start;
            for (int x = 0; x < window_width; x++) {
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
                if (board_idx == env->state.robot_position) {
                    env->observations[obs_idx] = 3.0f;
                    continue;
                }
                
                // Use bitmask directly instead of board_state array
                env->observations[obs_idx] = (float)TEST_BIT(env->state.blocks, board_idx);
            }
        }
    }
    
    // Add player state information at the end
    int state_start = window_width * window_depth * window_height;
    env->observations[state_start] = (float)env->state.robot_orientation;
    env->observations[state_start + 1] = (float)env->state.robot_state;
    env->observations[state_start + 2] = (float)env->state.block_grabbed;
    env->observations[state_start + 3] = (float)(env->state.block_grabbed != -1);
}

void reset(CTowerClimb* env) {
    env->log = (Log){0};
    env->dones[0] = 0;
    env->level_number = 0;
    env->level = levels[env->level_number];
    levelToPuzzleState(&env->level, &env->state);
    compute_observations(env);
}


void illegal_move(CTowerClimb* env){
    env->rewards[0] = env->reward_illegal_move;
    env->log.episode_return += env->reward_illegal_move;
}

void death(CTowerClimb* env){
	env->rewards[0] = -1;
	env->log.episode_return -= 1;
	add_log(env->log_buffer, &env->log);
	reset(env);
}


int isGoal(const PuzzleState* s, const Level* lvl) {
    // 1) Check if player is at the goal cell
    if (s->robot_position - 100 != lvl->goal_location) return 0;

    return 1;  // all conditions satisfied
}

int move(PuzzleState* outState, int action, int mode, CTowerClimb* env, const Level* lvl){
    int new_position = outState->robot_position + BFS_DIRECTION_VECTORS_X[action] + BFS_DIRECTION_VECTORS_Z[action]*10;
    outState->robot_position = new_position;
    return 1;
}

int climb(PuzzleState* outState, int action, int mode, CTowerClimb* env, const Level* lvl){
    int cell_direct_above = outState->robot_position + lvl->size;
    int cell_next_above = cell_direct_above + BFS_DIRECTION_VECTORS_X[action] + BFS_DIRECTION_VECTORS_Z[action]*lvl->cols;
    int goal = lvl->goal_location;
    int can_climb = cell_direct_above < lvl->total_length && cell_next_above < lvl->total_length &&
     !TEST_BIT(outState->blocks, cell_direct_above) &&
      !TEST_BIT(outState->blocks, cell_next_above) && 
      cell_next_above != goal &&
      cell_direct_above != goal;
    if (can_climb){
        outState->robot_position = cell_next_above;
        outState->robot_state = 0;
        return 1;
    }
    return 0;
}

int drop(PuzzleState* outState, int action, int mode, CTowerClimb* env, const Level* lvl){
    int next_cell = outState->robot_position + BFS_DIRECTION_VECTORS_X[action] + BFS_DIRECTION_VECTORS_Z[action]*lvl->cols;
    int next_below_cell = next_cell - lvl->size;
    int next_double_below_cell = next_cell - 2*lvl->size;
    if (next_below_cell < 0){
        return 0;
    } 
    int step_down = next_double_below_cell >= 0 && TEST_BIT(outState->blocks, next_double_below_cell);
    if (step_down){
        outState->robot_position = next_below_cell;
        return 1;
    } else {
        outState->robot_position = next_below_cell;
        outState->robot_orientation = (outState->robot_orientation + 2) % 4;
        outState->robot_state = 1;
        return 1;
    }
}

int drop_from_hang(PuzzleState* outState, int action, int mode, const Level* lvl){
    int below_cell = outState->robot_position - lvl->size;
    while(below_cell > lvl->size && !TEST_BIT(outState->blocks, below_cell)){
        below_cell -= lvl->size;
    }
    if (below_cell < lvl->size){
        if (mode == PLG_MODE){
            return MOVE_ILLEGAL;
        }
        if (mode == RL_MODE){
            return MOVE_DEATH;
        }
    }
    outState->robot_position = below_cell+lvl->size;
    outState->robot_state = 0;
    return 1;
}
static inline int bfs_is_valid_position(int pos, const Level* level) {
    return (pos >= 0 && pos < level->total_length);
}

// Helper function to check block stability
static int bfs_is_block_stable(const PuzzleState* state, int position, const Level* level) {
    const int fs = level->size;
    const int positions[] = {
        position - fs,              // Bottom
        position - fs - 1,          // Left
        position - fs + 1,          // Right
        position - fs - level->cols, // Front
        position - fs + level->cols  // Back
    };
    
    for (int i = 0; i < 5; i++) {
        if (bfs_is_valid_position(positions[i], level) && TEST_BIT(state->blocks, positions[i])) {
            return 1;
        }
    }
    return 0;
}

int will_fall(PuzzleState* outState, int position, const Level* lvl){
    return bfs_is_valid_position(position, lvl) &&
     (TEST_BIT(outState->blocks, position) || position == lvl->goal_location)
      && !bfs_is_block_stable(outState, position, lvl);
}

int handle_block_falling(PuzzleState* outState, int* affected_blocks, int* blocks_to_move, int affected_blocks_count, const Level* lvl) {
    // Create queue for blocks that need to be checked for falling
    int bfs_queue[lvl->total_length];  // Max possible blocks to check
    int front = 0;
    int rear = 0;
    int fs = lvl->size;
    int cols = lvl->cols;

    // Add initially affected blocks to queue
    for (int i = 0; i < affected_blocks_count; i++) {
        if (affected_blocks[i] == -1){
            break;
        }
        bfs_queue[rear] = affected_blocks[i];
        rear++;
    }

    // First check all blocks above and adjacent to moved blocks
    for (int i = 0; i < lvl->cols; i++) {
        int block_pos = blocks_to_move[i];
        if (block_pos == -1){
            continue;
        }
        // Add block directly above
        int cell_above = block_pos + fs;  // Assuming 100 is floor height
        
        // If valid block above and unstable, add to queue
        if (will_fall(outState, cell_above, lvl)) {
            bfs_queue[rear++] = cell_above;
        }
        
        // Check edge-supported blocks
        int edge_blocks[4] = {
            cell_above - 1,      // left
            cell_above + 1,      // right
            cell_above - cols,     // front (assuming 10 is width)
            cell_above + cols      // back
        };
        
        // Add valid edge blocks to queue
        for (int j = 0; j < 4; j++) {
            if (will_fall(outState, edge_blocks[j], lvl)) {
                bfs_queue[rear++] = edge_blocks[j];
            }
        }
    }

    // Process queue until empty
    while (front < rear) {
        int current = bfs_queue[front++];
        int falling_position = current;
        int found_support = 0;

        // Check if block is goal (2)
        if (current == lvl->goal_location) {
            // Goal block is falling - level failed
            return 0;
        }

        // Remove block from current position
        CLEAR_BIT(outState->blocks, current);

        // Keep moving down until support found or bottom reached
        while (!found_support && falling_position >= fs) {  // 100 represents one floor down
            // Place block temporarily to check stability
            SET_BIT(outState->blocks, falling_position);            
            // Check if block is stable
            if (bfs_is_block_stable(outState, falling_position, lvl)) {
                found_support = 1;
            } else {
                // Remove block if not stable
                CLEAR_BIT(outState->blocks, falling_position);
                falling_position -= fs;  // Move down one level
            }
        }

        // Check blocks that might be affected by this fall
        int original_above = current + fs;  // Block directly above original position
        if (will_fall(outState, original_above, lvl)) {
            bfs_queue[rear++] = original_above;
        }

        // Check edge blocks that might be affected
        int edge_blocks[4] = {
            original_above - 1,   // Left
            original_above + 1,   // Right
            original_above - cols,  // Front (assuming 10 is the width)
            original_above + cols   // Back
        };

        for (int i = 0; i < 4; i++) {
            if (will_fall(outState, edge_blocks[i], lvl)) {
                bfs_queue[rear++] = edge_blocks[i];
            }
        }
    }
    return 1;
}

int push(PuzzleState* outState, int action, const Level* lvl, int mode, CTowerClimb* env){
    int first_block_index = outState->robot_position + BFS_DIRECTION_VECTORS_X[outState->robot_orientation] + BFS_DIRECTION_VECTORS_Z[outState->robot_orientation]*lvl->cols;
    int block_offset = (outState->robot_orientation == 3) ? -lvl->cols :  // North
                          (outState->robot_orientation == 1) ? lvl->cols :    // South
                          (outState->robot_orientation == 0) ? 1 :    // East
                          (outState->robot_orientation == 2) ? -1 :   // West
                          -lvl->cols;
                          
    int* blocks_to_move = calloc(lvl->cols, sizeof(int));
    for(int i = 0; i < lvl->cols; i++) {
        blocks_to_move[i] = (i == 0) ? first_block_index : -1;
    }
    for (int i = 0; i < lvl->cols; i++){
        int b_address = first_block_index + i*block_offset;
        if(i!=0 && blocks_to_move[i-1] == -1){
            break;
        }
        if(TEST_BIT(outState->blocks, b_address)){
            blocks_to_move[i] = b_address;
        }
    }
    int affected_blocks[lvl->cols];
    int count = 0;
    for (int i = 0; i < lvl->cols; i++){
        // invert conditional 
        int b_index = blocks_to_move[i];
        if (b_index == -1){
            continue;
        }
        if(i==0){
            CLEAR_BIT(outState->blocks, b_index);
        }
        int grid_pos = b_index % lvl->size;
        int x = grid_pos % lvl->cols;
        int z = grid_pos / lvl->cols;
        // Check if movement would cross floor boundaries
        if ((x == 0 && block_offset == -1) ||           // Don't move left off floor
            (x == lvl->cols-1 && block_offset == 1) ||       // Don't move right off floor
            (z == 0 && block_offset == -lvl->cols) ||        // Don't move forward off floor
            (z == lvl->rows-1 && block_offset == lvl->cols)) {    // Don't move back off floor
            continue;
        }

        // If we get here, movement is safe
        SET_BIT(outState->blocks, b_index + block_offset);
        affected_blocks[count] = b_index + block_offset;
        count++;
    }
    outState->block_grabbed = -1;
    return handle_block_falling(outState, affected_blocks, blocks_to_move,count, lvl);
}

int pull(PuzzleState* outState, int action, const Level* lvl, int mode, CTowerClimb* env){
    int pull_block = outState->robot_position + BFS_DIRECTION_VECTORS_X[outState->robot_orientation] + BFS_DIRECTION_VECTORS_Z[outState->robot_orientation]*10;
    int block_offset = (action == 3) ? -lvl->cols :  // North
                          (action == 1) ? lvl->cols :    // South
                          (action == 0) ? 1 :    // East
                          (action == 2) ? -1 :   // West
                          -lvl->cols;
    int block_in_front = TEST_BIT(outState->blocks, pull_block);
    int cell_below_next_position = outState->robot_position + block_offset - lvl->size;
    int backwards_walkable = bfs_is_valid_position(cell_below_next_position, lvl) && TEST_BIT(outState->blocks, cell_below_next_position);
    if (block_in_front){
        CLEAR_BIT(outState->blocks, pull_block);
        SET_BIT(outState->blocks, outState->robot_position);
        if (backwards_walkable){
            outState->block_grabbed = outState->robot_position;
            outState->robot_position = outState->robot_position + block_offset;
        }
        else {
            outState->robot_position = cell_below_next_position;
            outState->robot_state = 1;
            outState->block_grabbed = -1;
        }
    }
    int blocks_to_move[1];
    blocks_to_move[0] = pull_block;
    int affected_blocks[1] = {-1};
    return handle_block_falling(outState, affected_blocks, blocks_to_move, 1, lvl);
}

int shimmy_normal(PuzzleState* outState, int action, const Level* lvl, int local_direction, int mode, CTowerClimb* env){
    int next_cell = outState->robot_position + BFS_DIRECTION_VECTORS_X[local_direction] + BFS_DIRECTION_VECTORS_Z[local_direction]*lvl->cols;
    int above_next_cell = next_cell + lvl->size;
    if (bfs_is_valid_position(above_next_cell, lvl) && !TEST_BIT(outState->blocks, above_next_cell)){
        outState->robot_position = next_cell;
        return 1;
    }
    return 0;
}

int wrap_around(PuzzleState* outState, int action, const Level* lvl, int mode, CTowerClimb* env){
    int action_idx = (action == ACTION_LEFT) ? 0 : 1;
    int grid_pos = outState->robot_position % lvl->size;
    int x = grid_pos % lvl->cols;
    int z = grid_pos / lvl->cols;
    int current_floor = outState->robot_position / lvl->size;
    int new_x = x + wrap_x[outState->robot_orientation][action_idx];
    int new_z = z + wrap_z[outState->robot_orientation][action_idx];
    int new_pos = new_x + new_z*lvl->cols + current_floor*lvl->size;
    if (TEST_BIT(outState->blocks, new_pos + lvl->size)){
        return 0;
    }
    outState->robot_position = new_pos;
    outState->robot_orientation = wrap_orientation[outState->robot_orientation][action_idx];
    return 1;
}

int climb_from_hang(PuzzleState* outState, int action, const Level* lvl, int next_cell, int mode, CTowerClimb* env){
    int climb_index = next_cell + lvl->size;
    int direct_above_index = outState->robot_position + lvl->size;
    int can_climb = bfs_is_valid_position(climb_index, lvl) && bfs_is_valid_position(direct_above_index, lvl) 
    && !TEST_BIT(outState->blocks, climb_index) && !TEST_BIT(outState->blocks, direct_above_index);
    if (can_climb){
        outState->robot_position = climb_index;
        outState->robot_state = 0;
        return 1;
    }
    return 0;
}

int applyAction(PuzzleState* outState, int action, const Level* lvl, int mode, CTowerClimb* env) {
    // necessary variables
    int next_dx = BFS_DIRECTION_VECTORS_X[outState->robot_orientation];
    int next_dz = BFS_DIRECTION_VECTORS_Z[outState->robot_orientation];   
    int grid_pos = outState->robot_position % lvl->size;
    int x = grid_pos % lvl->cols;
    int z = grid_pos / lvl->cols;
    int next_cell = outState->robot_position + next_dx + next_dz*lvl->cols;
    int next_below_cell = next_cell - lvl->size;
    int walkable = bfs_is_valid_position(next_below_cell, lvl) && TEST_BIT(outState->blocks, next_below_cell);
    int block_in_front = bfs_is_valid_position(next_cell, lvl) && (TEST_BIT(outState->blocks, next_cell) || next_cell == lvl->goal_location);
    int movement_action = (action >= 0 && action < 4);
    int move_orient_check = (action == outState->robot_orientation);
    int standing_and_holding_nothing = outState->robot_state == 0 && outState->block_grabbed == -1;
    int hanging = outState->robot_state == 1;
    // Handle movement actions with common orientation check
    if (standing_and_holding_nothing && movement_action) {
        // If orientation doesn't match action, just rotate
        if (!move_orient_check) {
            outState->robot_orientation = action;
            return 1;
        }
        // Now handle the actual movement cases
        if (walkable && !block_in_front) {
            return move(outState, action, mode, env, lvl);
        }
        if (block_in_front) {
            return climb(outState, action, mode, env, lvl);
        }
        if (!block_in_front && !walkable) {
            return drop(outState, action, mode, env, lvl);
        }
    }

    if(hanging && movement_action){
        if (action == ACTION_UP){
            return climb_from_hang(outState, action, lvl, next_cell, mode, env);
        }
        if (action == ACTION_DOWN){
            return 0;
        }
        int local_direction = outState->robot_orientation;
        if (action == ACTION_LEFT){
            local_direction = (outState->robot_orientation + 3) % 4;
        }
        if (action == ACTION_RIGHT){
            local_direction = (outState->robot_orientation + 1) % 4;
        }
        int shimmy_cell = next_cell + BFS_DIRECTION_VECTORS_X[local_direction] + BFS_DIRECTION_VECTORS_Z[local_direction]*lvl->cols;
        int shimmy_path_cell = outState->robot_position + BFS_DIRECTION_VECTORS_X[local_direction] + BFS_DIRECTION_VECTORS_Z[local_direction]*lvl->cols;
        int basic_shimmy = bfs_is_valid_position(shimmy_cell, lvl) && bfs_is_valid_position(shimmy_path_cell, lvl) && TEST_BIT(outState->blocks, shimmy_cell) && !TEST_BIT(outState->blocks, shimmy_path_cell);
        int rotation_shimmy = bfs_is_valid_position(shimmy_path_cell, lvl) && TEST_BIT(outState->blocks, shimmy_path_cell);
        int in_bounds = x + next_dx >= 0 && x + next_dx < lvl->cols && z + next_dz >= 0 && z + next_dz < lvl->rows;
        int wrap_shimmy = in_bounds && !TEST_BIT(outState->blocks, shimmy_path_cell) && !TEST_BIT(outState->blocks, shimmy_cell);
        
        if(basic_shimmy){
            return shimmy_normal(outState, action, lvl, local_direction, mode, env);
        }
        else if(rotation_shimmy){
            //rotate shimmy
            static const int LEFT_TURNS[] = {3, 0, 1, 2};   // RIGHT->UP, DOWN->RIGHT, LEFT->DOWN, UP->LEFT
            static const int RIGHT_TURNS[] = {1, 2, 3, 0};  // RIGHT->DOWN, DOWN->LEFT, LEFT->UP, UP->RIGHT

            outState->robot_orientation = (action == ACTION_LEFT) ? 
                LEFT_TURNS[outState->robot_orientation] : 
                RIGHT_TURNS[outState->robot_orientation];
            outState->robot_state = 1;
            return 1;
        }
        else if(wrap_shimmy){
            return wrap_around(outState, action, lvl, mode, env);
        }

    }

    // drop from hang action 
    if (action == ACTION_DROP && !standing_and_holding_nothing) {
        return drop_from_hang(outState, action, mode, lvl);
    }
    // grab action
    if (action == ACTION_GRAB && standing_and_holding_nothing 
    && block_in_front){
        if (outState->block_grabbed == -1){
            outState->block_grabbed = next_cell;
            return 1;
        } 
    } 
    if (action == ACTION_GRAB && outState->block_grabbed != -1){
        outState->block_grabbed = -1;
        return 1;
    }
    
    // push or pull block 
    if (movement_action && block_in_front && outState->block_grabbed != -1){
        int result = 0;
        if (outState->robot_orientation == action){
            result = push(outState, action, lvl, mode, env);
        } else if(outState->robot_orientation == (action+2)%4){
            result = pull(outState, action, lvl, mode, env);
        } else {
            outState->robot_orientation = action;
            outState->block_grabbed = -1;
            result = 1;
        }
        // block fell on top of player
        if (TEST_BIT(outState->blocks, outState->robot_position)){
            if (mode == PLG_MODE){
                return MOVE_ILLEGAL;
            }
            if (mode == RL_MODE){
                return MOVE_DEATH;
            }
        }
        return result;
    }
    return 0;
    
}



void next_level(CTowerClimb* env){
    if(env->level_number == 2){
        env->rewards[0] = 1;
        env->log.episode_return += 1;
        reset(env);
        return;
    }
    env->level_number += 1;
    env->level = levels[env->level_number];
    levelToPuzzleState(&env->level, &env->state);

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
    // Create next state
    PuzzleState nextState = env->state;
    int move_result = applyAction(&nextState, env->actions[0], &env->level, RL_MODE, env);
    
    if (move_result == MOVE_ILLEGAL) {
        illegal_move(env);
        return;
    }
    if (move_result == MOVE_DEATH){
        death(env);
        return;
    }
    
    // Update state
    env->state = nextState;
    
    // Check for goal state
    if (isGoal(&env->state, &env->level)) {
        next_level(env);
    }
    
    // Update observations
    compute_observations(env);
    // if(action != NOOP){
    //     print_observation_window(env);
    // }
}


typedef struct BFSNode {
    PuzzleState state;
    int depth;      // how many moves from start
    int parent;     // index in BFS array of who generated me
    int action;     // which action led here (if you want to reconstruct the path)
} BFSNode;

static BFSNode* queueBuffer = NULL;
static int front = 0, back = 0;

// hash table for visited states
typedef struct VisitedNode {
    PuzzleState state;
    uint64_t hashVal;
    struct VisitedNode* next;
} VisitedNode;

static VisitedNode* visitedTable[TABLE_SIZE];


// Helper to incorporate a 32-bit integer into the hash one byte at a time.
static inline uint64_t fnv1a_hash_int(uint64_t h, int value) {
    // Break the int into 4 bytes (assuming 32-bit int).
    // This ensures consistent hashing regardless of CPU endianness.
    unsigned char bytes[4];
    bytes[0] = (unsigned char)((value >>  0) & 0xFF);
    bytes[1] = (unsigned char)((value >>  8) & 0xFF);
    bytes[2] = (unsigned char)((value >> 16) & 0xFF);
    bytes[3] = (unsigned char)((value >> 24) & 0xFF);

    for (int i = 0; i < 4; i++) {
        h ^= bytes[i];
        h *= FNV_PRIME;
    }
    return h;
}

uint64_t hashPuzzleState(const PuzzleState *s) {
    uint64_t h = FNV_OFFSET;

    // 1) Hash the 125-byte bitmask
    for (int i = 0; i < BLOCK_BYTES; i++) {
        h ^= s->blocks[i];
        h *= FNV_PRIME;
    }

    // 2) Hash the int fields (position, orientation, state, block_grabbed)
    h = fnv1a_hash_int(h, s->robot_position);
    h = fnv1a_hash_int(h, s->robot_orientation);
    h = fnv1a_hash_int(h, s->robot_state);
    h = fnv1a_hash_int(h, s->block_grabbed);

    return h;
}
// Compares two puzzle states fully
int equalPuzzleState(const PuzzleState* a, const PuzzleState* b) {
    // compare bitmask
    if (memcmp(a->blocks, b->blocks, BLOCK_BYTES) != 0) return 0;
    // compare other fields
    if (a->robot_position != b->robot_position) return 0;
    if (a->robot_orientation != b->robot_orientation) return 0;
    if (a->robot_state != b->robot_state) return 0;
    if (a->block_grabbed != b->block_grabbed) return 0;
    return 1;
}

void resetVisited(void) {
    // If we want to free old chains, we do that, or just memset to 0 for now.
    memset(visitedTable, 0, sizeof(visitedTable));
}

// Helper function to find a node in the hash table
static VisitedNode* findNode(const PuzzleState* s, uint64_t hv, size_t idx) {
    VisitedNode* node = visitedTable[idx];
    while (node) {
        if (node->hashVal == hv && equalPuzzleState(&node->state, s)) {
            return node;
        }
        node = node->next;
    }
    return NULL;
}

int isVisited(const PuzzleState* s) {
    uint64_t hv = hashPuzzleState(s);
    size_t idx = (size_t)(hv % TABLE_SIZE);
    return findNode(s, hv, idx) != NULL;
}

void markVisited(const PuzzleState* s) {
    uint64_t hv = hashPuzzleState(s);
    size_t idx = (size_t)(hv % TABLE_SIZE);

    // Return if already present
    if (findNode(s, hv, idx)) {
        return;
    }

    // Insert new node
    VisitedNode* node = (VisitedNode*)malloc(sizeof(VisitedNode));
    node->state = *s;
    node->hashVal = hv;
    node->next = visitedTable[idx];
    visitedTable[idx] = node;
}

void print_level_detailed(const Level* lvl, const unsigned char* bitmask, int robot_position) {
    printf("\nLevel layout (Height layers from bottom to top):\n");
    
    // For each height level (total_length / size gives us height)
    int height = lvl->total_length / lvl->size;
    for (int y = 0; y < height; y++) {
        printf("\nLayer %d:\n", y);
        
        // For each depth row
        for (int z = 0; z < lvl->rows; z++) {
            printf("  ");  // Indent
            
            // For each width position
            for (int x = 0; x < lvl->cols; x++) {
                // Calculate index in the same way as the original level layout
                int idx = x + lvl->cols * z + lvl->size * y;
                
                if (idx == lvl->spawn_location) {
                    printf("S ");
                    continue;
                }      
                if (idx == robot_position) {
                    printf("R ");
                    continue;
                }
                // Print appropriate symbol based on bitmask test
                if (idx == lvl->goal_location) {
                    printf("2 ");  // Goal location
                } else if (TEST_BIT(bitmask, idx)) {
                    printf("■ ");  // Block present
                } else {
                    printf("□ ");  // Empty space
                }
            }
            printf("\n");
        }
    }
    printf("\n");
}

void print_state(const PuzzleState* s, const Level* lvl){
    print_level_detailed(lvl, s->blocks, s->robot_position);
    printf("robot position: %d\n", s->robot_position);
    printf("robot orientation: %d\n", s->robot_orientation);
    printf("robot state: %d\n", s->robot_state);
    printf("block grabbed: %d\n", s->block_grabbed);
}

// This function fills out up to MAX_NEIGHBORS BFSNodes in 'outNeighbors'
// from a given BFSNode 'current'. It returns how many neighbors it produced.
int getNeighbors(const BFSNode* current, BFSNode* outNeighbors, const Level* lvl) {
    int count = 0;
    // We'll read the current BFSNode's puzzle state
    const PuzzleState* curState = &current->state;

    // Try each action
    for (int i = 0; i < 6; i++) {
        int action = i;

        // 1) Make a copy of the current puzzle state
        PuzzleState newState = *curState;

        // 2) Attempt to apply the action to newState
        int success = applyAction(&newState, action, lvl, PLG_MODE, NULL);

        if (!success) {
            // Move was invalid, skip
            continue;
        }

        // 3) If valid, build a BFSNode
        BFSNode neighbor;
        neighbor.state = newState;
        neighbor.depth = current->depth + 1;
        neighbor.parent = -1;   // BFS sets or overwrites this later
        neighbor.action = action; // record which action led here

        // 4) Add to 'outNeighbors' array
        outNeighbors[count++] = neighbor;

        // If you only allow up to 6 total, we can break if we reach that
        if (count >= MAX_NEIGHBORS) break;
    }

    return count; // how many valid neighbors we produced
}

// Example BFS
int bfs(const PuzzleState* start, int maxDepth, const Level* lvl) {
    // Clear or init your BFS queue
    queueBuffer = (BFSNode*)malloc(MAX_BFS_SIZE * sizeof(BFSNode));
    if (!queueBuffer) {
        printf("Failed to allocate memory for BFS queue\n");
        return 0;
    }
    
    front = 0;
    back = 0;
    
    // Enqueue start node
    BFSNode startNode;
    startNode.state = *start;  // copy puzzle state
    startNode.depth = 0;
    startNode.parent = -1;
    startNode.action = -1;
    queueBuffer[back++] = startNode;

    // BFS loop
    while (front < back) {

        if (back >= MAX_BFS_SIZE) {
            printf("BFS queue overflow! Increase MAX_BFS_SIZE or optimize search.\n");
            return 0;
        }
        BFSNode current = queueBuffer[front];
        int currentIndex = front;
        front++;
        // If current.state is the goal, reconstruct path
        if (isGoal(&current.state, lvl)) {
            printf("Found solution path of length %d!\n", current.depth);
            
            // Store nodes in order
            BFSNode* path = (BFSNode*)malloc((current.depth + 1) * sizeof(BFSNode));
            BFSNode node = current;
            int idx = current.depth;
            
            // Walk backwards to get path
            while (idx >= 0) {
                path[idx] = node;
                if (node.parent != -1) {
                    node = queueBuffer[node.parent];
                }
                idx--;
            }
            
            // Print in forward order
            printf("\nStep 0 (Start):\n");
            printf("  Position: %d\n", path[0].state.robot_position);
            printf("  Orientation: %d\n", path[0].state.robot_orientation);
            printf("  State: %d\n", path[0].state.robot_state);
            printf("  Block grabbed: %d\n", path[0].state.block_grabbed);
            for (int i = 1; i <= current.depth; i++) {
                int block_in_front = path[i].state.robot_position + 
                    (path[i].state.robot_orientation == 0 ? 1 :    // Right
                     path[i].state.robot_orientation == 1 ? lvl->cols :    // Down
                     path[i].state.robot_orientation == 2 ? -1 :   // Left
                     -lvl->cols);  // Up (orientation == 3)
                printf("\nStep %d:\n", i);
                printf("  Action taken: %d\n", path[i].action);
                printf("  Position: %d\n", path[i].state.robot_position);
                printf("  Orientation: %d\n", path[i].state.robot_orientation);
                printf("  State: %d\n", path[i].state.robot_state);
                printf("  Block grabbed: %d\n", path[i].state.block_grabbed);
                printf("  Block in front: %d (Has block: %d)\n", block_in_front, 
                    bfs_is_valid_position(block_in_front, lvl) && 
                    (TEST_BIT(path[i].state.blocks, block_in_front) || block_in_front == lvl->goal_location));
            }
            
            free(path);
            return 1;
        }

        if (current.depth < maxDepth) {
            // generate neighbors
            BFSNode neighbors[MAX_NEIGHBORS];
            int nCount = getNeighbors(&current, neighbors, lvl);
            for (int i = 0; i < nCount; i++) {
                PuzzleState* nxt = &neighbors[i].state;
                // if not visited
                if (!isVisited(nxt)) {
                    markVisited(nxt);

                    // fill BFS node fields
                    neighbors[i].depth = current.depth + 1;
                    neighbors[i].parent = currentIndex;
                    // Enqueue
                    queueBuffer[back++] = neighbors[i];
                }
            }
        }        
    }
    free(queueBuffer);
    queueBuffer = NULL;
    // If we exit while, no solution found within maxDepth
    printf("No solution within %d moves.\n", maxDepth);
    return 0;
}


int verify_level(Level level, int max_moves){
    // converting level to puzzle state
    PuzzleState state;
    levelToPuzzleState(&level, &state);
    // reset visited hash table
    resetVisited();
    markVisited(&state);
    // Run BFS
    int solvable = bfs(&state, max_moves, &level);
    return solvable;
}

Level gen_level(int goal_level) {
    // Initialize an illegal level in case we need to return early
    Level illegal_level = {0};
    
    // Allocate board memory
    int* board = (int*)calloc(row_max * col_max * depth_max, sizeof(int));
    if (!board) {
        return illegal_level;
    }

    int legal_width_size = 8;
    int legal_depth_size = 8;
    int area = depth_max * col_max;
    int spawn_created = 0;
    int spawn_index = -1;
    int goal_created =0;
    int goal_index = -1;
    for(int y= 0; y < row_max; y++){
        for(int z = depth_max - 1; z >= 0; z--){
            for(int x = 0; x< col_max; x++){
                int block_index = x + col_max * z + area * y;
                if (x >= 1 && x < legal_width_size && z >= 1 && z < legal_depth_size && 
                y >= 1 && y < goal_level && (z <= (legal_depth_size - y))){
                    int chance = (rand() % 4 ==0) ? 1 : 0;
                    board[block_index] = chance;
                    // create spawn point above an existing block
                    if (spawn_created == 0 && y == 2 && board[block_index - area] == 1){
                        spawn_created = 1;
                        spawn_index = block_index;
                        board[spawn_index] = 0;
                    }
                }
                if (goal_created ==0 && y == goal_level && (board[block_index + col_max  - area]  ==1 ||  board[block_index - 1  - area] == 1|| board[block_index + 1 - area] ==1 )){
                    goal_created = 1;
                    goal_index = block_index;
                    board[goal_index] = 2;
                }
                
            }
        }
    }

    if (!spawn_created || spawn_index < 0) {
        printf("no spawn found\n");
        free(board);
        return illegal_level;
    }

    
    if (!goal_created || goal_index < 0) {
        printf("no goal found\n");
        printf("goal index: %d\n", goal_index);
        free(board);
        return illegal_level;
    }

    Level level = {
        .map = board,
        .rows = row_max,
        .cols = col_max,
        .size = row_max * col_max,
        .total_length = row_max * col_max * depth_max,
        .goal_location = goal_index,
        .spawn_location = spawn_index
    };
    return level;
}


static void init_random_levels(int goal_level) {
    time_t t;
    for(int i = 0; i < 3; i++) {
        srand((unsigned) time(&t) + i); // Increment seed for each level
        levels[i] = gen_level(goal_level);
        // guarantee a map is created
        while(levels[i].map == NULL || verify_level(levels[i],32) == 0){
            levels[i] = gen_level(goal_level);
        }
    }
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
    client->previousRobotPosition = env->state.robot_position;
    
    // Initialize visual position to match starting robot position
    int floor = env->state.robot_position / env->level.size;
    int grid_pos = env->state.robot_position % env->level.size;
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
    if (env->state.robot_orientation == 0) { // Facing +x
        client->visualPosition.x += 0.4f * reverse;
    } else if (env->state.robot_orientation == 1) { // Facing +z
        client->visualPosition.z += 0.4f * reverse;
    } else if (env->state.robot_orientation == 2) { // Facing -x
        client->visualPosition.x -= 0.4f * reverse;
    } else if (env->state.robot_orientation == 3) { // Facing -z
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
    int floor = env->state.robot_position / sz;
    int grid_pos = env->state.robot_position % sz;
    int x = grid_pos % cols;
    int z = grid_pos / cols;

    if (env->state.robot_state == DEFAULT && client->animState == ANIM_HANGING) {
        client->animState = ANIM_IDLE;
        client->animFrameCounter = 0;
        UpdateModelAnimation(client->robot, client->animations[4], 0);
        client->isMoving = false;
        client->visualPosition = client->targetPosition;
    }

    if (env->state.block_grabbed != -1 && client->animState != ANIM_GRABBING && client->animState != ANIM_START_GRABBING)
    {
        client->animState = ANIM_START_GRABBING;
        client->isMoving = true;
        client->animFrameCounter = 0;
        UpdateModelAnimation(client->robot, client->animations[3], 0); 
        client->visualPosition = client->targetPosition;

    } else if (env->state.block_grabbed == -1 && client->animState == ANIM_GRABBING) {
        // Reset to idle when block is released
        client->animState = ANIM_IDLE;
        client->animFrameCounter = 0;
        UpdateModelAnimation(client->robot, client->animations[4], 0);
    }

    // Check if robot position has changed
    if (env->state.robot_position != client->previousRobotPosition) {
        if (client->isMoving) {
            client->visualPosition = client->targetPosition;
        }
        
        client->isMoving = true;
            
        // Calculate new target position
        int floor = env->state.robot_position / env->level.size;
        int grid_pos = env->state.robot_position % env->level.size;
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
        } else if (env->state.robot_state == HANGING){
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
                if (env->state.robot_orientation == UP) {  // Facing up (-z)
                    moving_right = client->targetPosition.x > client->visualPosition.x;
                } else if (env->state.robot_orientation == DOWN) {  // Facing down (+z)
                    moving_right = client->targetPosition.x < client->visualPosition.x;
                } else if (env->state.robot_orientation == RIGHT) {  // Facing right (+x)
                    moving_right = client->targetPosition.z < client->visualPosition.z;
                } else if (env->state.robot_orientation == LEFT) {  // Facing left (-x)
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
        client->previousRobotPosition = env->state.robot_position;
    }

    // Update animation if moving
    if (client->isMoving) {
        // Use appropriate animation based on state
        if (client->animState == ANIM_CLIMBING) {
            if (stored_orientation == -1) {
                stored_orientation = env->state.robot_orientation;
            }
            int current_orientation = env->state.robot_orientation;
            env->state.robot_orientation = stored_orientation;
            client->animFrameCounter += 6;
            UpdateModelAnimation(client->robot, client->animations[1], client->animFrameCounter);
            if (client->animFrameCounter >= client->animations[1].frameCount) {  // Adjust frame count for climbing animation
                client->isMoving = false;
                client->animState = ANIM_IDLE;
                client->animFrameCounter = 0;
                UpdateModelAnimation(client->robot, client->animations[4], client->animFrameCounter);
                client->visualPosition = client->targetPosition;
                stored_orientation = -1;
                env->state.robot_orientation = current_orientation;
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
            if (env->state.robot_orientation == UP) {  // Facing -z
                client->visualPosition.x = Lerp(client->visualPosition.x, client->targetPosition.x, progress);
            } else if (env->state.robot_orientation == DOWN) {  // Facing +z
                client->visualPosition.x = Lerp(client->visualPosition.x, client->targetPosition.x, progress);
            } else if (env->state.robot_orientation == RIGHT) {  // Facing +x
                client->visualPosition.z = Lerp(client->visualPosition.z, client->targetPosition.z, progress);
            } else if (env->state.robot_orientation == LEFT) {  // Facing -x
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
        int floor = i / sz;
        int grid_pos = i % sz;
        int x = grid_pos % cols;
        int z = grid_pos / cols;
        if(TEST_BIT(env->state.blocks, i)){
            
                DrawModel(client->cube, (Vector3){x * 1.0f, floor * 1.0f, z * 1.0f}, client->scale, WHITE);
            
            if(i == env->state.block_grabbed){
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
        if (i == env->level.goal_location){
            EndShaderMode();
            DrawCube(
            (Vector3){x * 1.0f, floor * 1.0f, z * 1.0f},
            1.0f, 1.0f, 1.0f,
            PUFF_CYAN
            );
            BeginShaderMode(client->shader);
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
    rlRotatef(-90.0f + env->state.robot_orientation * 90.0f, 0, 0, 1);
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

