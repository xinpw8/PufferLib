#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <time.h>

#define row_max 10
#define col_max 10
#define depth_max 10

#define BLOCK_BYTES 125
// FNV-1a hash function
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL

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

typedef struct PuzzleState PuzzleState;
struct PuzzleState {
    unsigned char blocks[BLOCK_BYTES];
    int robot_position;
    int robot_orientation;
    int robot_state;
    int block_grabbed;
};

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

int isGoal(const PuzzleState* s, const Level* lvl) {
    // 1) Check if player is at the goal cell
    if (s->robot_position - 100 != lvl->goal_location) return 0;

    return 1;  // all conditions satisfied
}

int move(PuzzleState* outState, int action){
    int new_position = outState->robot_position + BFS_DIRECTION_VECTORS_X[action] + BFS_DIRECTION_VECTORS_Z[action]*10;
    outState->robot_position = new_position;
    return 1;
}

int climb(PuzzleState* outState, int action){
    int cell_direct_above = outState->robot_position + 100;
    int cell_next_above = cell_direct_above + BFS_DIRECTION_VECTORS_X[action] + BFS_DIRECTION_VECTORS_Z[action]*10;
    int can_climb = cell_direct_above < 1000 && cell_next_above < 1000 && !TEST_BIT(outState->blocks, cell_direct_above) && !TEST_BIT(outState->blocks, cell_next_above);
    if (can_climb){
        outState->robot_position = cell_next_above;
        outState->robot_state = 0;
        return 1;
    }
    return 0;
}

int drop(PuzzleState* outState, int action){
    int next_cell = outState->robot_position + BFS_DIRECTION_VECTORS_X[action] + BFS_DIRECTION_VECTORS_Z[action]*10;
    int next_below_cell = next_cell - 100;
    int next_double_below_cell = next_cell - 200;
    if (next_below_cell < 0) return 0;
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

int drop_from_hang(PuzzleState* outState, int action){
    int below_cell = outState->robot_position - 100;
    while(below_cell >100 && !TEST_BIT(outState->blocks, below_cell)){
        below_cell -= 100;
    }
    outState->robot_position = below_cell+100;
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
    int bfs_queue[1000];  // Max possible blocks to check
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
    for (int i = 0; i < 10; i++) {
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

int push(PuzzleState* outState, int action, const Level* lvl){
    int first_block_index = outState->robot_position + BFS_DIRECTION_VECTORS_X[outState->robot_orientation] + BFS_DIRECTION_VECTORS_Z[outState->robot_orientation]*lvl->cols;
    int block_offset = (outState->robot_orientation == 3) ? -lvl->cols :  // North
                          (outState->robot_orientation == 1) ? lvl->cols :    // South
                          (outState->robot_orientation == 0) ? 1 :    // East
                          (outState->robot_orientation == 2) ? -1 :   // West
                          -lvl->cols;
    int blocks_to_move[10] = {first_block_index, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    for (int i = 0; i < 10; i++){
        int b_address = first_block_index + i*block_offset;
        if(i!=0 && blocks_to_move[i-1] == -1){
            break;
        }
        if(TEST_BIT(outState->blocks, b_address)){
            blocks_to_move[i] = b_address;
        }
    }
    int affected_blocks[10];
    int count = 0;
    for (int i = 0; i < 10; i++){
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

int pull(PuzzleState* outState, int action, const Level* lvl){
    int pull_block = outState->robot_position + BFS_DIRECTION_VECTORS_X[outState->robot_orientation] + BFS_DIRECTION_VECTORS_Z[outState->robot_orientation]*10;
    int block_offset = (action == 3) ? -lvl->cols :  // North
                          (action == 1) ? lvl->cols :    // South
                          (action == 0) ? 1 :    // East
                          (action == 2) ? -1 :   // West
                          -lvl->cols;
    int block_in_front = TEST_BIT(outState->blocks, pull_block);
    int cell_below_next_position = outState->robot_position + block_offset - 100;
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

int shimmy_normal(PuzzleState* outState, int action, const Level* lvl, int local_direction){
    int next_cell = outState->robot_position + BFS_DIRECTION_VECTORS_X[local_direction] + BFS_DIRECTION_VECTORS_Z[local_direction]*lvl->cols;
    int above_next_cell = next_cell + 100;
    if (bfs_is_valid_position(above_next_cell, lvl) && !TEST_BIT(outState->blocks, above_next_cell)){
        outState->robot_position = next_cell;
        return 1;
    }
    return 0;
}

int wrap_around(PuzzleState* outState, int action, const Level* lvl){
    int action_idx = (action == ACTION_LEFT) ? 0 : 1;
    int x = outState->robot_position % lvl->cols;
    int z = outState->robot_position / lvl->cols;
    int current_floor = outState->robot_position / lvl->size;
    int new_x = x + wrap_x[outState->robot_orientation][action_idx];
    int new_z = z + wrap_z[outState->robot_orientation][action_idx];
    int new_pos = new_x + new_z*lvl->cols + current_floor*lvl->size;
    if (TEST_BIT(outState->blocks, new_pos + 100)){
        return 0;
    }
    outState->robot_position = new_pos;
    outState->robot_orientation = wrap_orientation[outState->robot_orientation][action_idx];
    return 1;
}

int climb_from_hang(PuzzleState* outState, int action, const Level* lvl, int next_cell){
    int climb_index = next_cell + 100;
    int direct_above_index = outState->robot_position + 100;
    int can_climb = bfs_is_valid_position(climb_index, lvl) && bfs_is_valid_position(direct_above_index, lvl) 
    && !TEST_BIT(outState->blocks, climb_index) && !TEST_BIT(outState->blocks, direct_above_index);
    if (can_climb){
        outState->robot_position = climb_index;
        outState->robot_state = 0;
        return 1;
    }
    return 0;
}

int applyAction(PuzzleState* outState, int action, const Level* lvl) {
    // necessary variables
    int next_dx = BFS_DIRECTION_VECTORS_X[outState->robot_orientation];
    int next_dz = BFS_DIRECTION_VECTORS_Z[outState->robot_orientation];   
    int x = outState->robot_position % lvl->cols;
    int z = outState->robot_position / lvl->cols;
    int next_cell = outState->robot_position + next_dx + next_dz*lvl->cols;
    int next_below_cell = next_cell - 100;
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
            return move(outState, action);
        }
        if (block_in_front) {
            return climb(outState, action);
        }
        if (!block_in_front && !walkable) {
            return drop(outState, action);
        }
    }

    if(hanging && movement_action){
        if (action == ACTION_UP){
            return climb_from_hang(outState, action, lvl, next_cell);
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
            return shimmy_normal(outState, action, lvl, local_direction);
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
            return wrap_around(outState, action, lvl);
        }

    }

    // drop from hang action 
    if (action == ACTION_DROP && !standing_and_holding_nothing) {
        return drop_from_hang(outState, action);
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
            result = push(outState, action, lvl);
        } else if(outState->robot_orientation == (action+2)%4){
            result = pull(outState, action, lvl);
        } else {
            outState->robot_orientation = action;
            outState->block_grabbed = -1;
            result = 1;
        }
        // block fell on top of player
        if (TEST_BIT(outState->blocks, outState->robot_position)){
            result = 0;
        }
        return result;
    }
    return 0;
    
}

// This function fills out up to MAX_NEIGHBORS BFSNodes in 'outNeighbors'
// from a given BFSNode 'current'. It returns how many neighbors it produced.
int getNeighbors(const BFSNode* current, BFSNode* outNeighbors, const Level* lvl) {
    int count = 0;
    static const int ALL_ACTIONS[6] = {
        ACTION_RIGHT,
        ACTION_DOWN,
        ACTION_LEFT,
        ACTION_UP,
        ACTION_GRAB,
        ACTION_DROP
    };

    // We'll read the current BFSNode's puzzle state
    const PuzzleState* curState = &current->state;

    // Try each action
    for (int i = 0; i < 6; i++) {
        int action = ALL_ACTIONS[i];

        // 1) Make a copy of the current puzzle state
        PuzzleState newState = *curState;

        // 2) Attempt to apply the action to newState
        int success = applyAction(&newState, action, lvl);

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

static const int tutorial[108] = {
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
    0,1,1,1,1,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    // floor 3
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,2,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
};
static const int tutorial_two[144] = {
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
    0,1,1,1,1,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    // floor 3
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,1,1,1,1,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    // floor 3
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,2,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
};

static const int level_one_map[288] = {
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

static const int level_two_map[392] = {
    // floor 1
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,0,0,0,0,
    // floor 2 
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    // floor 3
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,1,1,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    // floor 4
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,1,1,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    // floor 5
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    // floor 6
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,1,0,
    0,0,1,1,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    // floor 7
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,1,0,
    0,0,1,1,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    // floor 8
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,2,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
};

static const int level_three_map[432] = {
    // floor 1
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,
    0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,
    // floor 2
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,1,0,0,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    // floor 3
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,1,0,0,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    // floor 4
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,1,0,0,1,0,1,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    // floor 5
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    // floor 6
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    // floor 7
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    // floor 8
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,2,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
};

static  Level level_one = {
    .map = (int*)level_one_map, 
    .rows = 6,
    .cols = 6,
    .size = 36,
    .total_length = 288,
    .goal_location = 262,
    .spawn_location = 64,
};

static  Level level_two = {
    .map = (int*)level_two_map,
    .rows = 7,
    .cols = 7,
    .size = 49,
    .total_length = 392,
    .goal_location = 378,
    .spawn_location = 85,
};

static  Level level_three = {
    .map = (int*)level_three_map,
    .rows = 6,
    .cols = 9,
    .size = 54,
    .total_length = 432,
    .goal_location = 406,
    .spawn_location = 94,
};

static const Level level_tutorial = {
    .map = (int*)tutorial,
    .rows = 6,
    .cols = 6,
    .size = 36,
    .total_length = 108,
    .goal_location = 64,
    .spawn_location = 62,
};

static const Level level_tutorial_two = {
    .map = (int*)tutorial_two,
    .rows = 6,
    .cols = 6,
    .size = 36,
    .total_length = 144,
    .goal_location = 108,
    .spawn_location = 62,
};

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
                y >= 1 && y < goal_level && (z < (legal_depth_size - y))){
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

void print_level(const int* board, int legal_width_size, int legal_depth_size, int legal_height_size, int spawn_index) {
    printf("\nLevel layout (Height layers from bottom to top):\n");
    
    // For each height level
    for (int y = 0; y < legal_height_size; y++) {
        printf("\nLayer %d:\n", y);
        
        // For each depth row
        for (int z = 0; z < legal_depth_size; z++) {
            printf("  ");  // Indent

            // For each width position
            for (int x = 0; x < legal_width_size; x++) {
                // Get value from board array
                int idx = x + legal_width_size * z + (legal_width_size * legal_depth_size) * y;
                int val = board[idx];
                if (idx == spawn_index) {
                    printf("S ");
                    continue;
                } 
                // Print appropriate symbol
                if (val == 0) printf("□ ");    // Empty
                else if (val == 1) printf("■ "); // Block
                else printf("%d ", val);
            }
            printf("\n");
        }
    }
    printf("\n");
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

static Level levels[3];  // Array to store the actual level objects

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
    // levels[0] = level_one;
    // levels[1] = level_two;
    // levels[2] = level_three;
}

