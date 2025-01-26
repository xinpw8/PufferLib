#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <time.h>

#define row_max 10
#define col_max 10
#define depth_max 10

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

Level gen_level(int max_moves, int goal_level) {
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
                y >= 1 && y < goal_level && z <= (legal_depth_size - y)){
                    int chance = (rand() % 2 ==0) ? 1 : 0;
                    board[block_index] = chance;
                    // create spawn point above an existing block
                    if (spawn_created == 0 && y == 2 && board[block_index - area] == 1){
                        spawn_created = 1;
                        spawn_index = block_index;
                        board[spawn_index] = 0;
                    }
                }
                if (goal_created ==0 && y == goal_level && (board[block_index + col_max  - area]  ==1 )){
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

static Level levels[3];  // Array to store the actual level objects

static void init_random_levels(int goal_level) {
    time_t t;
    for(int i = 0; i < 3; i++) {
        srand((unsigned) time(&t) + i); // Increment seed for each level
        levels[i] = gen_level(9, goal_level);
        // guarantee a map is created
        while(levels[i].map == NULL){
            levels[i] = gen_level(9, goal_level);
        }
        print_level(levels[i].map, 10, 10, 10, levels[i].spawn_location);
    }
    // levels[0] = level_one;
    // levels[1] = level_two;
    // levels[2] = level_three;
}