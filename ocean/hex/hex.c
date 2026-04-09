#include "hex.h"
#include <math.h>
#include <time.h>
#include <stdio.h>

void demo() {
    Hex env = {0};
    allocate_chex(&env);
    c_reset(&env);
    c_render(&env);

    int tick = 0;
    while (!WindowShouldClose()) {
        bool move_made = false;
        
        // Hold Shift + Left Click to play manually

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            Vector2 mouse = GetMousePosition();
            
            int screen_width = GetScreenWidth();
            int screen_height = GetScreenHeight();
            float radius = 22.0f;
            float sqrt3 = 1.73205f;
            float hex_width = sqrt3 * radius;
            float hex_height = 2.0f * radius;
            
            float total_width = hex_width * BOARD_SIZE + hex_width * 0.5f * BOARD_SIZE;
            float total_height = hex_height * 0.75f * BOARD_SIZE;
            
            float start_x = screen_width / 2.0f - total_width / 2.0f + hex_width / 2.0f;
            float start_y = screen_height / 2.0f - total_height / 2.0f + hex_height / 2.0f;
            
            // Inverse map:
            int r = (int)roundf((mouse.y - start_y) / (hex_height * 0.75f));
            int c = (int)roundf((mouse.x - start_x) / hex_width - r * 0.5f);
            
            if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
                env.actions[0] = r * BOARD_SIZE + c;
                move_made = true;
            }
        }
        

        if (move_made) {
            c_step(&env);
        }
        
        c_render(&env);
        tick++;
    }
    
    free_allocated_chex(&env);
    c_close(&env);
}


void speed_test(){
    Hex env = {0};
    allocate_chex(&env);
    c_reset(&env);
    clock_t start = clock();

    int num_steps = 1000000;
    for (int i = 0; i < num_steps; i++) {
        env.actions[0] = compute_legal_move(&env);
        c_step(&env);
    }
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time for %d steps: %.2f seconds\n", num_steps, elapsed);
    printf("SPS: %.2fM\n", num_steps / elapsed / 1e6);

    free_allocated_chex(&env);
    c_close(&env);

}

int main() {
    demo();
    // speed_test();
    return 0;
}
