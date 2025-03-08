#include <time.h>
#include <unistd.h>
#include "gpudrive.h"
#include "puffernet.h"


void demo() {
    GPUDrive env = {
        .num_agents = 4,
        .active_agent_count = 4,
    };
    allocate(&env);
    c_reset(&env);
    Client* client = make_client(&env);

    // Initialize mouse control variables
    Vector2 prev_mouse_pos = GetMousePosition();
    bool is_dragging = false;
    float camera_move_speed = 0.5f;  // Adjust this to control movement sensitivity

    while (!WindowShouldClose()) {
        // Handle mouse drag for camera movement
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            prev_mouse_pos = GetMousePosition();
            is_dragging = true;
        }
        
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            is_dragging = false;
        }
        
        if (is_dragging) {
            Vector2 current_mouse_pos = GetMousePosition();
            Vector2 delta = {
                (current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
                -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed
            };

            // Update camera position (only X and Y)
            client->camera.position.x += delta.x;
            client->camera.position.y += delta.y;
            
            // Update camera target (only X and Y)
            client->camera.target.x += delta.x;
            client->camera.target.y += delta.y;

            prev_mouse_pos = current_mouse_pos;
        }

        // Handle mouse wheel for zoom
        float wheel = GetMouseWheelMove();
        if (wheel != 0) {
            float zoom_factor = 1.0f - (wheel * 0.1f);
            // Adjust camera position for zoom while maintaining height
            client->camera.position.x = client->camera.target.x + (client->camera.position.x - client->camera.target.x) * zoom_factor;
            client->camera.position.y = client->camera.target.y + (client->camera.position.y - client->camera.target.y) * zoom_factor;
        }

        c_step(&env);
        c_render(client, &env);
    }

    close_client(client);
    free_allocated(&env);
}

void performance_test() {
    long test_time = 10;
    GPUDrive env = {
        .num_agents = 4,
        .active_agent_count = 4,
    };
    allocate(&env);
    c_reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % 5;
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(&env);
}

int main() {
    demo();
    // performance_test();
    return 0;
}
