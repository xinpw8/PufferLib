#include <time.h>
#include <unistd.h>
#include "gpudrive.h"
#include "puffernet.h"

// Human control functions
void handle_human_input(GPUDrive* env) {
    float steering_delta = M_PI/8.0f;
    int active_idx = env->active_agent_indices[env->human_agent_idx];
    int (*actions)[2] = (int(*)[2])env->actions;
    
    // Reset the human-controlled agent's actions
    actions[active_idx][0] = 0;
    actions[active_idx][1] = 0;
    
    // Apply keyboard inputs to the human-controlled agent
    if(IsKeyDown(KEY_W)) {
        actions[active_idx][0] = 1;
    }
    if(IsKeyDown(KEY_S)) {
        actions[active_idx][0] = -1;
    }
    if(IsKeyDown(KEY_A)) {
        actions[active_idx][1] = -steering_delta;
    }
    if(IsKeyDown(KEY_D)) {
        actions[active_idx][1] = steering_delta;
    }
    
    // Allow switching between agents with number keys
    for(int i = 0; i < env->active_agent_count; i++) {
        if(IsKeyPressed(KEY_ONE + i)) {
            env->human_agent_idx = i;
            printf("Switched to controlling agent %d (index %d)\n", 
                   i, env->active_agent_indices[i]);
            break;
        }
    }
}

// Camera control functions
void handle_camera_controls(Client* client) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;
    
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
        client->camera.position.x = client->camera.target.x + 
            (client->camera.position.x - client->camera.target.x) * zoom_factor;
        client->camera.position.y = client->camera.target.y + 
            (client->camera.position.y - client->camera.target.y) * zoom_factor;
    }
}

void demo() {
    GPUDrive env = {
        .dynamics_model = CLASSIC,
        .human_agent_idx = 0,
    };
    allocate(&env);
    c_reset(&env);
    Client* client = make_client(&env);
    printf("Human controlling agent index: %d\n", env.active_agent_indices[env.human_agent_idx]);
    
    while (!WindowShouldClose()) {
        // Handle camera controls
        handle_camera_controls(client);
        int (*actions)[2] = (int(*)[2])env.actions;
        // Reset all agent actions at the beginning of each frame
        for(int i = 0; i < env.active_agent_count; i++) {
            actions[i][0] = 0;
            actions[i][1] = 0;
        }
        // Handle human input for the controlled agent
        // handle_human_input(&env);
        c_step(&env);
        c_render(client, &env);
    }

    close_client(client);
    free_allocated(&env);
}

void performance_test() {
    long test_time = 10;
    GPUDrive env = {
        .dynamics_model = CLASSIC,
        .human_agent_idx = 0,
    };
    allocate(&env);
    c_reset(&env);

    long start = time(NULL);
    int i = 0;
    int (*actions)[2] = (int(*)[2])env.actions;
    
    while (time(NULL) - start < test_time) {
        // Set random actions for all agents
        for(int j = 0; j < env.active_agent_count; j++) {
             actions[j][0] = 0;  // -1, 0, or 1
             actions[j][1] = 0;  // Random steering
        }
        
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (i*env.active_agent_count) / (end - start));
    free_allocated(&env);
}

int main() {
    // demo();
    performance_test();
    return 0;
}
