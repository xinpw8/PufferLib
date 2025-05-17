#include "terraform.h"

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
        client->camera.position.z += delta.y;
        
        // Update camera target (only X and Y)
        client->camera.target.x += delta.x;
        client->camera.target.z += delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {
            client->camera.position.x - client->camera.target.x,
            client->camera.position.y - client->camera.target.y,
            client->camera.position.z - client->camera.target.z
        };
        
        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;
        
        // Update the camera position based on the scaled direction
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

void demo() {
    //Weights* weights = load_weights("resources/pong_weights.bin", 133764);
    //LinearLSTM* net = make_linearlstm(weights, 1, 8, 3);

    Terraform env = {.size = 512, .num_agents = 8};
    allocate(&env);

    Client* client = make_client(&env);

    c_reset(&env);
    while (!WindowShouldClose()) {
        handle_camera_controls(env.client);
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = UP;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = DOWN;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = RIGHT;
        } else {
            env.actions[0] = NOOP;
            //forward_linearlstm(net, env.observations, env.actions);
        }
        c_step(&env);
        c_render(&env);
    }
    //free_linearlstm(net);
    //free(weights);
    free_allocated(&env);
    close_client(client);
}

void test_performance(int timeout) {
    Terraform env = {
        .size = 128,
        .num_agents = 8,
    };
    allocate(&env);
    c_reset(&env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        for (int i = 0; i < env.num_agents; i++) {
            env.actions[5*i] = rand() % 5;
            env.actions[5*i + 1] = rand() % 5;
            env.actions[5*i + 2] = rand() % 3;
            env.actions[5*i + 3] = rand() % 3;
        }

        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_steps / (end - start);
    printf("Test Environment SPS: %f\n", sps);
    free_allocated(&env);
}

int main() {
    // test_performance(10);
    demo();
}

