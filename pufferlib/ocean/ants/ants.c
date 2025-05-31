// for local testing of c code, build with:
// bash scripts/build_ocean.sh ants local

// #define MAX_ANTS_PER_COLONY 100
#define NUM_COLONIES 2

#include <time.h>
#include <math.h>       // for sqrt, atan2, fabs
#include "ants.h"
#include "puffernet.h"

// Function to toggle tracing mode and setup path
// This function must be defined at the top level, not inside another function.
void toggle_tracing(AntsEnv* env) {
    Ant* tracing_ant = &env->ants[0]; // Assuming ant 0 is the tracer
    static int original_colony_id_for_tracer = 0; // Default to 0
    if (!env->is_tracing_text) { // When about to START tracing
        original_colony_id_for_tracer = tracing_ant->colony_id;
    }

    env->is_tracing_text = !env->is_tracing_text;
    if (env->is_tracing_text) {
        printf("Starting text trace!\n");
        setup_pufferlib_trace_path(env); // Define the waypoints
        env->current_trace_waypoint_index = 0;
        tracing_ant->has_food = false; // Ensure ant is not in "has_food" state
        // tracing_ant->colony_id = 1;    // Omitted: keep sprite orientation consistent
    } else {
        printf("Stopping text trace.\n");
        // tracing_ant->colony_id = original_colony_id_for_tracer; // Omitted: keep sprite orientation consistent
    }
}

int demo() {
    AntsEnv env = {
        .num_ants      = 1,
        .width         = WINDOW_WIDTH,
        .height        = WINDOW_HEIGHT,
        .reward_food   = 0.1f,
        .reward_delivery = 1.0f,
        .reward_death  = -1.0f,
        .cell_size     = 1,
    };

    allocate_ants_env(&env);
    c_reset(&env);

    Weights* weights = load_weights("resources/ants_weights.bin", 266501);
    LinearLSTM* net = NULL;
    if (weights) {
        int logit_sizes[] = {32, 32}; // Two hidden layers of size 32
        net = make_linearlstm(weights, env.num_ants, env.obs_size, logit_sizes, 2);
    }

    printf("Environment initialized. Starting render loop...\n");
    printf("Ants: %d, Observation size: %d\n", env.num_ants, env.obs_size);
    printf("PRESS '5' TO START/STOP PHEROMONE TRACING 'pufferlib 3.0'\n\n");

    env.client = make_client(1, env.width, env.height);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_FIVE)) {
            toggle_tracing(&env);
        }

        if (env.is_tracing_text) {
            if (env.current_trace_waypoint_index < env.num_trace_waypoints) {
                Ant* ant = &env.ants[0]; // Assuming ant 0 is the one tracing
                TraceWaypoint current_target_wp = env.trace_waypoints[env.current_trace_waypoint_index];
                Vector2 target_pos = current_target_wp.position;

                float dx = target_pos.x - ant->position.x;
                float dy = target_pos.y - ant->position.y;
                float distance_to_target = sqrt(dx*dx + dy*dy);
                float arrival_threshold = ANT_SPEED * 1.5f;

                // Make arrival threshold tighter for non-pen_down waypoints to avoid skipping
                if (env.current_trace_waypoint_index > 0 &&
                    !env.trace_waypoints[env.current_trace_waypoint_index - 1].pen_down) {
                    arrival_threshold = ANT_SPEED * 0.8f;
                }

                if (distance_to_target < arrival_threshold) {
                    env.current_trace_waypoint_index++;
                    if (env.current_trace_waypoint_index >= env.num_trace_waypoints) {
                        printf("Finished trace path!\n");
                        toggle_tracing(&env); // This will set is_tracing_text to false
                    }
                } else {
                    float target_angle = atan2(dy, dx);
                    float angle_diff = wrap_angle(target_angle - ant->direction);

                    if (fabs(angle_diff) < TURN_ANGLE * 0.5f) {
                        env.actions[0] = ACTION_MOVE_FORWARD;
                    } else if (angle_diff < 0) {
                        env.actions[0] = ACTION_TURN_LEFT;
                    } else {
                        env.actions[0] = ACTION_TURN_RIGHT;
                    }
                }
            } else { // Should ideally not be reached if toggle_tracing works correctly
                if (env.is_tracing_text) { // If still tracing but no waypoints, stop.
                    toggle_tracing(&env);
                }
            }
            // When tracing, pheromone dropping is controlled by waypoint's pen_down,
            // which is handled in step_ant based on current_trace_waypoint_index.
            // Ensure manual pheromone dropping is off.
            env.ant_is_dropping_pheromone[0] = false;
        } else { // Not tracing text, so allow user or AI control
            // Default action if no other input overrides it
            env.actions[0] = ACTION_MOVE_FORWARD; 
            env.ant_is_dropping_pheromone[0] = false; // Default no pheromone drop

            if (IsKeyDown(KEY_LEFT_SHIFT)) { // User control
                if (IsKeyDown(KEY_LEFT)) {
                    env.actions[0] = ACTION_TURN_LEFT;
                } else if (IsKeyDown(KEY_RIGHT)) {
                    env.actions[0] = ACTION_TURN_RIGHT;
                }
                // If only shift is pressed without left/right, it defaults to ACTION_MOVE_FORWARD set above.
                
                env.ant_is_dropping_pheromone[0] = IsKeyDown(KEY_SPACE);
            } else { // AI / Default behavior when no user input and not tracing
                if (net) {
                    // Example: forward_linearlstm(net, (float*)env.observations, env.actions);
                    // For now, using random actions as placeholder for AI
                    env.actions[0] = rand() % 3; // ACTION_MOVE_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT
                } else {
                    env.actions[0] = rand() % 3;
                }
                // AI does not use the spacebar for pheromones in this setup
                env.ant_is_dropping_pheromone[0] = false; 
            }
        }

        c_step(&env);
        c_render(&env);

        if (env.tick % 1000 == 0 && env.log.n > 0) {
            printf("Tick %d: Episodes completed: %.0f, Avg score: %.2f, Avg return: %.2f\n",
                   env.tick, env.log.n, env.log.score / env.log.n, env.log.episode_return / env.log.n);
        }
    }

    printf("Closing environment...\n");

    if (net)    free_linearlstm(net);
    if (weights) free(weights);
    close_client(env.client);
    free_ants_env(&env);

    return 0;
}

void performance_test(long test_time) {
    AntsEnv env = {
        .num_ants      = 1024,
        .width         = 1280,
        .height        = 720,
        .reward_food   = 0.1f,
        .reward_delivery = 1.0f,
        .reward_death  = -1.0f,
        .cell_size     = 1,
    };

    allocate_ants_env(&env);
    c_reset(&env);

    long start = time(NULL);
    long steps = 0;

    while (time(NULL) - start < test_time) {
        for (int i = 0; i < env.num_ants; i++) {
            env.actions[i] = rand() % 4;
        }
        c_step(&env);
        steps++;
    }

    long end = time(NULL);
    float sps = (float)env.num_ants * steps / (end - start);
    free_ants_env(&env);
    printf("Ant Colony Environment SPS: %.0f\n", sps);
}

int main() {
    srand(time(NULL));

    printf("Ant Colony Environment Demo\n");
    printf("Controls:\n");
    printf("- Hold SHIFT to control the first ant\n");
    printf("- A/D or LEFT/RIGHT to turn\n");
    printf("- SPACE to drop pheromone\n");
    printf("- ESC to exit\n\n");

    demo();

    // Uncomment for performance testing
    // printf("\nRunning performance test...\n");
    // performance_test(10);

    return 0;
}
