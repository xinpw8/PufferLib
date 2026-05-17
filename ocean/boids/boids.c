#include <time.h>
#include "boids.h"
#include "puffernet.h"

static const int BOIDS_HIDDEN_SIZE = 128;
static const int BOIDS_NUM_LAYERS = 4;
static const char* BOIDS_WEIGHTS_PATH = "resources/boids/boids_weights.bin";

static void apply_defaults(Boids* env) {
    env->num_agents = 64;
    env->report_interval = 1;
    env->margin_turn_factor = 1.0f;
    env->cohesion_factor = 0.6f;
    env->separation_factor = 1.4f;
    env->alignment_factor = 0.8f;
    env->rng = (unsigned)time(NULL);
}

static int allocate_buffers(Boids* env) {
    env->observations = (float*)calloc(env->num_agents * BOIDS_OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(env->num_agents * 2, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (float*)calloc(env->num_agents, sizeof(float));
    if (env->observations == NULL || env->actions == NULL
            || env->rewards == NULL || env->terminals == NULL) {
        fprintf(stderr, "ERROR: failed to allocate boids demo buffers\n");
        return 0;
    }
    return 1;
}

static void free_buffers(Boids* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

static long file_size_bytes(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n;
}

static PufferNet* load_policy(void) {
    long bytes = file_size_bytes(BOIDS_WEIGHTS_PATH);
    if (bytes < 0) {
        fprintf(stderr, "Could not open %s. Train first and copy a checkpoint:\n"
                        "    mkdir -p resources/boids\n"
                        "    cp checkpoints/boids/<run_id>/<step>.bin %s\n",
                BOIDS_WEIGHTS_PATH, BOIDS_WEIGHTS_PATH);
        return NULL;
    }

    Weights* weights = load_weights(BOIDS_WEIGHTS_PATH);
    if (weights == NULL) return NULL;
    int logit_sizes[2] = {3, 3};
    return make_puffernet(weights, 64, BOIDS_OBS_SIZE, BOIDS_HIDDEN_SIZE,
        BOIDS_NUM_LAYERS, logit_sizes, 2);
}

static void generate_random_actions(Boids* env) {
    for (unsigned i = 0; i < env->num_agents; i++) {
        env->actions[i*2] = (float)(rand() % 3);
        env->actions[i*2 + 1] = (float)(rand() % 3);
    }
}

static void print_step_log(const char* label, int steps, const Log* total) {
    float n = total->n > 0.0f ? total->n : 1.0f;
    printf("%s: %d steps\n", label, steps);
    printf("  score               %+0.4f\n", total->score / n);
    printf("  margin              %+0.4f\n", total->t_margin_turn_reward / n);
    printf("  cohesion            %+0.4f\n", total->t_cohesion_reward / n);
    printf("  separation          %+0.4f\n", total->t_separation_reward / n);
    printf("  alignment           %+0.4f\n", total->t_alignment_reward / n);
    printf("  speed               %+0.4f\n", total->t_speed_reward / n);
    printf("  action              %+0.4f\n", total->t_action_reward / n);
    printf("  avg speed           %0.4f\n", total->avg_speed / n);
    printf("  visual neighbor frac %0.4f\n", total->avg_visual_count / n);
    printf("  protected frac       %0.4f\n", total->avg_protected_count / n);
}

static void accumulate_log(Log* total, const Log* step) {
    total->score += step->score;
    total->n += step->n;
    total->t_margin_turn_reward += step->t_margin_turn_reward;
    total->t_cohesion_reward += step->t_cohesion_reward;
    total->t_separation_reward += step->t_separation_reward;
    total->t_alignment_reward += step->t_alignment_reward;
    total->t_speed_reward += step->t_speed_reward;
    total->t_action_reward += step->t_action_reward;
    total->avg_speed += step->avg_speed;
    total->avg_visual_count += step->avg_visual_count;
    total->avg_protected_count += step->avg_protected_count;
}

static int headless_eval(int steps, int trained) {
    Boids env = {0};
    apply_defaults(&env);
    if (!allocate_buffers(&env)) return 1;
    init(&env);
    c_reset(&env);

    PufferNet* net = trained ? load_policy() : NULL;
    if (trained && net == NULL) {
        c_close(&env);
        free_buffers(&env);
        return 1;
    }

    Log total = {0};
    for (int t = 0; t < steps; t++) {
        if (trained) {
            forward_puffernet(net, env.observations, env.actions);
        } else {
            generate_random_actions(&env);
        }
        c_step(&env);
        accumulate_log(&total, &env.log);
    }

    print_step_log(trained ? "Boids trained-policy eval" : "Boids random-policy eval",
        steps, &total);
    c_close(&env);
    free_buffers(&env);
    return 0;
}

static int demo(int trained) {
    Boids env = {0};
    apply_defaults(&env);
    if (!allocate_buffers(&env)) return 1;
    init(&env);
    c_reset(&env);

    PufferNet* net = trained ? load_policy() : NULL;
    if (trained && net == NULL) {
        c_close(&env);
        free_buffers(&env);
        return 1;
    }

    env.client = make_client(&env);
    if (env.client == NULL) {
        c_close(&env);
        free_buffers(&env);
        return 1;
    }

    while (!WindowShouldClose()) {
        if (trained) {
            forward_puffernet(net, env.observations, env.actions);
        } else {
            generate_random_actions(&env);
        }
        c_step(&env);
        c_render(&env);
    }

    c_close(&env);
    free_buffers(&env);
    return 0;
}

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));
    if (argc > 1 && strcmp(argv[1], "p") == 0) {
        return headless_eval(200000, 0);
    }
    if (argc > 1 && strcmp(argv[1], "random") == 0) {
        return demo(0);
    }
    if (argc > 1 && strcmp(argv[1], "eval") == 0) {
        int steps = argc > 2 ? atoi(argv[2]) : 1000;
        return headless_eval(steps, 1);
    }
    return demo(1);
}
