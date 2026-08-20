#ifndef PUFFERLIB_OCEAN_CLIFFORD_CLIFFORD_H
#define PUFFERLIB_OCEAN_CLIFFORD_CLIFFORD_H

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
typedef unsigned char obs_t;
#include "pufferenv.h"

#ifndef CLIFFORD_N_QUBITS
#define CLIFFORD_N_QUBITS 6
#endif
#if CLIFFORD_N_QUBITS < 1 || CLIFFORD_N_QUBITS > 32
#error "CLIFFORD_N_QUBITS must be in [1, 32]; tableau rows are stored in uint64_t"
#endif
#define CLIFFORD_DIM (2 * CLIFFORD_N_QUBITS)
#ifndef CLIFFORD_USE_SHORTCUT_GATES
#define CLIFFORD_USE_SHORTCUT_GATES 0
#endif
#ifndef CLIFFORD_PAIR_ONEHOT
#define CLIFFORD_PAIR_ONEHOT 0
#endif
#if CLIFFORD_PAIR_ONEHOT
#define CLIFFORD_OBS_SIZE (CLIFFORD_N_QUBITS * CLIFFORD_N_QUBITS * 16)
#else
#define CLIFFORD_OBS_SIZE (CLIFFORD_DIM * CLIFFORD_DIM)
#endif
#define CLIFFORD_SINGLE_QUBIT_ACTIONS (CLIFFORD_USE_SHORTCUT_GATES ? 5 : 2)
#define CLIFFORD_NUM_ACTIONS (CLIFFORD_SINGLE_QUBIT_ACTIONS * CLIFFORD_N_QUBITS \
    + (CLIFFORD_N_QUBITS * (CLIFFORD_N_QUBITS - 1)) / 2)

#define ACT_SIZES {CLIFFORD_NUM_ACTIONS}
#define OBS_SIZE CLIFFORD_OBS_SIZE
#define NUM_ATNS 1

#define GATE_H 0
#define GATE_S 1
#define GATE_V 2
#define GATE_HS 3
#define GATE_HV 4
#define GATE_CZ 5

#define CLIFFORD_CELL 36
#define CLIFFORD_MARGIN 24
#define CLIFFORD_HUD 88

typedef struct {
    int gate_kind;
    int q0;
    int q1;
} CliffordAction;

// Log is a flat float struct; trainer averages by n.
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float episode_cz_sum;
    float success_rate;
    float difficulty;
    float max_steps;
    float n;
    float success_count;
    float success_step_sum;
    float success_step_sq_sum;
    float success_cz_sum;
};

#if defined(__cplusplus)
static_assert(sizeof(Log) % sizeof(float) == 0, "Log must be float-packed");
#else
_Static_assert(sizeof(Log) % sizeof(float) == 0, "Log must be float-packed");
#endif

typedef struct {
    uint64_t state;
} XorShift64;

typedef struct Client Client;
struct Client {
    int cell;
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    Client* client;
    // Tableau columns packed as bitmasks: bit r of cols[c] is T[r, c].
    uint64_t cols[CLIFFORD_DIM];
    int difficulty;
    float difficulty_fraction;
    int max_steps;
    float single_qubit_cost;
    float cz_cost;
    float goal_bonus;
    float failure_penalty;
    float hamming_scale;
    float curriculum_threshold;
    float curriculum_step;
    int curriculum_window;
    int curriculum_max;
    int curriculum_count;
    int curriculum_successes;
    float episode_return;
    int episode_length;
    int episode_cz_count;
    int steps;
    int episode_max_steps;
    int last_action;
    XorShift64 xor_rng;
    unsigned int rng;
};
typedef Env Clifford;

static inline uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline void rng_seed(XorShift64* rng, uint64_t seed) {
    if (seed == 0) {
        seed = 0x123456789ABCDEFULL;
    }
    rng->state = seed;
}

static inline uint64_t rng_next_u64(XorShift64* rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 2685821657736338717ULL;
}

static inline int rng_below(XorShift64* rng, int upper) {
    if (upper <= 1) {
        return 0;
    }
    return (int)(rng_next_u64(rng) % (uint64_t)upper);
}

static inline float rng_float01(XorShift64* rng) {
    return (float)((rng_next_u64(rng) >> 40) * (1.0 / 16777216.0));
}

static inline void set_difficulty_level(Clifford* env, double difficulty_level) {
    if (difficulty_level < 0.0) {
        difficulty_level = 0.0;
    }
    double floor_level = floor(difficulty_level + 1e-12);
    env->difficulty = (int)floor_level;
    env->difficulty_fraction = (float)(difficulty_level - floor_level);
    if (env->difficulty_fraction <= 1e-6f) {
        env->difficulty_fraction = 0.0f;
    } else if (env->difficulty_fraction >= 1.0f - 1e-6f) {
        env->difficulty += 1;
        env->difficulty_fraction = 0.0f;
    }
}

static inline int sample_reset_difficulty(Clifford* env) {
    if (env->difficulty_fraction <= 0.0f) {
        return env->difficulty;
    }
    return env->difficulty + (rng_float01(&env->xor_rng) < env->difficulty_fraction ? 1 : 0);
}

static inline void copy_identity_cols(Clifford* env) {
    for (int col = 0; col < CLIFFORD_DIM; ++col) {
        env->cols[col] = 1ULL << col;
    }
}

static inline void reset_episode_state(Clifford* env) {
    env->steps = 0;
    env->episode_max_steps = env->max_steps;
    env->episode_return = 0.0f;
    env->episode_length = 0;
    env->episode_cz_count = 0;
    env->last_action = -1;
}

static inline int tableau_hamming(const Clifford* env) {
    int hamming = 0;
    for (int col = 0; col < CLIFFORD_DIM; ++col) {
        hamming += __builtin_popcountll(env->cols[col] ^ (1ULL << col));
    }
    return hamming;
}

static inline int is_identity(const Clifford* env) {
    return tableau_hamming(env) == 0;
}

static inline CliffordAction decode_action(int action_idx) {
    CliffordAction action;
    action.gate_kind = GATE_H;
    action.q0 = 0;
    action.q1 = -1;
    if (action_idx < 0 || action_idx >= CLIFFORD_NUM_ACTIONS) {
        return action;
    }
    const int n = CLIFFORD_N_QUBITS;
    const int single = CLIFFORD_SINGLE_QUBIT_ACTIONS * n;
    if (action_idx < single) {
        action.gate_kind = action_idx / n;
        action.q0 = action_idx % n;
        action.q1 = -1;
        return action;
    }
    int pair = action_idx - single;
    for (int src = 0; src < n; ++src) {
        int remaining = n - src - 1;
        if (pair < remaining) {
            action.gate_kind = GATE_CZ;
            action.q0 = src;
            action.q1 = src + 1 + pair;
            return action;
        }
        pair -= remaining;
    }
    return action;
}

static inline int sample_action(Clifford* env) {
    return rng_below(&env->xor_rng, CLIFFORD_NUM_ACTIONS);
}

static inline void apply_action(Clifford* env, int action_idx) {
    CliffordAction action = decode_action(action_idx);
    const int n = CLIFFORD_N_QUBITS;
    const int q = action.q0;
    const uint64_t x = env->cols[q];
    const uint64_t z = env->cols[n + q];
    if (action.gate_kind == GATE_H) {
        env->cols[q] = z;
        env->cols[n + q] = x;
    } else if (action.gate_kind == GATE_S) {
        env->cols[q] = x;
        env->cols[n + q] = z ^ x;
    } else if (action.gate_kind == GATE_V) {
        env->cols[q] = x ^ z;
        env->cols[n + q] = z;
    } else if (action.gate_kind == GATE_HS) {
        env->cols[q] = z;
        env->cols[n + q] = x ^ z;
    } else if (action.gate_kind == GATE_HV) {
        env->cols[q] = x ^ z;
        env->cols[n + q] = x;
    } else {
        env->cols[n + action.q0] ^= env->cols[action.q1];
        env->cols[n + action.q1] ^= env->cols[action.q0];
    }
}

static inline void write_observation(Clifford* env) {
    obs_t* obs = env->agents[0].observations;
#if CLIFFORD_PAIR_ONEHOT
    const int n = CLIFFORD_N_QUBITS;
    memset(obs, 0, OBS_SIZE);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int pattern = (int)((env->cols[j] >> i) & 1ULL);
            pattern |= (int)((env->cols[n + j] >> i) & 1ULL) << 1;
            pattern |= (int)((env->cols[j] >> (n + i)) & 1ULL) << 2;
            pattern |= (int)((env->cols[n + j] >> (n + i)) & 1ULL) << 3;
            obs[(i * n + j) * 16 + pattern] = 1;
        }
    }
#else
    const int dim = CLIFFORD_DIM;
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            obs[row * dim + col] = (unsigned char)((env->cols[col] >> row) & 1ULL);
        }
    }
#endif
}

static inline void reset_single(Clifford* env) {
    reset_episode_state(env);
    const int reset_difficulty = sample_reset_difficulty(env);
    if (reset_difficulty <= 0) {
        copy_identity_cols(env);
        write_observation(env);
        return;
    }

    do {
        copy_identity_cols(env);
        for (int step = 0; step < reset_difficulty; ++step) {
            apply_action(env, sample_action(env));
        }
    } while (is_identity(env));

    write_observation(env);
}

static inline void add_log(Clifford* env, int success) {
    Log* log = &env->log;
    float difficulty_level = (float)env->difficulty + env->difficulty_fraction;
    log->perf += success ? 1.0f : 0.0f;
    log->score += env->episode_return;
    log->episode_return += env->episode_return;
    log->episode_length += (float)env->episode_length;
    log->episode_cz_sum += (float)env->episode_cz_count;
    log->success_rate += success ? 1.0f : 0.0f;
    log->difficulty += difficulty_level;
    log->max_steps += (float)env->max_steps;
    log->n += 1.0f;
    if (success) {
        log->success_count += 1.0f;
        log->success_step_sum += (float)env->episode_length;
        log->success_step_sq_sum += (float)(env->episode_length * env->episode_length);
        log->success_cz_sum += (float)env->episode_cz_count;
    }
    if (env->curriculum_threshold > 0.0f && env->curriculum_window > 0) {
        env->curriculum_count += 1;
        env->curriculum_successes += success ? 1 : 0;
        if (env->curriculum_count >= env->curriculum_window) {
            float rate = (float)env->curriculum_successes
                / (float)env->curriculum_count;
            float level = (float)env->difficulty + env->difficulty_fraction;
            if (rate >= env->curriculum_threshold
                    && level + 1e-6f < (float)env->curriculum_max) {
                set_difficulty_level(env, level + env->curriculum_step);
            }
            env->curriculum_count = 0;
            env->curriculum_successes = 0;
        }
    }
}

static inline float gate_cost_reward(const Clifford* env, int gate_kind) {
    return gate_kind == GATE_CZ ? -env->cz_cost : -env->single_qubit_cost;
}

static inline int wrap_action(float raw) {
    if (!isfinite(raw)) {
        return 0;
    }
    int action_idx = ((int)raw) % CLIFFORD_NUM_ACTIONS;
    if (action_idx < 0) {
        action_idx += CLIFFORD_NUM_ACTIONS;
    }
    return action_idx;
}

static inline const char* gate_name(int gate_kind) {
    if (gate_kind == GATE_H) return "H";
    if (gate_kind == GATE_S) return "S";
    if (gate_kind == GATE_V) return "V";
    if (gate_kind == GATE_HS) return "HS";
    if (gate_kind == GATE_HV) return "HV";
    if (gate_kind == GATE_CZ) return "CZ";
    return "?";
}

void init(Clifford* env) {
    memset(&env->log, 0, sizeof(Log));
    env->last_action = -1;
}

void puf_close(Clifford* env) {
    if (env->client != NULL) {
        if (IsWindowReady()) {
            CloseWindow();
        }
        free(env->client);
        env->client = NULL;
    }
}

static Client* make_client(Clifford* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->cell = CLIFFORD_CELL;
    int dim = CLIFFORD_DIM;
    int width = CLIFFORD_MARGIN * 2 + dim * client->cell;
    int height = CLIFFORD_HUD + CLIFFORD_MARGIN * 2 + dim * client->cell;
    InitWindow(width, height, "puffer Clifford");
    SetTargetFPS(30);
    return client;
}

void puf_render(Clifford* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    const Color PUFF_RED = (Color){187, 0, 0, 255};
    const Color PUFF_CYAN = (Color){0, 187, 187, 255};
    const Color PUFF_WHITE = (Color){241, 241, 241, 241};
    const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
    const Color ZERO = (Color){18, 42, 42, 255};
    const Color DIAG = (Color){40, 90, 90, 255};

    int cell = env->client->cell;
    int dim = CLIFFORD_DIM;
    int n = CLIFFORD_N_QUBITS;
    int ox = CLIFFORD_MARGIN;
    int oy = CLIFFORD_HUD;

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawText(TextFormat("n=%d  steps=%d/%d  return=%.3f",
        n, env->steps, env->episode_max_steps, env->episode_return),
        10, 10, 20, PUFF_WHITE);
    if (env->last_action >= 0) {
        CliffordAction action = decode_action(env->last_action);
        if (action.gate_kind == GATE_CZ) {
            DrawText(TextFormat("last: %s q%d q%d",
                gate_name(action.gate_kind), action.q0, action.q1),
                10, 36, 20, PUFF_CYAN);
        } else {
            DrawText(TextFormat("last: %s q%d",
                gate_name(action.gate_kind), action.q0),
                10, 36, 20, PUFF_CYAN);
        }
    } else {
        DrawText("last: (reset)", 10, 36, 20, PUFF_CYAN);
    }
    DrawText(TextFormat("solved=%s  cz=%d",
        is_identity(env) ? "yes" : "no", env->episode_cz_count),
        10, 62, 18, PUFF_WHITE);

    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            int bit = (int)((env->cols[col] >> row) & 1ULL);
            Color color = bit ? PUFF_CYAN : (row == col ? DIAG : ZERO);
            DrawRectangle(ox + col * cell, oy + row * cell, cell - 2, cell - 2, color);
            if (bit) {
                DrawText("1", ox + col * cell + 12, oy + row * cell + 8, 18, PUFF_BACKGROUND);
            }
        }
    }
    DrawLine(ox + n * cell - 1, oy, ox + n * cell - 1, oy + dim * cell, PUFF_RED);
    DrawLine(ox, oy + n * cell - 1, ox + dim * cell, oy + n * cell - 1, PUFF_RED);
    EndDrawing();
    puf_web_vsync();
}

void puf_reset(Clifford* env) {
    if (env->agents[0].rewards != NULL) {
        env->agents[0].rewards[0] = 0.0f;
    }
    if (env->agents[0].terminals != NULL) {
        env->agents[0].terminals[0] = 0.0f;
    }
    reset_single(env);
}

void puf_step(Clifford* env) {
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;

    int action_idx = wrap_action(env->agents[0].actions[0]);
    CliffordAction action = decode_action(action_idx);
    apply_action(env, action_idx);
    env->last_action = action_idx;
    env->steps += 1;
    env->episode_length += 1;
    if (action.gate_kind == GATE_CZ) {
        env->episode_cz_count += 1;
    }

    int terminated = is_identity(env);
    int truncated = (!terminated && env->steps >= env->episode_max_steps);
    float reward = gate_cost_reward(env, action.gate_kind);
    float hamming_norm = 8.0f * (float)(CLIFFORD_N_QUBITS * CLIFFORD_N_QUBITS);
    reward -= env->hamming_scale * (float)tableau_hamming(env) / hamming_norm;
    if (terminated) {
        reward += env->goal_bonus;
    } else if (truncated) {
        reward += env->failure_penalty;
    }
    env->episode_return += reward;
    env->agents[0].rewards[0] = reward;
    env->agents[0].terminals[0] = (float)(terminated || truncated);

    if (terminated || truncated) {
        add_log(env, terminated);
        reset_single(env);
    } else {
        write_observation(env);
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "mean_cz", log->episode_cz_sum);
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "difficulty", log->difficulty);
    dict_set(out, "max_steps", log->max_steps);
    dict_set(out, "n", log->n);

    float success_rate = log->success_count;
    float mean_success_steps = success_rate > 0.0f ? log->success_step_sum / success_rate : 0.0f;
    float mean_success_cz = success_rate > 0.0f ? log->success_cz_sum / success_rate : 0.0f;
    float success_step_second = success_rate > 0.0f ? log->success_step_sq_sum / success_rate : 0.0f;
    float success_step_var = success_step_second - mean_success_steps * mean_success_steps;
    if (success_step_var < 0.0f) {
        success_step_var = 0.0f;
    }
    dict_set(out, "success_step_mean", mean_success_steps);
    dict_set(out, "success_step_std", sqrtf(success_step_var));
    dict_set(out, "mean_success_cz", mean_success_cz);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->client = NULL;
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;

    int requested_n = dict_get(kwargs, "n_qubits");
    if (requested_n != CLIFFORD_N_QUBITS) {
        fprintf(stderr,
            "clifford is compiled for n_qubits=%d, got n_qubits=%d\n",
            CLIFFORD_N_QUBITS, requested_n);
    }
    assert(requested_n == CLIFFORD_N_QUBITS);

    int requested_shortcuts = dict_get(kwargs, "use_shortcut_gates");
    if (requested_shortcuts != CLIFFORD_USE_SHORTCUT_GATES) {
        fprintf(stderr,
            "Clifford env was compiled with CLIFFORD_USE_SHORTCUT_GATES=%d but got use_shortcut_gates=%d\n",
            CLIFFORD_USE_SHORTCUT_GATES, requested_shortcuts);
    }
    assert(requested_shortcuts == CLIFFORD_USE_SHORTCUT_GATES);

    set_difficulty_level(env, dict_get(kwargs, "difficulty"));
    env->max_steps = dict_get(kwargs, "max_steps");
    env->single_qubit_cost = dict_get(kwargs, "single_qubit_cost");
    env->cz_cost = dict_get(kwargs, "cz_cost");
    env->goal_bonus = dict_get(kwargs, "goal_bonus");
    env->failure_penalty = dict_get(kwargs, "failure_penalty");
    env->hamming_scale = dict_get(kwargs, "hamming_scale");
    int requested_onehot = dict_get(kwargs, "pair_onehot");
    if (requested_onehot != CLIFFORD_PAIR_ONEHOT) {
        fprintf(stderr,
            "Clifford env was compiled with CLIFFORD_PAIR_ONEHOT=%d but got pair_onehot=%d\n",
            CLIFFORD_PAIR_ONEHOT, requested_onehot);
    }
    assert(requested_onehot == CLIFFORD_PAIR_ONEHOT);
    assert(env->max_steps > 0);
    assert(env->single_qubit_cost >= 0.0f);
    assert(env->cz_cost >= 0.0f);
    assert(env->goal_bonus >= 0.0f);
    assert(env->failure_penalty <= 0.0f);
    env->curriculum_threshold = dict_get(kwargs, "curriculum_threshold");
    env->curriculum_step = dict_get(kwargs, "curriculum_step");
    env->curriculum_window = dict_get(kwargs, "curriculum_window");
    env->curriculum_max = dict_get(kwargs, "curriculum_max");
    env->curriculum_count = 0;
    env->curriculum_successes = 0;
    assert(env->hamming_scale >= 0.0f);
    assert(env->curriculum_threshold >= 0.0f);
    assert(env->curriculum_step >= 0.0f);
    assert(env->curriculum_window >= 0);
    assert(env->curriculum_max >= 0);

    uint64_t seed_state = (uint64_t)(uint32_t)dict_get(kwargs, "seed");
    uint64_t mixed = seed_state;
    for (unsigned int i = 0; i <= env->rng; ++i) {
        mixed = splitmix64_next(&seed_state);
    }
    rng_seed(&env->xor_rng, mixed ^ (uint64_t)(env->rng + 1));

    init(env);
}

#endif
