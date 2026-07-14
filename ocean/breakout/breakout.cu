// LLM gen crap. Just an unoptimized GPU breakout for testing this training path.
#ifndef PUFFER_BREAKOUT_GPU_CU
#define PUFFER_BREAKOUT_GPU_CU

#define PUFFER_GPU_ENV_AVAILABLE 1
#define BREAKOUT_GPU_MAX_BRICKS 256
#define BREAKOUT_GPU_PI 3.14159265358979323846f

typedef struct GpuBreakout {
    Log log;
    int num_agents;
    int tag;
    int boundary_reached;
    int score;
    float paddle_x;
    float paddle_y;
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    float brick_x[BREAKOUT_GPU_MAX_BRICKS];
    float brick_y[BREAKOUT_GPU_MAX_BRICKS];
    float brick_states[BREAKOUT_GPU_MAX_BRICKS];
    int balls_fired;
    float initial_paddle_width;
    float paddle_width;
    float paddle_height;
    float paddle_speed;
    float ball_speed;
    float initial_ball_speed;
    float max_ball_speed;
    int hits;
    int width;
    int height;
    int num_bricks;
    int brick_rows;
    int brick_cols;
    int ball_width;
    int ball_height;
    int brick_width;
    int brick_height;
    int num_balls;
    int max_score;
    int half_max_score;
    int tick;
    int frameskip;
    unsigned char hit_brick;
    int continuous;
    unsigned int rng;
} GpuBreakout;

typedef struct GpuBreakoutCollisionInfo {
    float t;
    float overlap;
    float x;
    float y;
    float vx;
    float vy;
    int brick_index;
} GpuBreakoutCollisionInfo;

__host__ __device__ static inline void gpu_breakout_generate_brick_positions(GpuBreakout* env) {
    env->half_max_score = 0;
    for (int row = 0; row < env->brick_rows; row++) {
        for (int col = 0; col < env->brick_cols; col++) {
            int idx = row * env->brick_cols + col;
            env->brick_x[idx] = col * env->brick_width;
            env->brick_y[idx] = row * env->brick_height + Y_OFFSET;
            env->half_max_score += 7 - 3 * (idx / env->brick_cols / 2);
        }
    }
    env->max_score = 2 * env->half_max_score;
}

__device__ static inline unsigned int gpu_breakout_rand_r(unsigned int* seed) {
    unsigned int next = *seed;
    int result;

    next *= 1103515245U;
    next += 12345U;
    result = (unsigned int)(next / 65536U) % 2048U;

    next *= 1103515245U;
    next += 12345U;
    result <<= 10;
    result ^= (unsigned int)(next / 65536U) % 1024U;

    next *= 1103515245U;
    next += 12345U;
    result <<= 10;
    result ^= (unsigned int)(next / 65536U) % 1024U;

    *seed = next;
    return (unsigned int)result;
}

__device__ static inline void gpu_breakout_add_log(GpuBreakout* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.perf += env->score / (float)env->max_score;
    env->log.n += 1.0f;
}

__device__ static inline void gpu_breakout_compute_observations(GpuBreakout* env, obs_t* obs) {
    obs[0] = env->paddle_x / env->width;
    obs[1] = env->paddle_y / env->height;
    obs[2] = env->ball_x / env->width;
    obs[3] = env->ball_y / env->height;
    obs[4] = env->ball_vx / 512.0f;
    obs[5] = env->ball_vy / 512.0f;
    obs[6] = env->balls_fired / 5.0f;
    obs[7] = env->score / 864.0f;
    obs[8] = env->num_balls / 5.0f;
    obs[9] = env->paddle_width / (2.0f * HALF_PADDLE_WIDTH);
    for (int i = 0; i < env->num_bricks; i++) {
        obs[10 + i] = env->brick_states[i];
    }
}

__device__ static inline bool gpu_breakout_calc_vline_collision(float xw, float yw, float hw,
        float x, float y, float vx, float vy, float h, GpuBreakoutCollisionInfo* col) {
    float t_new = (xw - x) / vx;
    float topmost = fminf(yw + hw, y + h + vy * t_new);
    float botmost = fmaxf(yw, y + vy * t_new);
    float overlap_new = topmost - botmost;

    if (overlap_new > 0.0f && t_new > 0.0f && t_new <= 1.0f &&
            (t_new < col->t || (t_new == col->t && overlap_new > col->overlap))) {
        col->t = t_new;
        col->overlap = overlap_new;
        col->x = xw;
        col->y = y + vy * t_new;
        col->vx = -vx;
        col->vy = vy;
        return true;
    }
    return false;
}

__device__ static inline bool gpu_breakout_calc_hline_collision(float xw, float yw, float ww,
        float x, float y, float vx, float vy, float w, GpuBreakoutCollisionInfo* col) {
    float t_new = (yw - y) / vy;
    float rightmost = fminf(xw + ww, x + w + vx * t_new);
    float leftmost = fmaxf(xw, x + vx * t_new);
    float overlap_new = rightmost - leftmost;

    if (overlap_new > 0.0f && t_new > 0.0f && t_new <= 1.0f &&
            (t_new < col->t || (t_new == col->t && overlap_new > col->overlap))) {
        col->t = t_new;
        col->overlap = overlap_new;
        col->x = x + vx * t_new;
        col->y = yw;
        col->vx = vx;
        col->vy = -vy;
        return true;
    }
    return false;
}

__device__ static inline void gpu_breakout_calc_brick_collision(GpuBreakout* env, int idx,
        GpuBreakoutCollisionInfo* collision_info) {
    bool collision = false;
    if (env->ball_vx > 0) {
        if (gpu_breakout_calc_vline_collision(env->brick_x[idx], env->brick_y[idx], env->brick_height,
                env->ball_x + env->ball_width, env->ball_y, env->ball_vx, env->ball_vy,
                env->ball_height, collision_info)) {
            collision = true;
            collision_info->x -= env->ball_width;
        }
    }

    if (env->ball_vx < 0) {
        if (gpu_breakout_calc_vline_collision(env->brick_x[idx] + env->brick_width,
                env->brick_y[idx], env->brick_height, env->ball_x, env->ball_y,
                env->ball_vx, env->ball_vy, env->ball_height, collision_info)) {
            collision = true;
        }
    }

    if (env->ball_vy > 0) {
        if (gpu_breakout_calc_hline_collision(env->brick_x[idx], env->brick_y[idx], env->brick_width,
                env->ball_x, env->ball_y + env->ball_height, env->ball_vx, env->ball_vy,
                env->ball_width, collision_info)) {
            collision = true;
            collision_info->y -= env->ball_height;
        }
    }

    if (env->ball_vy < 0) {
        if (gpu_breakout_calc_hline_collision(env->brick_x[idx], env->brick_y[idx] + env->brick_height,
                env->brick_width, env->ball_x, env->ball_y, env->ball_vx, env->ball_vy,
                env->ball_width, collision_info)) {
            collision = true;
        }
    }
    if (collision) {
        collision_info->brick_index = idx;
    }
}

__device__ static inline int gpu_breakout_column_index(GpuBreakout* env, float x) {
    return (int)(x / env->brick_width);
}

__device__ static inline int gpu_breakout_row_index(GpuBreakout* env, float y) {
    return (int)((y - Y_OFFSET) / env->brick_height);
}

__device__ static inline void gpu_breakout_calc_all_brick_collisions(GpuBreakout* env,
        GpuBreakoutCollisionInfo* collision_info) {
    float ball_x = env->ball_x;
    float ball_x_dst = ball_x + env->ball_vx;
    float ball_y = env->ball_y;
    float ball_y_dst = ball_y + env->ball_vy;
    float ball_width = env->ball_width;
    float ball_height = env->ball_height;

    int row_from = gpu_breakout_row_index(env, ball_y < ball_y_dst ? ball_y : ball_y_dst);
    if (row_from < 0) {
        row_from = 0;
    }

    if (row_from > env->brick_rows) {
        return;
    }

    int column_from = gpu_breakout_column_index(env, ball_x < ball_x_dst ? ball_x : ball_x_dst);
    if (column_from < 0) {
        column_from = 0;
    }

    float ball_x_end = ball_x + ball_width;
    float ball_x_dst_end = ball_x_dst + ball_width;
    int column_to = gpu_breakout_column_index(env, ball_x_dst_end > ball_x_end ? ball_x_dst_end : ball_x_end);
    if (column_to >= env->brick_cols) {
        column_to = env->brick_cols - 1;
    }

    float ball_y_end = ball_y + ball_height;
    float ball_y_dst_end = ball_y_dst + ball_height;
    int row_to = gpu_breakout_row_index(env, ball_y_dst_end > ball_y_end ? ball_y_dst_end : ball_y_end);
    if (row_to >= env->brick_rows) {
        row_to = env->brick_rows - 1;
    }

    for (int row = row_from; row <= row_to; row++) {
        for (int column = column_from; column <= column_to; column++) {
            int brick_index = row * env->brick_cols + column;
            if (env->brick_states[brick_index] == 0.0f) {
                gpu_breakout_calc_brick_collision(env, brick_index, collision_info);
            }
        }
    }
}

__device__ static inline bool gpu_breakout_calc_paddle_ball_collisions(GpuBreakout* env,
        GpuBreakoutCollisionInfo* collision_info) {
    float base_angle = BREAKOUT_GPU_PI / 4.0f;

    if (env->ball_y + env->ball_height + env->ball_vy < env->paddle_y) {
        return false;
    }

    if (!gpu_breakout_calc_hline_collision(env->paddle_x, env->paddle_y, env->paddle_width,
            env->ball_x, env->ball_y + env->ball_height, env->ball_vx, env->ball_vy,
            env->ball_width, collision_info) || collision_info->t > 1.0f) {
        return false;
    }

    collision_info->y -= env->ball_height;
    collision_info->brick_index = BRICK_INDEX_PADDLE_COLLISION;

    env->hit_brick = false;
    float relative_intersection = ((env->ball_x + env->ball_width / 2) - env->paddle_x) / env->paddle_width;
    float angle = -base_angle + relative_intersection * 2.0f * base_angle;
    env->ball_vx = sinf(angle) * env->ball_speed * TICK_RATE;
    env->ball_vy = -cosf(angle) * env->ball_speed * TICK_RATE;
    env->hits += 1;
    if (env->hits % 4 == 0 && env->ball_speed < env->max_ball_speed) {
        env->ball_speed += 64;
    }
    if (env->score == env->half_max_score) {
        for (int i = 0; i < env->num_bricks; i++) {
            env->brick_states[i] = 0.0f;
        }
    }
    return true;
}

__device__ static inline void gpu_breakout_calc_all_wall_collisions(GpuBreakout* env,
        GpuBreakoutCollisionInfo* collision_info) {
    if (env->ball_vx < 0) {
        if (gpu_breakout_calc_vline_collision(0, 0, env->height,
                env->ball_x, env->ball_y, env->ball_vx, env->ball_vy, env->ball_height,
                collision_info)) {
            collision_info->brick_index = BRICK_INDEX_SIDEWALL_COLLISION;
        }
    }
    if (env->ball_vx > 0) {
        if (gpu_breakout_calc_vline_collision(env->width, 0, env->height,
                env->ball_x + env->ball_width, env->ball_y, env->ball_vx, env->ball_vy,
                env->ball_height, collision_info)) {
            collision_info->x -= env->ball_width;
            collision_info->brick_index = BRICK_INDEX_SIDEWALL_COLLISION;
        }
    }
    if (env->ball_vy < 0) {
        if (gpu_breakout_calc_hline_collision(0, 0, env->width,
                env->ball_x, env->ball_y, env->ball_vx, env->ball_vy, env->ball_width,
                collision_info)) {
            collision_info->brick_index = BRICK_INDEX_BACKWALL_COLLISION;
        }
    }
}

__device__ static inline void gpu_breakout_check_wall_bounds(GpuBreakout* env) {
    float offset = env->max_ball_speed * 1.1f * TICK_RATE;
    if (env->ball_x < 0) {
        env->ball_x += offset;
    }
    if (env->ball_x > env->width) {
        env->ball_x -= offset;
    }
    if (env->ball_y < 0) {
        env->ball_y += offset;
    }
}

__device__ static inline void gpu_breakout_destroy_brick(GpuBreakout* env, int brick_idx,
        float* reward) {
    float gained_points = 7 - 3 * ((brick_idx / env->brick_cols) / 2);

    env->score += gained_points;
    env->brick_states[brick_idx] = 1.0f;
    *reward += gained_points;

    if (brick_idx / env->brick_cols < 3) {
        env->ball_speed = env->max_ball_speed;
    }
}

__device__ static inline bool gpu_breakout_handle_collisions(GpuBreakout* env, float* reward) {
    GpuBreakoutCollisionInfo collision_info = {
        .t = 2.0f,
        .overlap = -1.0f,
        .x = 0.0f,
        .y = 0.0f,
        .vx = 0.0f,
        .vy = 0.0f,
        .brick_index = BRICK_INDEX_NO_COLLISION,
    };

    gpu_breakout_check_wall_bounds(env);

    gpu_breakout_calc_all_brick_collisions(env, &collision_info);
    gpu_breakout_calc_all_wall_collisions(env, &collision_info);
    gpu_breakout_calc_paddle_ball_collisions(env, &collision_info);
    if (collision_info.brick_index != BRICK_INDEX_PADDLE_COLLISION && collision_info.t <= 1.0f) {
        env->ball_x = collision_info.x;
        env->ball_y = collision_info.y;
        env->ball_vx = collision_info.vx;
        env->ball_vy = collision_info.vy;
        if (collision_info.brick_index >= 0) {
            gpu_breakout_destroy_brick(env, collision_info.brick_index, reward);
        }
        if (collision_info.brick_index == BRICK_INDEX_BACKWALL_COLLISION) {
            env->paddle_width = HALF_PADDLE_WIDTH;
        }
    }
    return collision_info.brick_index != BRICK_INDEX_NO_COLLISION;
}

__device__ static inline void gpu_breakout_reset_round(GpuBreakout* env) {
    env->balls_fired = 0;
    env->hit_brick = false;
    env->hits = 0;
    env->ball_speed = env->initial_ball_speed;
    env->paddle_width = env->initial_paddle_width;

    env->paddle_x = env->width / 2.0f - env->paddle_width / 2.0f;
    env->paddle_y = env->height - env->paddle_height - 10;

    env->ball_x = env->paddle_x + (env->paddle_width / 2.0f - env->ball_width / 2.0f);
    env->ball_y = env->height / 2.0f - 30;

    env->ball_vx = 0.0f;
    env->ball_vy = 0.0f;
}

__device__ static inline void gpu_breakout_reset_state(GpuBreakout* env) {
    env->score = 0;
    env->num_balls = 5;
    for (int i = 0; i < env->num_bricks; i++) {
        env->brick_states[i] = 0.0f;
    }
    gpu_breakout_reset_round(env);
    env->tick = 0;
}

__device__ static inline void gpu_breakout_step_frame(GpuBreakout* env, float action,
        float* reward, float* terminal) {
    float act = 0.0f;
    if (env->balls_fired == 0) {
        env->balls_fired = 1;
        float direction = BREAKOUT_GPU_PI / 3.25f;

        env->ball_vy = cosf(direction) * env->ball_speed * TICK_RATE;
        env->ball_vx = sinf(direction) * env->ball_speed * TICK_RATE;
        if (gpu_breakout_rand_r(&env->rng) % 2 == 0) {
            env->ball_vx = -env->ball_vx;
        }
    } else if (action == LEFT) {
        act = -1.0f;
    } else if (action == RIGHT) {
        act = 1.0f;
    }
    if (env->continuous) {
        act = action;
    }
    env->paddle_x += act * env->paddle_speed * TICK_RATE;
    if (env->paddle_x <= 0) {
        env->paddle_x = fmaxf(0, env->paddle_x);
    } else {
        env->paddle_x = fminf(env->width - env->paddle_width, env->paddle_x);
    }

    if (!gpu_breakout_handle_collisions(env, reward)) {
        env->ball_x += env->ball_vx;
        env->ball_y += env->ball_vy;
    }

    if (env->ball_y >= env->paddle_y + env->paddle_height) {
        env->num_balls -= 1;
        gpu_breakout_reset_round(env);
    }
    if (env->num_balls < 0 || env->score == env->max_score) {
        *terminal = 1.0f;
        gpu_breakout_add_log(env);
        gpu_breakout_reset_state(env);
    }
}

__global__ void gpu_breakout_reset_kernel(GpuBreakout* envs, obs_t* observations,
        float* rewards, float* terminals, int num_envs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) {
        return;
    }
    gpu_breakout_reset_state(&envs[idx]);
    rewards[idx] = 0.0f;
    terminals[idx] = 0.0f;
    gpu_breakout_compute_observations(&envs[idx], observations + (long)idx * OBS_SIZE);
}

__global__ void gpu_breakout_step_kernel(GpuBreakout* envs, const float* actions,
        obs_t* observations, float* rewards, float* terminals, int start, int count) {
    int rel = blockIdx.x * blockDim.x + threadIdx.x;
    if (rel >= count) {
        return;
    }
    int idx = start + rel;
    GpuBreakout* env = &envs[idx];
    rewards[idx] = 0.0f;
    terminals[idx] = 0.0f;

    float action = actions[(long)idx * NUM_ATNS];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        gpu_breakout_step_frame(env, action, &rewards[idx], &terminals[idx]);
    }

    gpu_breakout_compute_observations(env, observations + (long)idx * OBS_SIZE);
}

__global__ void gpu_breakout_log_kernel(GpuBreakout* envs, float* out, int num_envs, int clear) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    float aggregate[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < num_envs; i++) {
        float* env_log = (float*)&envs[i].log;
        if (envs[i].log.n != 0.0f) {
            for (int j = 0; j < 5; j++) {
                aggregate[j] += env_log[j];
            }
        }
        if (clear) {
            envs[i].log.perf = 0.0f;
            envs[i].log.score = 0.0f;
            envs[i].log.episode_return = 0.0f;
            envs[i].log.episode_length = 0.0f;
            envs[i].log.n = 0.0f;
        }
    }
    for (int j = 0; j < 5; j++) {
        out[j] = aggregate[j];
    }
}

static void gpu_breakout_host_init(GpuBreakout* env, Dict* kwargs, int idx) {
    memset(env, 0, sizeof(GpuBreakout));
    env->num_agents = 1;
    env->frameskip = dict_get(kwargs, "frameskip");
    env->width = dict_get(kwargs, "width");
    env->height = dict_get(kwargs, "height");
    env->initial_paddle_width = dict_get(kwargs, "paddle_width");
    env->paddle_height = dict_get(kwargs, "paddle_height");
    env->ball_width = dict_get(kwargs, "ball_width");
    env->ball_height = dict_get(kwargs, "ball_height");
    env->brick_width = dict_get(kwargs, "brick_width");
    env->brick_height = dict_get(kwargs, "brick_height");
    env->brick_rows = dict_get(kwargs, "brick_rows");
    env->brick_cols = dict_get(kwargs, "brick_cols");
    env->initial_ball_speed = dict_get(kwargs, "initial_ball_speed");
    env->max_ball_speed = dict_get(kwargs, "max_ball_speed");
    env->paddle_speed = dict_get(kwargs, "paddle_speed");
    env->continuous = dict_get(kwargs, "continuous");
    env->rng = (unsigned int)idx;
    env->num_bricks = env->brick_rows * env->brick_cols;
    if (env->num_bricks <= 0 || env->num_bricks > BREAKOUT_GPU_MAX_BRICKS || env->num_bricks > OBS_SIZE - 10) {
        fprintf(stderr, "Breakout GPU env supports 1..%d bricks and OBS_SIZE-10 slots; got %d\n",
            BREAKOUT_GPU_MAX_BRICKS, env->num_bricks);
        exit(1);
    }
    env->num_balls = -1;
    gpu_breakout_generate_brick_positions(env);
}

static void* puf_gpu_env_create(int total_agents, Dict* env_kwargs) {
    GpuBreakout* host_envs = (GpuBreakout*)calloc((size_t)total_agents, sizeof(GpuBreakout));
    for (int i = 0; i < total_agents; i++) {
        gpu_breakout_host_init(&host_envs[i], env_kwargs, i);
    }

    GpuBreakout* gpu_envs = NULL;
    cudaMalloc((void**)&gpu_envs, (size_t)total_agents * sizeof(GpuBreakout));
    cudaMemcpy(gpu_envs, host_envs, (size_t)total_agents * sizeof(GpuBreakout), cudaMemcpyHostToDevice);
    free(host_envs);
    return gpu_envs;
}

static void puf_gpu_env_reset(void* raw_envs, obs_t* observations, float* rewards,
        float* terminals, int total_agents) {
    GpuBreakout* envs = (GpuBreakout*)raw_envs;
    gpu_breakout_reset_kernel<<<grid_size(total_agents), BLOCK_SIZE>>>(
        envs, observations, rewards, terminals, total_agents);
}

static void puf_gpu_env_step(void* raw_envs, const float* actions, obs_t* observations,
        float* rewards, float* terminals, int start, int count, cudaStream_t stream) {
    GpuBreakout* envs = (GpuBreakout*)raw_envs;
    gpu_breakout_step_kernel<<<grid_size(count), BLOCK_SIZE, 0, stream>>>(
        envs, actions, observations, rewards, terminals, start, count);
}

static int puf_gpu_env_log(void* raw_envs, int total_agents, float* gpu_log, Dict* out, int clear) {
    GpuBreakout* envs = (GpuBreakout*)raw_envs;
    float host_log[5] = {0};
    gpu_breakout_log_kernel<<<1, 1>>>(envs, gpu_log, total_agents, clear);
    cudaMemcpy(host_log, gpu_log, sizeof(host_log), cudaMemcpyDeviceToHost);

    float n = host_log[4];
    if (n == 0.0f) {
        return 0;
    }
    for (int i = 0; i < 5; i++) {
        host_log[i] /= n;
    }
    Log aggregate;
    memcpy(&aggregate, host_log, sizeof(aggregate));
    puf_log(&aggregate, out);
    dict_set(out, "n", n);
    return 1;
}

static void puf_gpu_env_close(void* raw_envs) {
    cudaFree(raw_envs);
}

#endif
