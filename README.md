memorize this breakout reinforcement learning environment written in c, cython, and python. it runs very quickly because the expensive computations are all done effectively in c. breakout was a slow, crappy ale_py python env, running 1k sps on a 14900k + 4090 rtx. now it runs on cpu only >1 million sps/core because of writing more of it in c, or cython, but, crucially, not slow python, and not calling python. 

we need to copy the structure of the environment, using it to make the next atari game that will run at 1 million sps at least: enduro. 

the environment is being made from scratch. once it is laid out like the breakout env, and it runs (run command is python demo.py --env <env_name>
this env name will be, for clarity's sake, called racing, but should follow the rules i just itemized.


Step 1) remember all of the pasted code files. These will be copied to make a successful racing env (the enduro env from scratch).

Pause and wait for next prompt. Information will be provided as to logic, file structure, etc., of the racing env.

the directory it will be in is:
/puffertank/ocean_cython/PufferLib/pufferlib/environments/ocean/racing

comparison:
/puffertank/ocean_cython/PufferLib/pufferlib/environments/ocean/breakout


# environment.py (it is a factory file)
# /puffertank/ocean_cython/PufferLib/pufferlib/environments/ocean/environment.py
import pufferlib.emulation
import pufferlib.postprocess

# breakout is comparison
def make_breakout(num_envs=1):
    from .breakout import breakout
    return breakout.MyBreakout(num_envs=num_envs)

def make_racing(num_envs=1):
    from .racing import racing
    return racing.MyRacing(num_envs=num_envs)

MAKE_FNS = {
    'my_breakout': make_breakout,
    'my_racing': make_racing,
}

    if name in MAKE_FNS:
        return MAKE_FNS[name]
    else:
        raise ValueError(f'Invalid environment name: {name}')


# setup.py 
# /puffertank/ocean_cython/PufferLib/setup.py
from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize
import numpy...
extension_paths = [
    'pufferlib/environments/ocean/breakout/cy_breakout',
    'pufferlib/environments/ocean/racing/cy_racing',
]

extensions = [Extension(
    path.replace('/', '.'),
    [path + '.pyx'],
    include_dirs=[numpy.get_include(), 'raylib-5.0_linux_amd64/include'],
    library_dirs=['raylib-5.0_linux_amd64/lib'],
    libraries=["raylib"],
    runtime_library_dirs=["raylib-5.0_linux_amd64/lib"],
    extra_compile_args=['-DPLATFORM_DESKTOP'],
) for path in extension_paths]
 
setup(
    name="pufferlib",
    description="PufferAI Library"
    "PufferAI's library of RL tools and utilities",
    long_description_content_type="text/markdown",
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy==1.23.3',
        'opencv-python==3.4.17.63',
        'cython>=3.0.0',
        'rich',
        'rich_argparse',
        f'gym<={GYM_VERSION}',
        f'gymnasium<={GYMNASIUM_VERSION}',
        f'pettingzoo<={PETTINGZOO_VERSION}',
        'shimmy[gym-v21]',
        'psutil==5.9.5',
        'pynvml',
        'imageio',
    ],
    extras_require={
        'docs': docs,
        'ray': ray,
        'cleanrl': cleanrl,
        'common': common,
        **environments,
    },
    ext_modules = cythonize([
        "pufferlib/extensions.pyx",
        "c_gae.pyx",
        "pufferlib/environments/ocean/grid/c_grid.pyx",
        "pufferlib/environments/ocean/snake/c_snake.pyx",
        "pufferlib/environments/ocean/moba/c_moba.pyx",
        "pufferlib/environments/ocean/moba/puffernet.pyx",
        "pufferlib/environments/ocean/moba/c_precompute_pathing.pyx",
        *extensions,
    ], 
       #nthreads=6,
       #annotate=True,
       #compiler_directives={'profile': True},# annotate=True
    ),
    include_dirs=[numpy.get_include(), 'raylib-5.0_linux_amd64/include'],
    python_requires=">=3.8",
    license="MIT",
    author="Joseph Suarez",
    author_email="jsuarez@puffer.ai",
    url="https://github.com/PufferAI/PufferLib",
    keywords=["Puffer", "AI", "RL", "Reinforcement Learning"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

// breakout.c
// /puffertank/ocean_cython/PufferLib/pufferlib/environments/ocean/breakout/breakout.c
#include "breakout.h"

int main() {
    CBreakout env = {
        .frameskip = 4,
        .width = 576,
        .height = 330,
        .paddle_width = 62,
        .paddle_height = 8,
        .ball_width = 6,
        .ball_height = 7,
        .brick_width = 32,
        .brick_height = 12,
        .brick_rows = 6,
        .brick_cols = 18,
    };
    allocate(&env);
    reset(&env);
 
    Client* client = make_client(&env);

    while (!WindowShouldClose()) {
        // User can take control of the paddle
        env.actions[0] = 0;
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = 1;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 2;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 3;

        step(&env);
        render(client, &env);
    }
    close_client(client);
    free_allocated(&env);
    return 0;
}

// breakout.h
// /puffertank/ocean_cython/PufferLib/pufferlib/environments/ocean/breakout/breakout.h
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "raylib.h"

#define NOOP 0
#define FIRE 1
#define LEFT 2
#define RIGHT 3
#define HALF_MAX_SCORE 432
#define MAX_SCORE 864
#define HALF_PADDLE_WIDTH 31
#define Y_OFFSET 50
#define TICK_RATE 1.0f/60.0f

typedef struct CBreakout CBreakout;
struct CBreakout {
    float* observations;
    unsigned char* actions;
    float* rewards;
    unsigned char* dones;
    int score;
    float episode_return;
    float paddle_x;
    float paddle_y;
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    float* brick_x;
    float* brick_y;
    float* brick_states;
    int balls_fired;
    float paddle_width;
    float paddle_height;
    float ball_speed;
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
    int frameskip;
};

void generate_brick_positions(CBreakout* env) {
    for (int row = 0; row < env->brick_rows; row++) {
        for (int col = 0; col < env->brick_cols; col++) {
            int idx = row * env->brick_cols + col;
            env->brick_x[idx] = col*env->brick_width;
            env->brick_y[idx] = row*env->brick_height + Y_OFFSET;
        }
    }
}

void init(CBreakout* env) {
    env->num_bricks = env->brick_rows * env->brick_cols;
    assert(env->num_bricks > 0);

    env->brick_x = (float*)calloc(env->num_bricks, sizeof(float));
    env->brick_y = (float*)calloc(env->num_bricks, sizeof(float));
    env->brick_states = (float*)calloc(env->num_bricks, sizeof(float));
    env->num_balls = -1;
    generate_brick_positions(env);
}

void allocate(CBreakout* env) {
    init(env);
    env->observations = (float*)calloc(11 + env->num_bricks, sizeof(float));
    env->actions = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void free_initialized(CBreakout* env) {
    free(env->brick_x);
    free(env->brick_y);
    free(env->brick_states);
}

void free_allocated(CBreakout* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_initialized(env);
}

void compute_observations(CBreakout* env) {
    env->observations[0] = env->paddle_x;
    env->observations[1] = env->paddle_y;
    env->observations[2] = env->ball_x;
    env->observations[3] = env->ball_y;
    env->observations[4] = env->ball_vx;
    env->observations[5] = env->ball_vy;
    env->observations[6] = env->balls_fired;
    env->observations[8] = env->num_balls;
    env->observations[10] = env->paddle_width;
    for (int i = 0; i < env->num_bricks; i++) {
        env->observations[11 + i] = env->brick_states[i];
    }
}

bool check_collision_discrete(float x, float y, int width, int height,
        float other_x, float other_y, int other_width, int other_height) {
    if (x + width <= other_x || other_x + other_width <= x) {
        return false;
    }
    if (y + height <= other_y || other_y + other_height <= y) {
        return false;
    }
    return true;
}

bool handle_paddle_ball_collisions(CBreakout* env) {
    float base_angle = M_PI / 4.0f;

    // Check if ball is above the paddle
    if (env->ball_y + env->ball_height < env->paddle_y) {
        return false;
    }

    // Check for collision
    if (check_collision_discrete(env->paddle_x, env->paddle_y,
            env->paddle_width, env->paddle_height, env->ball_x,
            env->ball_y, env->ball_width, env->ball_height)) {
        float relative_intersection = ((env->ball_x +
            env->ball_width / 2) - env->paddle_x) / env->paddle_width;
        float angle = -base_angle + relative_intersection * 2 * base_angle;
        env->ball_vx = sin(angle) * env->ball_speed * TICK_RATE;
        env->ball_vy = -cos(angle) * env->ball_speed * TICK_RATE;
        env->hits += 1;
        if (env->hits % 4 == 0 && env->hits <= 12) {
            env->ball_speed += 64;
        }
        if (env->score == HALF_MAX_SCORE) {
            env->brick_states[0] = 0.0;
        }
        return true;
    }
    return false;
}

bool handle_wall_ball_collisions(CBreakout* env) {
    if (env->ball_x > 0 && env->ball_x
            + env->ball_width < env->width && env->ball_y > 0) {
        return false;
    }

    // Left Wall Collision
    if (check_collision_discrete(-Y_OFFSET, 0, Y_OFFSET, env->height,
            env->ball_x, env->ball_y, env->ball_width, env->ball_height)) {
        env->ball_x = 0;
        env->ball_vx *= -1;
        return true;
    }

    // Top Wall Collision
    if (check_collision_discrete(0, -Y_OFFSET, env->width, Y_OFFSET,
            env->ball_x, env->ball_y, env->ball_width, env->ball_height)) {
        env->ball_y = 0;
        env->ball_vy *= -1;
        env->paddle_width = HALF_PADDLE_WIDTH;
        return true;
    }

    // Right Wall Collision
    if (check_collision_discrete(env->width, 0, Y_OFFSET, env->height,
            env->ball_x, env->ball_y, env->ball_width, env->ball_height)) {
        env->ball_x = env->width - env->ball_width;
        env->ball_vx *= -1;
        return true;
    }

    return false;
}

bool handle_brick_ball_collisions(CBreakout* env) {
    if (env->ball_y > env->brick_y[env->num_bricks-1] + env->brick_height) {
        return false;
    }
    
    // Loop over bricks in reverse to check lower bricks first
    for (int brick_idx = env->num_bricks - 1; brick_idx >= 0; brick_idx--) {
        if (env->brick_states[brick_idx] == 1.0) {
            continue;
        }
        if (check_collision_discrete(env->brick_x[brick_idx],
                env->brick_y[brick_idx], env->brick_width, env->brick_height,
                env->ball_x, env->ball_y, env->ball_width, env->ball_height)) {
            env->brick_states[brick_idx] = 1.0;
            float score = 7 - 3 * (brick_idx / env->brick_cols / 2);
            env->rewards[0] += score;
            env->score += score;

            // Determine collision direction
            if (env->ball_y + env->ball_height <= env->brick_y[brick_idx] + (env->brick_height / 2)) {
                // Hit was from below the brick
                env->ball_vy *= -1;
                return true;
            } else if (env->ball_y >= env->brick_y[brick_idx] + (env->brick_height / 2)) {
                // Hit was from above the brick
                env->ball_vy *= -1;
                return true;
            } else if (env->ball_x + env->ball_width <= env->brick_x[brick_idx] + (env->brick_width / 2)) {
                // Hit was from the left
                env->ball_vx *= -1;
                return true;
            } else if (env->ball_x >= env->brick_x[brick_idx] + (env->brick_width / 2)) {
                // Hit was from the right
                env->ball_vx *= -1;
                return true;
            }
        }
    }
    return false;
}

void reset(CBreakout* env) {
    if (env->num_balls == -1 || env->score == MAX_SCORE) {
        env->score = 0;
        env->num_balls = 5;
        for (int i = 0; i < env->num_bricks; i++) {
            env->brick_states[i] = 0.0;
        }
        env->hits = 0;
        env->ball_speed = 256;
        env->paddle_width = 2 * HALF_PADDLE_WIDTH;
    }

    env->dones[0] = 0;
    env->balls_fired = 0;

    env->paddle_x = env->width / 2.0 - env->paddle_width / 2;
    env->paddle_y = env->height - env->paddle_height - 10;

    env->ball_x = env->paddle_x + (env->paddle_width / 2 - env->ball_width / 2);
    env->ball_y = env->height / 2 - 30;

    env->ball_vx = 0.0;
    env->ball_vy = 0.0;
}

void step(CBreakout* env) {
    env->rewards[0] = 0.0;
    int action = env->actions[0];

    for (int i = 0; i < env->frameskip; i++) {
        if (action == FIRE && env->balls_fired == 0) {
            env->balls_fired = 1;
            float direction = M_PI / 3.25f;

            if (rand() % 2 == 0) {
                env->ball_vx = sin(direction) * env->ball_speed * TICK_RATE;
                env->ball_vy = cos(direction) * env->ball_speed * TICK_RATE;
            } else {
                env->ball_vx = -sin(direction) * env->ball_speed * TICK_RATE;
                env->ball_vy = cos(direction) * env->ball_speed * TICK_RATE;
            }
        } else if (action == LEFT) {
            env->paddle_x -= 620 * TICK_RATE;
            env->paddle_x = fmaxf(0, env->paddle_x);
        } else if (action == RIGHT) {
            env->paddle_x += 620 * TICK_RATE;
            env->paddle_x = fminf(env->width - env->paddle_width, env->paddle_x);
        }

        handle_brick_ball_collisions(env);
        handle_paddle_ball_collisions(env);
        handle_wall_ball_collisions(env);

        env->ball_x += env->ball_vx;
        env->ball_y += env->ball_vy;

        if (env->ball_y >= env->paddle_y + env->paddle_height) {
            env->num_balls -= 1;
            env->dones[0] = 1;
        }
        if (env->score == MAX_SCORE) {
            env->dones[0] = 1;
        }
        if (env->dones[0] == 1) {
            env->episode_return = env->score;
            reset(env);
        }
    }
    compute_observations(env);
}

Color BRICK_COLORS[6] = {RED, ORANGE, YELLOW, GREEN, SKYBLUE, BLUE};

typedef struct Client Client;
struct Client {
    float width;
    float height;
};

Client* make_client(CBreakout* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;

    InitWindow(env->width, env->height, "PufferLib Ray Breakout");
    SetTargetFPS(15);

    //sound_path = os.path.join(*self.__module__.split(".")[:-1], "hit.wav")
    //self.sound = rl.LoadSound(sound_path.encode())

    return client;
}

void render(Client* client, CBreakout* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    DrawRectangle(env->paddle_x, env->paddle_y,
        env->paddle_width, env->paddle_height, (Color){0, 255, 255, 255});
    DrawRectangle(env->ball_x, env->ball_y,
        env->ball_width, env->ball_height, WHITE);

    for (int row = 0; row < env->brick_rows; row++) {
        for (int col = 0; col < env->brick_cols; col++) {
            int brick_idx = row * env->brick_cols + col;
            if (env->brick_states[brick_idx] == 1) {
                continue;
            }
            int x = env->brick_x[brick_idx];
            int y = env->brick_y[brick_idx];
            Color brick_color = BRICK_COLORS[row];
            DrawRectangle(x, y, env->brick_width, env->brick_height, brick_color);
        }
    }

    DrawText(TextFormat("Score: %i", env->score), 10, 10, 20, WHITE);
    DrawText(TextFormat("Balls: %i", env->num_balls), client->width - 80, 10, 20, WHITE);
    EndDrawing();

    //PlaySound(client->sound);
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

# breakout.py
# /puffertank/ocean_cython/PufferLib/pufferlib/environments/ocean/breakout/breakout.py

import numpy as np
import gymnasium

from raylib import rl

import pufferlib
from pufferlib.environments.ocean.breakout.cy_breakout import CyBreakout

class MyBreakout(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=128,
            frameskip=4, width=576, height=330,
            paddle_width=62, paddle_height=8,
            ball_width=6, ball_height=6,
            brick_width=32, brick_height=12,
            brick_rows=6, brick_cols=18):
        super().__init__()

        # env
        self.num_envs = num_envs
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval

        # sim hparams (px, px/tick)
        self.frameskip = frameskip
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_width = ball_width 
        self.ball_height = ball_height
        self.brick_width = brick_width
        self.brick_height = brick_height
        self.brick_rows = brick_rows
        self.brick_cols = brick_cols

        # spaces
        self.num_obs = 11 + brick_rows*brick_cols
        self.num_act = 4
        self.observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(self.num_obs,), dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.action_space = gymnasium.spaces.Discrete(self.num_act)
        self.single_action_space = self.action_space
        self.human_action = None

        self.emulated = None
        self.done = False
        self.buf = pufferlib.namespace(
            observations = np.zeros((self.num_agents, self.num_obs,), dtype=np.float32),
            rewards = np.zeros(self.num_agents, dtype=np.float32),
            terminals = np.zeros(self.num_agents, dtype=np.bool),
            truncations = np.zeros(self.num_agents, dtype=bool),
            masks = np.ones(self.num_agents, dtype=bool),
        )
        self.actions = np.zeros(self.num_agents, dtype=np.uint32)
        self.terminals_uint8 = np.zeros(self.num_agents, dtype=np.uint8)

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs = []

        for i in range(self.num_envs):
            # TODO: since single agent, could we just pass values by reference instead of (1,) array?
            self.c_envs.append(CyBreakout(self.frameskip, self.actions[i:i+1],
                self.buf.observations[i], self.buf.rewards[i:i+1], self.buf.terminals[i:i+1],
                self.width, self.height, self.paddle_width, self.paddle_height,
                self.ball_width, self.ball_height, self.brick_width, self.brick_height,
                self.brick_rows, self.brick_cols))
            self.c_envs[i].reset()

        return self.buf.observations, {}

    def step(self, actions):
        self.actions[:] = actions
        for i in range(self.num_envs):
            self.c_envs[i].step()

        # TODO: hacky way to convert uint8 to bool
        self.buf.terminals[:] = self.terminals_uint8.astype(bool)
        self.tick += 1

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, {})

    def render(self):
        self.c_envs[0].render()

def test_performance(timeout=10, atn_cache=1024):
    env = MyPong(num_envs=1000)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 2, (atn_cache, env.num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_envs * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()

# build_local.sh
# /puffertank/ocean_cython/PufferLib/pufferlib/environments/ocean/breakout/build_local.sh
clang -Wall -Wuninitialized -Wmisleading-indentation -fsanitize=address -ferror-limit=3 -g -o breakoutgame breakout.c -I./raylib-5.0_linux_amd64/include/ -L./raylib-5.0_linux_amd64/lib/ -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 -DPLATFORM_DESKTOP  

# build_web.sh
# /puffertank/ocean_cython/PufferLib/pufferlib/environments/ocean/breakout/build_web.sh

# cy_breakout.pyx
# /puffertank/ocean_cython/PufferLib/pufferlib/environments/ocean/breakout/cy_breakout.pyx
cimport numpy as cnp
from libc.stdlib cimport free

cdef extern from "breakout.h":
    ctypedef struct CBreakout:
        float* observations
        unsigned char* actions
        float* rewards
        unsigned char* dones
        int score
        float episode_return
        float paddle_x
        float paddle_y
        float ball_x
        float ball_y
        float ball_vx
        float ball_vy
        float* brick_x
        float* brick_y
        float* brick_states
        int balls_fired
        float paddle_width
        float paddle_height
        float ball_speed
        int hits
        int width
        int height
        int num_bricks
        int brick_rows
        int brick_cols
        int ball_width
        int ball_height
        int brick_width
        int brick_height
        int num_balls
        int frameskip

    ctypedef struct Client

    void init(CBreakout* env)
    void free_initialized(CBreakout* env)

    Client* make_client(CBreakout* env)
    void close_client(Client* client)
    void render(Client* client, CBreakout* env)
    void reset(CBreakout* env)
    void step(CBreakout* env)

cdef class CyBreakout:
    cdef:
        CBreakout env
        Client* client

    def __init__(self, int frameskip, cnp.ndarray actions,
            cnp.ndarray observations, cnp.ndarray rewards, cnp.ndarray dones,
            int width, int height, float paddle_width, float paddle_height,
            int ball_width, int ball_height, int brick_width, int brick_height,
            int brick_rows, int brick_cols):
        self.env = CBreakout(
            observations=<float*> observations.data,
            actions=<unsigned char*> actions.data,
            rewards=<float*> rewards.data,
            dones=<unsigned char*> dones.data,
            width=width,
            height=height,
            paddle_width=paddle_width,
            paddle_height=paddle_height,
            ball_width=ball_width,
            ball_height=ball_height,
            brick_width=brick_width,
            brick_height=brick_height,
            brick_rows=brick_rows,
            brick_cols=brick_cols,
            frameskip=frameskip,
        )
        init(&self.env)
        self.client = NULL

    def reset(self):
        reset(&self.env)

    def step(self):
        step(&self.env)

    def render(self):
        if self.client == NULL:
            self.client = make_client(&self.env)

        render(self.client, &self.env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free_initialized(&self.env)


/puffertank/ocean_cython/PufferLib/pufferlib/environments/ocean/racing/

In this directory, we'll have several files:

    environment.py (the factory file)
    cy_racing.pyx (Cython bindings to C code)
    racing.c (Core C environment)
    racing.h (C header)
    racing.py (Python interface)
    build_local.sh and build_web.sh (Build scripts for desktop and web)





     obtain the values of important rendering params here, e.g. total screen size (160x210), total action screen size (there is a sort of # black hud overlay that remains constant and black continuously. a scoreboard is superimposed on it; the action screen, which is where the player, road, # scenery, sky are actively rendered, occupies x=8 through x=160 and y=0 through y=155.) # the scoreboard occupies the space x=48 through x=112 and y=161 through y=191. on the scoreboard is the total score, represented as a 5-digit number starting at 00000    and progressing steadily according to speed, rendered like 00001   00002  00003. even at no speed, the score increases slowly. score occupies space from x=56 through x=104 and y=162 through y=173.
# the number of cars left to pass to complete the day is rendered from (72,179) through (104, 188). displayed on it is are 4 digits, the leftmost being a little car icon, then the number of cars left to pass for the day. this number starts at 200 for day 1. it counts down each time a (new) car is passed. 
# the day the player is on is rendered from (56, 179) through (64, 188). this is the requisite area that the rendering of a single digit occupies. only 1 digit is displayed, starting at 1 at the beginning of the game. as days are completed, this number advances in units of 1.

Game over conditions: the requisite number of cars must be passed before the end of the current day, or else the game ends. 
If the requisite number of cars is passed, then, when the current day ends, the next day begins, and the day counter increments by one.

