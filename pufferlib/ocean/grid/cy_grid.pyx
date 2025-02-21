# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: profile=True

from libc.stdlib cimport calloc, free, rand

cdef extern from "grid.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;
        float score;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Agent:
        float y;
        float x;
        float prev_y;
        float prev_x;
        float spawn_y;
        float spawn_x;
        int color;
        float direction;
        int held;

    ctypedef struct Grid:
        int width;
        int height;
        int num_agents;
        int horizon;
        int vision;
        float speed;
        int obs_size;
        int max_size;
        bint discretize;
        Log log;
        LogBuffer* log_buffer;
        Agent* agents;
        unsigned char* grid;
        int* counts;
        unsigned char* observations;
        float* actions;
        float* rewards;
        unsigned char* dones;

    ctypedef struct State:
        int width;
        int height;
        int num_agents;
        Agent* agents;
        unsigned char* grid;

    cdef:
        void create_maze_level(Grid* env, int width, int height, float difficulty, int seed)
        void load_locked_room_env(unsigned char* observations,
            unsigned int* actions, float* rewards, float* dones)
        void init_grid(Grid* env)
        void reset(Grid* env, int seed)
        void compute_observations(Grid* env)
        bint step(Grid* env)
        ctypedef struct Renderer
        Renderer* init_renderer(int cell_size, int width, int height)
        void render_global(Renderer*erenderer, Grid* env, float frac, float overlay)
        void clear_overlay(Renderer* renderer)
        void close_renderer(Renderer* renderer)
        void init_state(State* state, int max_size, int num_agents)
        void free_state(State* state)
        void get_state(Grid* env, State* state)
        void set_state(Grid* env, State* state)

import numpy as np
cimport numpy as cnp

cdef class CGrid:
    cdef:
        Grid* envs
        State* levels
        Renderer* client
        LogBuffer* logs
        int num_envs
        int num_maps
        int max_size

    def __init__(self, unsigned char[:, :] observations, float[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs, int num_maps,
            int size, int max_size):

        self.num_envs = num_envs
        self.num_maps = num_maps
        if size > max_size:
            max_size = size

        self.max_size = max_size

        self.client = NULL
        self.levels = <State*> calloc(num_maps, sizeof(State))
        self.envs = <Grid*> calloc(num_envs, sizeof(Grid))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i
        for i in range(num_envs):
            self.envs[i] = Grid(
                observations = &observations[i, 0],
                actions = &actions[i],
                rewards = &rewards[i],
                dones = &terminals[i],
                log_buffer = self.logs,
                max_size =  max_size,
                num_agents = 1,
                vision = 5,
                speed = 1,
                discretize = True,
            )
            init_grid(&self.envs[i])

        cdef float difficulty
        cdef int sz
        for i in range(num_maps):

            # RNG or fixed size
            if size == -1:
                sz = np.random.randint(5, max_size)
            else:
                sz = size

            if sz % 2 == 0:
                sz -= 1

            difficulty = np.random.rand()
            create_maze_level(&self.envs[0], sz, sz, difficulty, i)
            init_state(&self.levels[i], max_size, 1)
            get_state(&self.envs[0], &self.levels[i])

    def reset(self):
        cdef int i, idx
        for i in range(self.num_envs):
            idx = rand() % self.num_maps
            reset(&self.envs[i], i)
            set_state(&self.envs[i], &self.levels[idx])
            compute_observations(&self.envs[i])

    def step(self):
        cdef:
            int i, idx
            bint done
        
        for i in range(self.num_envs):
            done = step(&self.envs[i])
            if done:
                idx = rand() % self.num_maps
                reset(&self.envs[i], i)
                set_state(&self.envs[i], &self.levels[idx])

                if i == 0 and self.client != NULL:
                    clear_overlay(self.client)

    def render(self, int cell_size=16, float overlay=0.0):
        if self.client == NULL:
            import os
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            self.client = init_renderer(cell_size, self.max_size, self.max_size)
            os.chdir(cwd)

        render_global(self.client, &self.envs[0], 0, overlay)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log

    def close(self):
        if self.client != NULL:
            close_renderer(self.client)
            self.client = NULL

        #free_envs(self.envs, self.num_envs)
