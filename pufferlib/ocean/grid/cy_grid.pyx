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

        Agent* agents;
        unsigned char* grid;

        int horizon;
        int vision;
        float speed;
        bint discretize;
        int obs_size;

        int tick;
        float episode_return;
        int max_size;

        unsigned char* observations;
        float* actions;
        float* rewards;
        float* dones;

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
        void reset(Grid* env)
        bint step(Grid* env)
        ctypedef struct Renderer
        Renderer* init_renderer(int cell_size, int width, int height)
        void render_global(Renderer*erenderer, Grid* env, float frac)
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
        int num_envs
        int num_maps
        int num_finished
        float sum_returns

    def __init__(self, cnp.ndarray observations, cnp.ndarray actions,
                 cnp.ndarray rewards, cnp.ndarray terminals, int num_maps, int num_envs,
                 int max_size, str task):

        self.num_envs = num_envs
        self.num_maps = num_maps
        self.client = NULL
        self.levels = <State*> calloc(num_maps, sizeof(State))
        self.envs = <Grid*> calloc(num_envs, sizeof(Grid))
        #self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i, size
        for i in range(num_envs):
            size = np.random.randint(5, max_size)
            self.envs[i] = Grid(
                observations = &observations[i, 0],
                actions = &actions[i],
                rewards = &rewards[i],
                dones = &terminals[i],
                max_size =  max_size,
                num_agents = 1,
                horizon = 2*size*size,
                width = size,
                height = size,
                vision = 5,
                speed = 1,
                discretize = True,
            )
            init_grid(&self.envs[i])

        cdef float difficulty
        for i in range(num_maps):
            difficulty = np.random.rand()
            create_maze_level(&self.envs[0], size, size, difficulty, i)
            init_state(&self.levels[i], max_size, 1)
            get_state(&self.envs[0], &self.levels[i])

    def reset(self):
        cdef int i, idx
        for i in range(self.num_envs):
            idx = rand() % self.num_maps
            set_state(&self.envs[i], &self.levels[idx])

    def step(self):
        cdef:
            int i, idx
            bint done
        
        for i in range(self.num_envs):
            done = step(&self.envs[i])
            if done:
                self.num_finished += 1
                self.sum_returns += self.envs[i].episode_return
                idx = rand() % self.num_maps
                set_state(&self.envs[i], &self.levels[idx])

    def get_returns(self):
        cdef float returns = self.sum_returns / self.num_finished
        self.sum_returns = 0
        self.num_finished = 0
        return returns

    def has_key(self):
        cdef int num_keys = 0
        cdef int i
        for i in range(self.num_envs):
            if self.envs[i].agents[0].keys[5] == 1:
                num_keys += 1

        return num_keys

    def render(self, int cell_size=16, int width=80, int height=45):
        if self.client == NULL:
            import os
            path = os.path.abspath(os.getcwd())
            print(path)
            c_path = os.path.join(os.sep, *__file__.split('/')[:-1])
            print(c_path)
            os.chdir(c_path)
            self.client = init_renderer(cell_size, width, height)
            os.chdir(path)

        render_global(self.client, &self.envs[0], 0)

    def close(self):
        if self.client != NULL:
            close_renderer(self.client)
            self.client = NULL

        #free_envs(self.envs, self.num_envs)
