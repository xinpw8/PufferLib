# cy_blastar.pyx

import numpy
cimport numpy as cnp
cnp.import_array()
from libc.stdlib cimport calloc, free
from libc.math cimport fabs
from libc.string cimport memset

cdef extern from "blastar.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Bullet:
        float x
        float y
        float last_x
        float last_y
        bint active
        double travel_time

    ctypedef struct Enemy:
        float x
        float y
        float last_x
        float last_y
        bint active
        bint attacking
        int direction
        int width
        int height
        int crossed_screen
        Bullet bullet

    ctypedef struct Player:
        float x
        float y
        float last_x
        float last_y
        int score
        int lives
        Bullet bullet
        bint bulletFired
        bint playerStuck
        float explosion_timer

    ctypedef struct Log:
        float episode_return
        float episode_length
        float score
        float lives
        float vertical_closeness_rew
        float fired_bullet_rew
        int kill_streak
        float flat_below_enemy_rew
        float hit_enemy_with_bullet_rew
        float avg_score_difference

    ctypedef struct LogBuffer:
        Log* logs
        int length
        int idx

    ctypedef struct BlastarEnv:
        int screen_width
        int screen_height
        float player_width
        float player_height
        float last_bullet_distance
        bint gameOver
        int tick
        int playerExplosionTimer
        int enemyExplosionTimer
        int max_score
        int bullet_travel_time
        int kill_streak
        int enemy_respawns
        Player player
        Enemy enemy
        Bullet bullet
        float* observations       # [27]
        int* actions              # [6]
        float* rewards            # [1]
        unsigned char* terminals  # [1]
        LogBuffer* log_buffer
        Log log

    LogBuffer* allocate_logbuffer(int size)
    void free_logbuffer(LogBuffer* buffer)
    void add_log(LogBuffer* logs, Log* log)
    Log aggregate_and_clear(LogBuffer* logs)

    void init(BlastarEnv *env)
    void c_reset(BlastarEnv *env)
    void c_step(BlastarEnv *env)
    void close_client(Client* client)
    void c_render(Client* client, BlastarEnv* env)

    ctypedef struct Client:
        pass

    Client* make_client(BlastarEnv* env)

cdef class CyBlastar:
    cdef BlastarEnv* envs
    cdef Client* client
    cdef LogBuffer* logs
    cdef int num_envs

    def __init__(self,
                 float[:, :] observations,
                 int[:] actions,
                 float[:] rewards,
                 unsigned char[:] terminals,
                 int num_envs):

        self.num_envs = num_envs
        self.client = NULL
        self.envs = <BlastarEnv*> calloc(num_envs, sizeof(BlastarEnv))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i
        for i in range(num_envs):
            self.envs[i].observations = &observations[i, 0]
            self.envs[i].actions = &actions[i]
            self.envs[i].rewards = &rewards[i]
            self.envs[i].terminals = &terminals[i]
            self.envs[i].log_buffer = self.logs
            init(&self.envs[i])

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef BlastarEnv* env = &self.envs[0]
        if self.client == NULL:
            import os
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
            self.client = make_client(env)
            os.chdir(cwd)

        c_render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)
        free(self.logs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log