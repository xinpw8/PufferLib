# cy_blastar.pyx

import numpy
cimport numpy as cnp
cnp.import_array()
from libc.stdlib cimport calloc, free
from libc.math cimport fabs
from libc.string cimport memset

cdef extern from "blastar_env.h":
    int LOG_BUFFER_SIZE
    int REWARD_BUFFER_SIZE

    # Define the Bullet struct
    ctypedef struct Bullet:
        float x
        float y
        float last_x
        float last_y
        bint active
        double travel_time

    # Define the Enemy struct
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

    # Define the Player struct
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

    # Define the Log struct
    ctypedef struct Log:
        float episode_return
        float episode_length
        float score
        float lives
        float bullet_travel_rew
        float fired_bullet_rew
        float bullet_distance_to_enemy_rew
        float gradient_penalty_rew
        float flat_below_enemy_rew
        float danger_zone_penalty_rew
        float crashing_penalty_rew
        float hit_enemy_with_bullet_rew
        float hit_by_enemy_bullet_penalty_rew
        int enemy_crossed_screen
        float bad_guy_score

    # Define the LogBuffer struct
    ctypedef struct LogBuffer:
        Log* logs
        int length
        int idx

    ctypedef struct RewardBuffer:
        float* rewards
        int size
        int idx

    # Define the BlastarEnv struct
    ctypedef struct BlastarEnv:
        int screen_width
        int screen_height
        float player_width
        float player_height
        float enemy_width
        float enemy_height
        float player_bullet_width
        float player_bullet_height
        float enemy_bullet_width
        float enemy_bullet_height
        float last_bullet_distance
        bint gameOver
        int tick
        int playerExplosionTimer
        int enemyExplosionTimer
        int max_score
        int bullet_travel_time
        int kill_streak
        float bad_guy_score
        Player player
        Enemy enemy
        Bullet bullet
        RewardBuffer* reward_buffer
        float* observations       # [25]
        int* actions              # [6]
        float* rewards            # [1]
        unsigned char* terminals  # [1]
        LogBuffer* log_buffer
        Log log

    # Function declarations
    LogBuffer* allocate_logbuffer(int size)
    void free_logbuffer(LogBuffer* buffer)
    void add_log(LogBuffer* logs, Log* log)
    Log aggregate_and_clear(LogBuffer* logs)

    RewardBuffer* allocate_reward_buffer(int size)
    void free_reward_buffer(RewardBuffer* buffer)

    void init_blastar(BlastarEnv *env)
    void reset_blastar(BlastarEnv *env)
    void c_step(BlastarEnv *env)
    void close_client(Client* client)
    void c_render(Client* client, BlastarEnv* env)

    # Rendering functions
    ctypedef struct Client:
        pass

    Client* make_client(BlastarEnv* env)
    void close_client(Client* client)
    void c_render(Client* client, BlastarEnv* env)

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
            self.envs[i].reward_buffer = allocate_reward_buffer(REWARD_BUFFER_SIZE)

            # Initialize the environment without overwriting RL pointers
            init_blastar(&self.envs[i])

        assert self.envs != NULL, "Failed to allocate memory for BlastarEnv instances."
        assert self.logs != NULL, "Failed to allocate memory for LogBuffer."

        for i in range(self.num_envs):
            assert self.envs[i].observations != NULL, f"Observation buffer for env {i} is NULL."
            assert self.envs[i].actions != NULL, f"Action buffer for env {i} is NULL."
            assert self.envs[i].rewards != NULL, f"Reward buffer for env {i} is NULL."
            assert self.envs[i].terminals != NULL, f"Terminal buffer for env {i} is NULL."
            assert self.envs[i].reward_buffer != NULL, f"RewardBuffer for env {i} is NULL."
        
        


    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            assert self.envs[i].observations != NULL, f"Observation buffer for env {i} is NULL after reset."
            assert self.envs[i].actions != NULL, f"Action buffer for env {i} is NULL after reset."
            assert self.envs[i].rewards != NULL, f"Reward buffer for env {i} is NULL after reset."
            assert self.envs[i].terminals != NULL, f"Terminal buffer for env {i} is NULL after reset."
            assert self.envs[i].reward_buffer != NULL, f"RewardBuffer for env {i} is NULL after reset."
            if self.envs[i].reward_buffer != NULL:
                free_reward_buffer(self.envs[i].reward_buffer)
            self.envs[i].reward_buffer = allocate_reward_buffer(REWARD_BUFFER_SIZE)
            reset_blastar(&self.envs[i])


    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef BlastarEnv* env = &self.envs[0]
        if self.client == NULL and self.num_envs > 0:
            import os
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            self.client = make_client(env)
            os.chdir(cwd)

        if self.client != NULL:
            c_render(self.client, &self.envs[0])

    def close(self):
        if self.envs != NULL:
            for i in range(self.num_envs):
                assert self.envs[i].reward_buffer != NULL, f"RewardBuffer for env {i} is already NULL."
                assert self.envs[i].observations != NULL, f"Observation buffer for env {i} is already NULL."
                if self.envs[i].reward_buffer != NULL:
                    free_reward_buffer(self.envs[i].reward_buffer)
                    self.envs[i].reward_buffer = NULL
            assert self.logs != NULL, "LogBuffer is already NULL."
            free(self.envs)
            self.envs = NULL

        if self.logs != NULL:
            free_logbuffer(self.logs)
            self.logs = NULL

        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return {
            'episode_return': log.episode_return,
            'episode_length': log.episode_length,
            'score': log.score,
            'lives': log.lives,
            'bullet_travel_rew': log.bullet_travel_rew,
            'fired_bullet_rew': log.fired_bullet_rew,
            'bullet_distance_to_enemy_rew': log.bullet_distance_to_enemy_rew,
            'gradient_penalty_rew': log.gradient_penalty_rew,
            'flat_below_enemy_rew': log.flat_below_enemy_rew,
            'danger_zone_penalty_rew': log.danger_zone_penalty_rew,
            'crashing_penalty_rew': log.crashing_penalty_rew,
            'hit_enemy_with_bullet_rew': log.hit_enemy_with_bullet_rew,
            'hit_by_enemy_bullet_penalty_rew': log.hit_by_enemy_bullet_penalty_rew,
            'enemy_crossed_screen': log.enemy_crossed_screen,
            'bad_guy_score': log.bad_guy_score
        }
        