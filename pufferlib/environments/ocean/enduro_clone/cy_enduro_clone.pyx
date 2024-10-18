# cy_enduro_clone.pyx

cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

cimport numpy as cnp
from numpy cimport npy_intp

from libc.stdlib cimport calloc, free
from libc.stdint cimport int32_t, uint8_t


cdef extern from "enduro_clone.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return
        float episode_length
        float score

    ctypedef struct LogBuffer:
        pass

    LogBuffer* allocate_logbuffer(int size)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Car:
        int lane
        float y
        int passed

    ctypedef struct Enduro:
        float* observations
        int32_t actions
        float* rewards
        unsigned char* terminals
        unsigned char* truncateds
        LogBuffer* log_buffer
        Log log

        float width
        float height
        float hud_height
        float car_width
        float car_height
        int max_enemies
        float crash_noop_duration
        float day_length
        int initial_cars_to_pass
        float min_speed
        float max_speed

        float player_x
        float player_y
        float speed

        int score
        int day
        int step_count
        int numEnemies
        int carsToPass

        float collision_cooldown
        float action_height

        Car enemyCars[10]

    ctypedef struct Client:
        pass

    void init(Enduro* env)
    void reset(Enduro* env)
    void step(Enduro* env)

    Client* make_client(Enduro* env)
    void close_client(Client* client)
    void render(Client* client, Enduro* env)

cdef class CyEnduro:
    cdef:
        Enduro* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self,
                 cnp.ndarray[cnp.float32_t, ndim=2] observations,
                 cnp.ndarray[cnp.int32_t, ndim=1] actions,
                 cnp.ndarray[cnp.float32_t, ndim=1] rewards,
                 cnp.ndarray[cnp.uint8_t, ndim=1] terminals,
                 cnp.ndarray[cnp.uint8_t, ndim=1] truncateds,
                 int num_envs, float width, float height, float hud_height, float car_width, float car_height,
                 int max_enemies, float crash_noop_duration, float day_length,
                 int initial_cars_to_pass, float min_speed, float max_speed):

        cdef int i, j
        cdef float* observations_i_data
        cdef int32_t* actions_i
        cdef float* rewards_i
        cdef unsigned char* terminals_i
        cdef unsigned char* truncateds_i
        cdef Enduro* env

        self.num_envs = num_envs
        self.client = NULL
        self.envs = <Enduro*> calloc(num_envs, sizeof(Enduro))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        for i in range(num_envs):
            observations_i_data = &observations[i, 0]
            actions_i = &actions[i]
            rewards_i = &rewards[i]
            terminals_i = &terminals[i]
            truncateds_i = &truncateds[i]
            env = &self.envs[i]

            env.observations = observations_i_data
            env.actions = actions_i[0]
            env.rewards = rewards_i
            env.terminals = terminals_i
            env.truncateds = truncateds_i
            env.log_buffer = self.logs

            env.log.episode_return = 0.0
            env.log.episode_length = 0.0
            env.log.score = 0.0

            env.width = width
            env.height = height
            env.hud_height = hud_height
            env.car_width = car_width
            env.car_height = car_height
            env.max_enemies = max_enemies
            env.crash_noop_duration = crash_noop_duration
            env.day_length = day_length
            env.initial_cars_to_pass = initial_cars_to_pass
            env.min_speed = min_speed
            env.max_speed = max_speed

            env.player_x = 0.0
            env.player_y = 0.0
            env.speed = min_speed

            env.score = 0
            env.day = 1
            env.step_count = 0
            env.numEnemies = 0
            env.carsToPass = initial_cars_to_pass

            env.collision_cooldown = 0.0
            env.action_height = height - hud_height

            for j in range(max_enemies):
                env.enemyCars[j].lane = 0
                env.enemyCars[j].y = 0.0
                env.enemyCars[j].passed = 0

            init(env)

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            step(&self.envs[i])

    def render(self):
        cdef Enduro* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env)
        render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL
        free(self.envs)
        free_logbuffer(self.logs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return {'episode_return': log.episode_return,
                'episode_length': log.episode_length,
                'score': log.score}
